"""
Shared Gemini Embedding 2 client for the retrieval pipeline.

gemini-embedding-2 maps text, images, video and audio into one unified
embedding space, so an image (+caption) query embeds in the same space as a
text article — no separate image encoder needed. Output is L2-normalized by
the model (including truncated dims), so step 3 cosine similarity works as-is.

Unlike older models, gemini-embedding-2 has NO `task_type` parameter — the
retrieval task is given as a text prefix instead:
  - documents : "title: {title} | text: {content}"
  - queries   : "task: search result | query: {text}"
Applying these at both index and query time improves asymmetric retrieval.

Auth (two modes, auto-detected by build_client):
  - Vertex AI (token-based, no API key): set
        GOOGLE_GENAI_USE_VERTEXAI=true
        GOOGLE_CLOUD_PROJECT=<your-gcp-project>
        GOOGLE_CLOUD_LOCATION=<region, e.g. us-central1>   (optional, default global)
    and provide credentials via Application Default Credentials, i.e. either
        gcloud auth application-default login
    or  GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
  - API key (AI Studio): set GEMINI_API_KEY (or GOOGLE_API_KEY).
Install: pip install google-genai

Two access patterns (gemini-embedding-2 embeds ONE content per call):
  - embed_texts(...)      : one call per string, run concurrently with a thread
                            pool and paced by a per-minute token-bucket limiter.
  - embed_multimodal(...) : one call per (text, image) item, run concurrently
                            with a thread pool (each Content = one vector).
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

MODEL          = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2")
OUTPUT_DIM     = 1536    # 768 / 1536 / 3072; model auto-normalizes truncated dims
MAX_RETRIES    = int(os.environ.get("GEMINI_MAX_RETRIES", "15"))
BASE_BACKOFF   = 2.0     # seconds; exponential

# gemini-embedding-2 embeds exactly ONE content per call (no list batching), so
# texts are embedded one-per-request and parallelized with a thread pool. Vertex
# enforces TWO independent per-minute quotas on a fresh project, and BOTH must be
# respected or the job dies with 429:
#   - input *tokens*/min  (embed_content_input_tokens_per_minute, default 50k)
#   - *requests*/min      (online_prediction_requests_per_base_model)
# so requests are paced by two shared token-buckets. Raise GEMINI_TOKENS_PER_MINUTE
# and GEMINI_REQUESTS_PER_MINUTE after a quota increase to go faster; raise
# GEMINI_WORKERS for more in-flight requests (only helps if both budgets allow it).
GEMINI_WORKERS    = int(os.environ.get("GEMINI_WORKERS", "16"))
GEMINI_TPM        = int(os.environ.get("GEMINI_TOKENS_PER_MINUTE", "50000"))
GEMINI_RPM        = int(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "100"))
# Vietnamese subword ratio is ~2.5 chars/token; undercounting makes the token
# limiter overshoot the real quota and trip 429s, so estimate conservatively.
CHARS_PER_TOKEN   = 2.5   # rough token estimate for pacing (len(text)/this)
# gemini-embedding-2 bills an image as a roughly fixed token block (tiles), so a
# flat per-image estimate is added to the caption tokens for the token limiter.
IMAGE_TOKENS_EST  = int(os.environ.get("GEMINI_IMAGE_TOKENS", "600"))


def _est_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


class RateLimiter:
    """Thread-safe token bucket pacing `units` per minute (tokens or requests)."""

    def __init__(self, units_per_minute: float):
        self.rate     = units_per_minute / 60.0   # units per second
        # Cap burst at ~5s worth so concurrent workers can't dump a full minute
        # of requests at startup (the classic thundering-herd that trips 429s).
        self.capacity = max(1.0, units_per_minute / 12.0)
        self.tokens   = 0.0                        # start empty: pure rate, no burst
        self.last     = time.monotonic()
        self.lock     = threading.Lock()

    def acquire(self, need: float = 1.0) -> None:
        need = min(need, self.capacity)            # never block forever on one item
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= need:
                    self.tokens -= need
                    return
                wait = (need - self.tokens) / self.rate
            time.sleep(wait)


# Back-compat alias: callers previously imported TokenRateLimiter.
TokenRateLimiter = RateLimiter

# Task prefixes (gemini-embedding-2 has no task_type param)
DOC_PREFIX_FMT   = "title: {title} | text: {content}"
QUERY_PREFIX     = "task: search result | query: "


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def build_client():
    """Create a genai client.

    Uses Vertex AI (token-based auth via Application Default Credentials) when
    GOOGLE_GENAI_USE_VERTEXAI is truthy, otherwise falls back to an AI Studio
    API key (GEMINI_API_KEY / GOOGLE_API_KEY).
    """
    from google import genai

    if _truthy(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")):
        project  = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError(
                "Vertex AI mode requested (GOOGLE_GENAI_USE_VERTEXAI=true) but "
                "GOOGLE_CLOUD_PROJECT is not set. Also ensure credentials are "
                "available via `gcloud auth application-default login` or "
                "GOOGLE_APPLICATION_CREDENTIALS=<service-account.json>."
            )
        return genai.Client(vertexai=True, project=project, location=location)

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No credentials found. Either set GOOGLE_GENAI_USE_VERTEXAI=true "
            "(+ GOOGLE_CLOUD_PROJECT and ADC) to use Vertex AI, or set "
            "GEMINI_API_KEY / GOOGLE_API_KEY to use an AI Studio API key."
        )
    return genai.Client(api_key=key)


def _is_transient(e: Exception) -> bool:
    s = str(e).lower()
    return any(t in s for t in ("rate", "quota", "429", "503", "500", "deadline", "timeout", "unavailable"))


def _with_retry(fn, *args, **kwargs):
    """Call fn with exponential backoff on transient (rate-limit/5xx) errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRIES - 1 or not _is_transient(e):
                raise
            time.sleep(min(60.0, BASE_BACKOFF * (2 ** attempt)))


def embed_texts(client, texts: list[str], dim: int = OUTPUT_DIM,
                workers: int = GEMINI_WORKERS,
                limiter: "RateLimiter | None" = None,
                rpm_limiter: "RateLimiter | None" = None) -> torch.Tensor:
    """Embed a list of (already-prefixed) strings → Tensor[len(texts), dim].

    gemini-embedding-2 only accepts ONE content per embed_content call, so each
    string is its own request. Requests run concurrently (thread pool) and are
    paced by `limiter` (input tokens/min) and `rpm_limiter` (requests/min) to
    stay under both Vertex quotas. Output order matches `texts`.
    """
    from google.genai import types

    cfg = types.EmbedContentConfig(output_dimensionality=dim)
    out: list[list[float] | None] = [None] * len(texts)

    def _one(i: int, text: str):
        if limiter is not None:
            limiter.acquire(_est_tokens(text))
        if rpm_limiter is not None:
            rpm_limiter.acquire(1.0)
        resp = _with_retry(client.models.embed_content,
                           model=MODEL, contents=[text], config=cfg)
        return i, resp.embeddings[0].values

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, i, t) for i, t in enumerate(texts)]
        for fut in as_completed(futures):
            i, vec = fut.result()
            out[i] = vec

    return torch.tensor(out, dtype=torch.float32)


def embed_multimodal_item(client, text: str | None, image_bytes: bytes,
                          mime_type: str = "image/jpeg",
                          dim: int = OUTPUT_DIM) -> list[float]:
    """Embed one (text + image) item as a single interleaved Content → vector."""
    from google.genai import types

    parts = []
    if text:
        parts.append(text)
    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    cfg  = types.EmbedContentConfig(output_dimensionality=dim)
    resp = _with_retry(client.models.embed_content,
                        model=MODEL, contents=parts, config=cfg)
    return resp.embeddings[0].values


def embed_multimodal_concurrent(client, items: list[tuple], workers: int = 12,
                                dim: int = OUTPUT_DIM,
                                limiter: "RateLimiter | None" = None,
                                rpm_limiter: "RateLimiter | None" = None):
    """
    Embed many (key, text, image_path, mime) items concurrently.

    Image bytes are read inside each worker (low memory, parallel I/O). Requests
    are paced by `limiter` (input tokens/min — caption tokens + a flat per-image
    estimate) and `rpm_limiter` (requests/min) to stay under the Vertex quotas.
    Yields (key, vector) as each completes (order not preserved), so the caller
    can checkpoint progress incrementally.
    """
    def _job(item):
        key, text, image_path, mime = item
        if limiter is not None:
            limiter.acquire(_est_tokens(text or "") + IMAGE_TOKENS_EST)
        if rpm_limiter is not None:
            rpm_limiter.acquire(1.0)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return key, embed_multimodal_item(client, text, image_bytes, mime, dim)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_job, it) for it in items]
        for fut in as_completed(futures):
            yield fut.result()
