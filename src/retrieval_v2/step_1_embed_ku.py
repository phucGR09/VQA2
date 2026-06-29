"""
Step 1 — Build Knowledge Unit (KU) embeddings from article database.

Two backends (--backend):
  - local  (default): Jina/BGE text encoder via SentenceTransformer. Each article
                      is split into overlapping token-windowed chunks (KUs).
  - gemini          : Gemini Embedding 2 API. Each article is one KU (no chunking;
                      8192-token limit), embedded with the document prefix
                      "title: {title} | text: {content}".

Output is written to a per-backend subfolder so the two never clash:
    outputs/ku_features/<backend>/ku_embeddings.pt
  {
    "ku_ids":      List[str]  — e.g. "art_001__chunk_0"
    "article_ids": List[str]  — parent article for each KU
    "embeddings":  Tensor[N_ku, D]
  }

Crash recovery: KU texts are embedded in parts of CKPT_CHUNK texts. Each finished
part is saved atomically to <backend>/parts/; rerunning after a crash skips parts
already on disk. Parts assume the database is unchanged between runs — if the
total KU count differs, stale parts are wiped and embedding restarts.

Gemini auth: set GEMINI_API_KEY (or GOOGLE_API_KEY). Install: pip install google-genai

Usage:
    python -m src.retrieval_v2.step_1_embed_ku                      # local
    python -m src.retrieval_v2.step_1_embed_ku --backend gemini
    python -m src.retrieval_v2.step_1_embed_ku --device cuda:0      # local only
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from src.retrieval_v2.utils import atomic_save

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_PATH    = "./data/merged_7_database.json"
OUTPUT_ROOT      = Path(__file__).parent / "outputs" / "ku_features"

# local backend
EMBED_MODEL      = "BAAI/bge-m3"
EMBED_BATCH_SIZE = 16    # KU texts per forward pass
KU_MAX_TOKENS    = 1024  # token window per KU chunk (BGE-M3 supports up to 8192)
KU_STRIDE        = 256   # overlap between consecutive chunks

# gemini backend (gemini-embedding-2 embeds one content per call; concurrency
# and per-minute token pacing are configured via env in gemini_embed.py)

CKPT_CHUNK       = int(os.environ.get("CKPT_CHUNK", "75"))  # KU texts per checkpoint part
# ─────────────────────────────────────────────────────────────────────────────


def _chunk_by_tokens(text: str, tokenizer, max_tokens: int, stride: int) -> list[str]:
    ids    = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    window = max_tokens - 2
    step   = window - stride
    chunks = []
    for start in range(0, len(ids), step):
        chunks.append(tokenizer.decode(ids[start : start + window]))
        if start + window >= len(ids):
            break
    return chunks


def build_kus_local(db: dict, tokenizer) -> tuple[list[str], list[str], list[str]]:
    """Split each article into overlapping KU chunks, return (ku_ids, article_ids, texts)."""
    ku_ids, article_ids, texts = [], [], []
    for article_id, article in tqdm(db.items(), desc="Building KUs"):
        full = f"{article.get('title', '')} {article.get('content', '')}".strip()
        tok_len = len(tokenizer(full, truncation=False, add_special_tokens=False)["input_ids"])

        if tok_len <= KU_MAX_TOKENS - 2:
            chunks = [full]
        else:
            chunks = _chunk_by_tokens(full, tokenizer, KU_MAX_TOKENS, KU_STRIDE)

        for i, chunk in enumerate(chunks):
            ku_ids.append(f"{article_id}__chunk_{i}")
            article_ids.append(article_id)
            texts.append(chunk)

    return ku_ids, article_ids, texts


def build_kus_gemini(db: dict) -> tuple[list[str], list[str], list[str]]:
    """One KU per article with the Gemini document prefix."""
    from src.retrieval_v2.gemini_embed import DOC_PREFIX_FMT

    ku_ids, article_ids, texts = [], [], []
    for article_id, article in tqdm(db.items(), desc="Building KUs"):
        text = DOC_PREFIX_FMT.format(
            title=article.get("title", "").strip(),
            content=article.get("content", "").strip(),
        )
        ku_ids.append(f"{article_id}__chunk_0")
        article_ids.append(article_id)
        texts.append(text)
    return ku_ids, article_ids, texts


def _check_part_dir(part_dir: Path, n_total: int) -> None:
    """Wipe stale parts if the KU count changed since the last (interrupted) run."""
    meta_path = part_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("n_total") == n_total:
            return
        print("[!] KU count changed since last run — discarding stale parts.")
        for p in part_dir.glob("part_*.pt*"):
            p.unlink()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"n_total": n_total}, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "gemini"], default="local",
                        help="Embedding backend (default: local BGE-M3)")
    parser.add_argument("--device", default="cuda", help="Device for local backend")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.backend
    part_dir   = output_dir / "parts"

    print(f"Loading database: {DATABASE_PATH}")
    with open(DATABASE_PATH, encoding="utf-8") as f:
        db = json.load(f)
    print(f"  {len(db)} articles")

    # ── Set up backend: encode_part(lo, hi) → Tensor ────────────────────────────
    if args.backend == "local":
        from sentence_transformers import SentenceTransformer

        print(f"Loading model: {EMBED_MODEL}")
        model = SentenceTransformer(EMBED_MODEL, device=args.device, trust_remote_code=True)
        model.max_seq_length = KU_MAX_TOKENS
        ku_ids, article_ids, texts = build_kus_local(db, model.tokenizer)

        def encode_part(lo: int, hi: int) -> torch.Tensor:
            return model.encode(
                texts[lo:hi],
                batch_size=EMBED_BATCH_SIZE,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).cpu()
    else:
        from src.retrieval_v2.gemini_embed import (
            GEMINI_RPM, GEMINI_TPM, GEMINI_WORKERS, OUTPUT_DIM, RateLimiter,
            build_client, embed_texts,
        )

        print(f"Connecting to Gemini Embedding 2 (dim={OUTPUT_DIM})")
        print(f"  pacing: {GEMINI_WORKERS} workers, "
              f"{GEMINI_TPM:,} tokens/min, {GEMINI_RPM:,} requests/min budget")
        client = build_client()
        tok_limiter = RateLimiter(GEMINI_TPM)   # shared across all parts
        rpm_limiter = RateLimiter(GEMINI_RPM)
        ku_ids, article_ids, texts = build_kus_gemini(db)

        def encode_part(lo: int, hi: int) -> torch.Tensor:
            return embed_texts(client, texts[lo:hi], dim=OUTPUT_DIM,
                               workers=GEMINI_WORKERS, limiter=tok_limiter,
                               rpm_limiter=rpm_limiter)

    print(f"  {len(ku_ids)} KUs built")

    part_dir.mkdir(parents=True, exist_ok=True)
    _check_part_dir(part_dir, len(texts))

    n_parts = (len(texts) + CKPT_CHUNK - 1) // CKPT_CHUNK
    skipped = 0
    for p in tqdm(range(n_parts), desc="Embedding parts"):
        part_path = part_dir / f"part_{p:05d}.pt"
        lo, hi    = p * CKPT_CHUNK, min((p + 1) * CKPT_CHUNK, len(texts))

        if part_path.exists():
            try:
                if torch.load(part_path, weights_only=True).shape[0] == hi - lo:
                    skipped += 1
                    continue
            except Exception:
                pass  # corrupt part → re-embed

        atomic_save(encode_part(lo, hi), part_path)

    if skipped:
        print(f"[*] Resumed: {skipped}/{n_parts} parts reused from previous run.")

    print("Merging parts...")
    embeddings = torch.cat(
        [torch.load(part_dir / f"part_{p:05d}.pt", weights_only=True) for p in range(n_parts)],
        dim=0,
    )

    out_path = output_dir / "ku_embeddings.pt"
    atomic_save(
        {"ku_ids": ku_ids, "article_ids": article_ids, "embeddings": embeddings},
        out_path,
    )
    print(f"Saved {len(ku_ids)} KU embeddings → {out_path}")

    for p in part_dir.glob("part_*.pt"):
        p.unlink()
    (part_dir / "meta.json").unlink(missing_ok=True)
    try:
        part_dir.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
