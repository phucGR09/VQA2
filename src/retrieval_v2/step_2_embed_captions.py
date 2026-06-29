"""
Step 2 — Embed query items for image→article retrieval.

Two backends (--backend):
  - local  (default): BGE-M3 text encoder over caption text
                      (generated_caption + original_caption). Same model as the
                      local KU embeddings in step 1 → shared text space.
  - gemini          : Gemini Embedding 2, multimodal. Each query is the actual
                      image interleaved with its generated caption, embedded with
                      the query prefix "task: search result | query: {caption}".
                      Lands in the same unified space as the Gemini KU embeddings.

Output is written to a per-backend subfolder so the two never clash:
    outputs/caption_features/<backend>/caption_embeddings.pt
  {
    "image_ids": List[str]
    "embeddings": Tensor[N_img, D]
  }

Crash recovery:
  - Progress is checkpointed atomically every SAVE_EVERY batches (and on
    Ctrl+C / crash via finally) to <backend>/caption_embeddings_ckpt.pt.
  - Rerunning skips images already embedded and continues from where it stopped.

Gemini auth: set GEMINI_API_KEY (or GOOGLE_API_KEY). Install: pip install google-genai

Usage:
    python -m src.retrieval_v2.step_2_embed_captions                   # local
    python -m src.retrieval_v2.step_2_embed_captions --backend gemini
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.retrieval_v2.utils import atomic_save

# ── Config ────────────────────────────────────────────────────────────────────
GROUPS_PATH      = "./data/groups.json"
CAPTION_PATH     = "/raid/ltnghia01/phucpv/VQA/image_caption_image_only.json"
IMAGE_DIR        = Path("./data/database_image_compress")   # {image_id}.jpg
OUTPUT_ROOT      = Path(__file__).parent / "outputs" / "caption_features"

EMBED_MODEL      = "BAAI/bge-m3"   # local backend
EMBED_BATCH_SIZE = 64              # local backend
GEMINI_WORKERS   = 12              # concurrent embed_content calls (gemini)
SAVE_EVERY       = 20              # batches/checkpoints between saves
# ─────────────────────────────────────────────────────────────────────────────


def get_query_image_ids(groups_path: str) -> list[str]:
    with open(groups_path, encoding="utf-8") as f:
        groups = json.load(f)
    seen, ids = set(), []
    for image_ids in groups.values():
        for iid in image_ids:
            if iid not in seen:
                seen.add(iid)
                ids.append(iid)
    return ids


def build_caption_text(cap_data: dict) -> str:
    gen  = (cap_data.get("generated_caption") or "").strip()
    orig = (cap_data.get("original_caption")  or "").strip()
    return f"{gen} {orig}".strip() if orig else gen


def load_progress(ckpt_path: Path, final_path: Path) -> tuple[list[str], list[torch.Tensor]]:
    for path in (ckpt_path, final_path):
        if not path.exists():
            continue
        try:
            data       = torch.load(path, weights_only=True)
            done_ids   = list(data["image_ids"])
            emb_chunks = [data["embeddings"]] if data["embeddings"].numel() > 0 else []
            print(f"[*] Resuming from {path.name}: {len(done_ids)} done.")
            return done_ids, emb_chunks
        except Exception:
            print(f"[!] Could not load {path.name}, ignoring it.")
    return [], []


def save_ckpt(ckpt_path: Path, done_ids: list[str], emb_chunks: list[torch.Tensor]) -> None:
    embeddings = torch.cat(emb_chunks, dim=0) if emb_chunks else torch.empty(0)
    atomic_save({"image_ids": done_ids, "embeddings": embeddings}, ckpt_path)


def run_local(todo, captions, args, ckpt_path, done_ids, emb_chunks) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, device=args.device, trust_remote_code=True)

    batches_since_save = 0
    try:
        for start in tqdm(range(0, len(todo), args.batch_size), desc="Embedding captions"):
            batch_ids = todo[start : start + args.batch_size]
            texts     = [build_caption_text(captions.get(iid, {})) for iid in batch_ids]

            embs = model.encode(
                texts,
                batch_size=args.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            emb_chunks.append(embs.cpu())
            done_ids.extend(batch_ids)

            batches_since_save += 1
            if batches_since_save >= SAVE_EVERY:
                save_ckpt(ckpt_path, done_ids, emb_chunks)
                batches_since_save = 0
    except KeyboardInterrupt:
        print("\n[!] Interrupted — saving checkpoint...")
    finally:
        save_ckpt(ckpt_path, done_ids, emb_chunks)


def run_gemini(todo, captions, args, ckpt_path, done_ids, emb_chunks) -> None:
    from src.retrieval_v2.gemini_embed import (
        GEMINI_RPM, GEMINI_TPM, OUTPUT_DIM, QUERY_PREFIX, RateLimiter,
        build_client, embed_multimodal_concurrent,
    )

    print(f"Connecting to Gemini Embedding 2 (dim={OUTPUT_DIM}, workers={GEMINI_WORKERS})")
    print(f"  pacing: {GEMINI_TPM:,} tokens/min, {GEMINI_RPM:,} requests/min budget")
    client      = build_client()
    tok_limiter = RateLimiter(GEMINI_TPM)
    rpm_limiter = RateLimiter(GEMINI_RPM)

    # (image_id, query_text, image_path, mime) — skip images missing on disk
    items, missing = [], 0
    for iid in todo:
        img_path = IMAGE_DIR / f"{iid}.jpg"
        if not img_path.exists():
            missing += 1
            continue
        gen  = (captions.get(iid, {}).get("generated_caption") or "").strip()
        text = f"{QUERY_PREFIX}{gen}" if gen else QUERY_PREFIX
        items.append((iid, text, str(img_path), "image/jpeg"))
    if missing:
        print(f"[!] {missing} images not found in {IMAGE_DIR} — skipped.")

    since_save = 0
    try:
        with tqdm(total=len(items), desc="Embedding images") as bar:
            for iid, vec in embed_multimodal_concurrent(
                client, items, workers=GEMINI_WORKERS, dim=OUTPUT_DIM,
                limiter=tok_limiter, rpm_limiter=rpm_limiter,
            ):
                emb_chunks.append(torch.tensor(vec, dtype=torch.float32).unsqueeze(0))
                done_ids.append(iid)
                bar.update(1)

                since_save += 1
                if since_save >= SAVE_EVERY * args.batch_size:
                    save_ckpt(ckpt_path, done_ids, emb_chunks)
                    since_save = 0
    except KeyboardInterrupt:
        print("\n[!] Interrupted — saving checkpoint...")
    finally:
        save_ckpt(ckpt_path, done_ids, emb_chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "gemini"], default="local",
                        help="Embedding backend (default: local BGE-M3)")
    parser.add_argument("--device",     default="cuda", help="Device for local backend")
    parser.add_argument("--batch_size", type=int, default=EMBED_BATCH_SIZE)
    parser.add_argument("--limit",      type=int, default=0,
                        help="Process only N images (0 = all)")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.backend
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = output_dir / "caption_embeddings_ckpt.pt"
    final_path = output_dir / "caption_embeddings.pt"

    print(f"Loading groups: {GROUPS_PATH}")
    all_ids = get_query_image_ids(GROUPS_PATH)

    print(f"Loading captions: {CAPTION_PATH}")
    with open(CAPTION_PATH, encoding="utf-8") as f:
        captions = json.load(f)

    done_ids, emb_chunks = load_progress(ckpt_path, final_path)
    skip = set(done_ids)
    todo = [iid for iid in all_ids if iid not in skip]
    print(f"Query images: {len(all_ids)} total, {len(todo)} to process.")

    if args.limit:
        todo = todo[: args.limit]
        print(f"[*] Limit set: processing only {args.limit} images.")

    if not todo:
        print("[*] Nothing to process.")
    elif args.backend == "local":
        run_local(todo, captions, args, ckpt_path, done_ids, emb_chunks)
    else:
        run_gemini(todo, captions, args, ckpt_path, done_ids, emb_chunks)

    remaining = set(all_ids) - set(done_ids) if not args.limit else set()
    if remaining:
        print(f"[*] Checkpoint saved ({len(done_ids)} done, {len(remaining)} remaining). "
              f"Re-run the same command to continue.")
        return

    embeddings = torch.cat(emb_chunks, dim=0) if emb_chunks else torch.empty(0)
    atomic_save({"image_ids": done_ids, "embeddings": embeddings}, final_path)
    ckpt_path.unlink(missing_ok=True)
    print(f"Saved {len(done_ids)} embeddings → {final_path}")


if __name__ == "__main__":
    main()
