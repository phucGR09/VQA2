"""
Step 2 — Embed query images with Jina CLIP v2 image encoder.

Reads query image IDs from groups.json, loads images in batches, and saves
embeddings. Single-GPU, batch-based processing.

Crash recovery:
  - Progress is checkpointed atomically every SAVE_EVERY batches (and on
    Ctrl+C / crash via finally) to image_embeddings_ckpt.pt.
  - Rerunning skips images already embedded (or permanently failed) and
    continues from where it stopped.
  - Unreadable images are skipped and remembered, so they are not retried.
  - On CUDA OOM the batch is retried image-by-image; after MAX_OOM total
    OOMs the run aborts (progress is still saved).

Output: src/retrieval_v2/outputs/image_features/image_embeddings.pt
  {
    "image_ids": List[str]
    "embeddings": Tensor[N_img, D]
  }

Usage:
    python -m src.retrieval_v2.step_2_embed_images --device cuda:0
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel
from tqdm import tqdm

from src.retrieval_v2.utils import atomic_save, is_oom

# ── Config ────────────────────────────────────────────────────────────────────
GROUPS_PATH  = "./data/groups.json"
IMAGE_DIR    = "./data/database_image_compress"
OUTPUT_DIR   = Path(__file__).parent / "outputs" / "image_features"
CKPT_PATH    = OUTPUT_DIR / "image_embeddings_ckpt.pt"
FINAL_PATH   = OUTPUT_DIR / "image_embeddings.pt"

EMBED_MODEL      = "jinaai/jina-clip-v2"
EMBED_BATCH_SIZE = 64
SAVE_EVERY       = 20  # batches between checkpoint saves
MAX_OOM          = 10
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


def load_progress() -> tuple[list[str], list[torch.Tensor], set[str]]:
    """Resume from checkpoint if present, else from a previous final output."""
    for path in (CKPT_PATH, FINAL_PATH):
        if not path.exists():
            continue
        try:
            data = torch.load(path, weights_only=True)
        except Exception:
            print(f"[!] Could not load {path.name}, ignoring it.")
            continue
        done_ids   = list(data["image_ids"])
        emb_chunks = [data["embeddings"]] if data["embeddings"].numel() > 0 else []
        failed_ids = set(data.get("failed_ids", []))
        print(f"[*] Resuming from {path.name}: {len(done_ids)} done, {len(failed_ids)} failed.")
        return done_ids, emb_chunks, failed_ids
    return [], [], set()


def save_ckpt(done_ids: list[str], emb_chunks: list[torch.Tensor], failed_ids: set[str]) -> None:
    embeddings = torch.cat(emb_chunks, dim=0) if emb_chunks else torch.empty(0)
    atomic_save(
        {"image_ids": done_ids, "embeddings": embeddings, "failed_ids": sorted(failed_ids)},
        CKPT_PATH,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=EMBED_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="Process only N images (0 = all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_ids = get_query_image_ids(GROUPS_PATH)
    done_ids, emb_chunks, failed_ids = load_progress()

    skip = set(done_ids) | failed_ids
    todo = [iid for iid in all_ids if iid not in skip]
    print(f"Query images: {len(all_ids)} total, {len(todo)} to process.")

    if args.limit:
        todo = todo[: args.limit]
        print(f"[*] Limit set: processing only {args.limit} images.")

    if not todo:
        print("[*] Nothing to process.")
    else:
        print(f"Loading model: {EMBED_MODEL}")
        model = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        model = model.to(args.device)
        model.eval()
        image_dir = Path(IMAGE_DIR)

        def encode(images: list) -> torch.Tensor:
            with torch.no_grad():
                embs = model.encode_image(images)
                embs = torch.as_tensor(embs, dtype=torch.float32)
                embs = F.normalize(embs, dim=-1)
            return embs.cpu()

        oom_count          = 0
        batches_since_save = 0
        try:
            for start in tqdm(range(0, len(todo), args.batch_size), desc="Embedding images"):
                batch_ids = todo[start : start + args.batch_size]

                images, valid = [], []
                for iid in batch_ids:
                    path = image_dir / f"{iid}.jpg"
                    try:
                        images.append(Image.open(path).convert("RGB"))
                        valid.append(iid)
                    except Exception as e:
                        print(f"\n  [warn] unreadable image {path}: {e}")
                        failed_ids.add(iid)
                if not images:
                    continue

                try:
                    embs = encode(images)
                except Exception as e:
                    if not is_oom(e):
                        raise
                    torch.cuda.empty_cache()
                    oom_count += 1
                    print(f"\n[!] OOM {oom_count}/{MAX_OOM} — retrying batch image-by-image")
                    if oom_count >= MAX_OOM:
                        raise
                    chunks, ok_ids = [], []
                    for img, iid in zip(images, valid):
                        try:
                            chunks.append(encode([img]))
                            ok_ids.append(iid)
                        except Exception as e2:
                            if not is_oom(e2):
                                raise
                            torch.cuda.empty_cache()
                            oom_count += 1
                            failed_ids.add(iid)
                            print(f"\n[!] OOM {oom_count}/{MAX_OOM} — skipping {iid}")
                            if oom_count >= MAX_OOM:
                                raise
                    if not chunks:
                        continue
                    embs, valid = torch.cat(chunks, dim=0), ok_ids

                emb_chunks.append(embs)
                done_ids.extend(valid)

                batches_since_save += 1
                if batches_since_save >= SAVE_EVERY:
                    save_ckpt(done_ids, emb_chunks, failed_ids)
                    batches_since_save = 0

        except KeyboardInterrupt:
            print("\n[!] Interrupted — saving checkpoint...")
        finally:
            save_ckpt(done_ids, emb_chunks, failed_ids)

    effective_ids = set(todo) | set(done_ids) if args.limit else set(all_ids)
    remaining = effective_ids - set(done_ids) - failed_ids
    if remaining:
        print(f"[*] Checkpoint saved ({len(done_ids)} done, {len(remaining)} remaining). "
              f"Re-run the same command to continue.")
        return

    embeddings = torch.cat(emb_chunks, dim=0) if emb_chunks else torch.empty(0)
    atomic_save({"image_ids": done_ids, "embeddings": embeddings}, FINAL_PATH)
    CKPT_PATH.unlink(missing_ok=True)
    print(f"Saved {len(done_ids)} image embeddings → {FINAL_PATH}")
    if failed_ids:
        print(f"[!] {len(failed_ids)} images failed permanently: {sorted(failed_ids)[:10]}...")


if __name__ == "__main__":
    main()
