"""
Step 4 — Rerank retrieved articles using BGE-Reranker-v2-m3.

For each (image, candidate_article) pair the reranker receives:
  - the caption text (generated + original, from existing caption file)
  - the article title + content (truncated)

It outputs a relevance score; candidates are re-sorted and the top-K articles
are written to the output CSV.

Compared to the previous Qwen2.5-VL reranker:
  - Text-only cross-encoder (no image processing) — 10-50x faster per pair
  - Shared model family with BGE-M3 embeddings (same representation space)
  - Smaller memory footprint

Batching: candidate pairs for one image are scored in a single batched forward
pass (--batch_size controls how many pairs per pass).

Crash recovery:
  - Each finished image/group is appended immediately to a .jsonl backup
    (single_reranked.jsonl / group_reranked.jsonl) and flushed.
  - Rerunning skips ids already present in the .jsonl and continues.
  - The output CSV is rebuilt from the .jsonl on every exit (including
    Ctrl+C and crashes), so it always reflects all progress so far.
  - On CUDA OOM the batch is retried pair-by-pair; after MAX_OOM total OOMs
    the run aborts (progress saved).

Inputs:
    src/retrieval_v2/outputs/results/single_retrieval.csv  (or group_retrieval.csv)
    /raid/ltnghia01/phucpv/VQA/image_caption_image_only.json
    ./data/merged_7_database.json

Outputs:
    src/retrieval_v2/outputs/results/single_reranked.csv
    src/retrieval_v2/outputs/results/group_reranked.csv   (with --mode group/both)

Usage:
    python -m src.retrieval_v2.step_4_rerank --device cuda:0 --batch_size 32
    python -m src.retrieval_v2.step_4_rerank --device cuda:0 --mode group
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from src.retrieval_v2.utils import append_jsonl, is_oom, load_jsonl

# ── Config ────────────────────────────────────────────────────────────────────
CAPTION_PATH      = "/raid/ltnghia01/phucpv/VQA/image_caption_image_only.json"
DATABASE_PATH     = "./data/merged_7_database.json"
GROUPS_PATH       = "./data/groups.json"
RESULTS_DIR       = Path(__file__).parent / "outputs" / "results"

RERANK_MODEL      = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_N      = 50    # candidates scored from retrieval CSV
FINAL_TOP_K       = 10    # articles kept in output
ARTICLE_MAX_CHARS = 2000  # truncate article content to save tokens
RERANK_BATCH_SIZE = 32    # pairs per forward pass (much larger than VLM)
RERANK_MAX_LENGTH = 1024  # token budget for (caption, article) pair
MAX_OOM           = 10
# ─────────────────────────────────────────────────────────────────────────────


def load_model(device: str, load_in_4bit: bool = False):
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, quantization_config=bnb_cfg, device_map=device
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, torch_dtype=torch.float16
        ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
    model.eval()
    return model, tokenizer


def build_caption_text(img_id: str, captions: dict) -> str:
    cap  = captions.get(img_id, {})
    gen  = (cap.get("generated_caption") or "").strip()
    orig = (cap.get("original_caption")  or "").strip()
    return f"{gen} {orig}".strip() if orig else gen


def build_article_text(article: dict) -> str:
    title   = (article.get("title",   "") or "").strip()
    content = (article.get("content", "") or "").strip()
    return f"{title}\n{content[:ARTICLE_MAX_CHARS]}"


def score_pairs(
    pairs: list[tuple[str, str]],
    model, tokenizer, device: str,
) -> list[float]:
    """Score a batch of (caption, article) pairs; return raw logit scores."""
    queries = [p[0] for p in pairs]
    docs    = [p[1] for p in pairs]
    inputs  = tokenizer(
        queries, docs,
        padding=True, truncation=True, max_length=RERANK_MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        if logits.dim() == 2:
            logits = logits.squeeze(-1)
    return logits.float().cpu().tolist()


def score_candidates(
    caption: str,
    articles: list[dict],
    model, tokenizer, device: str,
    batch_size: int,
    oom_state: dict,
) -> list[float]:
    """Score all candidates in batches, falling back to per-pair on CUDA OOM."""
    pairs = [(caption, build_article_text(a)) for a in articles]

    # Sort by total char length so batches contain similar-length sequences,
    # minimising padding waste (padding=True pads all to longest in each batch).
    order        = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    sorted_pairs = [pairs[i] for i in order]

    flat_scores: list[float] = []
    for start in range(0, len(sorted_pairs), batch_size):
        batch = sorted_pairs[start : start + batch_size]
        try:
            flat_scores.extend(score_pairs(batch, model, tokenizer, device))
            continue
        except Exception as e:
            if not is_oom(e):
                raise
            torch.cuda.empty_cache()
            oom_state["count"] += 1
            print(f"\n[!] OOM {oom_state['count']}/{MAX_OOM} — retrying batch pair-by-pair")
            if oom_state["count"] >= MAX_OOM:
                raise

        for pair in batch:
            try:
                flat_scores.extend(score_pairs([pair], model, tokenizer, device))
            except Exception as e2:
                if not is_oom(e2):
                    raise
                torch.cuda.empty_cache()
                oom_state["count"] += 1
                print(f"\n[!] OOM {oom_state['count']}/{MAX_OOM} — scoring 0 for this candidate")
                if oom_state["count"] >= MAX_OOM:
                    raise
                flat_scores.append(0.0)

    # Unsort: map scores back to original candidate order
    scores = [0.0] * len(pairs)
    for sorted_idx, orig_idx in enumerate(order):
        scores[orig_idx] = flat_scores[sorted_idx]
    return scores


def write_csv(jsonl_path: Path, csv_path: Path) -> None:
    merged: dict[str, list[str]] = {}
    for r in load_jsonl(jsonl_path):
        merged[str(r["image_id"])] = r.get("articles", [])
    rows = [
        {"image_id": iid, **{f"article_{j}": a for j, a in enumerate(arts)}}
        for iid, arts in merged.items()
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved {len(rows)} rows → {csv_path}")


def rerank_single(captions: dict, db: dict, model, tokenizer, device: str, batch_size: int) -> None:
    jsonl_path = RESULTS_DIR / "single_reranked.jsonl"
    csv_path   = RESULTS_DIR / "single_reranked.csv"
    done       = {str(r["image_id"]) for r in load_jsonl(jsonl_path)}
    if done:
        print(f"[*] Resuming single rerank: {len(done)} images already done.")

    df           = pd.read_csv(RESULTS_DIR / "single_retrieval.csv")
    article_cols = sorted(c for c in df.columns if c.startswith("article_"))[:RERANK_TOP_N]
    oom_state    = {"count": 0}

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Reranking single"):
            img_id = str(row["image_id"])
            if img_id in done:
                continue

            caption    = build_caption_text(img_id, captions)
            candidates = [row[c] for c in article_cols if pd.notna(row.get(c)) and row[c] in db]

            if not candidates:
                append_jsonl(jsonl_path, {"image_id": img_id, "articles": []})
                continue

            scores = score_candidates(
                caption, [db[a] for a in candidates],
                model, tokenizer, device, batch_size, oom_state,
            )
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            top    = [a for a, _ in ranked[:FINAL_TOP_K]]
            append_jsonl(jsonl_path, {"image_id": img_id, "articles": top})
    finally:
        write_csv(jsonl_path, csv_path)


def rerank_group(
    captions: dict, groups: dict, db: dict, model, tokenizer, device: str, batch_size: int
) -> None:
    jsonl_path = RESULTS_DIR / "group_reranked.jsonl"
    csv_path   = RESULTS_DIR / "group_reranked.csv"
    done       = {str(r["image_id"]) for r in load_jsonl(jsonl_path)}
    if done:
        print(f"[*] Resuming group rerank: {len(done)} groups already done.")

    df           = pd.read_csv(RESULTS_DIR / "group_retrieval.csv")
    article_cols = sorted(c for c in df.columns if c.startswith("article_"))[:RERANK_TOP_N]
    df           = df.set_index("image_id")
    df.index     = df.index.astype(str)
    oom_state    = {"count": 0}

    try:
        for group_id, image_ids in tqdm(groups.items(), desc="Reranking groups"):
            if group_id in done or group_id not in df.index:
                continue

            candidates = [
                df.at[group_id, c] for c in article_cols
                if pd.notna(df.at[group_id, c]) and df.at[group_id, c] in db
            ]
            if not candidates:
                append_jsonl(jsonl_path, {"image_id": group_id, "articles": []})
                continue

            # Aggregate captions from all images in group
            group_caption = " ".join(
                build_caption_text(iid, captions) for iid in image_ids
            ).strip()

            scores = score_candidates(
                group_caption, [db[a] for a in candidates],
                model, tokenizer, device, batch_size, oom_state,
            )
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            top    = [a for a, _ in ranked[:FINAL_TOP_K]]
            append_jsonl(jsonl_path, {"image_id": group_id, "articles": top})
    finally:
        write_csv(jsonl_path, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--mode",        choices=["single", "group", "both"], default="both")
    parser.add_argument("--batch_size",  type=int, default=RERANK_BATCH_SIZE)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit (NF4) to save VRAM (requires bitsandbytes)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading captions: {CAPTION_PATH}")
    with open(CAPTION_PATH, encoding="utf-8") as f:
        captions = json.load(f)

    print(f"Loading database: {DATABASE_PATH}")
    with open(DATABASE_PATH, encoding="utf-8") as f:
        db = json.load(f)

    print(f"Loading rerank model: {RERANK_MODEL} (4bit={args.load_in_4bit})")
    model, tokenizer = load_model(args.device, load_in_4bit=args.load_in_4bit)

    try:
        if args.mode in ("single", "both"):
            rerank_single(captions, db, model, tokenizer, args.device, args.batch_size)

        if args.mode in ("group", "both"):
            with open(GROUPS_PATH, encoding="utf-8") as f:
                groups = json.load(f)
            rerank_group(captions, groups, db, model, tokenizer, args.device, args.batch_size)
    except KeyboardInterrupt:
        print("\n[!] Interrupted — progress saved to .jsonl; re-run the same command to resume.")


if __name__ == "__main__":
    main()
