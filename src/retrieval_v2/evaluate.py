"""
Evaluate retrieval results.

Computes Precision@K, Recall@K, MRR, Hits@1 for any result CSV.
Ground truth: image_id → article_id mapping from the database.

Usage:
    # All CSVs in the default results directory
    python -m src.retrieval_v2.evaluate

    # Specific file (single-image mode)
    python -m src.retrieval_v2.evaluate --result src/retrieval_v2/outputs/results/single_reranked.csv

    # Group mode: metrics computed per-image (each image inherits its group's results)
    python -m src.retrieval_v2.evaluate --result src/retrieval_v2/outputs/results/group_reranked.csv --mode group
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_PATH = "./data/merged_7_database.json"
GROUPS_PATH   = "./data/groups.json"
RESULTS_DIR   = Path(__file__).parent / "outputs" / "results"
TOP_K         = 10
# ─────────────────────────────────────────────────────────────────────────────


def load_ground_truth() -> dict[str, str]:
    with open(DATABASE_PATH, encoding="utf-8") as f:
        db = json.load(f)
    return {
        img["image_id"]: article_id
        for article_id, article in db.items()
        for img in article.get("images", [])
    }


def compute_metrics(rows: list[dict], gt: dict[str, str], top_k: int) -> dict:
    hit1, recall, rr = [], [], []
    for row in rows:
        img_id = str(row["image_id"])
        if img_id not in gt:
            continue
        retrieved = row["retrieved"][:top_k]
        relevant  = gt[img_id]
        hit       = relevant in retrieved
        hit1.append(int(bool(retrieved) and retrieved[0] == relevant))
        recall.append(int(hit))
        rr.append(next((1.0 / (i + 1) for i, a in enumerate(retrieved) if a == relevant), 0.0))

    n    = len(rr)
    hits = sum(recall)
    if n == 0:
        return {"n_evaluated": 0}
    return {
        "hits@1":              sum(hit1),
        f"hits@{top_k}":      hits,
        f"precision@{top_k}": round(hits / (n * top_k), 4),
        f"recall@{top_k}":    round(hits / n, 4),
        "mrr":                 round(sum(rr) / n, 4),
        "n_evaluated":         n,
    }


def evaluate_csv(result_path: Path, gt: dict, top_k: int, mode: str, groups: dict | None) -> dict:
    df           = pd.read_csv(result_path)
    article_cols = sorted(c for c in df.columns if c.startswith("article_"))[:top_k]

    if mode == "single":
        rows = [
            {"image_id": str(r["image_id"]),
             "retrieved": [r[c] for c in article_cols if pd.notna(r.get(c))]}
            for _, r in df.iterrows()
        ]
    else:
        assert groups is not None
        df = df.set_index("image_id")
        df.index = df.index.astype(str)
        rows = []
        for group_id, image_ids in groups.items():
            if group_id not in df.index:
                continue
            retrieved = [df.at[group_id, c] for c in article_cols if pd.notna(df.at[group_id, c])]
            for img_id in image_ids:
                rows.append({"image_id": img_id, "retrieved": retrieved})

    return compute_metrics(rows, gt, top_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default=None)
    parser.add_argument("--mode",   choices=["single", "group"], default=None,
                        help="Override mode; auto-detected from filename if omitted.")
    parser.add_argument("--top_k",  type=int, default=TOP_K)
    args = parser.parse_args()

    gt     = load_ground_truth()
    groups = None

    paths = [Path(args.result)] if args.result else sorted(RESULTS_DIR.glob("*.csv"))
    for path in paths:
        mode = args.mode or ("group" if "group" in path.stem else "single")
        if mode == "group" and groups is None:
            with open(GROUPS_PATH, encoding="utf-8") as f:
                groups = json.load(f)
        metrics = evaluate_csv(path, gt, args.top_k, mode, groups)
        print(f"{path.name}: {metrics}")


if __name__ == "__main__":
    main()
