import json

import pandas as pd


def load_ground_truth(database_path: str) -> dict[str, str]:
    """Returns {image_id: article_id} from database.json."""
    with open(database_path) as f:
        db = json.load(f)
    return {
        img["image_id"]: article_id
        for article_id, article in db.items()
        for img in article.get("images", [])
    }


def evaluate(result_path: str, database_path: str, top_k: int = 10) -> dict:
    gt = load_ground_truth(database_path)
    df = pd.read_csv(result_path)
    article_cols = sorted(c for c in df.columns if c.startswith("article_"))[:top_k]

    precision_list, recall_list, rr_list, hit1_list = [], [], [], []
    for _, row in df.iterrows():
        image_id = row["image_id"]
        if image_id not in gt:
            continue
        retrieved = [row[c] for c in article_cols if pd.notna(row[c])]
        relevant = gt[image_id]
        hit = relevant in retrieved
        precision_list.append(int(hit) / top_k)
        recall_list.append(int(hit))
        hit1_list.append(int(bool(retrieved) and retrieved[0] == relevant))
        rr_list.append(
            next((1.0 / (i + 1) for i, a in enumerate(retrieved) if a == relevant), 0.0)
        )

    n = len(rr_list)
    hits = sum(recall_list)
    return {
        "hits@1": sum(hit1_list),
        f"hits@{top_k}": hits,
        f"precision@{top_k}": hits / (n * top_k),
        f"recall@{top_k}": hits / n,
        "mrr": sum(rr_list) / n,
        "n_evaluated": n,
    }
