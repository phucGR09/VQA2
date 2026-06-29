import json

import pandas as pd


def _load_image_to_article(database_path: str) -> dict[str, str]:
    with open(database_path) as f:
        db = json.load(f)
    return {
        img["image_id"]: article_id
        for article_id, article in db.items()
        for img in article.get("images", [])
    }


def _flatten_to_image_level(
    result_path: str,
    groups_path: str,
    article_cols: list[str],
) -> list[dict]:
    """Expand each group row into one row per image, all sharing the group's retrieved articles."""
    with open(groups_path) as f:
        groups = json.load(f)
    df = pd.read_csv(result_path).set_index("image_id")
    df.index = df.index.astype(str)

    rows = []
    for group_id, image_ids in groups.items():
        if group_id not in df.index:
            continue
        retrieved = [df.at[group_id, c] for c in article_cols if pd.notna(df.at[group_id, c])]
        for image_id in image_ids:
            rows.append({"image_id": image_id, "retrieved": retrieved})
    return rows


def evaluate_groups(
    result_path: str,
    groups_path: str,
    database_path: str,
    top_k: int = 10,
) -> dict:
    gt = _load_image_to_article(database_path)

    df = pd.read_csv(result_path)
    article_cols = sorted(c for c in df.columns if c.startswith("article_"))[:top_k]

    # flatten: each image inherits its group's retrieved articles
    image_rows = _flatten_to_image_level(result_path, groups_path, article_cols)

    hit1_list, recall_list, rr_list = [], [], []
    for row in image_rows:
        image_id = row["image_id"]
        if image_id not in gt:
            continue
        retrieved = row["retrieved"]
        relevant = gt[image_id]
        hit = relevant in retrieved
        hit1_list.append(int(bool(retrieved) and retrieved[0] == relevant))
        recall_list.append(int(hit))
        rr_list.append(
            next((1.0 / (i + 1) for i, a in enumerate(retrieved) if a == relevant), 0.0)
        )

    n = len(rr_list)
    hits = sum(recall_list)
    if n == 0:
        return {"hits@1": 0, f"hits@{top_k}": 0, f"recall@{top_k}": 0.0,
                "mrr": 0.0, "n_evaluated": 0}
    return {
        "hits@1": sum(hit1_list),
        f"hits@{top_k}": hits,
        f"recall@{top_k}": hits / n,
        "mrr": sum(rr_list) / n,
        "n_evaluated": n,
    }
