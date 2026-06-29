"""
Build image groups keyed by article_id from database.json.

Output format: {article_id: [image_id, ...], ...}  →  data/groups.json
Only includes image_ids whose .jpg file exists in database_image_dir.
"""

import json
from pathlib import Path

from retrieval.config import RetrievalConfig


def build_image_groups(config: RetrievalConfig) -> dict[str, list[str]]:
    with open(config.database_path) as f:
        db = json.load(f)

    image_dir = Path(config.image_dir)
    groups: dict[str, list[str]] = {}

    idx = 0
    for article in db.values():
        image_ids = [
            img["image_id"]
            for img in article.get("images", [])
            if (image_dir / f"{img['image_id']}.jpg").exists()
        ]
        if image_ids:
            groups[str(idx)] = image_ids
            idx += 1

    Path(config.groups_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.groups_path).write_text(json.dumps(groups, ensure_ascii=False, indent=2))
    print(f"Saved {len(groups)} groups → {config.groups_path}")
    return groups


if __name__ == "__main__":
    build_image_groups(RetrievalConfig())
