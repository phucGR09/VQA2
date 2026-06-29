from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.retrieval.config import RetrievalConfig


class GroupVotingRetrieval:
    """Reciprocal Rank Fusion across per-image single-retrieval results within a group."""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self._result_dir = Path(config.result_group_dir)
        self._result_dir.mkdir(parents=True, exist_ok=True)

    def retrieve(
        self,
        single_result_path: str,
        image_groups: dict[str, list[str]],
    ) -> pd.DataFrame:
        single_df = pd.read_csv(single_result_path).set_index("image_id")
        article_cols = sorted(c for c in single_df.columns if c.startswith("article_"))

        rows = []
        for group_id, image_ids in image_groups.items():
            scores: dict[str, float] = defaultdict(float)
            for image_id in image_ids:
                if image_id not in single_df.index:
                    continue
                for rank, col in enumerate(article_cols):
                    article_id = single_df.at[image_id, col]
                    if pd.notna(article_id):
                        scores[article_id] += 1.0 / (rank + 1)  # RRF score

            top_articles = sorted(scores, key=scores.__getitem__, reverse=True)[: self.config.top_k]
            row = {"image_id": group_id, **{f"article_{j}": art for j, art in enumerate(top_articles)}}
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self._result_dir / "group_voting_retrieval.csv", index=False)
        return df
