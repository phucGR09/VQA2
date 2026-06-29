from pathlib import Path

import pandas as pd
import torch

from src.retrieval.config import RetrievalConfig


class SingleRetrieval:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self._result_dir = Path(config.result_single_dir)
        self._result_dir.mkdir(parents=True, exist_ok=True)

    def retrieve(self, caption_embed_path: str, article_embed_path: str) -> pd.DataFrame:
        cap_data = torch.load(caption_embed_path, weights_only=True)
        art_data = torch.load(article_embed_path, weights_only=True)
        print(f"Loaded caption embeddings: {cap_data['embeddings'].shape}, article embeddings: {art_data['embeddings'].shape}")
        image_ids: list[str] = cap_data["image_ids"]
        article_ids: list[str] = art_data["article_ids"]
        # embeddings are L2-normalized → dot product == cosine similarity
        scores = cap_data["embeddings"].float() @ art_data["embeddings"].float().T  # [N_img, N_art]
        print(f"Computed similarity scores: {scores.shape}")
        top_indices = scores.topk(self.config.top_k, dim=-1).indices  # [N_img, k]

        rows = [
            {"image_id": image_ids[i], **{f"article_{j}": article_ids[idx] for j, idx in enumerate(top_indices[i].tolist())}}
            for i in range(len(image_ids))
        ]
        df = pd.DataFrame(rows)
        df.to_csv(self._result_dir / "single_retrieval.csv", index=False)
        return df
