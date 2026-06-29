from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.retrieval.config import RetrievalConfig
from src.retrieval.embedding import TextEmbedder
from src.retrieval.model import load_vlm


class GroupSyncRetrieval:
    """Summarizes all captions in a group via LLM, embeds the summary, then retrieves."""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.vlm = load_vlm(config)
        self.embedder = TextEmbedder(
            config.embed_model_name, config.device, config.embed_batch_size,
            config.embed_max_tokens, config.embed_chunk_stride,
        )        
        self._result_dir = Path(config.result_group_dir)
        self._result_dir.mkdir(parents=True, exist_ok=True)

    def retrieve(
        self,
        captions: dict[str, str],
        image_groups: dict[str, list[str]],
        article_embed_path: str,
    ) -> pd.DataFrame:
        art_data = torch.load(article_embed_path, weights_only=True)
        article_ids: list[str] = art_data["article_ids"]
        art_embeddings = art_data["embeddings"].float()

        group_ids = list(image_groups.keys())
        summaries = [
            self._summarize([captions[iid] for iid in image_groups[gid] if iid in captions])
            for gid in tqdm(group_ids, desc="Summarizing groups")
        ]

        summary_embeddings = self.embedder.embed(summaries).float()
        scores = summary_embeddings @ art_embeddings.T  # [N_groups, N_art]
        top_indices = scores.topk(self.config.top_k, dim=-1).indices

        rows = [
            {"image_id": group_ids[i], **{f"article_{j}": article_ids[idx] for j, idx in enumerate(top_indices[i].tolist())}}
            for i in range(len(group_ids))
        ]
        df = pd.DataFrame(rows)
        df.to_csv(self._result_dir / "group_sync_retrieval.csv", index=False)
        return df

    def _summarize(self, captions: list[str]) -> str:
        bullet_list = "\n".join(f"- {c}" for c in captions)
        prompt = (
            "The following are descriptions of images from the same news event.\n"
            f"{bullet_list}\n\n"
            "Write a concise summary describing the event depicted across all images."
        )
        return self.vlm.generate_text(prompt, self.config.group_summary_max_tokens)
