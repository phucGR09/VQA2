"""
retrieval_utils.py
==================
Contriever-based passage retrieval for level4 Case B and level5 Case B.

Extracted from /copy/phase3_retrieval.py.  The training pipeline
(/copy/phase3_vqa_finetune.py, /copy/phase4_evaluation.py) continues to
import from phase3_retrieval.py directly so that file is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from VQA_baseline.baseline_config import RetrievalConfig


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PassageIndex:
    passages: List[str]       # raw text of each passage chunk
    embeddings: np.ndarray    # float32 [N_passages, D], L2-normalised
    article_ids: List[str]    # source article_id for each passage


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


class PassageRetriever:
    """Contriever dense retriever: encode → L2-normalise → dot product = cosine sim."""

    def __init__(self, cfg: RetrievalConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        """Returns float32 [len(texts), D], L2-normalised."""
        all_embs = []
        for i in range(0, len(texts), self.cfg.encode_batch_size):
            batch = texts[i : i + self.cfg.encode_batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.passage_max_tokens,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            embs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            embs = F.normalize(embs, dim=-1)
            all_embs.append(embs.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)

    def batch_retrieve(
        self,
        queries: List[str],
        index: PassageIndex,
    ) -> List[List[Tuple[str, str]]]:
        query_embs = self.encode(queries)           # [N_q, D]
        scores = index.embeddings @ query_embs.T    # [N_p, N_q]
        top_k = min(self.cfg.top_k, len(index.passages))

        results = []
        for q_idx in range(len(queries)):
            top_idx = np.argsort(scores[:, q_idx])[::-1][:top_k]
            results.append(
                [(index.article_ids[i], index.passages[i]) for i in top_idx]
            )
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Article chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_article(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Non-overlapping token windows → decoded back to strings."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_passage_index(
    database: Dict[str, dict],
    retriever: PassageRetriever,
    cfg: RetrievalConfig,
) -> PassageIndex:
    if not database:
        raise ValueError("database is empty – cannot build passage index")

    passages: List[str] = []
    article_ids: List[str] = []

    for article_id, article in database.items():
        chunks = chunk_article(
            article["content"], retriever.tokenizer, cfg.passage_chunk_tokens
        )
        passages.extend(chunks)
        article_ids.extend([article_id] * len(chunks))

    embeddings = retriever.encode(passages)
    return PassageIndex(passages=passages, embeddings=embeddings, article_ids=article_ids)
