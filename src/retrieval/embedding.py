from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.retrieval.dataloader import ArticleDataset


class TextEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        batch_size: int = 64,
        max_tokens: int = 4096,
        chunk_stride: int = 512,
        article_batch_size: int | None = None,
    ):
        self.batch_size = batch_size
        self.article_batch_size = article_batch_size or batch_size
        self.max_tokens = max_tokens
        self.chunk_stride = chunk_stride
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_tokens
        self._tok = self.model.tokenizer

    def embed(self, texts: list[str], show_progress: bool = True) -> torch.Tensor:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

    # ------------------------------------------------------------------
    # Chunking helpers (for articles that exceed max_tokens)
    # ------------------------------------------------------------------

    def _token_len(self, text: str) -> int:
        return len(self._tok(text, truncation=False, add_special_tokens=False)["input_ids"])

    def _chunk(self, text: str) -> list[str]:
        """Split text into overlapping token-window chunks."""
        ids = self._tok(text, truncation=False, add_special_tokens=False)["input_ids"]
        window = self.max_tokens - 2  # reserve 2 for [CLS]/[SEP]
        step = window - self.chunk_stride
        chunks = []
        for start in range(0, len(ids), step):
            chunk_ids = ids[start : start + window]
            chunks.append(self._tok.decode(chunk_ids))
            if start + window >= len(ids):
                break
        return chunks

    def _embed_chunked(self, text: str) -> torch.Tensor:
        """Embed a long text as mean-pooled chunk embeddings (re-normalized)."""
        chunks = self._chunk(text)
        chunk_embs = self.embed(chunks, show_progress=False)  # [N_chunks, D]
        return F.normalize(chunk_embs.mean(dim=0), dim=0)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def embed_articles(self, dataset: ArticleDataset, save_dir: str) -> None:
        article_ids = [dataset[i]["article_id"] for i in range(len(dataset))]
        texts = [dataset[i]["text"] for i in tqdm(range(len(dataset)), desc="Loading articles")]

        # Partition into short (fits in one pass) and long (needs chunking)
        short_idx, long_idx = [], []
        for i, text in enumerate(tqdm(texts, desc="Checking token lengths")):
            if self._token_len(text) <= self.max_tokens - 2:
                short_idx.append(i)
            else:
                long_idx.append(i)

        print(f"Normal: {len(short_idx)}  |  Chunked: {len(long_idx)} ({100*len(long_idx)/len(texts):.1f}%)")

        all_embs: list[torch.Tensor | None] = [None] * len(texts)

        # Batch-embed short articles (efficient, smaller batch to fit GPU)
        short_embs = self.model.encode(
            [texts[i] for i in short_idx],
            batch_size=self.article_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for out_i, src_i in enumerate(short_idx):
            all_embs[src_i] = short_embs[out_i]

        # Chunk-embed long articles one by one
        for src_i in tqdm(long_idx, desc="Chunking long articles"):
            all_embs[src_i] = self._embed_chunked(texts[src_i])

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"article_ids": article_ids, "embeddings": torch.stack(all_embs).cpu()},
            save_path / "article_embeddings.pt",
        )

    def embed_captions(self, captions: dict[str, str], save_dir: str) -> None:
        image_ids = list(captions.keys())
        texts = list(captions.values())
        embeddings = self.embed(texts)
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"image_ids": image_ids, "embeddings": embeddings.cpu()},
            save_path / "caption_embeddings.pt",
        )
