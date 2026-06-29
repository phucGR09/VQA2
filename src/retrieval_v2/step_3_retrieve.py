"""
Step 3 — Retrieve top-N articles per image (and optionally per group).

Cosine similarity between caption embeddings and KU embeddings; KU scores are
max-pooled to article level. Optionally aggregates per-image results within a
group via Reciprocal Rank Fusion (RRF).

With --hybrid: BM25 keyword retrieval is fused with dense retrieval via RRF.
BM25 is implemented as a precomputed sparse weight matrix on GPU, so scoring
107K images × 28K articles is a single batched sparse matmul per chunk —
no Python loops over the corpus.

Outputs (per --backend, must match steps 1 & 2):
    src/retrieval_v2/outputs/results/<backend>/single_retrieval.csv
    src/retrieval_v2/outputs/results/<backend>/group_retrieval.csv   (--mode group/both)

Usage:
    python -m src.retrieval_v2.step_3_retrieve --mode both                    # local
    python -m src.retrieval_v2.step_3_retrieve --mode both --backend gemini
    python -m src.retrieval_v2.step_3_retrieve --mode both --hybrid
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
KU_FEATURE_ROOT      = Path(__file__).parent / "outputs" / "ku_features"
CAPTION_FEATURE_ROOT = Path(__file__).parent / "outputs" / "caption_features"
OUTPUT_ROOT          = Path(__file__).parent / "outputs" / "results"
GROUPS_PATH        = "./data/groups.json"
DATABASE_PATH      = "./data/merged_7_database.json"
CAPTION_PATH       = "/raid/ltnghia01/phucpv/VQA/image_caption_image_only.json"

RETRIEVAL_TOP_N = 50    # candidates passed to reranker
FINAL_TOP_K     = 50    # articles in output CSV (when no reranker follows)
SCORE_CHUNK     = 2048  # images per similarity block (larger = faster on GPU)
RRF_K           = 60    # RRF constant
# ─────────────────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ── GPU-accelerated BM25 ──────────────────────────────────────────────────────
class BM25GPU:
    """
    BM25 Okapi scoring via a precomputed sparse weight matrix on GPU.

    At build time we compute a sparse [N_docs, V] matrix where each entry is
    IDF(t) × TF_norm(t, d).  At query time, scoring a batch of B queries is a
    single sparse @ dense matmul: [N_docs, V] @ [V, B] → [N_docs, B].

    This replaces per-query Python loops over the corpus and is orders of
    magnitude faster when B is large (e.g. SCORE_CHUNK = 2048).
    """
    _K1 = 1.5
    _B  = 0.75

    def __init__(self, corpus_texts: list[str], device: torch.device):
        import scipy.sparse as sp

        print("    Tokenising corpus …", flush=True)
        tokenized = [_tokenize(t) for t in corpus_texts]

        vocab = sorted({w for doc in tokenized for w in doc})
        self.v_idx: dict[str, int] = {w: i for i, w in enumerate(vocab)}
        V, N = len(vocab), len(corpus_texts)
        print(f"    vocab={V:,}  docs={N:,}", flush=True)

        doc_len = np.array([len(d) for d in tokenized], dtype=np.float32)
        avgdl   = float(doc_len.mean()) or 1.0

        df = np.zeros(V, dtype=np.float32)
        for doc in tokenized:
            for w in set(doc):
                t = self.v_idx.get(w)
                if t is not None:
                    df[t] += 1

        idf = np.log((N - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

        print("    Building sparse BM25 matrix …", flush=True)
        rows, cols, vals = [], [], []
        for d, (doc, dl) in enumerate(zip(tokenized, doc_len)):
            for w, freq in Counter(doc).items():
                t = self.v_idx.get(w)
                if t is None:
                    continue
                tf_norm = freq * (self._K1 + 1) / (
                    freq + self._K1 * (1 - self._B + self._B * dl / avgdl)
                )
                rows.append(d)
                cols.append(t)
                vals.append(float(idf[t] * tf_norm))

        sp_mat = sp.csr_matrix(
            (np.array(vals, np.float32),
             (np.array(rows, np.int32), np.array(cols, np.int32))),
            shape=(N, V),
        )
        coo     = sp_mat.tocoo()
        indices = torch.from_numpy(np.vstack([coo.row, coo.col]).copy()).long()
        values  = torch.from_numpy(coo.data.copy()).float()
        self.mat    = torch.sparse_coo_tensor(indices, values, (N, V)).coalesce().to(device)
        self.V      = V
        self.device = device
        print(f"    Sparse matrix: {len(vals):,} non-zeros on {device}", flush=True)

    def score_batch(self, query_token_lists: list[list[str]]) -> torch.Tensor:
        """Returns [B, N_docs] BM25 scores on GPU."""
        B = len(query_token_lists)
        q_rows, q_cols = [], []
        for j, tokens in enumerate(query_token_lists):
            for t in tokens:
                idx = self.v_idx.get(t)
                if idx is not None:
                    q_rows.append(j)
                    q_cols.append(idx)

        if q_rows:
            idx_t = torch.tensor([q_rows, q_cols], dtype=torch.long, device=self.device)
            val_t = torch.ones(len(q_rows), dtype=torch.float32, device=self.device)
            q_mat = torch.sparse_coo_tensor(idx_t, val_t, (B, self.V)).to_dense()
        else:
            q_mat = torch.zeros(B, self.V, device=self.device)

        # [N, V] sparse  @  [V, B] dense  →  [N, B]  →  [B, N]
        return torch.sparse.mm(self.mat, q_mat.T).T


def build_bm25_index(unique_articles: list[str], db: dict, device: torch.device) -> BM25GPU:
    corpus_texts = [
        f"{db.get(a, {}).get('title', '')} {db.get(a, {}).get('content', '')}"
        for a in unique_articles
    ]
    return BM25GPU(corpus_texts, device)


def load_caption_texts(image_ids: list[str]) -> dict[str, str]:
    print(f"Loading caption texts for BM25: {CAPTION_PATH}")
    with open(CAPTION_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    texts = {}
    for iid in image_ids:
        cap  = raw.get(iid, {})
        gen  = (cap.get("generated_caption") or "").strip()
        orig = (cap.get("original_caption")  or "").strip()
        texts[iid] = f"{gen} {orig}".strip() if orig else gen
    return texts
# ─────────────────────────────────────────────────────────────────────────────


def retrieve_single(
    image_ids: list[str],
    query_embs: torch.Tensor,
    article_ids: list[str],
    ku_embs: torch.Tensor,
    top_n: int,
    bm25_index: BM25GPU | None = None,
    bm25_texts: dict | None = None,
) -> pd.DataFrame:
    unique_articles = list(dict.fromkeys(article_ids))
    art_to_idx      = {a: i for i, a in enumerate(unique_articles)}
    device          = ku_embs.device
    ku_art_idx      = torch.tensor([art_to_idx[a] for a in article_ids], device=device)
    n_art           = len(unique_articles)
    k               = min(top_n, n_art)

    query_embs_f = query_embs.float()
    ku_embs_t    = ku_embs.float().T
    hybrid       = bm25_index is not None and bm25_texts is not None

    rows = []
    for start in tqdm(range(0, len(image_ids), SCORE_CHUNK), desc="Scoring"):
        chunk_ids  = image_ids[start : start + SCORE_CHUNK]
        chunk_embs = query_embs_f[start : start + SCORE_CHUNK]
        c          = len(chunk_ids)

        # ── Dense scores ──────────────────────────────────────────────────────
        scores_ku  = chunk_embs @ ku_embs_t                      # [c, N_ku]
        art_scores = scores_ku.new_full((c, n_art), float("-inf"))
        art_scores.scatter_reduce_(
            1, ku_art_idx.unsqueeze(0).expand(c, -1), scores_ku, reduce="amax"
        )                                                         # [c, n_art]

        if hybrid:
            # ── Dense RRF — vectorised via double-argsort on GPU ──────────────
            dense_rank_of = (
                art_scores.argsort(dim=1, descending=True)
                          .argsort(dim=1)
            )                                                     # [c, n_art] GPU int
            dense_rrf = 1.0 / (RRF_K + dense_rank_of.float() + 1)  # [c, n_art] GPU float

            # ── BM25 scores — single sparse matmul on GPU ─────────────────────
            token_lists  = [_tokenize(bm25_texts.get(iid, "")) for iid in chunk_ids]
            bm25_scores  = bm25_index.score_batch(token_lists)    # [c, n_art] GPU float
            bm25_rank_of = (
                (-bm25_scores).argsort(dim=1).argsort(dim=1)
            )                                                      # [c, n_art] GPU int
            bm25_rrf = 1.0 / (RRF_K + bm25_rank_of.float() + 1)  # [c, n_art] GPU float

            # ── RRF merge + topk — all on GPU ─────────────────────────────────
            rrf_scores = dense_rrf + bm25_rrf                     # [c, n_art]
            top_idx    = rrf_scores.topk(k, dim=1).indices        # [c, k]

            for i, img_id in enumerate(chunk_ids):
                row = {"image_id": img_id}
                for j, idx in enumerate(top_idx[i].tolist()):
                    row[f"article_{j}"] = unique_articles[idx]
                rows.append(row)
        else:
            top_idx = art_scores.topk(k, dim=1).indices
            for i, img_id in enumerate(chunk_ids):
                row = {"image_id": img_id}
                for j, idx in enumerate(top_idx[i].tolist()):
                    row[f"article_{j}"] = unique_articles[idx]
                rows.append(row)

    return pd.DataFrame(rows)


def retrieve_group(single_df: pd.DataFrame, groups: dict, top_k: int) -> pd.DataFrame:
    article_cols = sorted(c for c in single_df.columns if c.startswith("article_"))
    indexed      = single_df.set_index("image_id")
    rows = []
    for group_id, image_ids in groups.items():
        scores: dict[str, float] = defaultdict(float)
        for img_id in image_ids:
            if img_id not in indexed.index:
                continue
            for rank, col in enumerate(article_cols):
                art_id = indexed.at[img_id, col]
                if pd.notna(art_id):
                    scores[art_id] += 1.0 / (rank + 1)
        top = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        rows.append({"image_id": group_id, **{f"article_{j}": a for j, a in enumerate(top)}})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["single", "group", "both"], default="both")
    parser.add_argument("--backend", choices=["local", "gemini"], default="local",
                        help="Which embedding backend's features to retrieve over "
                             "(must match step 1 & 2). Default: local")
    parser.add_argument("--hybrid", action="store_true",
                        help="Enable BM25 + dense RRF fusion (GPU sparse matmul)")
    parser.add_argument("--device", default=None,
                        help="Device to use, e.g. cuda, cuda:0, cuda:3, cpu (default: auto)")
    args = parser.parse_args()

    ku_embed_path      = KU_FEATURE_ROOT      / args.backend / "ku_embeddings.pt"
    caption_embed_path = CAPTION_FEATURE_ROOT / args.backend / "caption_embeddings.pt"
    output_dir         = OUTPUT_ROOT / args.backend
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading KU embeddings ({args.backend}): {ku_embed_path}")
    ku_data     = torch.load(ku_embed_path, weights_only=True)
    article_ids = ku_data["article_ids"]
    ku_embs     = ku_data["embeddings"].to(device)
    print(f"  {len(ku_data['ku_ids'])} KUs, dim={ku_embs.shape[1]}")

    print(f"Loading caption embeddings ({args.backend}): {caption_embed_path}")
    if not caption_embed_path.exists():
        raise FileNotFoundError(
            f"{caption_embed_path} not found — run step_2_embed_captions "
            f"--backend {args.backend} first."
        )
    cap_data   = torch.load(caption_embed_path, weights_only=True)
    image_ids  = cap_data["image_ids"]
    query_embs = cap_data["embeddings"].to(device)
    print(f"  {len(image_ids)} captions, dim={query_embs.shape[1]}")

    bm25_index = None
    bm25_texts = None
    if args.hybrid:
        print(f"Loading article database: {DATABASE_PATH}")
        with open(DATABASE_PATH, encoding="utf-8") as f:
            db = json.load(f)
        unique_articles = list(dict.fromkeys(article_ids))
        print(f"  Building GPU BM25 index over {len(unique_articles)} articles…")
        bm25_index = build_bm25_index(unique_articles, db, device)
        bm25_texts = load_caption_texts(image_ids)
        print("  BM25 index ready.")

    single_df = None

    if args.mode in ("single", "both"):
        print(f"Single retrieval → top-{RETRIEVAL_TOP_N} per image "
              f"({'hybrid' if args.hybrid else 'dense'})…")
        single_df = retrieve_single(
            image_ids, query_embs, article_ids, ku_embs, RETRIEVAL_TOP_N,
            bm25_index=bm25_index, bm25_texts=bm25_texts,
        )
        out = output_dir / "single_retrieval.csv"
        single_df.to_csv(out, index=False)
        print(f"  Saved → {out}")

    if args.mode in ("group", "both"):
        if single_df is None:
            single_df = pd.read_csv(output_dir / "single_retrieval.csv")
        with open(GROUPS_PATH, encoding="utf-8") as f:
            groups = json.load(f)
        print(f"Group retrieval (RRF): {len(groups)} groups…")
        group_df = retrieve_group(single_df, groups, FINAL_TOP_K)
        out = output_dir / "group_retrieval.csv"
        group_df.to_csv(out, index=False)
        print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
