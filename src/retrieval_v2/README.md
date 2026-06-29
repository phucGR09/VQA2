# Retrieval v2 — Image-to-Article Pipeline

Given a query image (or a group of images), retrieve the most relevant Vietnamese news articles from a 28K-article database.

## Architecture overview

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   OFFLINE (once)                    │
                    │                                                     │
                    │  Article DB (28,655 articles)                       │
                    │       │                                             │
                    │  Step 1: chunk → embed with BGE-M3                  │
                    │       └─→  ku_embeddings.pt  (62,864 KUs × 1024)   │
                    └─────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────┐
                    │                  QUERY (per run)                    │
                    │                                                     │
                    │  Query images (107,952 images)                      │
                    │       │                                             │
                    │  Step 2a: caption text → embed with BGE-M3          │
                    │       └─→  caption_embeddings.pt (107,952 × 1024)  │
                    │                                                     │
                    │  Step 2b: image pixels → embed with Jina CLIP v2   │
                    │       └─→  image_embeddings.pt  (N_img × 1024)     │
                    └─────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────┐
                    │               RETRIEVAL + RERANKING                 │
                    │                                                     │
                    │  Step 3: cosine sim (caption↔KU) [+ BM25 hybrid]   │
                    │       └─→  single_retrieval.csv  top-50 per image  │
                    │       └─→  group_retrieval.csv   top-50 per group  │
                    │                                                     │
                    │  Step 4: BGE-Reranker-v2-m3 cross-encoder          │
                    │       └─→  single_reranked.csv   top-10 per image  │
                    │       └─→  group_reranked.csv    top-10 per group  │
                    └─────────────────────────────────────────────────────┘
```

## Data

| Path | Description |
|---|---|
| `./data/merged_7_database.json` | 28,655 articles; each has `title`, `content`, `images[]` |
| `./data/groups.json` | 27,931 groups → list of image IDs (avg 4.1 images/group) |
| `./data/database_image_compress/` | Compressed JPG images keyed by `image_id` |
| `/raid/ltnghia01/phucpv/VQA/image_caption_image_only.json` | Per-image captions: `generated_caption` + `original_caption` |

Total image-article ground-truth pairs: **113,554**

## Embedding backends

Steps 1–3 support two interchangeable embedding backends via `--backend`:

| Backend | Step 1 (KU) | Step 2 (query) | Dim | Space |
|---|---|---|---|---|
| `local` (default) | BGE-M3 over chunked article text | BGE-M3 over caption text | 1024 | text-only |
| `gemini` | Gemini Embedding 2, 1 KU/article, doc prefix | Gemini Embedding 2, **multimodal** (image + caption) | 1536 | unified text+image |

Each backend writes to its **own subfolder**, so results never clash:
`outputs/{ku_features,caption_features,results}/<backend>/…`

**Gemini setup** (`gemini-embedding-2`, GA April 2026 — unified multimodal space, model auto-normalizes embeddings):

```bash
pip install google-genai
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY

# Step 1 — articles → KU embeddings (1 KU/article, doc prefix "title: … | text: …")
python -m src.retrieval_v2.step_1_embed_ku       --backend gemini

# Step 2 — query images → multimodal embeddings (image + generated_caption,
#          query prefix "task: search result | query: …"), 12 concurrent calls
python -m src.retrieval_v2.step_2_embed_captions --backend gemini

# Step 3 — retrieve over the gemini features
python -m src.retrieval_v2.step_3_retrieve --mode both --backend gemini
python -m src.retrieval_v2.step_3_retrieve --mode both --backend gemini --hybrid
```

Notes:
- `--backend` must match across steps 1, 2 and 3 (step 3 reads the matching subfolder).
- All steps are crash-safe and resumable — re-run the same command to continue.
- Tip: run step 2 first with `--limit 100` to sanity-check API key/quota before the full ~108K images.
- Cost (gemini-embedding-2): text $0.20/1M tok, image $0.45/1M tok → full run ≈ $22–76 (Batch API halves it; not used here).

## Steps

### Step 1 — Embed Knowledge Units (articles)

```bash
# local (BGE-M3)
python -m src.retrieval_v2.step_1_embed_ku [--device cuda:0]
# gemini
python -m src.retrieval_v2.step_1_embed_ku --backend gemini
```

- Splits each article (`title + content`) into overlapping token windows (max 1024 tokens, stride 256)
- Embeds every chunk with **BGE-M3** text encoder
- Output: `outputs/ku_features/ku_embeddings.pt`
  - `ku_ids`: e.g. `"abc123__chunk_0"`
  - `article_ids`: parent article for each KU
  - `embeddings`: `[62864, 1024]`

Crash-safe: progress saved per 5000 KUs in `outputs/ku_features/parts/`.

---

### Step 2a — Embed captions (text query)

```bash
# local (BGE-M3, caption text)
python -m src.retrieval_v2.step_2_embed_captions [--device cuda:0] [--batch_size 64]
# gemini (multimodal: image + generated_caption)
python -m src.retrieval_v2.step_2_embed_captions --backend gemini
```

- Reads all unique image IDs from `groups.json`
- `local`: concatenates `generated_caption + original_caption`, embeds with **BGE-M3** (shared text space with KU embeddings)
- `gemini`: embeds the **image** interleaved with its `generated_caption` (same unified space as Gemini KU embeddings)
- Output: `outputs/caption_features/<backend>/caption_embeddings.pt`
  - `image_ids`, `embeddings`: `[107952, 1024]` (local) / `[N_img, 1536]` (gemini)

Crash-safe: checkpoint every 20 batches to `caption_embeddings_ckpt.pt`.

---

### Step 2b — Embed images (visual query)

```bash
python -m src.retrieval_v2.step_2_embed_images [--device cuda:0] [--batch_size 64]
```

- Encodes raw images with **Jina CLIP v2** image encoder
- Output: `outputs/image_features/image_embeddings.pt`
  - `image_ids`, `embeddings`: `[N_img, 1024]`

> Note: Jina CLIP v2 and BGE-M3 live in **different embedding spaces** (1024-dim each, but not interchangeable). Image embeddings are not yet fused into step 3 retrieval.

Crash-safe: checkpoint every 20 batches; OOM retries image-by-image.

---

### Step 3 — Retrieve top-N candidates

```bash
# Dense only
python -m src.retrieval_v2.step_3_retrieve --mode both [--device cuda:0]

# Dense + BM25 hybrid (better recall)
python -m src.retrieval_v2.step_3_retrieve --mode both --hybrid [--device cuda:0]
```

**Dense retrieval**
- Scores every (caption_embedding, KU_embedding) pair via cosine similarity
- Max-pools KU scores to article level
- GPU-batched: 2048 images × all KUs per block

**Hybrid mode (`--hybrid`)**
- Adds GPU-accelerated BM25 (sparse matmul) on caption text vs article text
- Fuses dense + BM25 rankings via Reciprocal Rank Fusion (RRF, k=60)

**Group retrieval**
- Aggregates per-image rankings within each group via RRF
- Groups share a single article list (top-50)

Outputs (under `outputs/results/<backend>/`):
- `single_retrieval.csv` — 107,952 rows × 50 article columns
- `group_retrieval.csv`  — 27,931 rows × 50 article columns

---

### Step 4 — Rerank with cross-encoder

```bash
python -m src.retrieval_v2.step_4_rerank --device cuda:0 --batch_size 32 [--load_in_4bit]
```

- Cross-encoder: **BGE-Reranker-v2-m3** (text-only, no image)
- Input per pair: `(caption_text, article_title + content[:2000 chars])`
- Scores top-50 candidates, keeps top-10
- Pairs sorted by length before batching to minimise padding waste

Crash-safe: each finished image appended to `.jsonl`; CSV rebuilt on exit/interrupt.

Outputs:
- `outputs/results/single_reranked.csv` — 107,952 rows × 10 article columns
- `outputs/results/group_reranked.csv`  — 27,931 rows × 10 article columns

---

### Evaluate

```bash
python -m src.retrieval_v2.evaluate
```

Auto-detects mode from filename (`group_*` → group mode, expands groups to per-image metrics).

## Current results

| File | hits@1 | hits@10 | recall@10 | MRR | n_eval |
|---|---|---|---|---|---|
| single_retrieval | 10,032 | 21,390 | 0.1981 | 0.1245 | 107,952 |
| single_reranked | 8,957 | 29,529 | 0.2735 | 0.1344 | 107,952 |
| group_retrieval | 14,712 | 29,924 | 0.2635 | 0.1738 | 113,554 |
| **group_reranked** | **15,902** | **46,995** | **0.4139** | **0.2185** | 113,554 |

Retrieval ceiling (correct article in top-50): **~46%** for single mode.

## Known bottlenecks

1. **Low retrieval ceiling** — 53.7% of single images have no correct article in top-50. Caption text alone is often insufficient to match the exact source article.
2. **hits@1 regression after reranking (single)** — BGE-Reranker is text-only; without visual context it sometimes demotes the correct article from rank 1.
3. **Image embeddings incomplete** — Only 3,000 / 107,952 images embedded with Jina CLIP v2. Not yet used in step 3.

## Suggested next steps

| Priority | Change | Expected gain |
|---|---|---|
| High | Run step 3 with `--hybrid` | Better keyword recall (names, orgs, events) |
| High | Increase `RETRIEVAL_TOP_N` 50 → 100 | Direct ceiling improvement |
| Medium | Complete step 2b (all 107,952 images) + embed KUs with Jina CLIP v2 text encoder + fuse CLIP image→text score in step 3 | Multimodal retrieval |
| Low | Fine-tune BGE-M3 / reranker on domain data | Long-term quality |

## Output file layout

```
src/retrieval_v2/
├── outputs/                          # <backend> = local | gemini
│   ├── ku_features/<backend>/
│   │   └── ku_embeddings.pt
│   ├── caption_features/<backend>/
│   │   └── caption_embeddings.pt
│   ├── image_features/
│   │   └── image_embeddings.pt
│   └── results/<backend>/
│       ├── single_retrieval.csv
│       ├── group_retrieval.csv
│       ├── single_reranked.csv   ← single_reranked.jsonl (crash backup)
│       └── group_reranked.csv    ← group_reranked.jsonl  (crash backup)
├── step_1_embed_ku.py
├── step_2_embed_captions.py
├── step_2_embed_images.py
├── step_3_retrieve.py
├── step_4_rerank.py
├── evaluate.py
├── gemini_embed.py        # shared Gemini Embedding 2 client (--backend gemini)
└── utils.py
```



export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=project-2c01ba8c-924c-44c3-8ee
export GOOGLE_CLOUD_LOCATION=us-central1
export GEMINI_EMBED_MODEL=gemini-embedding-2-preview
export GEMINI_TOKENS_PER_MINUTE=50000          # đổi theo quota thật sau khi xin tăng
export GEMINI_WORKERS=16