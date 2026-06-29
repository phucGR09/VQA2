# Retrieval Module

Image-to-article retrieval pipeline: given a set of news images, find the most relevant news articles from a database. The pipeline runs in sequential phases controlled by `retrieval.py`.

## Architecture Overview

```
Images → [caption] → Captions → [embed_captions] → Caption Embeddings ─┐
                                                                         ├→ [single/group] → Results → [eval]
Articles → [embed_articles] → Article Embeddings ────────────────────────┘
```

The core idea is to bridge the image-text modality gap via VLM-generated captions, then do semantic similarity search in embedding space.

## Pipeline Phases

Run each phase in order:

```bash
python -m src.retrieval.retrieval --phase <PHASE> [--device cuda:0] [--batch_size 8] [--batch_num 8] [--batch_idx 0]
```

| Phase | What it does | Output |
|-------|-------------|--------|
| `caption` | Generates text captions for images using Qwen2.5-VL-7B | `outputs/retrieval/captions/captions_<idx>.json` |
| `merge_captions` | Merges all per-shard caption files into one | `outputs/retrieval/captions/captions.json` |
| `embed_articles` | Encodes all articles (title + content) into vectors | `outputs/retrieval/article_features/article_embeddings.pt` |
| `embed_captions` | Encodes all captions into vectors | `outputs/retrieval/caption_features/caption_embeddings.pt` |
| `single` | Cosine similarity search: each image vs all articles | `outputs/retrieval/results/single_retrieval.csv` |
| `group_vote` | Aggregates per-image results within a group via RRF | `outputs/retrieval/results/group_voting_retrieval.csv` |
| `group_sync` | Summarizes group captions with LLM, then retrieves | `outputs/retrieval/results/group_sync_retrieval.csv` |
| `eval` | Computes Precision@K, Recall@K, MRR on all result CSVs | stdout |

### Parallel Captioning (Sharding)

The `caption` phase supports sharding for multi-GPU parallelism:

```bash
# Run 8 shards in parallel across 8 GPUs
for i in {0..7}; do
  python -m src.retrieval.retrieval --phase caption --device cuda:$i --batch_num 8 --batch_idx $i &
done
wait
python -m src.retrieval.retrieval --phase merge_captions
```

## File Structure

```
retrieval/
├── retrieval.py              # Entry point — CLI, orchestrates all phases
├── config.py                 # RetrievalConfig dataclass (all paths & hyperparams)
├── dataloader.py             # ArticleDataset, ImageDataset (torch Dataset)
├── caption_image.py          # ImageCaptioner — VLM-based image captioning
├── embedding.py              # TextEmbedder — sentence-transformer embeddings
├── single_retrieval.py       # SingleRetrieval — per-image cosine similarity search
├── group_voting_retrieval.py # GroupVotingRetrieval — Reciprocal Rank Fusion across a group
├── group_sync_retrieval.py   # GroupSyncRetrieval — LLM group summary → embed → retrieve
├── evaluation.py             # evaluate() — Precision@K, Recall@K, MRR
├── model.py                  # load_vlm() factory
├── models/
│   └── qwen.py               # QwenVLModel — Qwen2.5-VL wrapper (batch captioning + text gen)
├── preprocess.py             # get_image_transform() — standard ImageNet normalization
└── utils.py                  # load_json, save_json, load_embeddings helpers
```

## Key Classes

### `ImageCaptioner` ([caption_image.py](caption_image.py))
Generates text descriptions for images using a VLM (Qwen2.5-VL-7B-Instruct).
- Supports batched inference with automatic single-image fallback on batch errors
- Shards the dataset so multiple processes can run in parallel
- Saves per-shard JSON files; `merge_captions` phase combines them

### `TextEmbedder` ([embedding.py](embedding.py))
Encodes text into L2-normalized embeddings using `intfloat/multilingual-e5-large`.
- Works for both articles and captions
- Saves embeddings as `.pt` files with corresponding ID lists

### `SingleRetrieval` ([single_retrieval.py](single_retrieval.py))
Retrieves top-K articles for each image individually.
- Computes cosine similarity via matrix multiplication (embeddings are pre-normalized)
- Outputs `single_retrieval.csv` with columns `image_id, article_0, ..., article_{K-1}`

### `GroupVotingRetrieval` ([group_voting_retrieval.py](group_voting_retrieval.py))
Aggregates single-image results across an image group using **Reciprocal Rank Fusion (RRF)**.
- Each image votes for articles; higher-ranked articles get more weight (`1 / rank`)
- Suitable when images in a group are independently captioned and retrieved

### `GroupSyncRetrieval` ([group_sync_retrieval.py](group_sync_retrieval.py))
Builds a single group-level query by summarizing all captions in the group via the LLM.
- More coherent than voting when images in a group depict the same event
- More expensive: requires an LLM forward pass per group

### `QwenVLModel` ([models/qwen.py](models/qwen.py))
Thin wrapper around `Qwen/Qwen2.5-VL-7B-Instruct` from HuggingFace Transformers.
- `generate_caption_batch(image_paths, prompt, max_new_tokens)` — batch image captioning
- `generate_text(prompt, max_new_tokens)` — text-only generation (used for group summaries)

## Configuration

All parameters live in `RetrievalConfig` ([config.py](config.py)):

| Field | Default | Description |
|-------|---------|-------------|
| `image_dir` | `./data/database_image_compress` | Directory of input `.jpg` images |
| `database_path` | `./data/merged_7_database.json` | Article database JSON |
| `groups_path` | `/data/groups.json` | Image group assignments JSON `{group_id: [image_id, ...]}` |
| `caption_dir` | `outputs/retrieval/captions` | Where captions are saved |
| `caption_feature_dir` | `outputs/retrieval/caption_features` | Caption embedding `.pt` files |
| `article_feature_dir` | `outputs/retrieval/article_features` | Article embedding `.pt` files |
| `result_dir` | `outputs/retrieval/results` | Retrieval result CSVs |
| `caption_model_name` | `Qwen/Qwen2.5-VL-7B-Instruct` | VLM for captioning & summarization |
| `embed_model_name` | `intfloat/multilingual-e5-large` | Sentence transformer for embeddings |
| `top_k` | `10` | Number of articles to retrieve per image/group |
| `embed_batch_size` | `64` | Batch size for embedding |
| `group_summary_max_tokens` | `512` | Max tokens for LLM group summary |

## Data Formats

**Article database** (`merged_7_database.json`):
```json
{
  "<article_id>": {
    "title": "...",
    "content": "...",
    "images": [{"image_id": "<image_id>"}, ...]
  }
}
```

**Image groups** (`groups.json`):
```json
{
  "<group_id>": ["<image_id_1>", "<image_id_2>", ...]
}
```

**Result CSV** (`single_retrieval.csv`, `group_voting_retrieval.csv`, `group_sync_retrieval.csv`):
```
image_id,article_0,article_1,...,article_9
img_001,art_42,art_7,...
```

## Evaluation Metrics

`evaluate()` in [evaluation.py](evaluation.py) computes, given ground-truth `{image_id: article_id}`:

- **Precision@K** — fraction of retrieved articles that are relevant (binary: 0 or 1/K)
- **Recall@K** — fraction of relevant articles retrieved (binary: 0 or 1)
- **MRR** — Mean Reciprocal Rank: `1/rank` of the first correct article, 0 if not in top-K

## Dependencies

- `transformers`, `torch` — VLM inference (Qwen2.5-VL)
- `sentence-transformers` — text embedding (multilingual-e5-large)
- `torchvision` — image preprocessing
- `pandas` — result CSV handling
- `Pillow` — image loading
- `tqdm` — progress bars
