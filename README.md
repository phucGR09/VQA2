# Vietnamese Visual Question Answering (VQA)

A multi-level baseline system for Vietnamese VQA, progressing from zero-shot vision-language models to a fully fine-tuned multimodal architecture (visual encoder + MLP projector + LoRA-adapted LLM).

---

## Project Structure

```
VQA/
├── src/
│   ├── VQA_baseline/           # Main source — all levels
│   │   ├── baseline_config.py  # Model registries and training configs
│   │   ├── data_utils.py       # Data loading helpers
│   │   ├── flatten_dataset.py  # Dataset preprocessing
│   │   ├── level2_image_qa.py  # Level 2: Image + Question (zero-shot)
│   │   ├── level3_article_qa.py# Level 3: Article + Question (no image)
│   │   ├── level4_rag_vqa.py   # Level 4: RAG — Image + Article + Question
│   │   ├── level5_dataset.py   # Dataset classes for fine-tuning
│   │   ├── level5_train.py     # Level 5: Fine-tuning (MLP pre-train + LoRA VQA)
│   │   ├── level5_eval.py      # Level 5: Evaluation
│   │   ├── metrics.py          # BLEU, ROUGE, BERTScore, CIDEr
│   │   ├── model_registry.py   # Zero-shot model wrappers
│   │   ├── prompt_utils.py     # Prompt templates
│   │   ├── retrieval_utils.py  # Contriever passage retrieval (Level 4 Case B)
│   │   └── utils.py            # ArticleSelector, QASample, result helpers
│   ├── config.py               # Pipeline-level configs and registries
│   ├── model.py                # MLPProjector and VQAModel
│   └── visual_encoder.py       # VisualEncoderWrapper (multi-model)
├── data/
│   ├── database.json           # Article database
│   ├── image_qa.json           # Full QA dataset
│   └── splits/
│       ├── train_set.json      # Training split (with captions)
│       └── test_split.json     # Test split
├── outputs/
│   ├── checkpoints/            # Saved model weights (see below)
│   └── evaluation/             # Evaluation results and CSV reports
├── compress_jpg.py             # Image preprocessing utility
├── convert_to_jpg.py           # Image format conversion utility
├── merge_jpg_folders.py        # Merge image folders utility
└── requirements.txt
```

---

## Installation

```bash
# 1. Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 2. Install remaining dependencies
pip install -r requirements.txt
```

> **Optional:** Install `flash-attn` for faster training (~2–3× speedup, ~30% less VRAM):
> ```bash
> pip install flash-attn --no-build-isolation
> ```

---

## Data Layout

Place your data under `data/` at the project root:

```
data/
├── database.json          # Article database  {article_id: {title, content, url}}
├── image_qa.json          # Full QA set       [{image_path, question, answer, article_id, caption}]
├── images/                # All database images
│   ├── <image_id>.jpg
│   └── ...
└── splits/
    ├── train_set.json     # Training split (must include caption field)
    └── test_split.json    # Test split
```

---

## Baseline Levels

All scripts are run from the **`src/`** directory.

```bash
cd src
```

### Level 2 — Image + Question (zero-shot)

```bash
python VQA_baseline/level2_image_qa.py \
    --models vintern_1b_v3_5 qwen2vl_7b internvl3_8b \
    --test_split ../data/splits/test_split.json \
    --images_dir ../data/images \
    --output_dir ../outputs/evaluation \
    --device cuda:0
```

Use `--models all` to run every model in the registry.

### Level 3 — Article + Question (no image)

```bash
python VQA_baseline/level3_article_qa.py \
    --models qwen2.5_7b llama3.1_8b \
    --test_split ../data/splits/test_split.json \
    --database ../data/database.json \
    --output_dir ../outputs/evaluation \
    --device cuda:0
```

### Level 4 — RAG VQA (Image + Retrieved Article + Question)

```bash
# Case A: ground-truth article (ceiling)
python VQA_baseline/level4_rag_vqa.py \
    --models qwen2vl_7b internvl3_8b \
    --case A \
    --test_split ../data/splits/test_split.json \
    --images_dir ../data/images \
    --database ../data/database.json \
    --output_dir ../outputs/evaluation \
    --device cuda:0

# Case B: retrieved article (real-world)
python VQA_baseline/level4_rag_vqa.py \
    --case B \
    --retrieval_results ../outputs/retrieval/summary.csv \
    ...
```

---

## Level 5 — Fine-tuned VQA

Training runs in two sequential stages. Both stages must use the **same** `--visual_model` and `--llm_model` pair.

### Stage 1 — MLP Projector Pre-training

Trains the MLP projector on image → caption with the visual encoder and LLM frozen.

```bash
python VQA_baseline/level5_train.py --stage mlp \
    --visual_model clip_vit_l14 \
    --llm_model qwen2.5_7b \
    --train_split ../data/splits/train_set.json \
    --images_dir ../data/images \
    --checkpoint_dir ../outputs/checkpoints \
    --device cuda:0
```

Output: `outputs/checkpoints/task1_mlp_best.pt`

### Stage 2 — VQA Fine-tuning (LoRA)

Fine-tunes the MLP + LoRA adapters on (image, article, question) → answer with the visual encoder frozen.

```bash
python VQA_baseline/level5_train.py --stage vqa \
    --visual_model clip_vit_l14 \
    --llm_model qwen2.5_7b \
    --train_split ../data/splits/train_set.json \
    --images_dir ../data/images \
    --checkpoint_dir ../outputs/checkpoints \
    --task1_checkpoint ../outputs/checkpoints/task1_mlp_best.pt \
    --device cuda:0
```

Output: `outputs/checkpoints/task2_vqa_best.pt`

#### Optional training flags

| Flag | Default | Description |
|---|---|---|
| `--flash_attn` | off | Enable Flash Attention 2 (requires `flash-attn`) |
| `--selector` | `bm25` | Article sentence selector: `bm25`, `dense`, `first` |
| `--top_k_sentences` | 5 | Sentences injected as context |
| `--mlp_epochs` | 2 | Override MLP training epochs |
| `--vqa_epochs` | 3 | Override VQA training epochs |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--max_samples` | all | Cap training samples (quick experiments) |

### Level 5 Evaluation

```bash
# Case A — ground-truth article
python VQA_baseline/level5_eval.py \
    --checkpoint ../outputs/checkpoints/task2_vqa_best.pt \
    --case A \
    --test_split ../data/splits/test_split.json \
    --images_dir ../data/images \
    --output_dir ../outputs/evaluation \
    --device cuda:0

# Case B — retrieved article
python VQA_baseline/level5_eval.py \
    --checkpoint ../outputs/checkpoints/task2_vqa_best.pt \
    --case B \
    --database ../data/database.json \
    --retrieval_results ../outputs/retrieval/summary.csv \
    --output_dir ../outputs/evaluation \
    --device cuda:0
```

---

## Model Weights and Checkpoints

```
outputs/checkpoints/
├── task1_mlp_best.pt      # Stage 1: MLP projector weights
│                          #   keys: epoch, visual_model_key, llm_model_key,
│                          #         mlp_config, mlp_state_dict, loss
└── task2_vqa_best.pt      # Stage 2: MLP + LoRA adapter weights
                           #   keys: epoch, visual_model_key, llm_model_key,
                           #         mlp_config, mlp_state_dict,
                           #         lora_target_modules, lora_adapter_state_dict, loss
```

> The checkpoint embeds `visual_model_key` and `llm_model_key`. Stage 2 will raise an error if the checkpoint's `d_llm` does not match the selected LLM — re-run Stage 1 with the same model pair if this happens.

---

## Supported Models

### Visual Encoders (`--visual_model`)

| Key | Model |
|---|---|
| `clip_vit_l14` | openai/clip-vit-large-patch14 |
| `clip_vit_b32` | openai/clip-vit-base-patch32 |
| `clip_vit_b16` | openai/clip-vit-base-patch16 |
| `clip_vit_l14_336` | openai/clip-vit-large-patch14-336 |
| `siglip_so400m` | google/siglip-so400m-patch14-384 |
| `dinov2_large` | facebook/dinov2-large |
| `blip2_vitg` | Salesforce/blip2-opt-2.7b |
| `resnet50` | microsoft/resnet-50 |
| `resnet101` | microsoft/resnet-101 |

### LLMs (`--llm_model`)

| Key | Model |
|---|---|
| `qwen2.5_7b` | Qwen/Qwen2.5-7B-Instruct |
| `qwen2.5_3b` | Qwen/Qwen2.5-3B-Instruct |
| `qwen2.5_14b` | Qwen/Qwen2.5-14B-Instruct |
| `llama3.1_8b` | meta-llama/Llama-3.1-8B-Instruct |
| `llama3.2_3b` | meta-llama/Llama-3.2-3B-Instruct |
| `gemma2_9b` | google/gemma-2-9b-it |
| `gemma3_12b` | google/gemma-3-12b-it |
| `mistral_7b` | mistralai/Mistral-7B-Instruct-v0.3 |
| `phi3.5_mini` | microsoft/Phi-3.5-mini-instruct |
| `internvl3_8b` | OpenGVLab/InternVL3-8B |
| `vintern_1b_v3_5` | 5CD-AI/Vintern-1B-v3_5 |

---

## Evaluation Metrics

Results are saved as CSV and text reports under `outputs/evaluation/`.  
Metrics computed: **BLEU-1/2/3/4**, **ROUGE-L**, **BERTScore (F1)**, **CIDEr**.
