from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


# ─────────────────────────────────────────────────────────────────────────────
# Model registries
# ─────────────────────────────────────────────────────────────────────────────

# Visual encoder registry
# Keys: short slug used everywhere in configs and checkpoint filenames
# d_v       : hidden dim of last_hidden_state (feeds MLP projector input)
# n_tokens  : number of visual tokens after encoding
# embed_dim : CLIP-projection / similarity-search dim (used in phase0/phase2)
# family    : used by VisualEncoderWrapper to select load path

VISUAL_MODELS: Dict[str, Dict[str, Any]] = {
    "clip_vit_l14": {
        "id":        "openai/clip-vit-large-patch14",
        "family":    "clip",
        "d_v":       1024,
        "n_tokens":  257,   # 1 CLS + 16×16 patches  (224/14=16)
        "embed_dim": 768,
    },
    "clip_vit_b32": {
        "id":        "openai/clip-vit-base-patch32",
        "family":    "clip",
        "d_v":       768,
        "n_tokens":  50,    # 1 CLS + 7×7 patches  (224/32=7)
        "embed_dim": 512,
    },
    "clip_vit_b16": {
        "id":        "openai/clip-vit-base-patch16",
        "family":    "clip",
        "d_v":       768,
        "n_tokens":  197,   # 1 CLS + 14×14 patches  (224/16=14)
        "embed_dim": 512,
    },
    "clip_vit_l14_336": {
        "id":        "openai/clip-vit-large-patch14-336",
        "family":    "clip",
        "d_v":       1024,
        "n_tokens":  577,   # 1 CLS + 24×24 patches  (336/14=24)
        "embed_dim": 768,
    },
    "siglip_so400m": {
        "id":        "google/siglip-so400m-patch14-384",
        "family":    "siglip",
        "d_v":       1152,
        "n_tokens":  729,   # 27×27 patches  (no CLS, 384/14≈27)
        "embed_dim": 1152,
    },
    "blip2_vitg": {
        "id":        "Salesforce/blip2-opt-2.7b",
        "family":    "blip2",
        "d_v":       1408,
        "n_tokens":  257,   # ViT-g/14 output before Q-Former
        "embed_dim": 1408,
    },
    "dinov2_large": {
        "id":        "facebook/dinov2-large",
        "family":    "dinov2",
        "d_v":       1024,
        "n_tokens":  256,   # 16×16 patches, no CLS in pooled output
        "embed_dim": 1024,
    },
    "resnet50": {
        "id":        "microsoft/resnet-50",
        "family":    "resnet",
        "d_v":       2048,
        "n_tokens":  49,    # 7×7 spatial grid at final stage
        "embed_dim": 2048,
    },
    "resnet101": {
        "id":        "microsoft/resnet-101",
        "family":    "resnet",
        "d_v":       2048,
        "n_tokens":  49,
        "embed_dim": 2048,
    },
}

# LLM registry
# d_llm: hidden embedding dim — must match MLP projector output

LLM_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen2.5_7b": {
        "id":     "Qwen/Qwen2.5-7B-Instruct",
        "family": "qwen",
        "d_llm":  3584,
    },
    "qwen2.5_3b": {
        "id":     "Qwen/Qwen2.5-3B-Instruct",
        "family": "qwen",
        "d_llm":  2048,
    },
    "qwen2.5_14b": {
        "id":     "Qwen/Qwen2.5-14B-Instruct",
        "family": "qwen",
        "d_llm":  5120,
    },
    "llama3.1_8b": {
        "id":     "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama",
        "d_llm":  4096,
    },
    "llama3.2_3b": {
        "id":     "meta-llama/Llama-3.2-3B-Instruct",
        "family": "llama",
        "d_llm":  3072,
    },
    "gemma2_9b": {
        "id":     "google/gemma-2-9b-it",
        "family": "gemma",
        "d_llm":  3584,
    },
    "gemma2_2b": {
        "id":     "google/gemma-2-2b-it",
        "family": "gemma",
        "d_llm":  2304,
    },
    "mistral_7b": {
        "id":     "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "mistral",
        "d_llm":  4096,
    },
    "phi3.5_mini": {
        "id":     "microsoft/Phi-3.5-mini-instruct",
        "family": "phi",
        "d_llm":  3072,
    },
}

# LoRA attention projection module names per LLM family.
# Phi uses a fused qkv_proj; all others use separate projections.
LORA_TARGETS: Dict[str, List[str]] = {
    "qwen":    ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gemma":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi":     ["qkv_proj", "o_proj"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp_config(visual_key: str, llm_key: str, d_hidden: int = 2048) -> "MLPProjectorConfig":
    """Return an MLPProjectorConfig with dims auto-derived from the registries."""
    if visual_key not in VISUAL_MODELS:
        raise KeyError(f"Unknown visual model key '{visual_key}'. Available: {list(VISUAL_MODELS)}")
    if llm_key not in LLM_MODELS:
        raise KeyError(f"Unknown LLM key '{llm_key}'. Available: {list(LLM_MODELS)}")
    return MLPProjectorConfig(
        d_v=VISUAL_MODELS[visual_key]["d_v"],
        d_hidden=d_hidden,
        d_llm=LLM_MODELS[llm_key]["d_llm"],
    )


def get_lora_targets(llm_key: str) -> List[str]:
    """Return the correct LoRA target module names for a given LLM key."""
    if llm_key not in LLM_MODELS:
        raise KeyError(f"Unknown LLM key '{llm_key}'. Available: {list(LLM_MODELS)}")
    family = LLM_MODELS[llm_key]["family"]
    if family not in LORA_TARGETS:
        raise KeyError(
            f"No LoRA target list defined for family '{family}'. "
            "Add it to LORA_TARGETS in config.py."
        )
    return LORA_TARGETS[family]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 – Paths & Embedding
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    data_dir: Path = Path("data")
    images_dir: Path = Path("../../tracked/database_images")
    embedded_image_dir: Path = Path("data/embedded_image")
    embedded_article_dir: Path = Path("data/embedded_article")
    splits_dir: Path = Path("data/splits")
    database_path: Path = Path("../../tracked/database.json")
    image_qa_path: Path = Path("../../image_vqa.json")
    master_records_path: Path = Path("data/splits/master_records.json")
    retrieval_output_dir: Path = Path("outputs/retrieval")

    def get_embedded_image_dir(self, backend: str) -> Path:
        """Return backend-specific image embedding dir.

        e.g. backend="clip"   → data/embedded_image/clip/
             backend="gemini" → data/embedded_image/gemini/

        Both phases (0 and 2) must call this with the same backend so their
        vector spaces are compatible.
        """
        return Path(self.embedded_image_dir) / backend

    def make_dirs(self):
        for d in [
            self.embedded_image_dir,
            self.embedded_article_dir,
            self.splits_dir,
            self.retrieval_output_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Visual encoder used for Phase 3/4 VQA (vision encoder + MLP projector)."""
    visual_model_key: str = "clip_vit_l14"

    @property
    def model_name(self) -> str:
        return VISUAL_MODELS[self.visual_model_key]["id"]

    @property
    def embed_dim(self) -> int:
        return VISUAL_MODELS[self.visual_model_key]["embed_dim"]

    @property
    def d_v(self) -> int:
        return VISUAL_MODELS[self.visual_model_key]["d_v"]

    @property
    def n_visual_tokens(self) -> int:
        return VISUAL_MODELS[self.visual_model_key]["n_tokens"]


@dataclass
class EmbedderConfig:
    """
    Controls which image embedding backend is used in Phase 0 (DB build)
    and Phase 2 (query embedding).  Both phases MUST use the same backend
    so their embedding spaces are compatible.

    backend : "clip"   → local GPU inference via HuggingFace CLIPModel
              "gemini" → Google AI Studio API (gemini-embedding-2-preview)
    """
    backend: str = "clip"                           # "clip" | "gemini"

    # ── CLIP settings ────────────────────────────────────────────────────────
    clip_model_name: str = "openai/clip-vit-large-patch14"
    clip_embed_dim: int = 768
    clip_batch_size: int = 64
    clip_num_workers: int = 8
    clip_fp16: bool = True

    # ── Gemini settings ──────────────────────────────────────────────────────
    gemini_model_name: str = "gemini-embedding-2-preview"
    gemini_embed_dim: int = 3072
    # Name of the environment variable that holds the API key (set in .env)
    gemini_api_key_env: str = "GOOGLE_API_KEY"

    @property
    def embed_dim(self) -> int:
        """Active embedding dimensionality for the chosen backend."""
        return self.clip_embed_dim if self.backend == "clip" else self.gemini_embed_dim


@dataclass
class ArticleEmbedConfig:
    """CLIP-based article text embedding (Phase 0). Always uses CLIP."""
    batch_size: int = 32
    max_tokens: int = 750    # max content tokens = 10 chunks × 75
    chunk_size: int = 75     # CLIP max is 77 tokens; 77 − 2 special tokens = 75
    fp16: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 – Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ViTConfig:
    """Visual encoder output shape. Kept for backward compatibility.
    Prefer reading dims from ModelConfig or VISUAL_MODELS registry."""
    d_v: int = 1024           # hidden dim of ViT last hidden state (NOT clip projection dim)
    n_visual_tokens: int = 257  # 1 CLS + 16×16 patches at 224px / 14px patch size


@dataclass
class MLPProjectorConfig:
    """Two-layer MLP that maps visual features to LLM token space.
    Prefer building via build_mlp_config(visual_key, llm_key)."""
    d_v: int = 1024       # must match visual encoder hidden dim
    d_hidden: int = 2048  # intermediate dim
    d_llm: int = 3584     # must match LLM hidden dim


@dataclass
class LLMConfig:
    llm_model_key: str = "qwen2.5_7b"

    @property
    def model_name(self) -> str:
        return LLM_MODELS[self.llm_model_key]["id"]

    @property
    def d_llm(self) -> int:
        return LLM_MODELS[self.llm_model_key]["d_llm"]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 – Task 1: MLP Projector Pre-training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task1Config:
    epochs: int = 2
    batch_size: int = 8
    gradient_accumulation_steps: int = 64   # effective batch = 512
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05              # 5% of total optimizer steps
    max_caption_length: int = 128
    num_workers: int = 4
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_name: str = "task1_mlp_best.pt"
    log_every_n_steps: int = 1              # log every N optimizer steps


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 – Task 2: VQA Fine-tuning with LoRA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Leave empty to auto-derive from LLMConfig.llm_model_key via get_lora_targets()
    target_modules: List[str] = field(
        default_factory=lambda: []
    )
    bias: str = "none"


@dataclass
class Task2Config:
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 128  # effective batch = 512
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    max_context_tokens: int = 256   # max tokens per retrieved passage in prompt
    max_answer_tokens: int = 128    # max answer length (for truncation guard)
    max_total_length: int = 2048    # max total input sequence length
    num_workers: int = 0            # 0 avoids multiprocessing issues with pre-computed retrieval
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_name: str = "task2_vqa_best.pt"
    task1_checkpoint: str = "outputs/checkpoints/task1_mlp_best.pt"
    log_every_n_steps: int = 1


@dataclass
class RetrievalConfig:
    model_name: str = "facebook/contriever-msmarco"
    top_k: int = 5
    passage_max_tokens: int = 128   # max tokens when embedding each passage chunk
    passage_chunk_tokens: int = 128  # token size of each passage chunk when splitting articles
    encode_batch_size: int = 64     # batch size for Contriever encoding


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 – Image Retrieval
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageRetrievalConfig:
    """Retrieval search parameters for phase2_retrieval.py.

    Embedding settings (batch size, fp16, model) have moved to EmbedderConfig.
    """
    pre_top_k: int = 15           # candidate images fetched before article aggregation
    top_k: int = 10               # final number of articles returned
    db_chunk_size: int = 100_000  # embeddings processed per chunk to avoid OOM


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    batch_size: int = 8
    max_new_tokens: int = 128
    num_workers: int = 0            # 0: pre-computed retrieval, no multiprocessing needed
    task2_checkpoint: str = "outputs/checkpoints/task2_vqa_best.pt"
    results_dir: str = "outputs/evaluation"
    results_file: str = "eval_results.txt"
    # BERTScore: multilingual model suitable for Vietnamese
    bertscore_model: str = "xlm-roberta-base"
