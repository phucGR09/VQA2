"""
baseline_config.py
==================
Baseline-scoped subset of /copy/config.py.

Only the items actually needed by VQA_baseline/ are kept here:
  - VISUAL_MODELS / LLM_MODELS / LORA_TARGETS  — registries
  - MLPProjectorConfig / RetrievalConfig        — dataclasses
  - build_mlp_config / get_lora_targets         — registry helpers

Pipeline-only config (PathConfig, EmbedderConfig, Task1/2Config,
ImageRetrievalConfig, EvalConfig, etc.) intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Visual encoder registry
# ─────────────────────────────────────────────────────────────────────────────
# d_v       : hidden dim of last_hidden_state → MLP projector input
# n_tokens  : visual token count after encoding
# embed_dim : CLIP-projection / similarity-search dim (phase0/phase2)
# family    : used by VisualEncoderWrapper to select load path

VISUAL_MODELS: Dict[str, Dict[str, Any]] = {
    "clip_vit_l14": {
        "id":        "openai/clip-vit-large-patch14",
        "family":    "clip",
        "d_v":       1024,
        "n_tokens":  257,
        "embed_dim": 768,
    },
    "clip_vit_b32": {
        "id":        "openai/clip-vit-base-patch32",
        "family":    "clip",
        "d_v":       768,
        "n_tokens":  50,
        "embed_dim": 512,
    },
    "clip_vit_b16": {
        "id":        "openai/clip-vit-base-patch16",
        "family":    "clip",
        "d_v":       768,
        "n_tokens":  197,
        "embed_dim": 512,
    },
    "clip_vit_l14_336": {
        "id":        "openai/clip-vit-large-patch14-336",
        "family":    "clip",
        "d_v":       1024,
        "n_tokens":  577,
        "embed_dim": 768,
    },
    "siglip_so400m": {
        "id":        "google/siglip-so400m-patch14-384",
        "family":    "siglip",
        "d_v":       1152,
        "n_tokens":  729,
        "embed_dim": 1152,
    },
    "blip2_vitg": {
        "id":        "Salesforce/blip2-opt-2.7b",
        "family":    "blip2",
        "d_v":       1408,
        "n_tokens":  257,
        "embed_dim": 1408,
    },
    "dinov2_large": {
        "id":        "facebook/dinov2-large",
        "family":    "dinov2",
        "d_v":       1024,
        "n_tokens":  256,
        "embed_dim": 1024,
    },
    "resnet50": {
        "id":        "microsoft/resnet-50",
        "family":    "resnet",
        "d_v":       2048,
        "n_tokens":  49,
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


# ─────────────────────────────────────────────────────────────────────────────
# LLM registry
# ─────────────────────────────────────────────────────────────────────────────
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
    "gemma3_12b": {
        "id":     "google/gemma-3-12b-it",
        "family": "gemma",
        "d_llm":  3840,
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
    "internvl3_8b": {
        "id":                "OpenGVLab/InternVL3-8B",
        "family":            "qwen",      # backbone is Qwen2.5-7B (hidden_size=3584)
        "d_llm":             3584,
        "trust_remote_code": True,
    },
    "vintern_1b_v3_5":{
        "id":                "5CD-AI/Vintern-1B-v3_5",
        "family":            "vintern", 
        "d_llm":             896,
        "trust_remote_code": True,
    }
}


# LoRA attention projection module names per LLM family.
# Phi uses a fused qkv_proj; all others use separate projections.
LORA_TARGETS: Dict[str, List[str]] = {
    "qwen":    ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gemma":   ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "vintern": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi":     ["qkv_proj", "o_proj"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def patch_custom_model_class(llm_key: str) -> None:
    """
    Patch transformers for compatibility with custom VLM classes (e.g. InternVLChatModel)
    that predate the all_tied_weights_keys API added in newer transformers.

    Two-pronged approach:
      1. Wrap all_tied_weights_keys on PreTrainedModel with a try/except fallback so the
         property never propagates AttributeError (InternVLChatModel config may be missing
         fields the property accesses).
      2. Also patch _move_missing_keys_from_meta_to_device as a safety net in case the
         method isn't found via MRO for some classes.

    Must be called BEFORE AutoModelForCausalLM.from_pretrained. Idempotent.
    """
    if not LLM_MODELS.get(llm_key, {}).get("trust_remote_code", False):
        return
    import transformers
    import transformers.modeling_utils as _mu

    if getattr(transformers, "_vlm_compat_patched", False):
        return
    transformers._vlm_compat_patched = True

    # 1. Wrap all_tied_weights_keys on PreTrainedModel with a safe fallback.
    #    The property may raise AttributeError internally for InternVLChatModel;
    #    Python then calls nn.Module.__getattr__ which also fails, giving a
    #    misleading "has no attribute" error even though the descriptor exists.
    _existing = getattr(_mu.PreTrainedModel, "all_tied_weights_keys", None)
    # cached_property uses .func, regular property uses .fget
    _orig_fget = getattr(_existing, "fget", None) or getattr(_existing, "func", None)

    def _safe_tied_keys(self):
        if _orig_fget is not None:
            try:
                return _orig_fget(self)
            except Exception:
                pass
        return {}

    # Use a no-op setter so that cached_property's write-back (self.x = val) doesn't crash
    _mu.PreTrainedModel.all_tied_weights_keys = property(_safe_tied_keys, lambda self, v: None)

    # 2. Also patch _move_missing_keys_from_meta_to_device as a belt-and-suspenders
    #    fallback for classes that define their own broken all_tied_weights_keys.
    _orig_move = getattr(_mu.PreTrainedModel, "_move_missing_keys_from_meta_to_device", None)
    if _orig_move is not None:
        def _safe_move(self, *args, **kwargs):
            if not hasattr(self, "all_tied_weights_keys"):
                type(self).all_tied_weights_keys = property(lambda s: {})
            return _orig_move(self, *args, **kwargs)
        _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _safe_move

_ATTN_PROJECTION_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",   # LLaMA / Qwen / Gemma / Mistral
    "wqkv", "wo",                               # InternLM2
    "query_key_value", "dense",                 # Falcon / BLOOM
    "c_attn", "c_proj",                         # GPT-2 style
}


def resolve_lora_targets(llm_key: str, model) -> List[str]:
    """
    Return LoRA target module names for *model*.

    Tries the registry-configured targets first; if none of them exist as leaf
    Linear layers in the model, falls back to auto-detecting all attention
    projection layers by matching against a known-name set.
    """
    import torch.nn as nn

    configured = get_lora_targets(llm_key)
    leaf_names = {name.split(".")[-1] for name, m in model.named_modules()
                  if isinstance(m, nn.Linear)}

    matched = [t for t in configured if t in leaf_names]
    if matched:
        return matched

    # Configured targets not found — auto-detect from attention projection names
    detected = sorted(leaf_names & _ATTN_PROJECTION_NAMES)
    if detected:
        print(
            f"[LoRA] Configured targets {configured} not found in model. "
            f"Auto-detected: {detected}"
        )
        return detected

    raise ValueError(
        f"Could not find LoRA target modules for '{llm_key}'. "
        f"Linear leaf names in model: {sorted(leaf_names)}"
    )


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
            "Add it to LORA_TARGETS in baseline_config.py."
        )
    return LORA_TARGETS[family]


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLPProjectorConfig:
    """Two-layer MLP mapping visual features to LLM token space.
    Prefer building via build_mlp_config(visual_key, llm_key)."""
    d_v: int = 1024       # must match visual encoder hidden dim
    d_hidden: int = 2048  # intermediate dim
    d_llm: int = 3584     # must match LLM hidden dim


@dataclass
class RetrievalConfig:
    """Contriever passage retrieval settings — kept for retrieval_utils.py compatibility."""
    model_name: str = "facebook/contriever-msmarco"
    top_k: int = 5
    passage_max_tokens: int = 128
    passage_chunk_tokens: int = 128
    encode_batch_size: int = 64


# ─────────────────────────────────────────────────────────────────────────────
# Level 5 — Training configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Level5LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)  # auto-derived if empty
    bias: str = "none"


@dataclass
class Level5MlpTrainConfig:
    """Task 1: MLP projector pre-training on image → caption."""
    epochs: int = 2
    batch_size: int = 8
    gradient_accumulation_steps: int = 64   # effective batch = 512
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    max_caption_length: int = 128
    num_workers: int = 4
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_name: str = "task1_mlp_best.pt"


@dataclass
class Level5VqaTrainConfig:
    """Task 2: VQA fine-tuning — MLP + LoRA on (image, article, question) → answer."""
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 64  # effective batch = 512
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    top_k_sentences: int = 5                # ArticleSelector sentences injected as context
    selector_method: str = "bm25"           # "bm25" | "dense" | "first"
    max_context_tokens: int = 256
    max_total_length: int = 2048
    num_workers: int = 4
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_name: str = "task2_vqa_best.pt"
    task1_checkpoint: str = "outputs/checkpoints/task1_mlp_best.pt"
