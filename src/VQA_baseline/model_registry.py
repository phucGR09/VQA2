"""
model_registry.py
=================
Central registry for all VLM / text-only LLM models used across baseline levels.

Architecture
------------
  VLMConfig          – dataclass holding all static dims, IDs, flags per model
  BaseModelWrapper   – abstract interface every family must implement
  Concrete wrappers:
    InternVLWrapper        covers InternVL3-8B, InternVL3-78B, Vintern-1B-v3
    Vintern35Wrapper       covers Vintern-1B-v3.5 (InternVL2.5-1B based)
    Qwen2VLWrapper         covers Qwen2.5-VL-7B / 72B-Instruct
    GemmaVLWrapper         covers Gemma 3 / Gemma 4 vision models
    MistralVLWrapper       covers Mistral-Small-3.2-24B-Instruct
    BLIP2Wrapper           covers BLIP-2 family (classic baseline)
    Phi4MultimodalWrapper  covers Phi-4-multimodal-instruct
    TextOnlyWrapper        covers Vistral, Qwen2.5-text, PhoGPT, Llama-3.1,
                           SeaLLMs (Level 3 only)

  build_wrapper(key, device)  – factory: looks up registry, returns correct wrapper

Usage
-----
  from baseline.model_registry import MODEL_REGISTRY, build_wrapper

  wrapper = build_wrapper("vintern_1b_v3_5", device="cuda:0")
  wrapper.load()
  answer = wrapper.generate_answer(image_path, question, context)
  wrapper.unload()
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

import gc
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

# Compatibility fix: newer transformers calls `model.all_tied_weights_keys`
# in _move_missing_keys_from_meta_to_device, but some installs have the call
# without the corresponding property on PreTrainedModel.  Patch it here so
# all models (including custom InternVL3) work transparently.
from transformers import PreTrainedModel as _PTM
if not hasattr(_PTM, "all_tied_weights_keys"):
    # post_init() in newer transformers sets this as an instance attribute;
    # _move_missing_keys_from_meta_to_device then reads it.  Models whose
    # custom code predates the property (e.g. InternVL3) never call post_init
    # for the outer model, so the attribute is never set.  A read-write
    # property with an instance-dict fallback satisfies both callers.
    def _atk_get(self):
        return self.__dict__.get(
            "_all_tied_weights_keys_val",
            {k: None for k in (getattr(self, "_tied_weights_keys", None) or [])},
        )
    def _atk_set(self, v):
        self.__dict__["_all_tied_weights_keys_val"] = v
    _PTM.all_tied_weights_keys = property(_atk_get, _atk_set)




def _torch_version_tuple() -> tuple:
    v = torch.__version__.split("+")[0]
    parts = v.split(".")
    nums: List[int] = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def _attn_impl_kwargs() -> Dict[str, str]:
    # Torch < 2.6 does not support flex attention mask functions.
    if _torch_version_tuple() < (2, 6, 0):
        return {"attn_implementation": "eager"}
    return {}


def _from_pretrained_with_attn(model_cls, *args, **kwargs):
    attn_kwargs = _attn_impl_kwargs()
    if attn_kwargs:
        try:
            return model_cls.from_pretrained(*args, **kwargs, **attn_kwargs)
        except TypeError:
            pass
    return model_cls.from_pretrained(*args, **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VLMConfig:
    # ── Identity ──────────────────────────────────────────────────────────────
    model_key: str
    model_id: str          # HuggingFace model ID or local path
    family: str            # loader strategy: internvl | qwen2vl | gemma_vl |
                           #   mistral_vl | blip2 | phi4_vl | text_only

    # ── Visual encoder ────────────────────────────────────────────────────────
    visual_encoder: str = ""       # short name for logging
    processor_id: Optional[str] = None
    img_size: int = 448            # input resolution per tile
    patch_size: int = 14
    visual_hidden_dim: int = 1024  # last_hidden_state hidden dim (D_v)
    n_visual_tokens: int = 256     # tokens per tile (before dynamic expansion)

    # ── LLM backbone ──────────────────────────────────────────────────────────
    llm_id: str = ""
    llm_hidden_dim: int = 4096

    # ── Runtime ───────────────────────────────────────────────────────────────
    max_context_length: int = 4096
    max_new_tokens: int = 256
    supports_vietnamese: bool = True
    trust_remote_code: bool = False

    # ── Dynamic tile / multi-image flags (InternVL) ───────────────────────────
    dynamic_tiles: bool = False
    max_tiles: int = 1      # max image tiles for dynamic resolution

    # ── Loading ───────────────────────────────────────────────────────────────
    dtype: str = "bfloat16"
    is_vl_model: bool = True  # False → text-only model (Level 3)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — add new models here
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, VLMConfig] = {

    # ── InternVL family ───────────────────────────────────────────────────────
    "vintern_1b_v3": VLMConfig(
        model_key="vintern_1b_v3",
        model_id="5CD-AI/Vintern-1B-v3",
        family="internvl",
        visual_encoder="InternViT-300M",
        img_size=448,
        patch_size=14,
        visual_hidden_dim=1024,
        n_visual_tokens=256,
        llm_id="Qwen2-0.5B",
        llm_hidden_dim=896,
        max_context_length=4096,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=True,
        dynamic_tiles=True,
        max_tiles=6,
    ),
    "vintern_1b_v3_5": VLMConfig(
        model_key="vintern_1b_v3_5",
        model_id="5CD-AI/Vintern-1B-v3_5",
        family="vintern35",
        visual_encoder="InternViT-300M",
        img_size=448,
        patch_size=14,
        visual_hidden_dim=1024,
        n_visual_tokens=256,
        llm_id="Qwen2.5-0.5B",
        llm_hidden_dim=896,
        max_context_length=1700,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=True,
        dynamic_tiles=True,
        max_tiles=4,   # 4 tiles + thumbnail = 5×256 = 1280 visual tokens, fits in 1700 limit
    ),
    "internvl3_8b": VLMConfig(
        model_key="internvl3_8b",
        model_id="OpenGVLab/InternVL3-8B",
        family="internvl",
        visual_encoder="InternViT-300M",
        img_size=448,
        patch_size=14,
        visual_hidden_dim=1024,
        n_visual_tokens=256,
        llm_id="Qwen2.5-7B",
        llm_hidden_dim=3584,
        max_context_length=8192,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=True,
        dynamic_tiles=True,
        max_tiles=12,
    ),
    # ── Qwen2.5-VL family ─────────────────────────────────────────────────────
    "qwen2vl_7b": VLMConfig(
        model_key="qwen2vl_7b",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        family="qwen2vl",
        visual_encoder="Qwen2-ViT",
        img_size=448,
        patch_size=14,
        visual_hidden_dim=1280,
        n_visual_tokens=256,
        llm_id="Qwen2.5-7B",
        llm_hidden_dim=3584,
        max_context_length=32768,
        max_new_tokens=64,
        supports_vietnamese=True,
        trust_remote_code=False,
        dynamic_tiles=True,
        max_tiles=4,
    ),

    # ── Gemma vision family (Gemma 3 + Gemma 4) ───────────────────────────────
    "gemma4_26b_moe": VLMConfig(
        model_key="gemma4_26b_moe",
        model_id="google/gemma-4-26b-a4b-it",
        family="gemma_vl",
        visual_encoder="SigLIP-400M",
        img_size=896,
        patch_size=14,
        visual_hidden_dim=1152,
        n_visual_tokens=256,
        llm_id="Gemma4-MoE-26B",
        llm_hidden_dim=2048,
        max_context_length=262144,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=False,
        dynamic_tiles=False,
        max_tiles=1,
    ),
    "gemma3_12b": VLMConfig(
        model_key="gemma3_12b",
        model_id="google/gemma-3-12b-it",
        family="gemma_vl",
        visual_encoder="SigLIP-400M",
        img_size=896,
        patch_size=14,
        visual_hidden_dim=1152,
        n_visual_tokens=256,
        llm_id="Gemma3-12B",
        llm_hidden_dim=3072,
        max_context_length=131072,
        max_new_tokens=32,
        supports_vietnamese=True,
        trust_remote_code=False,
        dynamic_tiles=False,
        max_tiles=1,
    ),
    "gemma3_27b": VLMConfig(
        model_key="gemma3_27b",
        model_id="google/gemma-3-27b-it",
        family="gemma_vl",
        visual_encoder="SigLIP-400M",
        img_size=896,
        patch_size=14,
        visual_hidden_dim=1152,
        n_visual_tokens=256,
        llm_id="Gemma3-27B",
        llm_hidden_dim=4096,
        max_context_length=131072,
        max_new_tokens=32,
        supports_vietnamese=True,
        trust_remote_code=False,
        dynamic_tiles=False,
        max_tiles=1,
    ),

    # ── Pixtral-12B (Mistral's native VLM) ───────────────────────────────────
    # Mistral-Small-3.2 is a text-only LLM; Pixtral-12B is the correct choice.
    # Uses vLLM (family="pixtral_vllm") because Pixtral-12B-2409 requires the
    # mistral_common tokenizer which is not supported by the HF transformers
    # pipeline (AutoModelForImageTextToText / AutoProcessor will fail).
    "pixtral_12b": VLMConfig(
        model_key="pixtral_12b",
        model_id="mistralai/Pixtral-12B-2409",
        family="pixtral_vllm",
        visual_encoder="Pixtral-ViT-400M",
        img_size=1024,
        patch_size=16,
        visual_hidden_dim=1024,
        n_visual_tokens=4096,   # up to (1024/16)^2 = 4096 patches at full res
        llm_id="Mistral-NeMo-12B",
        llm_hidden_dim=5120,
        max_context_length=131072,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=False,
        dynamic_tiles=False,
        max_tiles=1,
    ),

    # ── BLIP-2 (classic pre-LLM baseline) ────────────────────────────────────
    "blip2_opt_27b": VLMConfig(
        model_key="blip2_opt_27b",
        model_id="Salesforce/blip2-opt-2.7b",
        family="blip2",
        visual_encoder="ViT-g/14",
        img_size=224,
        patch_size=14,
        visual_hidden_dim=1408,
        n_visual_tokens=32,         # Q-Former compresses to 32 query tokens
        llm_id="OPT-2.7B",
        llm_hidden_dim=2560,
        max_context_length=512,
        max_new_tokens=128,
        supports_vietnamese=False,
        trust_remote_code=False,
        dynamic_tiles=False,
        max_tiles=1,
    ),

    # ── Phi-4-multimodal ──────────────────────────────────────────────────────
    "phi4_multimodal": VLMConfig(
        model_key="phi4_multimodal",
        model_id="microsoft/Phi-4-multimodal-instruct",
        family="phi4_vl",
        visual_encoder="CLIP-ViT-L/14",
        img_size=448,
        patch_size=14,
        visual_hidden_dim=1024,
        n_visual_tokens=1024,
        llm_id="Phi-4-14B",
        llm_hidden_dim=5120,
        max_context_length=131072,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=True,
        dynamic_tiles=False,
        max_tiles=1,
    ),

    # ── Text-only models (Level 3) ─────────────────────────────────────────────
    "vistral_7b": VLMConfig(
        model_key="vistral_7b",
        model_id="Viet-Mistral/Vistral-7B-Chat",
        family="text_only",
        visual_encoder="",
        img_size=0,
        patch_size=0,
        visual_hidden_dim=0,
        n_visual_tokens=0,
        llm_id="Mistral-7B",
        llm_hidden_dim=4096,
        max_context_length=4096,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=False,
        is_vl_model=False,
    ),
    "qwen2.5_7b_text": VLMConfig(
        model_key="qwen2.5_7b_text",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        family="text_only",
        visual_encoder="",
        img_size=0,
        patch_size=0,
        visual_hidden_dim=0,
        n_visual_tokens=0,
        llm_id="Qwen2.5-7B",
        llm_hidden_dim=3584,
        max_context_length=32768,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=False,
        is_vl_model=False,
    ),
    "qwen2.5_72b_text": VLMConfig(
        model_key="qwen2.5_72b_text",
        model_id="Qwen/Qwen2.5-72B-Instruct",
        family="text_only",
        visual_encoder="",
        img_size=0,
        patch_size=0,
        visual_hidden_dim=0,
        n_visual_tokens=0,
        llm_id="Qwen2.5-72B",
        llm_hidden_dim=8192,
        max_context_length=131072,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=False,
        is_vl_model=False,
    ),
    "phogpt_4b": VLMConfig(
        model_key="phogpt_4b",
        model_id="vinai/PhoGPT-4B-Chat",
        family="text_only",
        visual_encoder="",
        img_size=0,
        patch_size=0,
        visual_hidden_dim=0,
        n_visual_tokens=0,
        llm_id="PhoGPT-4B",
        llm_hidden_dim=2560,
        max_context_length=8192,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=True,
        is_vl_model=False,
    ),
    "llama31_8b_text": VLMConfig(
        model_key="llama31_8b_text",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        family="text_only",
        visual_encoder="",
        img_size=0,
        patch_size=0,
        visual_hidden_dim=0,
        n_visual_tokens=0,
        llm_id="Llama-3.1-8B",
        llm_hidden_dim=4096,
        max_context_length=131072,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=False,
        is_vl_model=False,
    ),
    "seallms_7b": VLMConfig(
        model_key="seallms_7b",
        model_id="SeaLLMs/SeaLLMs-v3-7B-Chat",
        family="text_only",
        visual_encoder="",
        img_size=0,
        patch_size=0,
        visual_hidden_dim=0,
        n_visual_tokens=0,
        llm_id="Mistral-7B (SEA-tuned)",
        llm_hidden_dim=4096,
        max_context_length=32768,
        max_new_tokens=256,
        supports_vietnamese=True,
        trust_remote_code=False,
        is_vl_model=False,
    ),
}

# ── Convenience groupings per level ──────────────────────────────────────────
LEVEL2_MODELS = [
    "qwen2vl_7b", "qwen2vl_72b",
    "internvl3_8b",
    "gemma4_26b_moe",
    "vintern_1b_v3_5",
    "pixtral_12b", "blip2_opt_27b", "phi4_multimodal",
]
LEVEL3_MODELS = [
    "qwen2.5_7b_text", "qwen2.5_72b_text",
    "gemma4_26b_moe",      # text-only run of the same VL model
    "vistral_7b", "phogpt_4b",
    "llama31_8b_text", "seallms_7b",
]
LEVEL4_MODELS = [
    "qwen2vl_7b", "qwen2vl_72b",
    "internvl3_8b",
    "gemma4_26b_moe",
    "vintern_1b_v3_5", "pixtral_12b",
    "gemma3_27b",
    "gemma3_12b",
]


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base wrapper
# ─────────────────────────────────────────────────────────────────────────────

class BaseModelWrapper(ABC):
    """
    All concrete wrappers must implement:
      load()              → load model + processor onto self.device
      generate_answer()   → run inference, return decoded string
      unload()            → free GPU memory

    `context` is the article passage (None for Level 2 / text-only image levels).
    """

    def __init__(self, config: VLMConfig, device: str = "cuda:0"):
        self.config = config
        self.device = device
        self._loaded = False

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate_answer(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str: ...

    def generate_batch(
        self,
        questions: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        contexts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,
    ) -> List[str]:
        """
        Default batched inference fallback.

        Subclasses should override for true batching.
        """
        if image_paths is None:
            image_paths = [None] * len(questions)
        if contexts is None:
            contexts = [None] * len(questions)
        if images is not None and len(images) != len(questions):
            raise ValueError("images must have same length as questions")
        if not (len(questions) == len(image_paths) == len(contexts)):
            raise ValueError("questions, image_paths, contexts must have same length")
        if images is not None and image_paths is None:
            raise ValueError("images provided but image_paths is None")
        outputs: List[str] = []
        for q, img, ctx in zip(questions, image_paths, contexts):
            outputs.append(self.generate_answer(q, image_path=img, context=ctx))
        return outputs

    def unload(self) -> None:
        for attr in ("model", "processor", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @property
    def _dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16

    def _open_image(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _resolve_images(
        self,
        image_paths: Optional[List[Optional[str]]],
        images: Optional[List[Optional[Image.Image]]],
    ) -> Optional[List[Optional[Image.Image]]]:
        if images is None:
            if image_paths is None:
                return None
            return [self._open_image(p) if p else None for p in image_paths]
        if image_paths is None:
            return images
        if len(images) != len(image_paths):
            raise ValueError("images and image_paths must have same length")
        resolved: List[Optional[Image.Image]] = []
        for img, path in zip(images, image_paths):
            if img is None and path:
                resolved.append(self._open_image(path))
            else:
                resolved.append(img)
        return resolved

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key={self.config.model_key}, device={self.device})"


# ─────────────────────────────────────────────────────────────────────────────
# InternVL / Vintern wrapper
# ─────────────────────────────────────────────────────────────────────────────

class InternVLWrapper(BaseModelWrapper):
    """
    Covers: InternVL3-8B, InternVL3-78B, Vintern-1B-v3.
    Uses dynamic tile preprocessing from the InternVL repo.
    """

    # ImageNet normalisation used by InternVL
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )
        # Avoid device_map entirely — any device_map value (including dicts)
        # triggers infer_auto_device_map in newer transformers which calls
        # model.all_tied_weights_keys, a property InternVL3's custom code
        # doesn't define. Load on CPU then move to GPU manually instead.
        self.model = _from_pretrained_with_attn(
            AutoModel,
            self.config.model_id,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        ).eval().to(self.device)
        self._loaded = True

    # ── Image preprocessing ───────────────────────────────────────────────────

    def _build_transform(self):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                (self.config.img_size, self.config.img_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._MEAN, std=self._STD),
        ])

    @staticmethod
    def _find_closest_aspect_ratio(ar, target_ratios, w, h, img_size):
        best_diff = float("inf")
        best = (1, 1)
        area = w * h
        for r in target_ratios:
            diff = abs(ar - r[0] / r[1])
            if diff < best_diff or (diff == best_diff and area > 0.5 * img_size * img_size * r[0] * r[1]):
                best_diff = diff
                best = r
        return best

    def _dynamic_preprocess(self, image: Image.Image, max_num: int):
        w, h = image.size
        ar = w / h
        s = self.config.img_size
        min_n = 1
        target_ratios = sorted(
            {(i, j) for n in range(min_n, max_num + 1)
             for i in range(1, n + 1) for j in range(1, n + 1)
             if min_n <= i * j <= max_num},
            key=lambda x: x[0] * x[1],
        )
        ratio = self._find_closest_aspect_ratio(ar, target_ratios, w, h, s)
        target_w, target_h = s * ratio[0], s * ratio[1]
        blocks = ratio[0] * ratio[1]
        resized = image.resize((target_w, target_h))
        tiles = []
        for idx in range(blocks):
            box = (
                (idx % ratio[0]) * s,
                (idx // ratio[0]) * s,
                ((idx % ratio[0]) + 1) * s,
                ((idx // ratio[0]) + 1) * s,
            )
            tiles.append(resized.crop(box))
        # always append a thumbnail tile as global context
        if len(tiles) > 1:
            tiles.append(image.resize((s, s)))
        return tiles

    def _load_pixel_values(self, image_path: str) -> torch.Tensor:
        img = self._open_image(image_path)
        transform = self._build_transform()
        max_num = self.config.max_tiles if self.config.dynamic_tiles else 1
        tiles = self._dynamic_preprocess(img, max_num)
        pixel_values = torch.stack([transform(t) for t in tiles])
        return pixel_values.to(self._dtype).to(self.device)

    # ── Inference ────────────────────────────────────────────────────────────

    def generate_answer(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        prompt = self._build_prompt(question, context)
        gen_cfg = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if image_path:
            pixel_values = self._load_pixel_values(image_path)
            response = self.model.chat(
                self.tokenizer, pixel_values, prompt, gen_cfg,
                history=None, return_history=False,
            )
        else:
            response = self.model.chat(
                self.tokenizer, None, prompt, gen_cfg,
                history=None, return_history=False,
            )
        return response.strip()

    def generate_batch(
        self,
        questions: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        contexts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,
    ) -> List[str]:
        # InternVL chat API is single-sample; fall back to loop.
        return super().generate_batch(
            questions, image_paths=image_paths, contexts=contexts, images=images
        )

    def _build_prompt(self, question: str, context: Optional[str]) -> str:
        if context is None:
            return question  # level runner pre-formatted the full prompt
        return "\n\n".join([
            f"Dưới đây là nội dung bài viết liên quan:\n{context}",
            f"Câu hỏi: {question}",
            "Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác.",
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Vintern-1B-v3.5 wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Vintern35Wrapper(InternVLWrapper):
    """
    Covers Vintern-1B-v3.5 (5CD-AI/Vintern-1B-v3_5), based on InternVL2.5-1B.

    Key differences from v3:
      - Requires ``<image>\\n`` prefix in the query when an image is present.
      - Uses beam search (num_beams=3) and repetition_penalty=2.5.
      - Loaded with low_cpu_mem_usage=True and use_flash_attn=False.
      - Supports up to 12 dynamic tiles (config.max_tiles=12).
    """

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = self._load_tokenizer_v35()
        self.model = _from_pretrained_with_attn(
            AutoModel,
            self.config.model_id,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval().to(self.device)
        self._loaded = True

    def _load_tokenizer_v35(self):
        # transformers 5.x always uses the tokenizers (Rust) backend and requires
        # tokenizer.json. Vintern-1B-v3.5's HF repo has no tokenizer.json, so
        # every AutoTokenizer path fails. Qwen2Tokenizer (the slow class) reads
        # the tiktoken vocab directly and never touches the Rust backend.
        try:
            from transformers import Qwen2Tokenizer
            return Qwen2Tokenizer.from_pretrained(
                self.config.model_id, trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot load tokenizer for {self.config.model_id}.\n"
                f"Error: {e}\n"
                "The model repo lacks tokenizer.json (required by transformers 5.x).\n"
                "Fix A: pip install 'transformers>=4.46,<5.0'\n"
                "Fix B: manually copy tokenizer.json from Qwen/Qwen2.5-0.5B into "
                "the model's HF cache snapshot directory."
            ) from e

    def _gen_config(self) -> dict:
        return dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            num_beams=3,
            repetition_penalty=2.5,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate_answer(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        prompt = self._build_prompt(question, context)
        gen_cfg = self._gen_config()

        if image_path:
            pixel_values = self._load_pixel_values(image_path)
            if not prompt.startswith("<image>"):
                prompt = "<image>\n" + prompt
            response = self.model.chat(
                self.tokenizer, pixel_values, prompt, gen_cfg,
                history=None, return_history=False,
            )
        else:
            response = self.model.chat(
                self.tokenizer, None, prompt, gen_cfg,
                history=None, return_history=False,
            )
        return response.strip()

    def generate_batch(
        self,
        questions: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        contexts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,
    ) -> List[str]:
        if image_paths is None:
            image_paths = [None] * len(questions)
        if contexts is None:
            contexts = [None] * len(questions)
        gen_cfg = self._gen_config()
        results: List[str] = []
        for q, img_path, ctx in zip(questions, image_paths, contexts):
            prompt = self._build_prompt(q, ctx)
            if img_path:
                pixel_values = self._load_pixel_values(img_path)
                if not prompt.startswith("<image>"):
                    prompt = "<image>\n" + prompt
                response = self.model.chat(
                    self.tokenizer, pixel_values, prompt, gen_cfg,
                    history=None, return_history=False,
                )
            else:
                response = self.model.chat(
                    self.tokenizer, None, prompt, gen_cfg,
                    history=None, return_history=False,
                )
            results.append(response.strip())
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Qwen2.5-VL wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Qwen2VLWrapper(BaseModelWrapper):
    """Covers Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-72B-Instruct."""

    # cap visual tokens per image: 768 patches × 28×28 px = good quality, ~3× fewer tokens than default
    _MAX_PIXELS = 768 * 28 * 28

    def load(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            min_pixels=4 * 28 * 28,
            max_pixels=self._MAX_PIXELS,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = _from_pretrained_with_attn(
            Qwen2_5_VLForConditionalGeneration,
            self.config.model_id,
            torch_dtype=self._dtype,
            device_map={"": self.device},
        ).eval()
        self._loaded = True

    def generate_answer(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        prompt_text = self._build_prompt_text(question, context)
        content: list = []
        images = []

        if image_path:
            content.append({"type": "image"})
            images.append(self._open_image(image_path))

        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def generate_batch(
        self,
        questions: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        contexts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,
    ) -> List[str]:
        if image_paths is None:
            image_paths = [None] * len(questions)
        if contexts is None:
            contexts = [None] * len(questions)
        images = self._resolve_images(image_paths, images)
        if not (len(questions) == len(image_paths) == len(contexts) == len(images)):
            raise ValueError("questions, image_paths, contexts must have same length")

        has_image = [img is not None for img in images]
        if any(has_image) and not all(has_image):
            raise ValueError("Qwen2VLWrapper.generate_batch does not support mixed image/no-image batches")

        texts: List[str] = []
        image_inputs: List[Image.Image] = []
        for q, img, ctx in zip(questions, images, contexts):
            prompt_text = self._build_prompt_text(q, ctx)
            content: list = []
            if img is not None:
                content.append({"type": "image"})
                image_inputs.append(img)
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        results = [s.strip() for s in self.processor.batch_decode(generated, skip_special_tokens=True)]
        del inputs, output_ids, generated
        torch.cuda.empty_cache()
        return results

    def _build_prompt_text(self, question: str, context: Optional[str]) -> str:
        if context is None:
            return question  # level runner pre-formatted the full prompt
        return "\n\n".join([
            f"Dưới đây là nội dung bài viết liên quan:\n{context}",
            f"Câu hỏi: {question}",
            "Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác.",
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Gemma vision wrapper  (Gemma 3 + Gemma 4)
# ─────────────────────────────────────────────────────────────────────────────

class GemmaVLWrapper(BaseModelWrapper):
    """
    Covers google/gemma-4-31b-it, google/gemma-4-26b-a4b-it, google/gemma-3-27b-it.

    Gemma 4 uses Gemma4ForConditionalGeneration; Gemma 3 uses
    Gemma3ForConditionalGeneration.  AutoModelForImageTextToText resolves
    the correct class automatically for both generations.

    When image_path is None (Level 3 text-only run) the image content block
    is simply omitted — the model degrades gracefully to pure text mode.
    """

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.processor.tokenizer.padding_side = "left"
        attn_kwargs = _attn_impl_kwargs()
        if not attn_kwargs:
            # torch >= 2.6: prefer flash_attention_2 if available, else let transformers choose
            try:
                import flash_attn  # noqa: F401
                attn_kwargs = {"attn_implementation": "flash_attention_2"}
            except ImportError:
                pass
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            torch_dtype=self._dtype,
            device_map={"": self.device},
            **attn_kwargs,
        ).eval()
        self._loaded = True

    def generate_answer(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        prompt_text = self._build_prompt_text(question, context)
        images = []
        content: list = []

        if image_path:
            images.append(self._open_image(image_path))
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        # apply_chat_template produces the text scaffold only (tokenize=False);
        # processor then handles pixel encoding and tokenization together.
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def generate_batch(
        self,
        questions: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        contexts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,
    ) -> List[str]:
        if image_paths is None:
            image_paths = [None] * len(questions)
        if contexts is None:
            contexts = [None] * len(questions)
        images = self._resolve_images(image_paths, images)
        if not (len(questions) == len(image_paths) == len(contexts) == len(images)):
            raise ValueError("questions, image_paths, contexts must have same length")

        if any(img is None for img in images):
            raise ValueError("Gemma batch has missing images; check paths/decoding or use --no_prefetch")

        has_image = [img is not None for img in images]
        if any(has_image) and not all(has_image):
            raise ValueError("GemmaVLWrapper.generate_batch does not support mixed image/no-image batches")

        texts: List[str] = []
        image_inputs: List[Image.Image] = []
        for q, img, ctx in zip(questions, images, contexts):
            prompt_text = self._build_prompt_text(q, ctx)
            content: list = []
            if img is not None:
                image_inputs.append(img)
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            texts.append(text)

        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return [s.strip() for s in self.processor.batch_decode(generated, skip_special_tokens=True)]

    def _build_prompt_text(self, question: str, context: Optional[str]) -> str:
        if context is None:
            return question  # level runner pre-formatted the full prompt
        return "\n\n".join([
            f"Dưới đây là nội dung bài viết liên quan:\n{context}",
            f"Câu hỏi: {question}",
            "Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác.",
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Pixtral wrapper  — vLLM offline inference
# ─────────────────────────────────────────────────────────────────────────────

class PixtralVLLMWrapper(BaseModelWrapper):
    """
    Covers mistralai/Pixtral-12B-2409.

    Pixtral uses Mistral's own mistral_common tokenizer which is incompatible
    with the HuggingFace transformers pipeline. vLLM's offline LLM class
    handles it correctly via tokenizer_mode="mistral".

    Images are encoded as base64 data-URIs and passed in the OpenAI-style
    chat message format that vLLM's Pixtral backend expects.
    """

    def load(self) -> None:
        from vllm import LLM, SamplingParams  # pip install vllm
        # Extract the cuda device index for tensor_parallel_size; vLLM manages
        # its own CUDA context so we just tell it which GPU to use.
        gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))

        self._llm = LLM(
            model=self.config.model_id,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            max_model_len=self.config.max_context_length,
        )
        self._sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=0.0,
        )
        self._loaded = True

    def unload(self) -> None:
        if hasattr(self, "_llm"):
            del self._llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_data_uri(image_path: str) -> str:
        """Read a local image file and return a base64 data-URI string."""
        import base64
        suffix = Path(image_path).suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "jpeg")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"

    def _build_messages(
        self,
        question: str,
        image_path: Optional[str],
        context: Optional[str],
    ) -> List[Dict]:
        prompt_text = self._build_prompt_text(question, context)
        content: list = []
        if image_path:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._to_data_uri(image_path)},
            })
        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def _build_prompt_text(self, question: str, context: Optional[str]) -> str:
        if context is None:
            return question  # level runner pre-formatted the full prompt
        return "\n\n".join([
            f"Dưới đây là nội dung bài viết liên quan:\n{context}",
            f"Câu hỏi: {question}",
            "Hãy trả lời bằng tiếng Việt dựa trên hình ảnh và bài viết trên, ngắn gọn và chính xác.",
        ])

    # ── Inference ─────────────────────────────────────────────────────────────

    def generate_answer(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        messages = self._build_messages(question, image_path, context)
        outputs = self._llm.chat([messages], sampling_params=self._sampling_params)
        return outputs[0].outputs[0].text.strip()

    def generate_batch(
        self,
        questions: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        contexts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,  # unused; vLLM reads paths
    ) -> List[str]:
        if image_paths is None:
            image_paths = [None] * len(questions)
        if contexts is None:
            contexts = [None] * len(questions)

        # Build one conversation per sample; vLLM batches them internally
        batch_messages = [
            self._build_messages(q, img_path, ctx)
            for q, img_path, ctx in zip(questions, image_paths, contexts)
        ]
        outputs = self._llm.chat(batch_messages, sampling_params=self._sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]


# ─────────────────────────────────────────────────────────────────────────────
# BLIP-2 wrapper
# ─────────────────────────────────────────────────────────────────────────────

class BLIP2Wrapper(BaseModelWrapper):
    """Classic baseline. Covers blip2-opt-2.7b. Simple image-captioning style QA."""

    def load(self) -> None:
        from transformers import Blip2ForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.model = _from_pretrained_with_attn(
            Blip2ForConditionalGeneration,
            self.config.model_id,
            torch_dtype=self._dtype,
            device_map={"": self.device},
        ).eval()
        self._loaded = True

    def generate_answer(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        prompt = self._build_prompt(question, context)
        if not image_path:
            raise ValueError("BLIP-2 requires an image path")
        img = self._open_image(image_path)
        inputs = self.processor(
            images=img, text=prompt, return_tensors="pt"
        ).to(self.device, self._dtype)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def generate_batch(
        self,
        questions: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        contexts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,
    ) -> List[str]:
        if image_paths is None and images is None:
            raise ValueError("BLIP-2 requires images or image_paths")
        if contexts is None:
            contexts = [None] * len(questions)
        images = self._resolve_images(image_paths, images)
        if image_paths is None:
            image_paths = [None] * len(questions)
        if not (len(questions) == len(image_paths) == len(contexts) == len(images)):
            raise ValueError("questions, image_paths, contexts must have same length")
        if any(img is None for img in images):
            raise ValueError("BLIP-2 requires images for all samples")

        prompts = [self._build_prompt(q, ctx) for q, ctx in zip(questions, contexts)]
        inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device, self._dtype)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        return [s.strip() for s in self.processor.batch_decode(output_ids, skip_special_tokens=True)]

    def _build_prompt(self, question: str, context: Optional[str]) -> str:
        if context is None:
            return question  # level runner pre-formatted the full prompt
        # OPT backbone is English-only; keep context injection in English
        return f"Context: {context}\nQuestion: {question}\nAnswer:"


# ─────────────────────────────────────────────────────────────────────────────
# Phi-4-multimodal wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Phi4MultimodalWrapper(BaseModelWrapper):
    """
    Covers microsoft/Phi-4-multimodal-instruct.

    Uses <|image_1|> placeholder token injected into the prompt when an image
    is present.  Requires trust_remote_code=True.
    """

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, trust_remote_code=True
        )
        self.model = _from_pretrained_with_attn(
            AutoModelForCausalLM,
            self.config.model_id,
            torch_dtype=self._dtype,
            trust_remote_code=True,
            device_map={"": self.device},
        ).eval()
        self._loaded = True

    def generate_answer(
        self,
        question: str,
        image_path: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        prompt_text = self._build_prompt_text(question, context)
        images = []

        if image_path:
            images.append(self._open_image(image_path))
            user_content = f"<|image_1|>\n{prompt_text}"
        else:
            user_content = prompt_text

        messages = [{"role": "user", "content": user_content}]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def generate_batch(
        self,
        questions: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        contexts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,
    ) -> List[str]:
        if image_paths is None:
            image_paths = [None] * len(questions)
        if contexts is None:
            contexts = [None] * len(questions)
        images = self._resolve_images(image_paths, images)
        if not (len(questions) == len(image_paths) == len(contexts) == len(images)):
            raise ValueError("questions, image_paths, contexts must have same length")

        has_image = [img is not None for img in images]
        if any(has_image) and not all(has_image):
            raise ValueError("Phi4MultimodalWrapper.generate_batch does not support mixed image/no-image batches")

        texts: List[str] = []
        image_inputs: List[Image.Image] = []
        for q, img, ctx in zip(questions, images, contexts):
            prompt_text = self._build_prompt_text(q, ctx)
            if img is not None:
                image_inputs.append(img)
                user_content = f"<|image_1|>\n{prompt_text}"
            else:
                user_content = prompt_text
            messages = [{"role": "user", "content": user_content}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return [s.strip() for s in self.processor.batch_decode(generated, skip_special_tokens=True)]

    def _build_prompt_text(self, question: str, context: Optional[str]) -> str:
        if context is None:
            return question  # level runner pre-formatted the full prompt
        return "\n\n".join([
            f"Dưới đây là nội dung bài viết liên quan:\n{context}",
            f"Câu hỏi: {question}",
            "Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác.",
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Text-only wrapper  (Vistral, Qwen2.5-text, PhoGPT, Llama-3.1, SeaLLMs — Level 3)
# ─────────────────────────────────────────────────────────────────────────────

class TextOnlyWrapper(BaseModelWrapper):
    """
    Causal LM with no vision. Used for Level 3 (article + question, no image).
    Covers Vistral-7B-Chat, Qwen2.5-7B/72B-Instruct, PhoGPT-4B-Chat,
    Llama-3.1-8B-Instruct, SeaLLMs-v3-7B-Chat.
    """

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = _from_pretrained_with_attn(
            AutoModelForCausalLM,
            self.config.model_id,
            torch_dtype=self._dtype,
            trust_remote_code=self.config.trust_remote_code,
            device_map={"": self.device},
        ).eval()
        self._loaded = True

    def generate_answer(
        self,
        question: str,
        image_path: Optional[str] = None,  # ignored
        context: Optional[str] = None,
    ) -> str:
        prompt = self._build_prompt(question, context)
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_context_length - self.config.max_new_tokens,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[:, enc.input_ids.shape[1]:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def generate_batch(
        self,
        questions: List[str],
        image_paths: Optional[List[Optional[str]]] = None,
        contexts: Optional[List[Optional[str]]] = None,
        images: Optional[List[Optional[Image.Image]]] = None,
    ) -> List[str]:
        if contexts is None:
            contexts = [None] * len(questions)
        if not (len(questions) == len(contexts)):
            raise ValueError("questions and contexts must have same length")

        prompts = [self._build_prompt(q, ctx) for q, ctx in zip(questions, contexts)]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_context_length - self.config.max_new_tokens,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[:, enc.input_ids.shape[1]:]
        return [s.strip() for s in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]

    def _build_prompt(self, question: str, context: Optional[str]) -> str:
        if context is None:
            return question  # level runner pre-formatted the full prompt
        return (
            f"Bài viết:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            "Hãy trả lời bằng tiếng Việt dựa trên bài viết trên, ngắn gọn và chính xác.\n"
            "Trả lời:"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_FAMILY_TO_WRAPPER = {
    "internvl":      InternVLWrapper,
    "vintern35":     Vintern35Wrapper,
    "qwen2vl":       Qwen2VLWrapper,
    "gemma_vl":      GemmaVLWrapper,
    "pixtral_vllm":  PixtralVLLMWrapper,
    "blip2":         BLIP2Wrapper,
    "phi4_vl":       Phi4MultimodalWrapper,
    "text_only":     TextOnlyWrapper,
}


def build_wrapper(model_key: str, device: str = "cuda:0") -> BaseModelWrapper:
    """Return an unloaded wrapper for the given registry key."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key '{model_key}'. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    cfg = MODEL_REGISTRY[model_key]
    cls = _FAMILY_TO_WRAPPER.get(cfg.family)
    if cls is None:
        raise ValueError(
            f"No wrapper implemented for family '{cfg.family}'. "
            f"Add it to _FAMILY_TO_WRAPPER in model_registry.py."
        )
    return cls(cfg, device=device)
