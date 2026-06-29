"""
VisualEncoderWrapper
====================
Normalises any supported HuggingFace visual backbone to a single interface:

    forward(pixel_values) -> Tensor [B, N_tokens, D_v]

Supported families
------------------
  clip      openai/clip-vit-*              CLIPModel  → .vision_model
  siglip    google/siglip-*               AutoModel  → model directly
  blip2     Salesforce/blip2-*            Blip2ForConditionalGeneration → .vision_model
  dinov2    facebook/dinov2-*             AutoModel  → model directly
  resnet    microsoft/resnet-*            AutoModel  → reshape [B,C,H,W] → [B,H*W,C]
  vit       generic ViT-like (fallback)   AutoModel  → .last_hidden_state

Image pre-processing
--------------------
  get_processor() returns an AutoImageProcessor that produces the correct
  pixel_values for each model family.  Use it exactly like CLIPProcessor:

      pixel_values = encoder.get_processor()(images=pil_img, return_tensors="pt")["pixel_values"]
"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoImageProcessor, AutoConfig


# ─────────────────────────────────────────────────────────────────────────────
# Family detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_family(model_name: str) -> str:
    n = model_name.lower()
    if "clip" in n:
        return "clip"
    if "blip2" in n or "blip-2" in n:
        return "blip2"
    if "resnet" in n:
        return "resnet"
    if "siglip" in n:
        return "siglip"
    if "dinov2" in n or "dino" in n:
        return "dinov2"
    return "vit"  # generic fallback: SigLIP variants, EVA-CLIP, etc.


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VisualEncoderWrapper(nn.Module):
    """
    Wraps a HuggingFace visual backbone and exposes:
      • forward(pixel_values) -> Tensor [B, N_tokens, D_v]
      • d_v            : int   hidden dim of each token
      • get_processor() : AutoImageProcessor for pixel_values pre-processing
      • freeze()        : freeze all encoder parameters

    Parameters
    ----------
    model_name : HuggingFace model ID (or local path)
    torch_dtype : dtype for model weights (default bfloat16)
    device_map  : passed to from_pretrained (e.g. "auto" or {"": "cuda:0"})
    device      : explicit device; used only when device_map is None
    """

    def __init__(
        self,
        model_name: str,
        torch_dtype=torch.bfloat16,
        device_map=None,
        device=None,
    ):
        super().__init__()
        self.model_name = model_name
        self._family = _detect_family(model_name)

        load_kwargs = {"torch_dtype": torch_dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self._encoder, self._d_v = self._load_encoder(model_name, self._family, load_kwargs)

        if device is not None and device_map is None:
            self._encoder = self._encoder.to(device)

        self._processor = AutoImageProcessor.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Loader helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_encoder(model_name: str, family: str, load_kwargs: dict):
        if family == "clip":
            from transformers import CLIPModel
            full = CLIPModel.from_pretrained(model_name, **load_kwargs)
            enc = full.vision_model
            d_v = enc.config.hidden_size
            return enc, d_v

        if family == "blip2":
            from transformers import Blip2ForConditionalGeneration
            full = Blip2ForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            enc = full.vision_model
            d_v = enc.config.hidden_size
            return enc, d_v

        if family == "resnet":
            from transformers import AutoModel
            enc = AutoModel.from_pretrained(model_name, **load_kwargs)
            # ResNet config stores channel counts per stage in hidden_sizes list
            cfg = AutoConfig.from_pretrained(model_name)
            d_v = cfg.hidden_sizes[-1]
            return enc, d_v

        # siglip, dinov2, generic vit
        from transformers import AutoModel
        enc = AutoModel.from_pretrained(model_name, **load_kwargs)
        cfg = AutoConfig.from_pretrained(model_name)
        # Most ViT-like models expose hidden_size at top level
        d_v = getattr(cfg, "hidden_size", None)
        if d_v is None:
            # some configs nest it under vision_config
            d_v = getattr(getattr(cfg, "vision_config", cfg), "hidden_size", None)
        if d_v is None:
            raise ValueError(
                f"Cannot auto-detect hidden_size for {model_name}. "
                "Add it manually to VISUAL_MODELS in config.py."
            )
        return enc, d_v

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Parameters
        ----------
        pixel_values : Tensor [B, C, H, W]

        Returns
        -------
        Tensor [B, N_tokens, D_v]
        """
        if self._family == "resnet":
            out = self._encoder(pixel_values)
            feat = out.last_hidden_state  # [B, C, H, W]
            B, C, H, W = feat.shape
            return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # All ViT-like models (clip vision_model, siglip, dinov2, blip2 vision_model, vit)
        out = self._encoder(pixel_values=pixel_values)
        return out.last_hidden_state  # [B, N, D_v]

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def d_v(self) -> int:
        return self._d_v

    def get_processor(self) -> AutoImageProcessor:
        return self._processor

    def freeze(self):
        for p in self._encoder.parameters():
            p.requires_grad = False
