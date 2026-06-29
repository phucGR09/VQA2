import torch
import torch.nn as nn
from torch import Tensor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import MLPProjectorConfig
from visual_encoder import VisualEncoderWrapper


class MLPProjector(nn.Module):
    # Linear(D_v → D_hidden) + LayerNorm + GELU + Linear(D_hidden → D_llm)

    def __init__(self, cfg: MLPProjectorConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cfg.d_v, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
            nn.GELU(),
            nn.Linear(cfg.d_hidden, cfg.d_llm),
        )

    def forward(self, visual_features: Tensor) -> Tensor:  # [B,N,D_v] → [B,N,D_llm]
        return self.proj(visual_features)


class VQAModel(nn.Module):
    """
    VQA model: VisualEncoderWrapper → MLPProjector → LLM.

    vision_encoder  : VisualEncoderWrapper — forward(pixel_values) → [B,N,D_v]
    mlp_projector   : MLPProjector        — maps [B,N,D_v] → [B,N,D_llm]
    llm             : AutoModelForCausalLM (optionally wrapped with LoRA)
    """

    def __init__(
        self,
        vision_encoder: VisualEncoderWrapper,
        mlp_projector: MLPProjector,
        llm,
    ):
        super().__init__()
        self.vision_model = vision_encoder   # kept as vision_model for checkpoint compat
        self.mlp_projector = mlp_projector
        self.llm = llm

    def forward(
        self,
        pixel_values: Tensor,    # [B, 3, H, W]
        input_ids: Tensor,       # [B, T]
        attention_mask: Tensor,  # [B, T]
        labels: Tensor,          # [B, T]  -100 at positions not used for loss
    ):
        with torch.no_grad():
            visual_feats = self.vision_model(pixel_values)  # [B, N, D_v]

        visual_tokens = self.mlp_projector(visual_feats)    # [B, N, D_llm]
        B, N = visual_tokens.shape[:2]

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        vis_mask = torch.ones(B, N, dtype=attention_mask.dtype, device=attention_mask.device)
        extended_mask = torch.cat([vis_mask, attention_mask], dim=1)

        vis_labels = torch.full((B, N), -100, dtype=labels.dtype, device=labels.device)
        extended_labels = torch.cat([vis_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=extended_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Tensor,    # [B, 3, H, W]
        input_ids: Tensor,       # [B, T]  left-padded prompt prefix
        attention_mask: Tensor,  # [B, T]
        max_new_tokens: int = 128,
        **generate_kwargs,
    ) -> Tensor:
        visual_feats = self.vision_model(pixel_values)      # [B, N, D_v]
        visual_tokens = self.mlp_projector(visual_feats)    # [B, N, D_llm]
        B, N = visual_tokens.shape[:2]

        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        vis_mask = torch.ones(B, N, dtype=attention_mask.dtype, device=attention_mask.device)
        extended_mask = torch.cat([vis_mask, attention_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
