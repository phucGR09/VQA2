"""
level5_train.py
===============
Level 5 — Fine-tuning pipeline

Two training stages run in order:

  Stage mlp:  MLP projector pre-training  (Task 1)
    frozen visual encoder + frozen LLM → train MLP on image → caption
    Output: task1_mlp_best.pt

  Stage vqa:  VQA fine-tuning with LoRA  (Task 2)
    frozen visual encoder, MLP (from stage 1) + LoRA adapters jointly trained
    Context: ArticleSelector on ground-truth article  (same RAG as Level 4 Case A)
    Output: task2_vqa_best.pt

Run
---
    # Stage 1 — MLP pre-train
    python VQA_baseline/level5_train.py --stage mlp \\
        --visual_model clip_vit_l14 --llm_model qwen2.5_7b \\
        --train_split data/splits/train_set.json \\
        --images_dir data/images \\
        --checkpoint_dir outputs/checkpoints

    # Stage 2 — VQA fine-tune
    python VQA_baseline/level5_train.py --stage vqa \\
        --visual_model clip_vit_l14 --llm_model qwen2.5_7b \\
        --train_split data/splits/train_split.json \\
        --images_dir data/images \\
        --checkpoint_dir outputs/checkpoints \\
        --task1_checkpoint outputs/checkpoints/task1_mlp_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional  # noqa: F401 (Optional used in run_* signatures)

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from VQA_baseline.baseline_config import (
    VISUAL_MODELS, LLM_MODELS,
    build_mlp_config, get_lora_targets, resolve_lora_targets, patch_custom_model_class,
    Level5LoRAConfig, Level5MlpTrainConfig, Level5VqaTrainConfig,
)
from VQA_baseline.level5_dataset import CaptionDataset, VQADataset, collate_fn
from VQA_baseline.utils import ArticleSelector
from model import MLPProjector, VQAModel
from visual_encoder import VisualEncoderWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Shared training loop
# ─────────────────────────────────────────────────────────────────────────────

def _load_llm_tokenizer(llm_model_key: str, llm_model_id: str, trust_remote_code: bool):
    if llm_model_key == "vintern_1b_v3_5":
        try:
            from transformers import Qwen2Tokenizer
            return Qwen2Tokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load tokenizer for vintern_1b_v3_5. "
                "The model repo lacks tokenizer.json; use Qwen2Tokenizer or install tiktoken."
            ) from exc
    return AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=trust_remote_code)

def _train_epoch(
    model: VQAModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    grad_accum_steps: int,
    trainable_params: list,
    device: torch.device,
    desc: str,
) -> float:
    accumulated_loss = 0.0
    total_loss = 0.0
    optimizer_step = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc=desc)):
        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values, input_ids, attention_mask, labels)
            loss = outputs.loss / grad_accum_steps

        loss.backward()
        accumulated_loss += loss.item()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1
            total_loss += accumulated_loss
            accumulated_loss = 0.0

    # flush remaining gradients if steps not divisible by grad_accum
    if accumulated_loss > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss
        optimizer_step += 1

    return total_loss / max(optimizer_step, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — MLP pre-train
# ─────────────────────────────────────────────────────────────────────────────

def run_mlp_pretrain(
    visual_model_key: str,
    llm_model_key: str,
    train_split_path: str,
    images_dir: str,
    cfg: Level5MlpTrainConfig,
    device: torch.device,
    flash_attn: bool = False,
    max_samples: Optional[int] = None,
) -> None:
    cfg_mlp = build_mlp_config(visual_model_key, llm_model_key)
    visual_model_id = VISUAL_MODELS[visual_model_key]["id"]
    llm_model_id    = LLM_MODELS[llm_model_key]["id"]

    print(f"[Level5-MLP] Visual encoder : {visual_model_key}  ({visual_model_id})")
    print(f"[Level5-MLP] LLM            : {llm_model_key}  ({llm_model_id})")
    print(f"[Level5-MLP] MLP dims       : d_v={cfg_mlp.d_v}  d_hidden={cfg_mlp.d_hidden}  d_llm={cfg_mlp.d_llm}")
    print(f"[Level5-MLP] Epochs={cfg.epochs}  batch={cfg.batch_size}  grad_accum={cfg.gradient_accumulation_steps}  lr={cfg.learning_rate}")

    vision_encoder = VisualEncoderWrapper(
        visual_model_id, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    vision_encoder.freeze()
    processor = vision_encoder.get_processor()

    trust_remote_code = LLM_MODELS[llm_model_key].get("trust_remote_code", False)

    llm_tokenizer = _load_llm_tokenizer(llm_model_key, llm_model_id, trust_remote_code)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "right"

    patch_custom_model_class(llm_model_key)
    llm_kwargs: dict = {"torch_dtype": torch.bfloat16, "trust_remote_code": trust_remote_code}
    if not trust_remote_code:
        llm_kwargs["device_map"] = {"": device}
    if flash_attn:
        llm_kwargs["attn_implementation"] = "flash_attention_2"
    llm = AutoModelForCausalLM.from_pretrained(llm_model_id, **llm_kwargs)
    if hasattr(llm, "language_model"):
        llm = llm.language_model.to(device)
    for p in llm.parameters():
        p.requires_grad = False

    mlp_projector = MLPProjector(cfg_mlp).to(device).to(torch.bfloat16)
    model = VQAModel(vision_encoder, mlp_projector, llm)

    dataset = CaptionDataset(
        train_split_path=Path(train_split_path),
        images_dir=Path(images_dir),
        processor=processor,
        tokenizer=llm_tokenizer,
        max_caption_length=cfg.max_caption_length,
        max_samples=max_samples,
    )
    if not dataset:
        raise ValueError("CaptionDataset is empty — check master_records_path has captions.")

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, llm_tokenizer.pad_token_id),
    )
    print(f"[Level5-MLP] Dataset : {len(dataset)} samples | {len(loader)} steps/epoch")

    trainable_params = list(mlp_projector.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    total_steps  = (len(loader) // cfg.gradient_accumulation_steps) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ckpt_path = Path(cfg.checkpoint_dir) / cfg.checkpoint_name
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.mlp_projector.train()
        model.vision_model.eval()
        model.llm.eval()

        avg_loss = _train_epoch(
            model, loader, optimizer, scheduler,
            cfg.gradient_accumulation_steps, trainable_params,
            device, desc=f"[Level5-MLP] Epoch {epoch}/{cfg.epochs}",
        )
        print(f"[Level5-MLP] Epoch {epoch}/{cfg.epochs} | loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":           epoch,
                "visual_model_key": visual_model_key,
                "llm_model_key":   llm_model_key,
                "mlp_config":      {"d_v": cfg_mlp.d_v, "d_hidden": cfg_mlp.d_hidden, "d_llm": cfg_mlp.d_llm},
                "mlp_state_dict":  mlp_projector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":            best_loss,
            }, ckpt_path)
            print(f"[Level5-MLP] Best checkpoint → {ckpt_path}  (loss={best_loss:.4f})")

    print(f"[Level5-MLP] Done. Best loss: {best_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — VQA fine-tune
# ─────────────────────────────────────────────────────────────────────────────

def run_vqa_finetune(
    visual_model_key: str,
    llm_model_key: str,
    train_split_path: str,
    images_dir: str,
    task1_checkpoint: str,
    cfg: Level5VqaTrainConfig,
    lora_cfg: Level5LoRAConfig,
    device: torch.device,
    flash_attn: bool = False,
    max_samples: Optional[int] = None,
) -> None:
    from peft import LoraConfig, TaskType, get_peft_model

    cfg_mlp      = build_mlp_config(visual_model_key, llm_model_key)
    visual_model_id = VISUAL_MODELS[visual_model_key]["id"]
    llm_model_id    = LLM_MODELS[llm_model_key]["id"]
    lora_targets = lora_cfg.target_modules or get_lora_targets(llm_model_key)

    print(f"[Level5-VQA] Visual encoder : {visual_model_key}  ({visual_model_id})")
    print(f"[Level5-VQA] LLM            : {llm_model_key}  ({llm_model_id})")
    print(f"[Level5-VQA] LoRA targets   : {lora_targets}")
    print(f"[Level5-VQA] Selector       : {cfg.selector_method}  top_k={cfg.top_k_sentences}")
    print(f"[Level5-VQA] Epochs={cfg.epochs}  batch={cfg.batch_size}  grad_accum={cfg.gradient_accumulation_steps}  lr={cfg.learning_rate}")

    vision_encoder = VisualEncoderWrapper(
        visual_model_id, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    vision_encoder.freeze()
    processor = vision_encoder.get_processor()

    trust_remote_code = LLM_MODELS[llm_model_key].get("trust_remote_code", False)

    llm_tokenizer = _load_llm_tokenizer(llm_model_key, llm_model_id, trust_remote_code)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "right"  # right-pad for teacher-forced training

    patch_custom_model_class(llm_model_key)
    llm_kwargs: dict = {"torch_dtype": torch.bfloat16, "trust_remote_code": trust_remote_code}
    if not trust_remote_code:
        llm_kwargs["device_map"] = {"": device}
    if flash_attn:
        llm_kwargs["attn_implementation"] = "flash_attention_2"
    llm = AutoModelForCausalLM.from_pretrained(llm_model_id, **llm_kwargs)
    if hasattr(llm, "language_model"):
        llm = llm.language_model.to(device)
    lora_targets = resolve_lora_targets(llm_model_key, llm)
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r, lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_targets, bias=lora_cfg.bias,
    )
    llm = get_peft_model(llm, peft_cfg)
    llm.print_trainable_parameters()

    ckpt1 = torch.load(task1_checkpoint, map_location=device)
    if "mlp_config" in ckpt1:
        from VQA_baseline.baseline_config import MLPProjectorConfig
        cfg_mlp = MLPProjectorConfig(**ckpt1["mlp_config"])
        expected_d_llm = LLM_MODELS[llm_model_key]["d_llm"]
        if cfg_mlp.d_llm != expected_d_llm:
            raise ValueError(
                f"Task 1 checkpoint was trained with d_llm={cfg_mlp.d_llm} "
                f"({ckpt1.get('llm_model_key', 'unknown')}), but current LLM "
                f"'{llm_model_key}' requires d_llm={expected_d_llm}. "
                f"Re-run --stage mlp with --llm_model {llm_model_key}."
            )
    mlp_projector = MLPProjector(cfg_mlp).to(device).to(torch.bfloat16)
    mlp_projector.load_state_dict(ckpt1["mlp_state_dict"])
    print(f"[Level5-VQA] MLP loaded from Task 1 (epoch={ckpt1['epoch']}, loss={ckpt1['loss']:.4f})")

    model = VQAModel(vision_encoder, mlp_projector, llm)

    selector = ArticleSelector(method=cfg.selector_method)
    dataset = VQADataset(
        train_split_path=Path(train_split_path),
        images_dir=Path(images_dir),
        processor=processor,
        tokenizer=llm_tokenizer,
        selector=selector,
        cfg=cfg,
        max_samples=max_samples,
    )
    if not dataset:
        raise ValueError("VQADataset is empty — check train_split.json.")

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_fn(b, llm_tokenizer.pad_token_id),
    )
    print(f"[Level5-VQA] Dataset : {len(dataset)} QA pairs | {len(loader)} steps/epoch")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    total_steps  = (len(loader) // cfg.gradient_accumulation_steps) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    ckpt_path = Path(cfg.checkpoint_dir) / cfg.checkpoint_name
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.mlp_projector.train()
        model.vision_model.eval()
        model.llm.train()  # LoRA adapters are trainable

        avg_loss = _train_epoch(
            model, loader, optimizer, scheduler,
            cfg.gradient_accumulation_steps, trainable_params,
            device, desc=f"[Level5-VQA] Epoch {epoch}/{cfg.epochs}",
        )
        print(f"[Level5-VQA] Epoch {epoch}/{cfg.epochs} | loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            lora_adapter = {k: v for k, v in llm.state_dict().items() if "lora_" in k}
            torch.save({
                "epoch":                epoch,
                "visual_model_key":     visual_model_key,
                "llm_model_key":        llm_model_key,
                "lora_target_modules":  lora_targets,
                "mlp_config":           {"d_v": cfg_mlp.d_v, "d_hidden": cfg_mlp.d_hidden, "d_llm": cfg_mlp.d_llm},
                "mlp_state_dict":       mlp_projector.state_dict(),
                "lora_adapter_state_dict": lora_adapter,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss":                 best_loss,
            }, ckpt_path)
            print(f"[Level5-VQA] Best checkpoint → {ckpt_path}  (loss={best_loss:.4f})")

    print(f"[Level5-VQA] Done. Best loss: {best_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Level 5 fine-tuning (--stage mlp | vqa)")
    parser.add_argument("--stage",           required=True, choices=["mlp", "vqa"],
                        help="mlp = Task 1 MLP pre-train  |  vqa = Task 2 VQA fine-tune")
    parser.add_argument("--visual_model",    default="clip_vit_l14")
    parser.add_argument("--llm_model",       default="qwen2.5_7b")
    parser.add_argument("--train_split",     default="data/splits/train_set.json")
    parser.add_argument("--images_dir",      default="../../data/images")
    parser.add_argument("--checkpoint_dir",  default="outputs/checkpoints")
    parser.add_argument("--device",          default="cuda:0")

    # Stage 1 overrides
    parser.add_argument("--mlp_epochs",     type=int,   default=None)
    parser.add_argument("--mlp_batch_size", type=int,   default=None)
    parser.add_argument("--mlp_lr",         type=float, default=None)

    # Stage 2 overrides
    parser.add_argument("--task1_checkpoint", default=None,
                        help="Task 1 .pt path (required for --stage vqa; default: checkpoint_dir/task1_mlp_best.pt)")
    parser.add_argument("--vqa_epochs",      type=int,   default=None)
    parser.add_argument("--vqa_batch_size",  type=int,   default=None)
    parser.add_argument("--vqa_lr",          type=float, default=None)
    parser.add_argument("--selector",        default="bm25", choices=["bm25", "dense", "first"])
    parser.add_argument("--top_k_sentences", type=int,   default=None)
    parser.add_argument("--lora_r",          type=int,   default=None)
    parser.add_argument("--lora_alpha",      type=int,   default=None)
    parser.add_argument("--flash_attn",      action="store_true",
                        help="Use Flash Attention 2 (requires flash-attn package). "
                             "Saves ~30%% VRAM and speeds up training ~2-3x.")
    parser.add_argument("--max_samples",    type=int, default=None,
                        help="Cap the number of training samples (useful for quick experiments).")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.stage == "mlp":
        cfg = Level5MlpTrainConfig(checkpoint_dir=args.checkpoint_dir)
        if args.mlp_epochs    is not None: cfg.epochs        = args.mlp_epochs
        if args.mlp_batch_size is not None: cfg.batch_size   = args.mlp_batch_size
        if args.mlp_lr        is not None: cfg.learning_rate = args.mlp_lr

        run_mlp_pretrain(
            visual_model_key=args.visual_model,
            llm_model_key=args.llm_model,
            train_split_path=args.train_split,
            images_dir=args.images_dir,
            cfg=cfg,
            device=device,
            flash_attn=args.flash_attn,
            max_samples=args.max_samples,
        )

    else:  # vqa
        task1_ckpt = args.task1_checkpoint or f"{args.checkpoint_dir}/task1_mlp_best.pt"
        if not Path(task1_ckpt).exists():
            raise FileNotFoundError(
                f"Task 1 checkpoint not found: {task1_ckpt}\n"
                "Run --stage mlp first."
            )

        cfg = Level5VqaTrainConfig(
            checkpoint_dir=args.checkpoint_dir,
            task1_checkpoint=task1_ckpt,
            selector_method=args.selector,
        )
        if args.vqa_epochs      is not None: cfg.epochs          = args.vqa_epochs
        if args.vqa_batch_size  is not None: cfg.batch_size       = args.vqa_batch_size
        if args.vqa_lr          is not None: cfg.learning_rate    = args.vqa_lr
        if args.top_k_sentences is not None: cfg.top_k_sentences  = args.top_k_sentences

        lora_cfg = Level5LoRAConfig()
        if args.lora_r     is not None: lora_cfg.r          = args.lora_r
        if args.lora_alpha is not None: lora_cfg.lora_alpha  = args.lora_alpha

        run_vqa_finetune(
            visual_model_key=args.visual_model,
            llm_model_key=args.llm_model,
            train_split_path=args.train_split,
            images_dir=args.images_dir,
            task1_checkpoint=task1_ckpt,
            cfg=cfg,
            lora_cfg=lora_cfg,
            device=device,
            flash_attn=args.flash_attn,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
