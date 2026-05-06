"""
level5_eval.py
==============
Level 5 — Fine-tuned VQA Model Evaluation

Evaluates a checkpoint produced by level5_train.py:
  --stage mlp  →  task1_mlp_best.pt   (MLP projector pre-trained)
  --stage vqa  →  task2_vqa_best.pt   (MLP + LoRA fine-tuned)

Two experimental cases:
  Case A — ground-truth article injected         →  ceiling performance
  Case B — article from Phase 2 image retrieval  →  real-world performance

The Case A − Case B gap = retrieval error propagation under fine-tuned conditions.
Compare with Level 4 gap to see whether fine-tuning reduces retrieval sensitivity.

Run
---
    # Case A (ground-truth article):
    python VQA_baseline/level5_eval.py \\
        --checkpoint outputs/checkpoints/task2_vqa_best.pt \\
        --case A \\
        --test_split data/splits/test_split.json \\
        --images_dir data/images \\
        --output_dir outputs/evaluation \\
        --device cuda:0

    # Both cases:
    python VQA_baseline/level5_eval.py \\
        --checkpoint outputs/checkpoints/task2_vqa_best.pt \\
        --case both \\
        --database data/database.json \\
        --retrieval_results outputs/retrieval/summary.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from VQA_baseline.metrics import compute_all_metrics, MetricsResult
from VQA_baseline.utils import (
    ArticleSelector,
    QASample,
    TimeTracker,
    flatten_qa_samples,
    load_retrieval_map,
    load_test_split,
    print_results_table,
    save_level_results,
)

LEVEL_NAME = "level5"


# ─────────────────────────────────────────────────────────────────────────────
# Load fine-tuned VQA model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _load_llm_tokenizer(llm_model_key: str, llm_model_id: str, trust_remote_code: bool):
    from transformers import AutoTokenizer
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

def load_finetuned_model(checkpoint_path: str | Path, device: torch.device):
    """
    Load VQAModel + processor + tokenizer from a task2_vqa_best.pt checkpoint.
    Reads visual_model_key, llm_model_key, and mlp_config directly from the
    checkpoint dict — no config overrides needed.

    Returns
    -------
    (vqa_model, processor, tokenizer, visual_key, llm_key)
    """
    from VQA_baseline.baseline_config import (
        VISUAL_MODELS, LLM_MODELS, MLPProjectorConfig, build_mlp_config, get_lora_targets,
        resolve_lora_targets, patch_custom_model_class,
    )
    from model import MLPProjector, VQAModel
    from visual_encoder import VisualEncoderWrapper
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = torch.load(checkpoint_path, map_location=device)

    visual_key = ckpt.get("visual_model_key", "clip_vit_l14")
    llm_key    = ckpt.get("llm_model_key",    "qwen2.5_7b")

    mlp_cfg_dict = ckpt.get("mlp_config", None)
    cfg_mlp = (
        MLPProjectorConfig(**mlp_cfg_dict)
        if mlp_cfg_dict
        else build_mlp_config(visual_key, llm_key)
    )

    lora_targets = ckpt.get("lora_target_modules", get_lora_targets(llm_key))

    visual_model_id = VISUAL_MODELS[visual_key]["id"]
    llm_model_id    = LLM_MODELS[llm_key]["id"]

    print(f"[Level5] Visual encoder : {visual_key}  ({visual_model_id})")
    print(f"[Level5] LLM            : {llm_key}  ({llm_model_id})")
    print(f"[Level5] MLP dims       : d_v={cfg_mlp.d_v}  d_llm={cfg_mlp.d_llm}")

    vision_encoder = VisualEncoderWrapper(
        visual_model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    vision_encoder.freeze()
    processor = vision_encoder.get_processor()

    trust_remote_code = LLM_MODELS[llm_key].get("trust_remote_code", False)

    llm_tokenizer = _load_llm_tokenizer(llm_key, llm_model_id, trust_remote_code)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "left"  # left-pad for autoregressive generation

    patch_custom_model_class(llm_key)
    llm_load_kwargs: dict = {"torch_dtype": torch.bfloat16, "trust_remote_code": trust_remote_code}
    if not trust_remote_code:
        llm_load_kwargs["device_map"] = {"": device}
    llm = AutoModelForCausalLM.from_pretrained(llm_model_id, **llm_load_kwargs)
    if hasattr(llm, "language_model"):
        llm = llm.language_model.to(device)
    lora_targets = resolve_lora_targets(llm_key, llm)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=lora_targets,
        bias="none",
    )
    llm = get_peft_model(llm, lora_cfg)

    mlp_projector = MLPProjector(cfg_mlp).to(device).to(torch.bfloat16)
    mlp_projector.load_state_dict(ckpt["mlp_state_dict"])
    llm.load_state_dict(ckpt["lora_adapter_state_dict"], strict=False)
    print(f"[Level5] Checkpoint loaded (epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f})")

    model = VQAModel(vision_encoder, mlp_projector, llm)
    model.vision_model.eval()
    model.mlp_projector.eval()
    model.llm.eval()

    return model, processor, llm_tokenizer, visual_key, llm_key


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — import from prompt_utils to keep training/eval in sync
# ─────────────────────────────────────────────────────────────────────────────

from VQA_baseline.prompt_utils import build_prompt as build_level5_prompt, SYSTEM_PROMPT  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop — one case at a time
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one_case(
    model,
    processor,
    tokenizer,
    samples: List[QASample],
    passage_lists: List[List[Tuple[str, str]]],
    case_label: str,
    device: torch.device,
    max_context_tokens: int = 256,
    max_total_length: int = 2048,
    max_new_tokens: int = 128,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Dict:
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    class _InferDataset(Dataset):
        def __init__(self, samples, passage_lists, processor, tokenizer, max_ctx, max_len):
            self.samples       = samples
            self.passage_lists = passage_lists
            self.processor     = processor
            self.tokenizer     = tokenizer
            self.max_ctx       = max_ctx
            self.max_len       = max_len

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            img = Image.open(s.image_path).convert("RGB")
            pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            _, prefix = build_level5_prompt(
                self.passage_lists[idx], s.question, answer="",
                max_context_tokens=self.max_ctx, tokenizer=self.tokenizer,
            )
            enc = self.tokenizer(
                prefix, max_length=self.max_len,
                truncation=True, return_tensors="pt",
            )
            return {
                "pixel_values":  pixel_values,
                "input_ids":     enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "answer":        s.answer,
                "question":      s.question,
                "image_id":      s.image_id,
            }

    def _collate(batch, pad_id):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        max_len = max(b["input_ids"].shape[0] for b in batch)
        B = len(batch)
        input_ids  = torch.full((B, max_len), pad_id, dtype=torch.long)
        attn_mask  = torch.zeros(B, max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            L = b["input_ids"].shape[0]
            input_ids[i, max_len - L:] = b["input_ids"]        # left-pad
            attn_mask[i, max_len - L:] = b["attention_mask"]
        return {
            "pixel_values":  pixel_values,
            "input_ids":     input_ids,
            "attention_mask": attn_mask,
            "answers":       [b["answer"]   for b in batch],
            "questions":     [b["question"] for b in batch],
            "image_ids":     [b["image_id"] for b in batch],
        }

    dataset = _InferDataset(
        samples, passage_lists, processor, tokenizer,
        max_context_tokens, max_total_length,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )

    predictions: List[str] = []
    references: List[str]  = []
    sample_records: List[Dict] = []

    tracker = TimeTracker(
        desc=f"[Level5-Case{case_label}] inference",
        total=len(samples),
    )

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        batch_size_actual = pixel_values.shape[0]

        with tracker:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        last_t = tracker.times[-1]
        per_sample_t = last_t / batch_size_actual
        for _ in range(batch_size_actual - 1):
            tracker._times.append(per_sample_t)

        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for pred, ref, q, img_id in zip(
            decoded, batch["answers"], batch["questions"], batch["image_ids"]
        ):
            predictions.append(pred.strip())
            references.append(ref)
            sample_records.append({
                "image_id":          img_id,
                "question":          q,
                "reference":         ref,
                "prediction":        pred.strip(),
                "inference_time_ms": per_sample_t,
            })

    tracker.close()
    try:
        metrics = compute_all_metrics(predictions, references, bertscore_device=str(device))
    except TypeError:
        metrics = compute_all_metrics(predictions, references)

    return {
        "case":    case_label,
        "metrics": metrics.to_dict(),
        "timing":  tracker.report,
        "samples": sample_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_level5(
    checkpoint_path: str | Path,
    test_split_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    database_path: Optional[str] = None,
    retrieval_results_path: Optional[str] = None,
    device_str: str = "cuda:0",
    case: str = "both",
    top_k_passages: int = 5,
    selector_method: str = "bm25",
    max_context_tokens: int = 256,
    max_total_length: int = 2048,
    max_new_tokens: int = 128,
    batch_size: int = 4,
    num_workers: int = 0,
    num_batches: int = 1,
    batch_idx: int = 0,
) -> Dict:
    if not (0 <= batch_idx < num_batches):
        raise ValueError(f"batch_idx={batch_idx} out of range for num_batches={num_batches}")

    device = torch.device(device_str)

    all_records = load_test_split(test_split_path)

    chunk = math.ceil(len(all_records) / num_batches)
    records = all_records[batch_idx * chunk : (batch_idx + 1) * chunk]

    samples = flatten_qa_samples(records, images_dir)

    if num_batches > 1:
        print(
            f"[Level5] Shard {batch_idx + 1}/{num_batches} — "
            f"records {batch_idx * chunk}–{batch_idx * chunk + len(records) - 1} "
            f"({len(records)} images, {len(samples)} QA samples)"
        )
    else:
        print(f"[Level5] Loaded {len(samples)} QA samples")

    if not samples:
        raise ValueError(f"No valid samples in shard {batch_idx}/{num_batches}. Check images_dir: {images_dir}")

    model, processor, tokenizer, visual_key, llm_key = load_finetuned_model(
        checkpoint_path, device
    )

    cases_to_run: List[str] = []
    if case in ("A", "both"):
        cases_to_run.append("A")
    if case in ("B", "both"):
        if not retrieval_results_path:
            print("[Level5] WARNING: Case B requires --retrieval_results. Skipping.")
        elif not database_path:
            print("[Level5] WARNING: Case B requires --database. Skipping.")
        else:
            cases_to_run.append("B")

    selector = ArticleSelector(method=selector_method)
    passage_lists: Dict[str, List] = {}

    if "A" in cases_to_run:
        passage_lists["A"] = [
            [(s.article_id, selector.select(s.article_content, s.question, top_k=top_k_passages))]
            for s in samples
        ]

    if "B" in cases_to_run:
        from VQA_baseline.data_utils import load_database
        print("[Level5] Loading Phase 2 retrieval map for Case B …")
        retrieval_map = load_retrieval_map(retrieval_results_path)
        database = load_database(database_path)
        passage_lists["B"] = [
            [(retrieval_map.get(s.image_id, s.article_id),
              selector.select(
                  database.get(retrieval_map.get(s.image_id, ""), {}).get("content", ""),
                  s.question, top_k=top_k_passages,
              ))]
            for s in samples
        ]

    model_label = f"finetuned_{visual_key}_{llm_key}"
    all_results: Dict = {}

    for case_label in cases_to_run:
        print(f"\n[Level5-Case{case_label}] ──────────────────────────────────────")
        result = evaluate_one_case(
            model, processor, tokenizer,
            samples, passage_lists[case_label],
            case_label=case_label,
            device=device,
            max_context_tokens=max_context_tokens,
            max_total_length=max_total_length,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        result["visual_key"] = visual_key
        result["llm_key"]    = llm_key
        key = f"{model_label}_case{case_label}"
        all_results[key] = result

        m = MetricsResult(**{k: v for k, v in result["metrics"].items()})
        print(f"\n[Level5-Case{case_label}] {model_label}:\n{m}")
        t = result["timing"]
        print(f"[Level5] Timing: avg={t.get('avg_ms')}ms  p95={t.get('p95_ms')}ms  total={t.get('total_s')}s")

    if "A" in cases_to_run and "B" in cases_to_run:
        mA = all_results[f"{model_label}_caseA"]["metrics"]
        mB = all_results[f"{model_label}_caseB"]["metrics"]
        print(f"\n[Level5] Case A−B gap (retrieval error propagation):")
        for metric in ("exact_match", "token_f1", "bertscore_f1", "cider"):
            gap = mA.get(metric, 0) - mB.get(metric, 0)
            print(f"  {metric:<20}: {gap:+.4f}")

    save_name = (
        f"{LEVEL_NAME}/batch_{batch_idx}_of_{num_batches}"
        if num_batches > 1
        else LEVEL_NAME
    )
    save_level_results(save_name, all_results, output_dir)
    print_results_table(save_name, all_results)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Level 5: Fine-tuned VQA evaluation")
    parser.add_argument("--checkpoint",         required=True,
                        help="Path to task2_vqa_best.pt checkpoint")
    parser.add_argument("--test_split",         default="data/splits/test1_set.json")
    parser.add_argument("--images_dir",         default="../../data/images")
    parser.add_argument("--database",           default=None)
    parser.add_argument("--retrieval_results",  default=None,
                        help="Path to Phase 2 summary.csv (required for Case B)")
    parser.add_argument("--output_dir",         default="outputs/evaluation")
    parser.add_argument("--device",             default="cuda:0")
    parser.add_argument("--case",               default="A", choices=["A", "B", "both"])
    parser.add_argument("--top_k_passages",     type=int, default=5)
    parser.add_argument("--selector",           default="bm25",
                        choices=["bm25", "dense", "first"])
    parser.add_argument("--max_context_tokens", type=int, default=256)
    parser.add_argument("--max_total_length",   type=int, default=2048)
    parser.add_argument("--max_new_tokens",     type=int, default=128)
    parser.add_argument("--batch_size",         type=int, default=4)
    parser.add_argument("--num_workers",        type=int, default=0)
    parser.add_argument(
        "--num_batches", type=int, default=1,
        help="Split test records into this many shards (for parallel jobs)"
    )
    parser.add_argument(
        "--batch_idx", type=int, default=0,
        help="Zero-based index of the shard to run (0 <= batch_idx < num_batches)"
    )
    args = parser.parse_args()

    run_level5(
        checkpoint_path=args.checkpoint,
        test_split_path=args.test_split,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        database_path=args.database,
        retrieval_results_path=args.retrieval_results,
        device_str=args.device,
        case=args.case,
        top_k_passages=args.top_k_passages,
        selector_method=args.selector,
        max_context_tokens=args.max_context_tokens,
        max_total_length=args.max_total_length,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        batch_idx=args.batch_idx,
    )


if __name__ == "__main__":
    main()
