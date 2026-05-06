"""
level2_image_qa.py
==================
Level 2 Baseline — Image + Question  (no article context)

This is the *standard VQA* baseline: the model receives only the image and
the question.  No article text is provided.  The gap between Level 2 and
Level 4 measures how much the article context contributes.

Prompt template (Vietnamese):
    [image]
    Câu hỏi: {question}
    Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác.

Run
---
    python copy/baseline/level2_image_qa.py \\
        --models vintern_1b_v3 qwen2vl_7b internvl3_8b blip2_opt_27b \\
        --test_split data/splits/test_split.json \\
        --images_dir data/images \\
        --output_dir outputs/evaluation \\
        --device cuda:0

    # Use 'all' to run every model in LEVEL2_MODELS:
    python copy/baseline/level2_image_qa.py --models all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── resolve copy/ root so sibling modules are importable ─────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from VQA_baseline.metrics import compute_all_metrics
from VQA_baseline.model_registry import (
    LEVEL2_MODELS,
    MODEL_REGISTRY,
    BaseModelWrapper,
    build_wrapper,
)
from VQA_baseline.utils import (
    QASample,
    TimeTracker,
    flatten_qa_samples,
    load_test_split,
    print_results_table,
    save_level_results,
)

LEVEL_NAME = "level2_image_qa"

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "Câu hỏi: {question}\n"
    "Hãy trả lời bằng tiếng Việt, ngắn gọn và chính xác."
)


# ─────────────────────────────────────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    wrapper: BaseModelWrapper,
    samples: List[QASample],
    batch_size: int = 4,
    num_workers: int = 0,        # unused; kept for API compatibility
    prefetch_images: bool = True, # unused; kept for API compatibility
) -> Dict:
    """
    Run inference grouped by image.

    Each unique image is opened once; all questions for that image are
    sub-batched and answered with the same PIL object, avoiding repeated
    image decoding.
    """
    from PIL import Image

    # Group samples by image path, preserving original encounter order
    image_groups: Dict[str, List[QASample]] = {}
    for s in samples:
        image_groups.setdefault(s.image_path, []).append(s)

    predictions: List[str] = []
    references: List[str] = []
    sample_records: List[Dict] = []

    tracker = TimeTracker(
        desc=f"[Level2] {wrapper.config.model_key}",
        total=len(samples),
    )

    for img_path, group in image_groups.items():
        # Decode image once for all questions in this group
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\n[Level2] Cannot open {img_path}: {e}")
            pil_img = None

        for i in range(0, len(group), batch_size):
            sub = group[i : i + batch_size]
            n = len(sub)
            questions = [PROMPT_TEMPLATE.format(question=s.question) for s in sub]
            img_paths = [img_path] * n
            # Reuse the same PIL image object for every question in the sub-batch
            images = [pil_img] * n

            with tracker:
                try:
                    preds = wrapper.generate_batch(
                        questions=questions,
                        image_paths=img_paths,
                        contexts=None,
                        images=images,
                    )
                except Exception as e:
                    print(f"\n[Level2] ERROR on batch for {img_path}: {e}")
                    if not Path(img_path).exists():
                        print(f"[Level2] Missing image: {img_path}")
                    preds = []
                    for q, s in zip(questions, sub):
                        try:
                            preds.append(wrapper.generate_answer(
                                question=q,
                                image_path=img_path,
                                context=None,
                            ))
                        except Exception as e2:
                            preds.append("")
                            print(f"\n[Level2] ERROR on {s.image_id}: {e2}")

            batch_t = tracker.times[-1]
            per_sample_t = batch_t / max(1, n)
            # __exit__ already advanced pbar by 1; advance remaining n-1
            tracker._pbar.update(n - 1)
            for _ in range(n - 1):
                tracker._times.append(per_sample_t)

            for pred, s in zip(preds, sub):
                predictions.append(pred)
                references.append(s.answer)
                sample_records.append({
                    "image_id":          s.image_id,
                    "question":          s.question,
                    "reference":         s.answer,
                    "prediction":        pred,
                    "inference_time_ms": per_sample_t,
                })

    tracker.close()
    metrics = compute_all_metrics(predictions, references)

    return {
        "model_key": wrapper.config.model_key,
        "model_id":  wrapper.config.model_id,
        "metrics":   metrics.to_dict(),
        "timing":    tracker.report,
        "samples":   sample_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_level2(
    model_keys: List[str],
    test_split_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 4,
    num_workers: int = 0,
    prefetch_images: bool = True,
    num_batches: int = 1,
    batch_idx: int = 0,
) -> Dict:
    """
    Evaluate all requested models for Level 2.

    Parameters
    ----------
    num_batches : int
        Split the test records into this many equal shards.
        Useful for running multiple jobs in parallel.
    batch_idx : int
        Zero-based index of the shard to run (0 ≤ batch_idx < num_batches).
        Results are saved under a per-shard sub-directory when num_batches > 1.

    Returns
    -------
    Dict keyed by model_key, each value is the result dict from evaluate_model().
    """
    if not (0 <= batch_idx < num_batches):
        raise ValueError(f"batch_idx={batch_idx} out of range for num_batches={num_batches}")

    # ── Device info ───────────────────────────────────────────────────────────
    import torch as _torch
    print(f"[Level2] Device : {device}")
    if device.startswith("cuda") and _torch.cuda.is_available():
        idx = int(device.split(":")[-1]) if ":" in device else 0
        print(f"[Level2] GPU    : {_torch.cuda.get_device_name(idx)}")
        free, total = _torch.cuda.mem_get_info(idx)
        print(f"[Level2] VRAM   : {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

    # ── Load & shard data ─────────────────────────────────────────────────────
    all_records = load_test_split(test_split_path)

    # Split at the record (image) level so all QAs for an image stay together
    chunk = math.ceil(len(all_records) / num_batches)
    records = all_records[batch_idx * chunk : (batch_idx + 1) * chunk]

    samples = flatten_qa_samples(records, images_dir)

    if num_batches > 1:
        print(
            f"[Level2] Shard {batch_idx + 1}/{num_batches} — "
            f"records {batch_idx * chunk}–{batch_idx * chunk + len(records) - 1} "
            f"({len(records)} images, {len(samples)} QA samples)"
        )
    else:
        print(f"[Level2] Loaded {len(samples)} QA samples from {len(records)} records")

    if not samples:
        raise ValueError(
            f"No valid samples in shard {batch_idx}/{num_batches}. "
            f"Check images_dir: {images_dir}"
        )

    # ── Filter to VL models only ──────────────────────────────────────────────
    vl_keys = [k for k in model_keys if MODEL_REGISTRY[k].is_vl_model]
    if len(vl_keys) < len(model_keys):
        skipped = set(model_keys) - set(vl_keys)
        print(f"[Level2] Skipping text-only models (not applicable to Level 2): {skipped}")

    all_results: Dict = {}

    for model_key in vl_keys:
        print(f"\n[Level2] ── {model_key} ────────────────────────────────────")
        cfg = MODEL_REGISTRY[model_key]
        print(f"  Model ID : {cfg.model_id}")
        print(f"  Family   : {cfg.family}  |  d_v={cfg.visual_hidden_dim}  d_llm={cfg.llm_hidden_dim}")

        wrapper = build_wrapper(model_key, device=device)
        wrapper.load()

        result = evaluate_model(
            wrapper, samples,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_images=prefetch_images,
        )
        all_results[model_key] = result

        wrapper.unload()

        # Print per-model summary
        from VQA_baseline.metrics import MetricsResult
        m = MetricsResult(**{k: v for k, v in result["metrics"].items()})
        print(f"\n[Level2] {model_key} — Metrics:\n{m}")
        t = result["timing"]
        print(f"[Level2] Timing: avg={t.get('avg_ms')}ms  p95={t.get('p95_ms')}ms  total={t.get('total_s')}s")

    # ── Save & print comparison ───────────────────────────────────────────────
    # When sharding, nest results under a per-shard sub-directory
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
    parser = argparse.ArgumentParser(description="Level 2: Image + Question baseline")
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help="Model keys to evaluate, or 'all' for default Level 2 set"
    )
    parser.add_argument("--test_split",  default="data/splits/test1_set.json")
    parser.add_argument("--images_dir",  default="../../../Eventa/webCrawl/src/database_image")
    parser.add_argument("--output_dir",  default="outputs/evaluation")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_prefetch", action="store_true",
                        help="Disable image prefetching in DataLoader workers")
    parser.add_argument(
        "--num_batches", type=int, default=1,
        help="Split test records into this many shards (for parallel jobs)"
    )
    parser.add_argument(
        "--batch_idx", type=int, default=0,
        help="Zero-based index of the shard to run (0 <= batch_idx < num_batches)"
    )
    args = parser.parse_args()

    model_keys = LEVEL2_MODELS if args.models == ["all"] else args.models

    # Validate keys
    unknown = [k for k in model_keys if k not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown model keys: {unknown}. Available: {sorted(MODEL_REGISTRY)}")

    run_level2(
        model_keys=model_keys,
        test_split_path=args.test_split,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_images=not args.no_prefetch,
        num_batches=args.num_batches,
        batch_idx=args.batch_idx,
    )


if __name__ == "__main__":
    main()
