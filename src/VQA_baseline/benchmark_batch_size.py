"""
benchmark_batch_size.py
=======================
Quick throughput benchmark for batched inference (Level 2/3/4).

Examples
--------
# Level 2, two batch sizes
python copy/VQA_baseline/benchmark_batch_size.py \
    --level 2 --models qwen2vl_7b --batch_sizes 1 4 8 \
    --test_split data/splits/test_split.json --images_dir data/images

# Level 3 (text-only)
python copy/VQA_baseline/benchmark_batch_size.py \
    --level 3 --models qwen2.5_7b_text --batch_sizes 2 4 8

# Level 4 (Case B)
python copy/VQA_baseline/benchmark_batch_size.py \
    --level 4 --case B --database data/database.json --batch_sizes 1 2 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from VQA_baseline.model_registry import (
    LEVEL2_MODELS,
    LEVEL3_MODELS,
    LEVEL4_MODELS,
    MODEL_REGISTRY,
    BaseModelWrapper,
    build_wrapper,
)
from VQA_baseline.utils import ArticleSelector, QASample, flatten_qa_samples, load_test_split


def _chunks(n: int, batch_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def _sync_if_cuda(device_str: str) -> None:
    if "cuda" in device_str and torch.cuda.is_available():
        torch.cuda.synchronize()


def _prepare_level2(samples: List[QASample]) -> Tuple[List[str], List[str]]:
    from VQA_baseline.level2_image_qa import PROMPT_TEMPLATE
    prompts = [PROMPT_TEMPLATE.format(question=s.question) for s in samples]
    image_paths = [s.image_path for s in samples]
    return prompts, image_paths


def _prepare_level3(
    samples: List[QASample],
    selector_method: str,
    top_k_sentences: int,
) -> Tuple[List[str], List[Optional[str]]]:
    from VQA_baseline.level3_article_qa import PROMPT_TEMPLATE

    selector = ArticleSelector(method=selector_method)
    contexts = [
        selector.select(s.article_content, s.question, top_k=top_k_sentences)
        for s in samples
    ]
    prompts = [
        PROMPT_TEMPLATE.format(context=ctx, question=s.question)
        for s, ctx in zip(samples, contexts)
    ]
    image_paths = [None] * len(samples)
    return prompts, image_paths


def _prepare_level4(
    samples: List[QASample],
    case: str,
    database_path: Optional[str],
    device: str,
    selector_method: str,
    top_k_sentences: int,
) -> Tuple[List[str], List[str]]:
    from VQA_baseline.level4_rag_vqa import PROMPT_TEMPLATE, _build_retrieved_contexts

    selector = ArticleSelector(method=selector_method)
    if case == "A":
        contexts = [
            selector.select(s.article_content, s.question, top_k=top_k_sentences)
            for s in samples
        ]
    else:
        if not database_path:
            raise ValueError("Case B requires --database")
        contexts = _build_retrieved_contexts(samples, database_path, device, top_k=top_k_sentences)

    prompts = [
        PROMPT_TEMPLATE.format(context=ctx, question=s.question)
        for s, ctx in zip(samples, contexts)
    ]
    image_paths = [s.image_path for s in samples]
    return prompts, image_paths


def _run_one_model(
    wrapper: BaseModelWrapper,
    prompts: List[str],
    image_paths: List[Optional[str]],
    batch_sizes: List[int],
    device: str,
    warmup: int,
) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    n = len(prompts)

    for bs in batch_sizes:
        if bs <= 0:
            continue
        # Warmup on the first batch to reduce startup noise.
        if warmup > 0:
            start, end = 0, min(bs, n)
            try:
                _sync_if_cuda(device)
                wrapper.generate_batch(
                    questions=prompts[start:end],
                    image_paths=image_paths[start:end],
                    contexts=None,
                )
                _sync_if_cuda(device)
            except Exception:
                pass

        total_time = 0.0
        total_samples = 0
        errors = 0

        for start, end in _chunks(n, bs):
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            try:
                wrapper.generate_batch(
                    questions=prompts[start:end],
                    image_paths=image_paths[start:end],
                    contexts=None,
                )
            except Exception:
                errors += 1
                # Fallback to single-sample to keep the run going
                for i in range(start, end):
                    wrapper.generate_answer(
                        question=prompts[i],
                        image_path=image_paths[i],
                        context=None,
                    )
            _sync_if_cuda(device)
            total_time += time.perf_counter() - t0
            total_samples += (end - start)

        sps = total_samples / total_time if total_time > 0 else 0.0
        ms = (total_time / total_samples * 1000.0) if total_samples > 0 else 0.0
        results[bs] = {
            "samples": float(total_samples),
            "total_s": round(total_time, 3),
            "samples_per_s": round(sps, 3),
            "ms_per_sample": round(ms, 3),
            "batch_errors": float(errors),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-size benchmark for VQA baselines")
    parser.add_argument("--level", type=int, choices=[2, 3, 4], required=True)
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--test_split", default="data/splits/test_split.json")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--database", default=None)
    parser.add_argument("--case", default="A", choices=["A", "B"])
    parser.add_argument("--selector", default="bm25", choices=["bm25", "dense", "first"])
    parser.add_argument("--top_k_sentences", type=int, default=5)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    if args.level == 2:
        default_models = LEVEL2_MODELS
    elif args.level == 3:
        default_models = LEVEL3_MODELS
    else:
        default_models = LEVEL4_MODELS

    model_keys = default_models if args.models == ["all"] else args.models
    unknown = [k for k in model_keys if k not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown model keys: {unknown}")

    records = load_test_split(args.test_split)
    samples = flatten_qa_samples(records, args.images_dir)
    if not samples:
        raise ValueError("No valid samples loaded")
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    if args.level == 2:
        prompts, image_paths = _prepare_level2(samples)
    elif args.level == 3:
        prompts, image_paths = _prepare_level3(
            samples, args.selector, args.top_k_sentences
        )
    else:
        prompts, image_paths = _prepare_level4(
            samples,
            case=args.case,
            database_path=args.database,
            device=args.device,
            selector_method=args.selector,
            top_k_sentences=args.top_k_sentences,
        )

    for model_key in model_keys:
        cfg = MODEL_REGISTRY[model_key]
        print(f"\n[Benchmark] {model_key} ({cfg.model_id})")

        wrapper = build_wrapper(model_key, device=args.device)
        wrapper.load()

        results = _run_one_model(
            wrapper,
            prompts,
            image_paths,
            batch_sizes=args.batch_sizes,
            device=args.device,
            warmup=args.warmup,
        )

        for bs in sorted(results.keys()):
            r = results[bs]
            print(
                f"  bs={bs:<3}  samples={int(r['samples'])}  "
                f"total={r['total_s']}s  "
                f"{r['samples_per_s']} samples/s  "
                f"{r['ms_per_sample']} ms/sample  "
                f"errors={int(r['batch_errors'])}"
            )

        wrapper.unload()


if __name__ == "__main__":
    main()
