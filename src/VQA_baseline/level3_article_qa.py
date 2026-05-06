"""
level3_article_qa.py
====================
Level 3 Baseline — Article + Question  (no image)

The model receives article text and the question but NO image.
This isolates whether visual context is necessary.

Comparison with Level 2 (image-only):
  Level 3 >> Level 2  →  questions are more text-grounded (weakness for visual dataset)
  Level 3 << Level 2  →  image carries non-redundant information (strength)

Only Case A (ground truth article) is meaningful here — there is no visual
retrieval to perform.  Article sentences are selected via BM25 (top-k).

Prompt template (Vietnamese):
    Bài viết:
    {selected_sentences}

    Câu hỏi: {question}
    Hãy trả lời bằng tiếng Việt dựa trên bài viết trên, ngắn gọn và chính xác.
    Trả lời:

Text models: Vistral-7B-Chat, Qwen2.5-7B-Instruct (text-only)

Run
---
    python copy/baseline/level3_article_qa.py \\
        --models vistral_7b qwen2.5_7b_text \\
        --test_split data/splits/test_split.json \\
        --images_dir data/images \\
        --output_dir outputs/evaluation \\
        --device cuda:0 \\
        --top_k_sentences 5 \\
        --selector bm25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from VQA_baseline.metrics import compute_all_metrics, MetricsResult
from VQA_baseline.model_registry import (
    LEVEL3_MODELS,
    MODEL_REGISTRY,
    BaseModelWrapper,
    build_wrapper,
)
from VQA_baseline.utils import (
    ArticleSelector,
    QASample,
    TimeTracker,
    flatten_qa_samples,
    load_test_split,
    print_results_table,
    save_level_results,
)

LEVEL_NAME = "level3_article_qa"

PROMPT_TEMPLATE = (
    "Bài viết:\n{context}\n\n"
    "Câu hỏi: {question}\n"
    "Hãy trả lời bằng tiếng Việt dựa trên bài viết trên, ngắn gọn và chính xác.\n"
    "Trả lời:"
)


# ─────────────────────────────────────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    wrapper: BaseModelWrapper,
    samples: List[QASample],
    selector: ArticleSelector,
    top_k: int = 5,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Dict:
    from torch.utils.data import DataLoader, Dataset

    contexts = [
        selector.select(s.article_content, s.question, top_k=top_k)
        for s in samples
    ]

    class _InferDataset(Dataset):
        def __init__(self, samples: List[QASample], contexts: List[str]):
            self.samples = samples
            self.contexts = contexts

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict:
            s = self.samples[idx]
            context = self.contexts[idx]
            prompt = PROMPT_TEMPLATE.format(context=context, question=s.question)
            return {
                "prompt": prompt,
                "context": context,
                "answer": s.answer,
                "question": s.question,
                "image_id": s.image_id,
            }

    def _collate(batch: List[Dict]) -> Dict:
        return {
            "prompts": [b["prompt"] for b in batch],
            "contexts": [b["context"] for b in batch],
            "answers": [b["answer"] for b in batch],
            "questions": [b["question"] for b in batch],
            "image_ids": [b["image_id"] for b in batch],
        }

    loader = DataLoader(
        _InferDataset(samples, contexts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    predictions: List[str] = []
    references: List[str]  = []
    sample_records: List[Dict] = []

    tracker = TimeTracker(
        desc=f"[Level3] {wrapper.config.model_key}",
        total=len(samples),
    )

    for batch in loader:
        batch_size_actual = len(batch["prompts"])

        with tracker:
            try:
                preds = wrapper.generate_batch(
                    questions=batch["prompts"],
                    image_paths=[None] * batch_size_actual,
                    contexts=None,
                )
            except Exception as e:
                print(f"\n[Level3] ERROR on batch: {e}")
                preds = []
                for q, image_id in zip(batch["prompts"], batch["image_ids"]):
                    try:
                        preds.append(wrapper.generate_answer(
                            question=q,
                            image_path=None,
                            context=None,
                        ))
                    except Exception as e2:
                        preds.append("")
                        print(f"\n[Level3] ERROR on {image_id}: {e2}")

        last_t = tracker.times[-1]
        per_sample_t = last_t / max(1, batch_size_actual)
        for _ in range(batch_size_actual - 1):
            tracker._times.append(per_sample_t)
            tracker._times[-1] = per_sample_t

        for pred, ref, q, ctx, img_id in zip(
            preds, batch["answers"], batch["questions"], batch["contexts"], batch["image_ids"]
        ):
            predictions.append(pred)
            references.append(ref)
            sample_records.append({
                "image_id":          img_id,
                "question":          q,
                "reference":         ref,
                "prediction":        pred,
                "context_used":      ctx,
                "inference_time_ms": per_sample_t,
            })

    tracker.close()
    metrics = compute_all_metrics(predictions, references)

    return {
        "model_key":   wrapper.config.model_key,
        "model_id":    wrapper.config.model_id,
        "top_k":       top_k,
        "selector":    selector.method,
        "metrics":     metrics.to_dict(),
        "timing":      tracker.report,
        "samples":     sample_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_level3(
    model_keys: List[str],
    test_split_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    top_k_sentences: int = 5,
    selector_method: str = "bm25",
    batch_size: int = 4,
    num_workers: int = 0,
) -> Dict:
    records = load_test_split(test_split_path)
    samples = flatten_qa_samples(records, images_dir)
    print(f"[Level3] Loaded {len(samples)} QA samples")

    if not samples:
        raise ValueError(f"No valid samples. Check images_dir: {images_dir}")

    vl_as_text = [k for k in model_keys if MODEL_REGISTRY[k].is_vl_model]
    if vl_as_text:
        print(f"[Level3] NOTE: VL models used in text-only mode (image_path=None): {vl_as_text}")

    selector = ArticleSelector(method=selector_method)
    all_results: Dict = {}

    for model_key in model_keys:
        print(f"\n[Level3] ── {model_key} ────────────────────────────────────")
        cfg = MODEL_REGISTRY[model_key]
        print(f"  Model ID  : {cfg.model_id}")
        print(f"  Family    : {cfg.family}  |  d_llm={cfg.llm_hidden_dim}")
        print(f"  Selector  : {selector_method}  top_k={top_k_sentences}")
    
        wrapper = build_wrapper(model_key, device=device)
        wrapper.load()

        result = evaluate_model(
            wrapper, samples, selector,
            top_k=top_k_sentences,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        all_results[model_key] = result

        wrapper.unload()

        m = MetricsResult(**{k: v for k, v in result["metrics"].items()})
        print(f"\n[Level3] {model_key} — Metrics:\n{m}")
        t = result["timing"]
        print(f"[Level3] Timing: avg={t.get('avg_ms')}ms  p95={t.get('p95_ms')}ms  total={t.get('total_s')}s")

    save_level_results(LEVEL_NAME, all_results, output_dir)
    print_results_table(LEVEL_NAME, all_results)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Level 3: Article + Question baseline")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--test_split",        default="data/splits/test1_set.json")
    parser.add_argument("--images_dir",        default="../../../Eventa/webCrawl/src/database_image")
    parser.add_argument("--output_dir",        default="outputs/evaluation")
    parser.add_argument("--device",            default="cuda:0")
    parser.add_argument("--top_k_sentences",   type=int, default=5,
                        help="Number of article sentences to inject as context")
    parser.add_argument("--selector",          default="bm25",
                        choices=["bm25", "dense", "first"],
                        help="Sentence selection method")
    parser.add_argument("--batch_size",        type=int, default=4)
    parser.add_argument("--num_workers",       type=int, default=0)
    args = parser.parse_args()

    model_keys = LEVEL3_MODELS if args.models == ["all"] else args.models
    unknown = [k for k in model_keys if k not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown model keys: {unknown}")

    run_level3(
        model_keys=model_keys,
        test_split_path=args.test_split,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        device=args.device,
        top_k_sentences=args.top_k_sentences,
        selector_method=args.selector,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
