"""
level4_rag_vqa.py
=================
Level 4 Baseline — Image + Article + Question  (zero-shot RAG-VQA, no fine-tuning)

The model receives all three inputs.  No training on the dataset is performed.
This directly answers: "Is fine-tuning necessary, or can a strong VLM just read
the article and answer?"

Two experimental cases:
  Case A — ground-truth article injected          →  measures ceiling of zero-shot RAG
  Case B — article from Phase 2 image retrieval   →  measures real-world zero-shot RAG

The Case A − Case B gap here is the *retrieval error propagation* measure under
zero-shot conditions.  Compare with Level 5 Case A − Case B to see whether
fine-tuning reduces sensitivity to retrieval quality.

Prompt template (Vietnamese):
    [image]
    Dưới đây là nội dung bài viết liên quan:
    {selected_sentences}

    Câu hỏi: {question}
    Hãy trả lời bằng tiếng Việt dựa trên hình ảnh và bài viết trên, ngắn gọn và chính xác.

Run
---
    # Case A only (ground-truth article):
    python copy/baseline/level4_rag_vqa.py \\
        --models vintern_1b_v3 qwen2vl_7b internvl3_8b \\
        --case A \\
        --test_split data/splits/test_split.json \\
        --images_dir data/images \\
        --database data/database.json \\
        --output_dir outputs/evaluation \\
        --device cuda:0

    # Case B (Phase 2 retrieved article):
    python copy/baseline/level4_rag_vqa.py --case B --database data/database.json \\
        --retrieval_results outputs/retrieval/summary.csv

    # Both cases at once:
    python copy/baseline/level4_rag_vqa.py --case both \\
        --database data/database.json --retrieval_results outputs/retrieval/summary.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from VQA_baseline.metrics import compute_all_metrics, MetricsResult
from VQA_baseline.model_registry import (
    LEVEL4_MODELS,
    MODEL_REGISTRY,
    BaseModelWrapper,
    build_wrapper,
)
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

LEVEL_NAME = "level4_rag_vqa"

PROMPT_TEMPLATE = (
    "Dưới đây là nội dung bài viết liên quan:\n{context}\n\n"
    "Câu hỏi: {question}\n"
    "Hãy trả lời bằng tiếng Việt dựa trên hình ảnh và bài viết trên, ngắn gọn và chính xác."
)


# ─────────────────────────────────────────────────────────────────────────────
# Single-model evaluation  (one case at a time)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_one_case(
    wrapper: BaseModelWrapper,
    samples: List[QASample],
    contexts: List[str],           # pre-built context per sample
    case_label: str,               # "A" or "B"
    selector: ArticleSelector,
    top_k: int,
    batch_size: int = 4,
    num_workers: int = 0,          # unused; kept for API compatibility
    prefetch_images: bool = True,  # unused; kept for API compatibility
) -> Dict:
    """
    Run inference grouped by image; each image is opened once.

    contexts  – one string per sample (full or pre-retrieved passage block).
    selector  – used only for Case A to trim the article to top-k sentences.
    top_k     – number of sentences to inject (Case A uses selector, Case B uses full retrieved text).
    """
    from PIL import Image

    effective_contexts = [
        selector.select(raw_ctx, s.question, top_k=top_k)
        for s, raw_ctx in zip(samples, contexts)
    ]

    # Flat global list — batch across images to fill batch_size properly
    flat: List[Tuple[QASample, str]] = list(zip(samples, effective_contexts))

    predictions: List[str] = []
    references: List[str] = []
    sample_records: List[Dict] = []

    tracker = TimeTracker(
        desc=f"[Level4-Case{case_label}] {wrapper.config.model_key}",
        total=len(samples),
    )

    # Small recent cache so Q&As from the same image in adjacent batches
    # don't re-open from disk (PIL open is cheap but cache avoids redundancy).
    _img_cache: Dict[str, Optional[Image.Image]] = {}

    for i in range(0, len(flat), batch_size):
        sub = flat[i : i + batch_size]
        n = len(sub)
        questions = [PROMPT_TEMPLATE.format(context=ctx, question=s.question) for s, ctx in sub]
        img_paths = [s.image_path for s, _ in sub]
        images: List[Optional[Image.Image]] = []
        for p in img_paths:
            if p not in _img_cache:
                try:
                    _img_cache[p] = Image.open(p).convert("RGB")
                except Exception as e:
                    print(f"\n[Level4-Case{case_label}] Cannot open {p}: {e}")
                    _img_cache[p] = None
            images.append(_img_cache[p])

        with tracker:
            try:
                preds = wrapper.generate_batch(
                    questions=questions,
                    image_paths=img_paths,
                    contexts=None,
                    images=images,
                )
            except Exception as e:
                print(f"\n[Level4-Case{case_label}] ERROR on batch: {e}")
                preds = []
                for q, (s, _ctx) in zip(questions, sub):
                    try:
                        preds.append(wrapper.generate_answer(
                            question=q,
                            image_path=s.image_path,
                            context=None,
                        ))
                    except Exception as e2:
                        preds.append("")
                        print(f"\n[Level4-Case{case_label}] ERROR on {s.image_id}: {e2}")

        batch_t = tracker.times[-1]
        per_sample_t = batch_t / max(1, n)
        # __exit__ already advanced pbar by 1; advance remaining n-1
        tracker._pbar.update(n - 1)
        for _ in range(n - 1):
            tracker._times.append(per_sample_t)

        for pred, (s, ctx) in zip(preds, sub):
            predictions.append(pred)
            references.append(s.answer)
            sample_records.append({
                "image_id":          s.image_id,
                "question":          s.question,
                "reference":         s.answer,
                "prediction":        pred,
                "context_used":      ctx[:300] + "..." if len(ctx) > 300 else ctx,
                "inference_time_ms": per_sample_t,
            })

    tracker.close()
    metrics = compute_all_metrics(predictions, references)

    return {
        "model_key": wrapper.config.model_key,
        "model_id":  wrapper.config.model_id,
        "case":      case_label,
        "top_k":     top_k,
        "selector":  selector.method,
        "metrics":   metrics.to_dict(),
        "timing":    tracker.report,
        "samples":   sample_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_level4(
    model_keys: List[str],
    test_split_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    database_path: Optional[str] = None,
    retrieval_results_path: Optional[str] = None,
    device: str = "cuda:0",
    case: str = "both",            # "A" | "B" | "both"
    top_k_sentences: int = 5,
    selector_method: str = "bm25",
    batch_size: int = 4,
    num_workers: int = 0,
    prefetch_images: bool = True,
    num_batches: int = 1,
    batch_idx: int = 0,
) -> Dict:
    """
    Parameters
    ----------
    num_batches : int
        Split the test records into this many equal shards.
    batch_idx : int
        Zero-based index of the shard to run (0 ≤ batch_idx < num_batches).
        Results are saved under a per-shard sub-directory when num_batches > 1.
    """
    if not (0 <= batch_idx < num_batches):
        raise ValueError(f"batch_idx={batch_idx} out of range for num_batches={num_batches}")

    # ── Load & shard data ─────────────────────────────────────────────────────
    all_records = load_test_split(test_split_path)

    # Split at the record (image) level so all QAs for an image stay together
    chunk = math.ceil(len(all_records) / num_batches)
    records = all_records[batch_idx * chunk : (batch_idx + 1) * chunk]

    samples = flatten_qa_samples(records, images_dir)

    if num_batches > 1:
        print(
            f"[Level4] Shard {batch_idx + 1}/{num_batches} — "
            f"records {batch_idx * chunk}–{batch_idx * chunk + len(records) - 1} "
            f"({len(records)} images, {len(samples)} QA samples)"
        )
    else:
        print(f"[Level4] Loaded {len(samples)} QA samples")

    if not samples:
        raise ValueError(
            f"No valid samples in shard {batch_idx}/{num_batches}. "
            f"Check images_dir: {images_dir}"
        )

    selector = ArticleSelector(method=selector_method)

    # Pre-build contexts
    cases_to_run: List[str] = []
    if case in ("A", "both"):
        cases_to_run.append("A")
    if case in ("B", "both"):
        if not retrieval_results_path:
            print("[Level4] WARNING: Case B requires --retrieval_results. Skipping Case B.")
        elif not database_path:
            print("[Level4] WARNING: Case B requires --database. Skipping Case B.")
        else:
            cases_to_run.append("B")

    contexts: Dict[str, List[str]] = {}
    if "A" in cases_to_run:
        contexts["A"] = [s.article_content for s in samples]
    if "B" in cases_to_run:
        from VQA_baseline.data_utils import load_database
        print("[Level4] Loading Phase 2 retrieval map for Case B …")
        retrieval_map = load_retrieval_map(retrieval_results_path)
        database = load_database(database_path)
        contexts["B"] = [
            database.get(retrieval_map.get(s.image_id, ""), {}).get("content", "")
            for s in samples
        ]

    all_results: Dict = {}

    for model_key in model_keys:
        cfg = MODEL_REGISTRY[model_key]
        if not cfg.is_vl_model:
            print(f"[Level4] Skipping text-only model {model_key}")
            continue

        print(f"\n[Level4] ── {model_key} ────────────────────────────────────")
        print(f"  Model ID : {cfg.model_id}")
        print(f"  Family   : {cfg.family}  |  d_v={cfg.visual_hidden_dim}  d_llm={cfg.llm_hidden_dim}")

        wrapper = build_wrapper(model_key, device=device)
        wrapper.load()

        for case_label in cases_to_run:
            result = evaluate_model_one_case(
                wrapper, samples, contexts[case_label],
                case_label=case_label,
                selector=selector,
                top_k=top_k_sentences,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_images=prefetch_images,
            )
            key = f"{model_key}_case{case_label}"
            all_results[key] = result

            m = MetricsResult(**{k: v for k, v in result["metrics"].items()})
            print(f"\n[Level4-Case{case_label}] {model_key}:\n{m}")
            t = result["timing"]
            print(f"[Level4] Timing: avg={t.get('avg_ms')}ms  p95={t.get('p95_ms')}ms  total={t.get('total_s')}s")

        # Case A−B gap
        if "A" in cases_to_run and "B" in cases_to_run:
            mA = all_results[f"{model_key}_caseA"]["metrics"]
            mB = all_results[f"{model_key}_caseB"]["metrics"]
            print(f"\n[Level4] {model_key} — Case A−B gap (retrieval error propagation):")
            for metric in ("exact_match", "token_f1", "bertscore_f1", "cider"):
                gap = mA.get(metric, 0) - mB.get(metric, 0)
                print(f"  {metric:<20}: {gap:+.4f}")

        wrapper.unload()

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
    parser = argparse.ArgumentParser(description="Level 4: Full RAG-VQA zero-shot baseline")
    parser.add_argument("--models",           nargs="+", default=["all"])
    parser.add_argument("--test_split",       default="data/splits/test1_set.json")
    parser.add_argument("--images_dir",       default="../../../Eventa/webCrawl/src/database_image")
    parser.add_argument("--database",          default=None,
                        help="Path to database.json (required for Case B)")
    parser.add_argument("--retrieval_results", default=None,
                        help="Path to Phase 2 summary.csv (required for Case B)")
    parser.add_argument("--output_dir",       default="outputs/evaluation")
    parser.add_argument("--device",           default="cuda:0")
    parser.add_argument("--case",             default="A", choices=["A", "B", "both"],
                        help="A=ground-truth article, B=retrieved, both=run both")
    parser.add_argument("--top_k_sentences",  type=int, default=5)
    parser.add_argument("--selector",         default="bm25",
                        choices=["bm25", "dense", "first"])
    parser.add_argument("--batch_size",       type=int, default=16)
    parser.add_argument("--num_workers",      type=int, default=0)
    parser.add_argument("--no_prefetch",      action="store_true",
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

    model_keys = LEVEL4_MODELS if args.models == ["all"] else args.models
    unknown = [k for k in model_keys if k not in MODEL_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown model keys: {unknown}")

    run_level4(
        model_keys=model_keys,
        test_split_path=args.test_split,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        database_path=args.database,
        retrieval_results_path=args.retrieval_results,
        device=args.device,
        case=args.case,
        top_k_sentences=args.top_k_sentences,
        selector_method=args.selector,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_images=not args.no_prefetch,
        num_batches=args.num_batches,
        batch_idx=args.batch_idx,
    )


if __name__ == "__main__":
    main()
