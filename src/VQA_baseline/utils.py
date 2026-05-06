"""
utils.py
========
Shared utilities for all baseline levels.

  load_test_split(path)          → List[Dict]  (records from test_split.json)
  flatten_qa_samples(records)    → List[QASample]
  ArticleSelector                → BM25 / dense sentence selection
  TimeTracker                    → tqdm-based per-sample timing
  ResultWriter                   → save JSON + human-readable text report
  print_results_table(results)   → console summary across models
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QASample:
    image_id: str
    article_id: str
    article_content: str   # full ground-truth article text
    question: str
    answer: str            # ground-truth answer
    image_path: str = ""   # resolved at load time


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_test_split(split_path: str | Path) -> List[Dict]:
    """Load the test_split.json (or any split) as a list of master-record dicts."""
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_qa_samples(
    records: List[Dict],
    images_dir: str | Path,
) -> List[QASample]:
    """
    Flatten master-record list to one QASample per QA pair.
    Skips records that have no questions or whose image file cannot be found.
    """
    from pathlib import Path as _P
    IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]
    images_dir = _P(images_dir)

    samples: List[QASample] = []
    for rec in records:
        image_id = rec["image_id"]

        # Resolve image path (try multiple extensions)
        img_path = ""
        for ext in IMAGE_EXTS:
            p = images_dir / f"{image_id}{ext}"
            if p.exists():
                img_path = str(p)
                break
        if not img_path:
            continue  # skip if image missing

        for qa in rec.get("questions", []):
            samples.append(QASample(
                image_id=image_id,
                article_id=rec.get("article_id", ""),
                article_content=rec.get("article_content", ""),
                question=qa["question"],
                answer=qa["answer"],
                image_path=img_path,
            ))
    return samples


def load_retrieval_map(csv_path: str | Path, rank: int = 1) -> Dict[str, str]:
    """
    Load Phase 2 summary.csv into {image_id: article_id}.
    rank=1 picks the top-1 retrieved article (article_id_1 column).
    """
    col = f"article_id_{rank}"
    mapping: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            article_id = row.get(col, "")
            if article_id:
                mapping[row["query_id"]] = article_id
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Article sentence selection
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences_vi(text: str) -> List[str]:
    """
    Simple sentence splitter for Vietnamese text.
    Splits on '. ', '.\n', '! ', '? ' and then strips empty entries.
    """
    # Split on sentence-ending punctuation followed by space or newline
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


class ArticleSelector:
    """
    Select the top-k most relevant sentences from an article given a question.

    method:
      "bm25"   – BM25Okapi (rank_bm25 library); lightweight, no GPU
      "dense"  – sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
      "first"  – naive: just take the first top_k sentences
    """

    def __init__(self, method: str = "bm25"):
        self.method = method
        self._encoder = None  # lazy-loaded for dense mode

    def select(
        self,
        article: str,
        question: str,
        top_k: int = 5,
    ) -> str:
        """Return selected sentences joined into a single string (original order)."""
        if not article:
            return ""
        sentences = _split_sentences_vi(article)
        if not sentences:
            return article[:2000]  # fallback: first 2000 chars
        if len(sentences) <= top_k:
            return " ".join(sentences)

        if self.method == "bm25":
            indices = self._bm25_rank(sentences, question, top_k)
        elif self.method == "dense":
            indices = self._dense_rank(sentences, question, top_k)
        else:
            indices = list(range(min(top_k, len(sentences))))

        indices_sorted = sorted(indices)
        return " ".join(sentences[i] for i in indices_sorted)

    # ------------------------------------------------------------------
    def _bm25_rank(self, sentences: List[str], question: str, top_k: int) -> List[int]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return list(range(min(top_k, len(sentences))))

        tokenized_corpus = [s.lower().split() for s in sentences]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(question.lower().split())
        top_k = min(top_k, len(sentences))
        return sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:top_k]

    def _dense_rank(self, sentences: List[str], question: str, top_k: int) -> List[int]:
        try:
            from sentence_transformers import SentenceTransformer
            import torch as _torch
        except ImportError:
            return self._bm25_rank(sentences, question, top_k)

        if self._encoder is None:
            self._encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        q_emb = self._encoder.encode(question, convert_to_tensor=True)
        s_emb = self._encoder.encode(sentences, convert_to_tensor=True)
        scores = _torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), s_emb).cpu().numpy()
        top_k = min(top_k, len(sentences))
        return sorted(
            sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:top_k]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Time tracker
# ─────────────────────────────────────────────────────────────────────────────

class TimeTracker:
    """
    Context-manager style per-sample timer with live tqdm progress bar.

    Usage
    -----
        tracker = TimeTracker("Level2 | vintern", total=len(samples))
        for sample in samples:
            with tracker:
                answer = model.generate(...)
        tracker.close()
        report = tracker.report  # dict with avg_ms, p95_ms, total_s, ...
    """

    def __init__(self, desc: str, total: int):
        self._pbar = tqdm(total=total, desc=desc, unit="sample", dynamic_ncols=True)
        self._times: List[float] = []
        self._t0: float = 0.0

    # Context-manager protocol
    def __enter__(self) -> "TimeTracker":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self._times.append(elapsed_ms)
        n = len(self._times)
        avg = np.mean(self._times)
        self._pbar.update(1)
        self._pbar.set_postfix({
            "avg": f"{avg:.0f}ms",
            "last": f"{elapsed_ms:.0f}ms",
            "n": n,
        })

    def close(self) -> None:
        self._pbar.close()

    @property
    def times(self) -> List[float]:
        """Per-sample inference times in milliseconds."""
        return self._times.copy()

    @property
    def report(self) -> Dict:
        if not self._times:
            return {}
        t = np.array(self._times)
        return {
            "n_samples":  int(len(t)),
            "total_s":    round(float(t.sum()) / 1000.0, 2),
            "avg_ms":     round(float(t.mean()), 1),
            "min_ms":     round(float(t.min()), 1),
            "max_ms":     round(float(t.max()), 1),
            "std_ms":     round(float(t.std()), 1),
            "p50_ms":     round(float(np.percentile(t, 50)), 1),
            "p95_ms":     round(float(np.percentile(t, 95)), 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Result writer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    image_id: str
    question: str
    reference: str
    prediction: str
    inference_time_ms: float


def save_level_results(
    level_name: str,
    all_model_results: Dict,
    output_dir: str | Path,
) -> None:
    """
    Save results for one level.

    Writes:
      outputs/evaluation/{level_name}/results.json   – full machine-readable dump
      outputs/evaluation/{level_name}/report.txt     – human-readable summary
    """
    out = Path(output_dir) / level_name
    out.mkdir(parents=True, exist_ok=True)

    # ── JSON dump ─────────────────────────────────────────────────────────────
    json_path = out / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, ensure_ascii=False, indent=2)

    # ── Text report ───────────────────────────────────────────────────────────
    txt_path = out / "report.txt"
    _write_text_report(txt_path, level_name, all_model_results)

    print(f"[{level_name}] Results → {out}")


def _write_text_report(path: Path, level_name: str, results: Dict) -> None:
    width = 60
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * width + "\n")
        f.write(f"VQA Baseline — {level_name}\n")
        f.write(f"Date   : {now}\n")
        f.write(f"Models : {len(results)}\n")
        f.write("=" * width + "\n\n")

        for model_key, data in results.items():
            f.write(f"── {model_key} " + "─" * max(0, width - len(model_key) - 4) + "\n")

            # Metrics
            m = data.get("metrics", {})
            f.write("  METRICS\n")
            for metric_name, val in m.items():
                if metric_name == "n_samples":
                    continue
                f.write(f"    {metric_name:<20}: {val:.4f}\n")
            f.write(f"    {'n_samples':<20}: {m.get('n_samples', '?')}\n")
            f.write("\n")

            # Timing
            t = data.get("timing", {})
            if t:
                f.write("  TIMING\n")
                f.write(f"    total_s   : {t.get('total_s', '?')} s\n")
                f.write(f"    avg_ms    : {t.get('avg_ms', '?')} ms\n")
                f.write(f"    p50_ms    : {t.get('p50_ms', '?')} ms\n")
                f.write(f"    p95_ms    : {t.get('p95_ms', '?')} ms\n")
                f.write(f"    min_ms    : {t.get('min_ms', '?')} ms\n")
                f.write(f"    max_ms    : {t.get('max_ms', '?')} ms\n")
            f.write("\n")

        # Comparison table
        f.write("=" * width + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("=" * width + "\n")
        header = f"{'Model':<22} {'EM':>6} {'F1':>6} {'BLEU4':>6} {'METEOR':>7} {'BS-F1':>7} {'CIDEr':>7}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for model_key, data in results.items():
            m = data.get("metrics", {})
            f.write(
                f"{model_key:<22} "
                f"{m.get('exact_match', 0):.4f} "
                f"{m.get('token_f1', 0):.4f} "
                f"{m.get('bleu4', 0):.4f} "
                f"{m.get('meteor', 0):>7.4f} "
                f"{m.get('bertscore_f1', 0):>7.4f} "
                f"{m.get('cider', 0):>7.4f}\n"
            )
        f.write("\n")


def print_results_table(level_name: str, all_model_results: Dict) -> None:
    """Print a compact comparison table to stdout."""
    width = 76
    print("\n" + "=" * width)
    print(f"  {level_name} — Results")
    print("=" * width)
    header = f"{'Model':<22} {'EM':>6} {'F1':>6} {'BLEU4':>6} {'METEOR':>7} {'BS-F1':>7} {'CIDEr':>7} {'avg_ms':>8}"
    print(header)
    print("-" * width)
    for model_key, data in all_model_results.items():
        m = data.get("metrics", {})
        t = data.get("timing", {})
        print(
            f"{model_key:<22} "
            f"{m.get('exact_match', 0):.4f} "
            f"{m.get('token_f1', 0):.4f} "
            f"{m.get('bleu4', 0):.4f} "
            f"{m.get('meteor', 0):>7.4f} "
            f"{m.get('bertscore_f1', 0):>7.4f} "
            f"{m.get('cider', 0):>7.4f} "
            f"{t.get('avg_ms', 0):>7.1f}ms"
        )
    print("=" * width + "\n")
