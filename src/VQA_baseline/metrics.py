"""
metrics.py
==========
All evaluation metrics for VQA baselines.

  compute_all_metrics(predictions, references) → MetricsResult

Metrics
-------
  exact_match   – normalised exact string match
  token_f1      – token overlap F1 (standard SQuAD metric)
  bleu4         – corpus BLEU-4 via nltk
  meteor        – per-sample METEOR via nltk
  bertscore     – BERTScore F1 with xlm-roberta-base (multilingual / Vietnamese)
  cider         – CIDEr corpus score via pycocoevalcap

All functions are safe to call with empty lists (return 0.0).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    exact_match: float       = 0.0
    token_f1: float          = 0.0
    bleu4: float             = 0.0
    meteor: float            = 0.0
    bertscore_p: float       = 0.0
    bertscore_r: float       = 0.0
    bertscore_f1: float      = 0.0
    cider: float             = 0.0
    n_samples: int           = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"  EM          : {self.exact_match:.4f}\n"
            f"  Token F1    : {self.token_f1:.4f}\n"
            f"  BLEU-4      : {self.bleu4:.4f}\n"
            f"  METEOR      : {self.meteor:.4f}\n"
            f"  BERTScore P : {self.bertscore_p:.4f}\n"
            f"  BERTScore R : {self.bertscore_r:.4f}\n"
            f"  BERTScore F1: {self.bertscore_f1:.4f}\n"
            f"  CIDEr       : {self.cider:.4f}\n"
            f"  N samples   : {self.n_samples}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, unicode-normalise, strip punctuation and extra whitespace."""
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenise(text: str) -> List[str]:
    return _normalise(text).split()


# ─────────────────────────────────────────────────────────────────────────────
# Exact Match
# ─────────────────────────────────────────────────────────────────────────────

def exact_match(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    hits = sum(_normalise(p) == _normalise(r) for p, r in zip(predictions, references))
    return hits / len(predictions)


# ─────────────────────────────────────────────────────────────────────────────
# Token F1  (SQuAD-style)
# ─────────────────────────────────────────────────────────────────────────────

def _single_token_f1(pred: str, ref: str) -> float:
    pred_tok = _tokenise(pred)
    ref_tok  = _tokenise(ref)
    common   = Counter(pred_tok) & Counter(ref_tok)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tok)
    recall    = n_common / len(ref_tok)
    return 2 * precision * recall / (precision + recall)


def token_f1(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    scores = [_single_token_f1(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


# ─────────────────────────────────────────────────────────────────────────────
# BLEU-4
# ─────────────────────────────────────────────────────────────────────────────

def bleu4(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        hyps = [_tokenise(p) for p in predictions]
        refs = [[_tokenise(r)] for r in references]
        smoother = SmoothingFunction().method1
        return corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
    except Exception as e:
        print(f"[metrics] BLEU-4 failed: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# METEOR
# ─────────────────────────────────────────────────────────────────────────────

def meteor(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    try:
        from nltk.translate.meteor_score import meteor_score as _meteor
        import nltk
        for resource in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
        scores = [_meteor([_tokenise(r)], _tokenise(p)) for p, r in zip(predictions, references)]
        return sum(scores) / len(scores)
    except Exception as e:
        print(f"[metrics] METEOR failed: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BERTScore
# ─────────────────────────────────────────────────────────────────────────────

def bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "xlm-roberta-base",
    device: str = "cuda:0",
    batch_size: int = 64,
) -> Tuple[float, float, float]:
    """Returns (precision, recall, f1) — all corpus-averaged."""
    if not predictions:
        return 0.0, 0.0, 0.0
    try:
        from bert_score import score as _bs
        P, R, F1 = _bs(
            predictions, references,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=False,
        )
        return float(P.mean()), float(R.mean()), float(F1.mean())
    except Exception as e:
        import traceback
        print(f"[metrics] BERTScore failed: {e}")
        traceback.print_exc()
        return 0.0, 0.0, 0.0


def bertscore_per_sample(
    predictions: List[str],
    references: List[str],
    model_type: str = "xlm-roberta-base",
) -> Tuple[List[float], List[float], List[float]]:
    """Returns per-sample (precision_list, recall_list, f1_list)."""
    if not predictions:
        return [], [], []
    try:
        from bert_score import score as _bs
        P, R, F1 = _bs(predictions, references, model_type=model_type, verbose=False)
        return P.tolist(), R.tolist(), F1.tolist()
    except Exception as e:
        print(f"[metrics] BERTScore failed: {e}")
        n = len(predictions)
        return [0.0] * n, [0.0] * n, [0.0] * n


# ─────────────────────────────────────────────────────────────────────────────
# CIDEr
# ─────────────────────────────────────────────────────────────────────────────

def cider(predictions: List[str], references: List[str]) -> Tuple[float, List[float]]:
    """Returns (corpus_score, per_sample_scores)."""
    if not predictions:
        return 0.0, []
    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        # Use NFC-normalised lowercase to keep Vietnamese diacritics intact
        refs = {i: [unicodedata.normalize("NFC", r).lower()] for i, r in enumerate(references)}
        hyps = {i: [unicodedata.normalize("NFC", p).lower()] for i, p in enumerate(predictions)}
        corpus_score, per_sample = scorer.compute_score(refs, hyps)
        return float(corpus_score), list(per_sample)
    except Exception as e:
        print(f"[metrics] CIDEr failed: {e}")
        import traceback; traceback.print_exc()
        return 0.0, [0.0] * len(predictions)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    bertscore_model: str = "xlm-roberta-base",
    bertscore_device: str = "cuda:0",
) -> MetricsResult:
    """
    Compute all metrics in one call.

    Parameters
    ----------
    predictions   : list of generated answers
    references    : list of ground truth answers (same order)
    bertscore_model : HuggingFace model ID for BERTScore

    Returns
    -------
    MetricsResult dataclass
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    em   = exact_match(predictions, references)
    f1   = token_f1(predictions, references)
    b4   = bleu4(predictions, references)
    met  = meteor(predictions, references)
    bs_p, bs_r, bs_f1 = bertscore(predictions, references, bertscore_model, device=bertscore_device)
    cid, _ = cider(predictions, references)

    return MetricsResult(
        exact_match=round(em, 4),
        token_f1=round(f1, 4),
        bleu4=round(b4, 4),
        meteor=round(met, 4),
        bertscore_p=round(bs_p, 4),
        bertscore_r=round(bs_r, 4),
        bertscore_f1=round(bs_f1, 4),
        cider=round(cid, 4),
        n_samples=len(predictions),
    )
