"""
level5_dataset.py
=================
Dataset classes for Level 5 fine-tuning.

  CaptionDataset  — Task 1 MLP pre-training (image → caption prediction)
  VQADataset      — Task 2 VQA fine-tuning  (image + article + question → answer)
  collate_fn      — shared right-padding collator for both datasets

VQADataset uses ArticleSelector on the ground-truth article — the same RAG
mechanism as Level 4 / Level 5 eval Case A.  No Contriever needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from VQA_baseline.data_utils import find_image_path
from VQA_baseline.prompt_utils import build_prompt
from VQA_baseline.utils import ArticleSelector


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 dataset
# ─────────────────────────────────────────────────────────────────────────────

class CaptionDataset(Dataset):
    """
    Reads image captions directly from train_split.json (flattened format).
    Returns (pixel_values, input_ids, attention_mask, labels) for caption prediction.
    Loss is computed on every caption token.
    """

    def __init__(
        self,
        train_split_path: Path,
        images_dir: Path,
        processor,
        tokenizer,
        max_caption_length: int = 128,
        max_samples: Optional[int] = None,
    ):
        with open(train_split_path, "r", encoding="utf-8") as f:
            train_records = json.load(f)

        self.records = [
            r for r in train_records
            if r.get("caption", "").strip()
        ]
        if max_samples is not None:
            self.records = self.records[:max_samples]

        self.images_dir = Path(images_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_caption_length = max_caption_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        img = Image.open(find_image_path(self.images_dir, record["image_id"])).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        enc = self.tokenizer(
            record["caption"],
            max_length=self.max_caption_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()  # predict every caption token

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 dataset
# ─────────────────────────────────────────────────────────────────────────────

class VQADataset(Dataset):
    """
    Flattens train_split.json to one item per QA pair.
    Context = ArticleSelector on the ground-truth article — same RAG as Level 4 Case A.
    Loss is computed on answer tokens only (labels=-100 for the prompt prefix).
    """

    def __init__(
        self,
        train_split_path: Path,
        images_dir: Path,
        processor,
        tokenizer,
        selector: ArticleSelector,
        cfg,  # Level5VqaTrainConfig
        max_samples: Optional[int] = None,
    ):
        with open(train_split_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.items: List[Dict] = []
        for rec in records:
            for qa in rec.get("questions", []):
                self.items.append({
                    "image_id":       rec["image_id"],
                    "article_id":     rec.get("article_id", ""),
                    "article_content": rec.get("article_content", ""),
                    "question":       qa["question"],
                    "answer":         qa["answer"],
                })
        if max_samples is not None:
            self.items = self.items[:max_samples]

        # Pre-compute selected context once at init — no GPU, no Contriever
        self.passages: List[List[Tuple[str, str]]] = [
            [(item["article_id"],
              selector.select(item["article_content"], item["question"],
                              top_k=cfg.top_k_sentences))]
            for item in self.items
        ]

        self.images_dir = Path(images_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]

        img = Image.open(find_image_path(self.images_dir, item["image_id"])).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        full_text, prefix_text = build_prompt(
            self.passages[idx],
            item["question"],
            item["answer"],
            self.cfg.max_context_tokens,
            self.tokenizer,
        )

        full_enc = self.tokenizer(
            full_text,
            max_length=self.cfg.max_total_length,
            truncation=True,
            return_tensors="pt",
        )
        prefix_enc = self.tokenizer(
            prefix_text,
            max_length=self.cfg.max_total_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        prefix_len = prefix_enc["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prefix_len] = -100  # only compute loss on answer tokens

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shared collator
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict], pad_token_id: int) -> Dict:
    """Right-pad text sequences to the longest in the batch."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    max_len = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)

    input_ids    = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    labels       = torch.full((B, max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L]     = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]
        labels[i, :L]        = b["labels"]

    return {
        "pixel_values":  pixel_values,
        "input_ids":     input_ids,
        "attention_mask": attention_mask,
        "labels":        labels,
    }
