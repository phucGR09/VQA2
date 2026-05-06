"""
data_utils.py
=============
Minimal data loading utilities needed by the baseline levels.

Extracted from /copy/dataloader.py — only the two functions used by
level4 (Case B) and level5.  Pipeline-only functions (load_image_qa,
build_master_records) are intentionally excluded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"]


def find_image_path(images_dir: Path, image_id: str) -> Path:
    """Locate an image file by image_id, trying common extensions."""
    for ext in IMAGE_EXTENSIONS:
        p = Path(images_dir) / f"{image_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image found for id '{image_id}' in {images_dir}")


def load_database(path: str | Path) -> Dict[str, Any]:
    """Load the article database JSON (article_id → article dict)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
