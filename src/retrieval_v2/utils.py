import json
import os
from pathlib import Path

import torch


def atomic_save(obj, path: Path) -> None:
    """torch.save to a temp file then rename, so a crash never leaves a corrupt file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_jsonl(path, obj: dict) -> None:
    """Append one record to a .jsonl backup file and flush immediately."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl(path) -> list[dict]:
    """Load all records from a .jsonl file, tolerating a truncated last line."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def is_oom(e: Exception) -> bool:
    return "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()
