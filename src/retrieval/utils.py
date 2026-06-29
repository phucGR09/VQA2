import json
from pathlib import Path

import torch


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def save_json(data: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_embeddings(path: str) -> tuple[list[str], torch.Tensor]:
    data = torch.load(path, weights_only=True)
    ids_key = next(k for k in data if k.endswith("_ids"))
    return data[ids_key], data["embeddings"]
