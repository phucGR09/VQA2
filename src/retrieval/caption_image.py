import json
import logging
import math
import re
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.retrieval.config import RetrievalConfig
from src.retrieval.dataloader import ImageDataset
from src.retrieval.model import load_vlm
from src.retrieval.models.qwen import QwenVLModel


def _make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"caption.{log_path.stem}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(ch)
    return logger


def _image_sizes(image_paths: list[str]) -> list[tuple[int, int]]:
    # PIL lazy-open reads only header — no pixel data loaded
    return [Image.open(p).size for p in image_paths]


class ImageCaptioner:
    def __init__(self, config: RetrievalConfig, batch_size: int = 1):
        self.config = config
        self.batch_size = batch_size
        self.model: QwenVLModel = load_vlm(config)
        self._caption_dir = Path(config.caption_dir)
        self._caption_dir.mkdir(parents=True, exist_ok=True)

    def caption_images(
        self,
        dataset: ImageDataset,
        batch_idx: int = 0,
        batch_num: int = 1,
    ) -> dict[str, str]:
        start, end = self._shard_range(len(dataset), batch_idx, batch_num)
        indices = list(range(start, end))

        logger = _make_logger(self._caption_dir / f"captions_{batch_idx}.log")
        captions: dict[str, str] = {}

        for i in tqdm(
            range(0, len(indices), self.batch_size),
            desc=f"Captioning shard {batch_idx}/{batch_num}",
        ):
            chunk = indices[i : i + self.batch_size]
            samples = [dataset[j] for j in chunk]
            image_ids = [s["image_id"] for s in samples]
            image_paths = [s["image_path"] for s in samples]

            sizes = _image_sizes(image_paths)
            size_info = "  ".join(
                f"{iid}={w}x{h}({w*h:,}px)" for iid, (w, h) in zip(image_ids, sizes)
            )
            logger.info("batch %d | %s", i // self.batch_size, size_info)

            try:
                results = self.model.generate_caption_batch(
                    image_paths,
                    self.config.caption_prompt,
                    self.config.caption_max_new_tokens,
                )
                for image_id, caption in zip(image_ids, results):
                    captions[image_id] = self._clean(caption)
            except Exception as batch_err:
                logger.error("Batch failed %s: %s", image_ids, batch_err, exc_info=True)
                for image_id, image_path in zip(image_ids, image_paths):
                    try:
                        caption = self.model.generate_caption(
                            image_path,
                            self.config.caption_prompt,
                            self.config.caption_max_new_tokens,
                        )
                        captions[image_id] = self._clean(caption)
                    except Exception as single_err:
                        logger.error(
                            "Image failed [%s]: %s", image_id, single_err, exc_info=True
                        )

        out_path = self._caption_dir / f"captions_{batch_idx}.json"
        out_path.write_text(json.dumps(captions, ensure_ascii=False, indent=2))
        return captions

    def load(self, batch_idx: int | None = None) -> dict[str, str]:
        if batch_idx is not None:
            return json.loads(
                (self._caption_dir / f"captions_{batch_idx}.json").read_text()
            )
        return json.loads((self._caption_dir / "captions.json").read_text())

    @staticmethod
    def _shard_range(total: int, batch_idx: int, batch_num: int) -> tuple[int, int]:
        shard_size = math.ceil(total / batch_num)
        start = batch_idx * shard_size
        return start, min(start + shard_size, total)

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
