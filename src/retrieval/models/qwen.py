import logging
import math
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


def _resize_to_max_pixels(image: Image.Image, max_pixels: int, patch_size: int = 28) -> Image.Image:
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    new_w = max(patch_size, round(w * scale / patch_size) * patch_size)
    new_h = max(patch_size, round(h * scale / patch_size) * patch_size)
    return image.resize((new_w, new_h), Image.LANCZOS)


@dataclass
class QwenConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "cuda"
    max_pixels: int = 512 * 28 * 28


class QwenVLModel:
    def __init__(self, config: QwenConfig):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.model_name, use_fast=False)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
        ).eval()

    @torch.inference_mode()
    def generate_caption(self, image_path: str, prompt: str, max_new_tokens: int = 256) -> str:
        return self.generate_caption_batch([image_path], prompt, max_new_tokens)[0]

    @torch.inference_mode()
    def generate_caption_batch(
        self,
        image_paths: list[str],
        prompt: str,
        max_new_tokens: int = 256,
    ) -> list[str]:
        messages_list = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            for _ in image_paths
        ]
        texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]
        images = [
            _resize_to_max_pixels(Image.open(p).convert("RGB"), self.config.max_pixels)
            for p in image_paths
        ]
        inputs = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(self.config.device_map)

        logger.info("input_ids shape: %s", inputs["input_ids"].shape)

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.processor.batch_decode(generated, skip_special_tokens=True)]

    @torch.inference_mode()
    def generate_text(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(self.config.device_map)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
