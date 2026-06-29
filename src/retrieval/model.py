from src.retrieval.config import RetrievalConfig
from src.retrieval.models.qwen import QwenConfig, QwenVLModel


def load_vlm(config: RetrievalConfig) -> QwenVLModel:
    qwen_cfg = QwenConfig(
        model_name=config.caption_model_name,
        device_map=config.device,
        max_pixels=config.caption_max_pixels,
    )
    return QwenVLModel(qwen_cfg)
