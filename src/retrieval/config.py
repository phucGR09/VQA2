from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    # Paths
    image_dir: str = "./data/database_image_compress"
    database_path: str = "./data/merged_7_database.json"
    groups_path: str = "./data/groups.json"
    caption_dir: str = "outputs/retrieval/captions"
    caption_feature_dir: str = "outputs/retrieval/caption_features"
    article_feature_dir: str = "outputs/retrieval/article_features"
    result_single_dir: str = "outputs/retrieval/results/single_image"
    result_group_dir: str = "outputs/retrieval/results/group_image"

    # Image
    image_size: int = 448

    # Caption model (VLM)
    caption_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    caption_max_new_tokens: int = 256
    caption_prompt: str = (
        "Describe the events, objects, people, and scene in this image in detail."
    )

    # Embedding model
    embed_model_name: str = "BAAI/bge-m3"
    embed_batch_size: int = 64      # for short texts (captions)
    embed_article_batch_size: int = 8  # for long texts (articles)
    embed_max_tokens: int = 4096
    embed_chunk_stride: int = 512


    # Retrieval
    top_k: int = 10
    device: str = "cuda"
    caption_max_pixels: int = 512 * 28 * 28  # ~401K px, limits visual tokens per image

    # Group retrieval
    group_summary_max_tokens: int = 512
