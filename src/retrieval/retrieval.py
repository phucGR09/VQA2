import argparse
import json
from pathlib import Path

from src.retrieval.caption_image import ImageCaptioner
from src.retrieval.config import RetrievalConfig
from src.retrieval.dataloader import ArticleDataset, ImageDataset
from src.retrieval.embedding import TextEmbedder
from src.retrieval.evaluation import evaluate
from src.retrieval.group_evaluation import evaluate_groups
from src.retrieval.group_sync_retrieval import GroupSyncRetrieval
from src.retrieval.group_voting_retrieval import GroupVotingRetrieval
from src.retrieval.single_retrieval import SingleRetrieval
from src.retrieval.utils import load_json

PHASES = [
    "caption",
    "merge_captions",
    "embed_articles",
    "embed_captions",
    "single",
    "group_vote",
    "group_sync",
    "eval_single",
    "eval_group",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-to-article retrieval pipeline")
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch_size", type=int, default=8, help="Images per VLM forward pass")
    parser.add_argument("--batch_num", type=int, default=8, help="Number of dataset shards")
    parser.add_argument("--batch_idx", type=int, default=0, help="Shard index this process handles (0 … batch_num-1)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RetrievalConfig(device=args.device)

    if args.phase == "caption":
        dataset = ImageDataset(config.image_dir)
        ImageCaptioner(config, batch_size=args.batch_size).caption_images(
            dataset, batch_idx=args.batch_idx, batch_num=args.batch_num
        )

    elif args.phase == "merge_captions":
        caption_dir = Path(config.caption_dir)
        merged: dict = {}
        for f in sorted(caption_dir.glob("captions_*.json")):
            if "_errors" not in f.name:
                merged.update(json.loads(f.read_text()))
        out = caption_dir / "captions.json"
        out.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        print(f"Merged {len(merged)} captions → {out}")

    elif args.phase == "embed_articles":
        dataset = ArticleDataset(config.database_path)
        TextEmbedder(
            config.embed_model_name, config.device, config.embed_batch_size,
            config.embed_max_tokens, config.embed_chunk_stride,
            article_batch_size=config.embed_article_batch_size,
        ).embed_articles(dataset, config.article_feature_dir)

    elif args.phase == "embed_captions":
        captions = load_json(f"{config.caption_dir}/captions_flat.json")
        TextEmbedder(
            config.embed_model_name, config.device, config.embed_batch_size,
            config.embed_max_tokens, config.embed_chunk_stride,
        ).embed_captions(captions, config.caption_feature_dir)
    elif args.phase == "single":
        SingleRetrieval(config).retrieve(
            f"{config.caption_feature_dir}/caption_embeddings.pt",
            f"{config.article_feature_dir}/article_embeddings.pt",
        )
        print(f"Completed single retrieval → {config.result_single_dir}/single_retrieval.csv")

    elif args.phase == "group_vote":
        groups = load_json(config.groups_path)
        GroupVotingRetrieval(config).retrieve(
            f"{config.result_single_dir}/single_retrieval.csv", groups
        )

    elif args.phase == "group_sync":
        captions = load_json(f"{config.caption_dir}/captions_flat.json")
        groups = load_json(config.groups_path)
        GroupSyncRetrieval(config).retrieve(
            captions, groups, f"{config.article_feature_dir}/article_embeddings.pt"
        )

    elif args.phase == "eval_single":
        for result_file in sorted(Path(config.result_single_dir).glob("*.csv")):
            metrics = evaluate(str(result_file), config.database_path, config.top_k)
            print(f"{result_file.name}: {metrics}")
    elif args.phase == "eval_group":
        for result_file in sorted(Path(config.result_group_dir).glob("*.csv")):
            metrics = evaluate_groups(
                str(result_file), config.groups_path, config.database_path, config.top_k
            )
            print(f"{result_file.name}: {metrics}")

if __name__ == "__main__":
    main()
