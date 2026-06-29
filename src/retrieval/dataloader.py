import json
from pathlib import Path

from torch.utils.data import Dataset


class ArticleDataset(Dataset):
    def __init__(self, database_path: str):
        with open(database_path) as f:
            db = json.load(f)
        self.article_ids = list(db.keys())
        self.articles = db

    def __len__(self) -> int:
        return len(self.article_ids)

    def __getitem__(self, idx: int) -> dict:
        article_id = self.article_ids[idx]
        article = self.articles[article_id]
        return {
            "article_id": article_id,
            "text": f"{article['title']} {article['content']}",
        }

    def get_image_to_article_map(self) -> dict[str, str]:
        return {
            img["image_id"]: article_id
            for article_id, article in self.articles.items()
            for img in article.get("images", [])
        }


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, image_ids: list[str] | None = None):
        image_dir = Path(image_dir)
        if image_ids is not None:
            self.image_paths = [image_dir / f"{iid}.jpg" for iid in image_ids]
        else:
            self.image_paths = sorted(image_dir.glob("*.jpg"))
        self.image_ids = [p.stem for p in self.image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image_id": self.image_ids[idx],
            "image_path": str(self.image_paths[idx]),
        }
