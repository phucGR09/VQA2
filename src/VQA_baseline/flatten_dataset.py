import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class PathConfig:
    def __init__(
        self,
        image_qa_train_path: Path = Path("../../splits/image_vqa_with_difficulty_cleaned_test.json"),
        image_qa_test1_path: Path = Path("../../splits/image_vqa_with_difficulty_cleaned_val1.json"),
        image_qa_test2_path: Path = Path("../../splits/image_vqa_with_difficulty_cleaned_val2.json"),
        image_qa_test3_path: Path = Path("../../splits/image_vqa_with_difficulty_cleaned_test.json"),
        database_path: Path = Path("../../../Eventa/webCrawl/src/merged_4_database.json"),
        output_dir: Path = Path("/data/splits"),
    ) -> None:
        self.image_qa_train_path = Path(image_qa_train_path)
        self.image_qa_test1_path = Path(image_qa_test1_path)
        self.image_qa_test2_path = Path(image_qa_test2_path)
        self.image_qa_test3_path = Path(image_qa_test3_path)
        self.database_path = Path(database_path)
        self.output_dir = Path(output_dir)


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_dict(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict, got {type(value).__name__}")
    return value


def _validate_image_qa_split(split_obj: Dict[str, Any], split_name: str) -> None:
    if not isinstance(split_obj, dict):
        raise ValueError(f"image_qa split '{split_name}' must be a dict")
    for image_id, payload in split_obj.items():
        if not isinstance(image_id, str):
            raise ValueError(f"image_id must be str, got {type(image_id).__name__}")
        if not isinstance(payload, dict):
            raise ValueError(f"payload for '{image_id}' must be a dict")
        qa_pairs = payload.get("qa")
        if not isinstance(qa_pairs, list):
            raise ValueError(f"qa for '{image_id}' must be list")
        for pair in qa_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                raise ValueError(
                    f"qa for '{image_id}' must be [question, answer, ...]"
                )


def _build_caption_index(database: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    caption_index: Dict[str, Tuple[str, str]] = {}
    for article_id, article in database.items():
        if not isinstance(article, dict):
            raise ValueError(f"article '{article_id}' must be a dict")
        for img in article.get("images", []):
            if not isinstance(img, dict):
                raise ValueError(f"images in article '{article_id}' must be dicts")
            image_id = img.get("image_id")
            caption = img.get("caption", "")
            if not isinstance(image_id, str):
                raise ValueError(f"image_id in article '{article_id}' must be str")
            if not isinstance(caption, str):
                raise ValueError(f"caption for '{image_id}' must be str")
            if image_id not in caption_index:
                caption_index[image_id] = (article_id, caption)
    return caption_index


def _build_records_for_split(
    split_obj: Dict[str, Dict[str, Any]],
    caption_index: Dict[str, Tuple[str, str]],
    database: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    records: List[Dict[str, Any]] = []
    invalid_article_id = 0
    invalid_qa_text = 0
    mismatched_article_id = 0
    for image_id, payload in split_obj.items():
        article_id = payload.get("article_id")
        if not isinstance(article_id, str):
            invalid_article_id += 1
            continue
        if article_id not in database:
            raise ValueError(f"article_id '{article_id}' not found in database")
        if image_id not in caption_index:
            raise ValueError(f"image_id '{image_id}' not found in database images")
        indexed_article_id, caption = caption_index[image_id]
        if indexed_article_id != article_id:
            mismatched_article_id += 1
            continue
        article = database[article_id]
        article_content = article.get("content")
        if not isinstance(article_content, str):
            raise ValueError(f"article '{article_id}' missing content string")
        questions = []
        for pair in payload["qa"]:
            q, a = pair[0], pair[1]
            if not isinstance(q, str) or not isinstance(a, str):
                invalid_qa_text += 1
                continue
            questions.append({"question": q, "answer": a})
        records.append(
            {
                "image_id": image_id,
                "article_id": article_id,
                "caption": caption,
                "article_content": article_content,
                "questions": questions,
            }
        )
    return records, invalid_article_id, invalid_qa_text, mismatched_article_id


def main() -> None:
    cfg = PathConfig()
    image_qa_train_raw = _load_json(cfg.image_qa_train_path)
    image_qa_test1_raw = _load_json(cfg.image_qa_test1_path)
    image_qa_test2_raw = _load_json(cfg.image_qa_test2_path)
    image_qa_test3_raw = _load_json(cfg.image_qa_test3_path)
    database_raw = _load_json(cfg.database_path)

    image_qa_train = _require_dict(image_qa_train_raw, "image_qa_train.json")
    image_qa_test1 = _require_dict(image_qa_test1_raw, "image_qa_test1.json")
    image_qa_test2 = _require_dict(image_qa_test2_raw, "image_qa_test2.json")
    image_qa_test3 = _require_dict(image_qa_test3_raw, "image_qa_test3.json")
    database = _require_dict(database_raw, "database.json")

    _validate_image_qa_split(image_qa_train, "train")
    _validate_image_qa_split(image_qa_test1, "test1")
    _validate_image_qa_split(image_qa_test2, "test2")
    _validate_image_qa_split(image_qa_test3, "test3")

    caption_index = _build_caption_index(database)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    split_specs = [
        ("train", image_qa_train, "train_set.json"),
        ("test1", image_qa_test1, "test1_set.json"),
        ("test2", image_qa_test2, "test2_set.json"),
        ("test3", image_qa_test3, "test3_set.json"),
    ]

    total_invalid_article_id = 0
    total_invalid_qa_text = 0
    total_mismatched_article_id = 0
    for split_name, split_obj, out_name in split_specs:
        records, invalid_article_id, invalid_qa_text, mismatched_article_id = _build_records_for_split(
            split_obj,
            caption_index,
            database,
        )
        total_invalid_article_id += invalid_article_id
        total_invalid_qa_text += invalid_qa_text
        total_mismatched_article_id += mismatched_article_id
        out_path = cfg.output_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=True)
        print(f"{split_name}: {len(records)} records")

    if total_invalid_article_id:
        print(f"skipped invalid article_id: {total_invalid_article_id}")
    if total_invalid_qa_text:
        print(f"skipped invalid qa text: {total_invalid_qa_text}")
    if total_mismatched_article_id:
        print(f"skipped mismatched article_id: {total_mismatched_article_id}")


if __name__ == "__main__":
    main()
