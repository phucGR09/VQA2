"""
Transform caption output format to simple {image_id: caption} mapping.

Input:  {image_id: {article_id, title, category, original_caption, generated_caption, ...}}
Output: {image_id: generated_caption}
"""

import json
from pathlib import Path


def transform(input_path: str, output_path: str) -> None:
    data = json.loads(Path(input_path).read_text())
    result = {
        image_id: entry["generated_caption"]
        for image_id, entry in data.items()
    }
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Transformed {len(result)} captions → {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to source caption JSON")
    parser.add_argument("--output", required=True, help="Path to save transformed JSON")
    args = parser.parse_args()

    transform(args.input, args.output)
