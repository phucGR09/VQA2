import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def compress_images(
    input_dir: str,
    output_dir: str,
    quality: int = 60,
    num_splits: int = 1,
    split_index: int = 0,
) -> None:
    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in src.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg"))
    files = files[split_index::num_splits]

    success, failed = 0, 0
    for path in tqdm(files, desc=f"Compressing [{split_index}/{num_splits}]", unit="img"):
        out_path = dst / path.relative_to(src)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(path) as img:
                img.convert("RGB").save(out_path, "JPEG", quality=quality, optimize=True)
            success += 1
        except Exception as e:
            print(f"[FAIL] {path}: {e}")
            failed += 1

    print(f"Done. Compressed: {success}  Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-compress JPG images with lower quality")
    parser.add_argument("input_dir", help="Folder containing source JPG images")
    parser.add_argument("output_dir", help="Folder to save compressed images")
    parser.add_argument("--quality", type=int, default=60, help="JPEG quality 1-95 (default: 60)")
    parser.add_argument("--num_splits", type=int, default=1, help="Total number of splits (default: 1)")
    parser.add_argument("--split_index", type=int, default=0, help="Which split to process, 0-based (default: 0)")
    args = parser.parse_args()

    if not (0 <= args.split_index < args.num_splits):
        raise ValueError(f"split_index {args.split_index} out of range for num_splits {args.num_splits}")

    compress_images(args.input_dir, args.output_dir, args.quality, args.num_splits, args.split_index)
