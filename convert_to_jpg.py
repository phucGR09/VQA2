import argparse
from pathlib import Path
from PIL import Image


QUALITY = 95


def convert_png(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background
    return img.convert("RGB")


def convert_jpeg(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def convert_webp(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background
    return img.convert("RGB")


STRATEGIES = {
    ".png": convert_png,
    ".jpeg": convert_jpeg,
    ".webp": convert_webp,
}


def convert_images(input_dir: str, output_dir: str) -> None:
    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files = [f for f in src.rglob("*") if f.suffix.lower() in STRATEGIES]

    success, failed = 0, 0
    for path in files:
        out_path = dst / path.relative_to(src).with_suffix(".jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(path) as img:
                rgb = STRATEGIES[path.suffix.lower()](img)
                rgb.save(out_path, "JPEG", quality=QUALITY)
            success += 1
        except Exception as e:
            print(f"[FAIL] {path}: {e}")
            failed += 1

    print(f"Done. Converted: {success}  Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Folder containing source images")
    parser.add_argument("output_dir", help="Folder to save converted JPG images")
    args = parser.parse_args()
    convert_images(args.input_dir, args.output_dir)
