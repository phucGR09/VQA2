import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def merge_folders(input_dirs: list[str], output_dir: str, on_conflict: str = "skip") -> None:
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    files: list[tuple[Path, str]] = []
    for input_dir in input_dirs:
        src = Path(input_dir)
        for f in src.rglob("*"):
            if f.suffix.lower() in (".jpg", ".jpeg"):
                files.append((f, input_dir))

    success, skipped, failed = 0, 0, 0
    for path, src_root in tqdm(files, desc="Merging", unit="img"):
        out_path = dst / path.name

        if out_path.exists():
            if on_conflict == "skip":
                skipped += 1
                continue
            elif on_conflict == "rename":
                stem, suffix = path.stem, path.suffix
                counter = 1
                while out_path.exists():
                    out_path = dst / f"{stem}_{counter}{suffix}"
                    counter += 1

        try:
            shutil.copy2(path, out_path)
            success += 1
        except Exception as e:
            print(f"[FAIL] {path}: {e}")
            failed += 1

    print(f"Done. Copied: {success}  Skipped: {skipped}  Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JPG folders into one output folder")
    parser.add_argument("output_dir", help="Destination folder")
    parser.add_argument("input_dirs", nargs="+", help="Source folders to merge")
    parser.add_argument(
        "--on_conflict",
        choices=["skip", "rename", "overwrite"],
        default="skip",
        help="What to do when a filename already exists in output (default: skip)",
    )
    args = parser.parse_args()
    merge_folders(args.input_dirs, args.output_dir, args.on_conflict)
