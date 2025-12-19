"""
Sample a random subset of images from a dataset folder into a new folder.
Default: pick 2000 images from data/food-101/images into data/subset_2000.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List


def gather_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def copy_subset(images: List[Path], out_dir: Path, limit: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    take = min(limit, len(images))
    chosen = random.sample(images, take)

    for idx, src in enumerate(chosen, start=1):
        dest = out_dir / src.name
        shutil.copy2(src, dest)
        if idx % 200 == 0 or idx == take:
            print(f"Copied {idx}/{take} images")


def main():
    parser = argparse.ArgumentParser(description="Sample a random subset of images")
    parser.add_argument("--source", default="data/food-101/images", help="Root folder containing images")
    parser.add_argument("--output", default="data/subset_2000", help="Destination folder for sampled images")
    parser.add_argument("--num", type=int, default=2000, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    source_root = Path(args.source)
    out_dir = Path(args.output)

    if not source_root.exists():
        raise SystemExit(f"Source folder not found: {source_root}")

    images = gather_images(source_root)
    if not images:
        raise SystemExit(f"No images found under: {source_root}")

    print(f"Found {len(images)} images under {source_root}")
    copy_subset(images, out_dir, args.num)
    print(f"Done. Sampled images are in {out_dir}")


if __name__ == "__main__":
    main()
