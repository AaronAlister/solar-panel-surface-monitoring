import argparse
import os
import shutil
import subprocess
import sys

import kagglehub


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
DATASET_REF = "pythonafroz/solar-panel-images"


def import_from_local_folder(source_dir: str, out_dir: str, limit: int | None = None) -> int:
    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    for root, _, files in os.walk(source_dir):
        for name in sorted(files):
            if limit is not None and copied >= limit:
                return copied
            if not name.lower().endswith(SUPPORTED_EXTS):
                continue
            src = os.path.join(root, name)
            dst = os.path.join(out_dir, f"kaggle_{copied + 1:05d}{os.path.splitext(name)[1].lower()}")
            shutil.copy2(src, dst)
            copied += 1
    return copied


def clear_existing_images(out_dir: str) -> int:
    if not os.path.isdir(out_dir):
        return 0
    removed = 0
    for name in os.listdir(out_dir):
        if not name.lower().endswith(SUPPORTED_EXTS):
            continue
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            os.remove(p)
            removed += 1
    return removed


def run_detector(input_dir: str) -> int:
    cmd = [sys.executable, "batch_solar_defect.py", "--input", input_dir, "--output", "output_results", "--viz"]
    result = subprocess.run(cmd, check=False)
    return result.returncode


def build_parser():
    p = argparse.ArgumentParser(description="Import Kaggle solar panel images into input_images for batch_solar_defect.py")
    p.add_argument("--input-dir", default="input_images", help="Destination image folder")
    p.add_argument("--limit", type=int, default=200, help="Maximum number of images to import")
    p.add_argument("--run", action="store_true", help="Run batch_solar_defect.py after importing")
    return p


def main():
    args = build_parser().parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None

    removed = clear_existing_images(args.input_dir)
    if removed > 0:
        print(f"Removed {removed} existing images from: {args.input_dir}")

    try:
        dataset_root = kagglehub.dataset_download(DATASET_REF)
    except Exception as e:
        print(f"Failed to download dataset '{DATASET_REF}': {e}")
        print("Install dependencies and auth first:")
        print("1) pip install kagglehub")
        print("2) Configure Kaggle credentials")
        return

    print(f"Downloaded dataset to: {dataset_root}")
    imported = import_from_local_folder(dataset_root, args.input_dir, limit=limit)
    print(f"Imported {imported} images from Kaggle dataset into: {args.input_dir}")

    if imported == 0:
        print("No images imported. Check dataset availability/path.")
        return

    if args.run:
        rc = run_detector(args.input_dir)
        if rc != 0:
            print(f"Detector exited with code: {rc}")


if __name__ == "__main__":
    main()
