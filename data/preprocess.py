"""Batch convert KITTI .bin files to preprocessed .npy range images."""

import os
import sys
import glob
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.projection import bin_to_range_image
from config.default import Config


def preprocess_all(config: Config, skip_existing: bool = True):
    """Walk kitti_raw/, convert all .bin to .npy in kitti_processed/."""
    raw_root = config.data_root
    out_root = config.processed_root
    os.makedirs(out_root, exist_ok=True)

    bin_files = sorted(glob.glob(os.path.join(raw_root, "**", "*.bin"), recursive=True))
    print(f"Found {len(bin_files)} .bin files in {raw_root}")

    converted, skipped = 0, 0
    for bin_path in tqdm(bin_files, desc="Converting"):
        # Build output path: kitti_raw/.../data/0000.bin -> kitti_processed/.../0000_range.npy
        rel = os.path.relpath(bin_path, raw_root)
        base = os.path.splitext(rel)[0]  # e.g. 2011_09_26/.../data/0000000000
        range_path = os.path.join(out_root, base + "_range.npy")
        mask_path = os.path.join(out_root, base + "_mask.npy")

        if skip_existing and os.path.exists(range_path) and os.path.exists(mask_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(range_path), exist_ok=True)
        range_img, mask = bin_to_range_image(bin_path, config=config)
        np.save(range_path, range_img)
        np.save(mask_path, mask)
        converted += 1

    print(f"Done: {converted} converted, {skipped} skipped (already exist)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="kitti_raw")
    parser.add_argument("--processed_root", default="kitti_processed")
    parser.add_argument("--force", action="store_true", help="Re-convert even if exists")
    args = parser.parse_args()

    cfg = Config(data_root=args.data_root, processed_root=args.processed_root)
    preprocess_all(cfg, skip_existing=not args.force)
