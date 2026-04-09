"""Inference script for FLASH super-resolution."""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from config.default import Config
from model.unet import FlashUNet
from data.dataset import RangeImageDataset, gather_files
from utils.reprojection import range_image_to_points
from utils.misc import get_device


@torch.no_grad()
def infer_single(model, range_path: str, config: Config, device: torch.device):
    """Run inference on a single frame.
    Returns:
        pred_range: (H, W) predicted range image (log-compressed)
        pred_pts: (N, 3) reprojected 3D points
    """
    dataset = RangeImageDataset([range_path], config)
    sample = dataset[0]
    inp = sample["input"].unsqueeze(0).to(device)
    mask = sample["mask"].numpy()[0]

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config.mixed_precision):
        pred = model(inp)

    pred_range = pred[0, 0].cpu().float().numpy()
    pred_pts = range_image_to_points(pred_range, mask, config)
    return pred_range, pred_pts


@torch.no_grad()
def infer_batch(model, file_list: list, config: Config, device: torch.device,
                output_dir: str = "infer_output"):
    """Run inference on a list of frames and save results."""
    os.makedirs(output_dir, exist_ok=True)
    dataset = RangeImageDataset(file_list, config)

    for i in tqdm(range(len(dataset)), desc="Inference"):
        sample = dataset[i]
        inp = sample["input"].unsqueeze(0).to(device)
        mask = sample["mask"].numpy()[0]

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config.mixed_precision):
            pred = model(inp)

        pred_range = pred[0, 0].cpu().float().numpy()
        pred_pts = range_image_to_points(pred_range, mask, config)

        # Save
        basename = os.path.splitext(os.path.basename(file_list[i]))[0]
        basename = basename.replace("_range", "")
        np.save(os.path.join(output_dir, f"{basename}_pred_range.npy"), pred_range)
        np.save(os.path.join(output_dir, f"{basename}_pred_pts.npy"), pred_pts)

    print(f"Saved {len(dataset)} predictions to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--output_dir", type=str, default="infer_output")
    parser.add_argument("--num_frames", type=int, default=None, help="Limit frames")
    args = parser.parse_args()

    config = Config.dev() if args.dev else Config()
    device = get_device()

    model = FlashUNet(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Get val files
    all_files = gather_files(config.processed_root)
    split = int(len(all_files) * 0.8)
    val_files = all_files[split:]
    if args.num_frames is not None:
        val_files = val_files[:args.num_frames]

    infer_batch(model, val_files, config, device, args.output_dir)


if __name__ == "__main__":
    main()
