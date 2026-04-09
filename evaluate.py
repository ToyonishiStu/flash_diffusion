"""Evaluation pipeline for FLASH / FLASH+ super-resolution."""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from config.default import Config
from model.unet import FlashUNet
from data.dataset import create_dataloaders
from utils.reprojection import range_image_to_points
from utils.metrics import (
    compute_all_metrics, compute_metrics_by_distance, compute_mae_by_distance,
)
from utils.misc import get_device


@torch.no_grad()
def evaluate(model, loader, device, config):
    """Run evaluation on all frames in loader.
    Returns:
        results: dict with mean metrics
        per_frame: list of per-frame metric dicts
    """
    model.eval()
    per_frame = []

    for batch in tqdm(loader, desc="Evaluating"):
        inp = batch["input"].to(device, non_blocking=True)
        target_t = batch["target"]
        mask_t = batch["mask"]

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=config.mixed_precision):
            pred_t = model(inp)

        # Process each frame in batch
        B = inp.size(0)
        for b in range(B):
            pred_np = pred_t[b, 0].cpu().float().numpy()
            gt_np = target_t[b, 0].numpy()
            mask_np = mask_t[b, 0].numpy()

            # Reproject to 3D
            pred_pts = range_image_to_points(pred_np, mask_np, config)
            gt_pts = range_image_to_points(gt_np, mask_np, config)

            # Compute metrics
            metrics = compute_all_metrics(
                pred_np, gt_np, mask_np, pred_pts, gt_pts,
                voxel_size=config.voxel_size, threshold=config.cd_threshold,
            )
            # Distance-based 3D metrics (detailed)
            dist_metrics = compute_metrics_by_distance(
                pred_pts, gt_pts,
                distance_ranges=[(0, 10), (10, 30), (30, 50), (50, 80)],
            )
            metrics["by_distance"] = dist_metrics

            # Distance-based MAE (main reporting ranges)
            mae_by_dist = compute_mae_by_distance(
                pred_np, gt_np, mask_np,
                distance_ranges=[(0, 30), (30, 60)],
            )
            metrics["mae_by_distance"] = mae_by_dist

            per_frame.append(metrics)

    # Aggregate
    keys = ["mae", "chamfer_distance", "iou", "precision", "recall", "f1"]
    agg = {}
    for k in keys:
        vals = [f[k] for f in per_frame if np.isfinite(f[k])]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")

    return agg, per_frame


def print_results(agg: dict, per_frame: list):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k in ["mae", "chamfer_distance", "iou", "precision", "recall", "f1"]:
        mean = agg.get(f"{k}_mean", float("nan"))
        std = agg.get(f"{k}_std", float("nan"))
        print(f"  {k:>20s}: {mean:.6f} +/- {std:.6f}")

    # MAE by distance
    if per_frame and "mae_by_distance" in per_frame[0]:
        print("\n--- MAE by Distance ---")
        ranges = per_frame[0]["mae_by_distance"].keys()
        for rng in ranges:
            vals = [f["mae_by_distance"][rng] for f in per_frame
                    if np.isfinite(f["mae_by_distance"][rng])]
            if vals:
                print(f"  {rng:>10s}: MAE={np.mean(vals):.4f}m +/- {np.std(vals):.4f}")

    # 3D metrics by distance
    if per_frame and "by_distance" in per_frame[0]:
        print("\n--- 3D Metrics by Distance ---")
        ranges = per_frame[0]["by_distance"].keys()
        for rng in ranges:
            cds = [f["by_distance"][rng]["cd"] for f in per_frame
                   if np.isfinite(f["by_distance"][rng]["cd"])]
            ious = [f["by_distance"][rng]["iou"] for f in per_frame]
            f1s = [f["by_distance"][rng]["f1"] for f in per_frame]
            if cds:
                print(f"  {rng:>10s}: CD={np.mean(cds):.6f}, "
                      f"IoU={np.mean(ious):.4f}, F1={np.mean(f1s):.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["baseline", "rafk", "mkdisc", "proposed"])
    parser.add_argument("--output", type=str, default="eval_results.npz")
    args = parser.parse_args()

    # Config
    if args.variant:
        config = Config.ablation(args.variant, dev=args.dev)
    else:
        config = Config.dev() if args.dev else Config()

    device = get_device()

    # Load model (no discriminator at eval time)
    model = FlashUNet(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    variant_str = args.variant or "flash"
    print(f"Loaded {variant_str} from epoch {ckpt.get('epoch', '?')}")

    # Data
    _, val_loader = create_dataloaders(config)

    # Evaluate
    agg, per_frame = evaluate(model, val_loader, device, config)
    print_results(agg, per_frame)

    # Save
    np.savez(args.output, agg=agg, per_frame=per_frame)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
