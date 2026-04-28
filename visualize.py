"""Visualization utilities for FLASH super-resolution."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from config.default import Config
from model.unet import FlashUNet
from data.dataset import RangeImageDataset, gather_files
from utils.reprojection import range_image_to_points
from utils.misc import get_device


def plot_range_image_comparison(input_img, pred_img, target_img, mask,
                                save_path: str = "vis_range_comparison.png"):
    """Plot input / prediction / ground truth / error map side by side."""
    fig, axes = plt.subplots(4, 1, figsize=(20, 10))

    titles = ["Input (bilinear 16->64)", "Prediction", "Ground Truth", "Absolute Error"]
    imgs = [input_img, pred_img, target_img, np.abs(pred_img - target_img) * mask]
    cmaps = ["viridis", "viridis", "viridis", "hot"]

    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        im = ax.imshow(img, aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, fraction=0.02)

    axes[-1].set_xlabel("Column")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_bev(pred_pts, gt_pts, save_path: str = "vis_bev.png", input_pts=None):
    """Bird's eye view: XY scatter colored by Z.

    If input_pts is provided, shows 3 panels: Input (16-beam bilinear) | GT | Prediction.
    Otherwise shows 2 panels: GT | Prediction (backward compatible).
    """
    if input_pts is not None:
        panels = [
            (input_pts, "Input (16-beam bilinear)"),
            (gt_pts,    "Ground Truth"),
            (pred_pts,  "Prediction"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(27, 8))
    else:
        panels = [
            (gt_pts,   "Ground Truth"),
            (pred_pts, "Prediction"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, (pts, title) in zip(axes, panels):
        if len(pts) > 50000:
            idx = np.random.choice(len(pts), 50000, replace=False)
            pts = pts[idx]
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3,
                        cmap="viridis", vmin=-2, vmax=2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        plt.colorbar(sc, ax=ax, label="Z (m)", fraction=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_histogram(pred_pts, gt_pts, save_path: str = "vis_error_hist.png"):
    """Error distribution histogram split by distance range."""
    from scipy.spatial import KDTree

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return

    tree_gt = KDTree(gt_pts)
    dists, _ = tree_gt.query(pred_pts)
    pred_r = np.linalg.norm(pred_pts, axis=1)

    ranges = [(0, 10), (10, 30), (30, 50), (50, 80)]
    fig, axes = plt.subplots(1, len(ranges), figsize=(20, 4))

    for ax, (rmin, rmax) in zip(axes, ranges):
        sel = (pred_r >= rmin) & (pred_r < rmax)
        if sel.sum() > 0:
            ax.hist(dists[sel], bins=50, range=(0, 1.0), density=True, alpha=0.7)
        ax.set_title(f"{rmin}-{rmax}m (n={sel.sum()})")
        ax.set_xlabel("NN Distance (m)")
        ax.set_ylabel("Density")
        ax.axvline(0.1, color="r", linestyle="--", alpha=0.5, label="threshold")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


@torch.no_grad()
def visualize_frames(checkpoint_path: str, config: Config = None,
                     num_frames: int = 3, output_dir: str = "vis_output"):
    """Generate visualizations for a few frames."""
    if config is None:
        config = Config.dev()

    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = FlashUNet(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Get val files
    all_files = gather_files(config.processed_root)
    split = int(len(all_files) * 0.8)
    val_files = all_files[split:]

    dataset = RangeImageDataset(val_files[:num_frames], config)

    for i in range(min(num_frames, len(dataset))):
        sample = dataset[i]
        inp = sample["input"].unsqueeze(0).to(device)
        target = sample["target"].numpy()[0]  # (64, 1024)
        mask = sample["mask"].numpy()[0]
        input_np = sample["input"].numpy()[0]  # (64, 1024)

        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=config.mixed_precision):
            pred = model(inp)
        pred_np = pred[0, 0].cpu().float().numpy()

        # Range image comparison
        plot_range_image_comparison(
            input_np, pred_np, target, mask,
            save_path=os.path.join(output_dir, f"range_compare_{i:03d}.png"),
        )

        # BEV
        pred_pts  = range_image_to_points(pred_np,  mask, config)
        gt_pts    = range_image_to_points(target,   mask, config)
        input_pts = range_image_to_points(input_np, mask, config)
        plot_bev(pred_pts, gt_pts, input_pts=input_pts,
                 save_path=os.path.join(output_dir, f"bev_{i:03d}.png"))

        # Error histogram
        plot_error_histogram(pred_pts, gt_pts,
                             save_path=os.path.join(output_dir, f"error_hist_{i:03d}.png"))

    print(f"All visualizations saved to {output_dir}/")


def plot_range_image_comparison_multi(images: dict, mask,
                                      save_path: str = "vis_multi_compare.png"):
    """Plot range images from multiple models side by side.
    Args:
        images: dict of {label: (H, W) ndarray}, e.g. {"FLASH": ..., "FLASH+": ..., "GT": ...}
        mask: (H, W) validity mask
    """
    n = len(images)
    fig, axes = plt.subplots(n, 1, figsize=(20, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, (label, img) in zip(axes, images.items()):
        im = ax.imshow(img, aspect="auto", cmap="viridis")
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax, fraction=0.02)

    axes[-1].set_xlabel("Column")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_histogram_overlay(models_pts: dict, gt_pts,
                                  save_path: str = "vis_error_overlay.png"):
    """Overlaid error histograms comparing multiple models.
    Args:
        models_pts: dict of {label: pred_pts ndarray}
        gt_pts: ground truth points
    """
    from scipy.spatial import KDTree

    if len(gt_pts) == 0:
        return

    tree_gt = KDTree(gt_pts)
    ranges = [(0, 30), (30, 60)]

    fig, axes = plt.subplots(1, len(ranges), figsize=(12, 4))
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]

    for ax, (rmin, rmax) in zip(axes, ranges):
        for (label, pts), color in zip(models_pts.items(), colors):
            if len(pts) == 0:
                continue
            dists, _ = tree_gt.query(pts)
            pred_r = np.linalg.norm(pts, axis=1)
            sel = (pred_r >= rmin) & (pred_r < rmax)
            if sel.sum() > 0:
                ax.hist(dists[sel], bins=50, range=(0, 1.0), density=True,
                        alpha=0.5, label=label, color=color)
        ax.set_title(f"{rmin}-{rmax}m")
        ax.set_xlabel("NN Distance (m)")
        ax.set_ylabel("Density")
        ax.axvline(0.1, color="k", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


@torch.no_grad()
def benchmark_fps(config: Config, num_warmup: int = 10,
                  num_iters: int = 50, device=None) -> float:
    """Measure inference FPS for a model configuration.
    Returns FPS (frames per second).
    """
    import time

    if device is None:
        device = get_device()

    model = FlashUNet(config).to(device).eval()
    dummy = torch.randn(1, 1, config.H, config.W, device=device)

    # Warmup
    for _ in range(num_warmup):
        _ = model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Timed
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start

    fps = num_iters / elapsed
    return fps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["baseline", "rafk", "mkdisc", "proposed"])
    parser.add_argument("--num_frames", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="vis_output")
    parser.add_argument("--benchmark", action="store_true", help="Run FPS benchmark")
    args = parser.parse_args()

    if args.variant:
        config = Config.ablation(args.variant, dev=args.dev)
    else:
        config = Config.dev() if args.dev else Config()

    if args.benchmark:
        fps = benchmark_fps(config)
        variant_str = args.variant or "flash"
        params = sum(p.numel() for p in FlashUNet(config).parameters())
        print(f"{variant_str}: {fps:.1f} FPS, {params:,} params")
    else:
        visualize_frames(args.checkpoint, config, args.num_frames, args.output_dir)
