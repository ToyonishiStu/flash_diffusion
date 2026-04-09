"""Cross-model comparison: FLASH (baseline) vs FLASH+ (proposed).

FLASH論文 Section IV の「既存モデル vs FLASH」の図・表に対応する形式で比較を生成する。
- 既存モデル枠 → FLASH (baseline)
- 提案モデル枠 → FLASH+ (proposed)

Outputs:
    {output_dir}/range_compare_{i:03d}.png   — Input | FLASH | FLASH+ | GT の4行
    {output_dir}/bev_compare_{i:03d}.png     — BEV 3列比較 (GT / FLASH / FLASH+)
    {output_dir}/error_hist_overlay_{i:03d}.png — 距離帯エラーヒストグラム重ね
    {output_dir}/paper_comparison_table.tex  — FLASH Table III 対応 LaTeX
"""

import os
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.default import Config
from model.unet import FlashUNet
from data.dataset import RangeImageDataset, gather_files
from utils.reprojection import range_image_to_points
from utils.misc import get_device
from visualize import (
    plot_range_image_comparison_multi,
    plot_error_histogram_overlay,
    benchmark_fps,
)


def load_model(variant: str, base_dir: str, config: Config, device) -> FlashUNet:
    """チェックポイントからモデルをロードする。"""
    ckpt_path = os.path.join(base_dir, variant, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"先に run_research.sh --full (または --dev) で学習してください。"
        )
    model = FlashUNet(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {variant} from epoch {ckpt.get('epoch', '?')}: {ckpt_path}")
    return model


def load_eval_metrics(variant: str, base_dir: str) -> dict | None:
    """評価済みの npz メトリクスをロードする。"""
    path = os.path.join(base_dir, variant, "eval_results.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "agg": data["agg"].item(),
        "per_frame": list(data["per_frame"]),
    }


def generate_latex_comparison(
    flash_metrics: dict | None,
    proposed_metrics: dict | None,
    output_path: str,
):
    """FLASH論文 Table III 対応の LaTeX 比較表を生成する。

    行 = [FLASH (既存), FLASH+ (提案)]
    列 = MAE near / MAE far / CD / IoU / F1 / FPS
    """
    rows = [
        ("FLASH",  flash_metrics),
        ("FLASH+", proposed_metrics),
    ]

    def _mae_range(per_frame, rng_key):
        if per_frame is None:
            return float("nan")
        vals = [
            f["mae_by_distance"][rng_key]
            for f in per_frame
            if "mae_by_distance" in f and np.isfinite(f["mae_by_distance"].get(rng_key, float("nan")))
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def _metric(agg, key):
        if agg is None:
            return float("nan")
        return agg.get(f"{key}_mean", float("nan"))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of FLASH (baseline) and the proposed FLASH+.",
        r"Evaluated on KITTI (sequence 0, 2{,}000 frames).}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & MAE$_{\text{near}}$ (m) & MAE$_{\text{far}}$ (m) "
        r"& CD & IoU & F1 \\",
        r"\midrule",
    ]

    # Find best values for bold formatting
    col_fns = {
        "mae_near": lambda m: _mae_range(m["per_frame"] if m else None, "0-30m"),
        "mae_far":  lambda m: _mae_range(m["per_frame"] if m else None, "30-60m"),
        "cd":  lambda m: _metric(m["agg"] if m else None, "chamfer_distance"),
        "iou": lambda m: _metric(m["agg"] if m else None, "iou"),
        "f1":  lambda m: _metric(m["agg"] if m else None, "f1"),
    }
    lower_better = {"mae_near", "mae_far", "cd"}
    best = {}
    for col, fn in col_fns.items():
        vals = [fn(m) for _, m in rows]
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            best[col] = min(finite) if col in lower_better else max(finite)

    def _fmt(val, col):
        if not np.isfinite(val):
            return "--"
        s = f"{val:.4f}"
        if col in best and np.isclose(val, best[col]):
            s = r"\textbf{" + s + "}"
        return s

    for label, m in rows:
        per_frame = m["per_frame"] if m else None
        agg = m["agg"] if m else None
        cells = [
            _fmt(_mae_range(per_frame, "0-30m"),  "mae_near"),
            _fmt(_mae_range(per_frame, "30-60m"), "mae_far"),
            _fmt(_metric(agg, "chamfer_distance"), "cd"),
            _fmt(_metric(agg, "iou"),  "iou"),
            _fmt(_metric(agg, "f1"),   "f1"),
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved: {output_path}")


def plot_bev_three(gt_pts, flash_pts, proposed_pts, save_path: str):
    """BEV 3列比較: GT / FLASH / FLASH+"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    pairs = [
        (axes[0], gt_pts,       "Ground Truth"),
        (axes[1], flash_pts,    "FLASH (baseline)"),
        (axes[2], proposed_pts, "FLASH+ (proposed)"),
    ]
    for ax, pts, title in pairs:
        if len(pts) == 0:
            ax.set_title(title)
            continue
        if len(pts) > 50000:
            idx = np.random.choice(len(pts), 50000, replace=False)
            pts = pts[idx]
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=0.3,
                        cmap="viridis", vmin=-2, vmax=2)
        ax.set_title(title, fontsize=13)
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


@torch.no_grad()
def run_comparison(
    base_dir: str,
    num_frames: int,
    output_dir: str,
    dev: bool,
):
    """メイン比較ルーチン。"""
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()

    # --- Config per variant ---
    cfg_flash    = Config.ablation("baseline", dev=dev)
    cfg_proposed = Config.ablation("proposed", dev=dev)

    # --- Load models ---
    print("\n[1/5] Loading models...")
    model_flash    = load_model("baseline", base_dir, cfg_flash,    device)
    model_proposed = load_model("proposed", base_dir, cfg_proposed, device)

    # --- Load val frames ---
    print("\n[2/5] Loading validation frames...")
    all_files = gather_files(cfg_flash.processed_root)
    split = int(len(all_files) * 0.8)
    val_files = all_files[split:]
    n = min(num_frames, len(val_files))
    if n == 0:
        print("WARNING: No validation files found. Skipping per-frame visualizations.")
    else:
        dataset_flash    = RangeImageDataset(val_files[:n], cfg_flash)
        dataset_proposed = RangeImageDataset(val_files[:n], cfg_proposed)

        print(f"\n[3/5] Generating per-frame visualizations ({n} frames)...")
        for i in range(n):
            s_f = dataset_flash[i]
            s_p = dataset_proposed[i]

            inp_t   = s_f["input"].unsqueeze(0).to(device)
            target  = s_f["target"].numpy()[0]
            mask    = s_f["mask"].numpy()[0]
            inp_np  = s_f["input"].numpy()[0]

            # Inference
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=cfg_flash.mixed_precision):
                pred_flash = model_flash(inp_t)[0, 0].cpu().float().numpy()

            inp_t_p = s_p["input"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=cfg_proposed.mixed_precision):
                pred_proposed = model_proposed(inp_t_p)[0, 0].cpu().float().numpy()

            # Range image comparison: Input | FLASH | FLASH+ | GT
            plot_range_image_comparison_multi(
                images={
                    "Input (bilinear 16→64)": inp_np,
                    "FLASH (baseline)":       pred_flash,
                    "FLASH+ (proposed)":      pred_proposed,
                    "Ground Truth":           target,
                },
                mask=mask,
                save_path=os.path.join(output_dir, f"range_compare_{i:03d}.png"),
            )

            # 3D reprojection
            gt_pts       = range_image_to_points(target,       mask, cfg_flash)
            flash_pts    = range_image_to_points(pred_flash,   mask, cfg_flash)
            proposed_pts = range_image_to_points(pred_proposed, mask, cfg_proposed)

            # BEV comparison (GT / FLASH / FLASH+)
            plot_bev_three(
                gt_pts, flash_pts, proposed_pts,
                save_path=os.path.join(output_dir, f"bev_compare_{i:03d}.png"),
            )

            # Error histogram overlay
            plot_error_histogram_overlay(
                models_pts={"FLASH": flash_pts, "FLASH+": proposed_pts},
                gt_pts=gt_pts,
                save_path=os.path.join(output_dir, f"error_hist_overlay_{i:03d}.png"),
            )
            print(f"  Frame {i+1}/{n} done.")

    # --- Load eval metrics ---
    print("\n[4/5] Loading evaluation metrics...")
    flash_metrics    = load_eval_metrics("baseline", base_dir)
    proposed_metrics = load_eval_metrics("proposed", base_dir)

    if flash_metrics is None:
        print("WARNING: No eval metrics for baseline. Run evaluation first.")
    if proposed_metrics is None:
        print("WARNING: No eval metrics for proposed. Run evaluation first.")

    # --- LaTeX table ---
    generate_latex_comparison(
        flash_metrics, proposed_metrics,
        output_path=os.path.join(output_dir, "paper_comparison_table.tex"),
    )

    # --- Terminal summary ---
    print("\n" + "=" * 60)
    print("FLASH vs FLASH+ — Comparison Summary")
    print("=" * 60)
    for label, m in [("FLASH (baseline)", flash_metrics), ("FLASH+ (proposed)", proposed_metrics)]:
        if m is None:
            print(f"  {label}: metrics not available")
            continue
        agg = m["agg"]
        per_frame = m["per_frame"]
        mae_near = np.nanmean([f["mae_by_distance"].get("0-30m", float("nan"))
                               for f in per_frame if "mae_by_distance" in f])
        mae_far  = np.nanmean([f["mae_by_distance"].get("30-60m", float("nan"))
                               for f in per_frame if "mae_by_distance" in f])
        print(f"  {label}:")
        print(f"    MAE near (0-30m): {mae_near:.4f} m")
        print(f"    MAE far (30-60m): {mae_far:.4f} m")
        print(f"    CD:   {agg.get('chamfer_distance_mean', float('nan')):.6f}")
        print(f"    IoU:  {agg.get('iou_mean', float('nan')):.4f}")
        print(f"    F1:   {agg.get('f1_mean', float('nan')):.4f}")
    print("=" * 60)

    print(f"\n[5/5] All outputs saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="FLASH vs FLASH+ cross-model comparison"
    )
    parser.add_argument("--base_dir",   type=str, default="experiments",
                        help="Base directory containing per-variant subdirs")
    parser.add_argument("--output_dir", type=str, default="vis_output/comparison")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--dev",        action="store_true")
    args = parser.parse_args()

    run_comparison(
        base_dir=args.base_dir,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        dev=args.dev,
    )


if __name__ == "__main__":
    main()
