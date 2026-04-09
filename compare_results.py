"""Compare results across FLASH+ ablation variants.

Generates:
- Terminal comparison table
- LaTeX table (for paper)
- Matplotlib bar chart figure
"""

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = ["baseline", "rafk", "mkdisc", "proposed"]
VARIANT_LABELS = {
    "baseline": "FLASH",
    "rafk": "+RAFK",
    "mkdisc": "+MKDisc",
    "proposed": "FLASH+",
}


def load_results(base_dir: str) -> dict:
    """Load per-frame eval results for all available variants."""
    results = {}
    for variant in VARIANTS:
        path = os.path.join(base_dir, variant, "eval_results.npz")
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            results[variant] = {
                "agg": data["agg"].item(),
                "per_frame": list(data["per_frame"]),
            }
    return results


def print_comparison_table(results: dict):
    """Print formatted comparison table to terminal."""
    metrics = ["mae", "chamfer_distance", "iou", "f1"]
    metric_labels = {"mae": "MAE (m)", "chamfer_distance": "CD (m²)",
                     "iou": "IoU", "f1": "F1"}

    header = f"{'Method':<12s}" + "".join(f"{metric_labels[m]:>14s}" for m in metrics)
    print("\n" + "=" * len(header))
    print("COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for variant in VARIANTS:
        if variant not in results:
            continue
        agg = results[variant]["agg"]
        label = VARIANT_LABELS[variant]
        row = f"{label:<12s}"
        for m in metrics:
            mean = agg.get(f"{m}_mean", float("nan"))
            std = agg.get(f"{m}_std", float("nan"))
            row += f"  {mean:>5.4f}±{std:.3f}"
        print(row)

    print("=" * len(header))

    # MAE by distance
    print("\nMAE by Distance Range:")
    for variant in VARIANTS:
        if variant not in results:
            continue
        per_frame = results[variant]["per_frame"]
        label = VARIANT_LABELS[variant]
        if "mae_by_distance" not in per_frame[0]:
            continue
        ranges = per_frame[0]["mae_by_distance"].keys()
        row = f"  {label:<10s}"
        for rng in ranges:
            vals = [f["mae_by_distance"][rng] for f in per_frame
                    if np.isfinite(f["mae_by_distance"][rng])]
            row += f"  {rng}: {np.mean(vals):.4f}m"
        print(row)


def generate_latex_table(results: dict, output_path: str):
    """Generate LaTeX table in FLASH paper format."""
    metrics = ["mae", "chamfer_distance", "iou", "f1"]
    metric_headers = ["MAE (m)", "CD (m$^2$)", "IoU", "F1"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study results.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Method & " + " & ".join(metric_headers) + r" \\",
        r"\midrule",
    ]

    best_vals = {}
    for m in metrics:
        vals = []
        for v in VARIANTS:
            if v in results:
                val = results[v]["agg"].get(f"{m}_mean", float("nan"))
                vals.append(val)
        if vals:
            if m in ("mae", "chamfer_distance"):
                best_vals[m] = min(v for v in vals if np.isfinite(v))
            else:
                best_vals[m] = max(v for v in vals if np.isfinite(v))

    for variant in VARIANTS:
        if variant not in results:
            continue
        agg = results[variant]["agg"]
        label = VARIANT_LABELS[variant]
        cells = []
        for m in metrics:
            mean = agg.get(f"{m}_mean", float("nan"))
            std = agg.get(f"{m}_std", float("nan"))
            cell = f"{mean:.4f}"
            if m in best_vals and np.isclose(mean, best_vals[m]):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {output_path}")


def generate_bar_chart(results: dict, output_path: str):
    """Generate grouped bar chart comparing variants."""
    metrics = ["mae", "chamfer_distance", "iou", "f1"]
    metric_labels = {"mae": "MAE (m)", "chamfer_distance": "CD (m²)",
                     "iou": "IoU", "f1": "F1"}

    available = [v for v in VARIANTS if v in results]
    if not available:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for ax, m in zip(axes, metrics):
        vals = []
        stds = []
        labels = []
        for i, v in enumerate(available):
            agg = results[v]["agg"]
            vals.append(agg.get(f"{m}_mean", 0))
            stds.append(agg.get(f"{m}_std", 0))
            labels.append(VARIANT_LABELS[v])

        bars = ax.bar(range(len(vals)), vals, yerr=stds, capsize=4,
                      color=colors[:len(vals)], alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(metric_labels[m])
        ax.set_title(metric_labels[m])

        # Highlight best
        if m in ("mae", "chamfer_distance"):
            best_idx = int(np.argmin(vals))
        else:
            best_idx = int(np.argmax(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved to {output_path}")


def run_statistical_tests(results: dict):
    """Run and print statistical tests on per-frame results."""
    from utils.stats import run_all_tests, print_test_results

    per_frame_results = {}
    for variant, data in results.items():
        per_frame_results[variant] = data["per_frame"]

    test_results = run_all_tests(per_frame_results)
    print_test_results(test_results)
    return test_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="experiments")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_stats", action="store_true", help="Skip statistical tests")
    args = parser.parse_args()

    output_dir = args.output_dir or args.base_dir
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(args.base_dir)
    if not results:
        print("No results found. Run ablation experiments first.")
        return

    print_comparison_table(results)
    generate_latex_table(results, os.path.join(output_dir, "ablation_table.tex"))
    generate_bar_chart(results, os.path.join(output_dir, "ablation_comparison.png"))

    if not args.no_stats and len(results) > 1:
        run_statistical_tests(results)


if __name__ == "__main__":
    main()
