"""Statistical tests for comparing FLASH+ ablation variants."""

import numpy as np
from scipy import stats


def paired_t_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test (two-sided).
    Returns: dict with t_stat, p_value, significant (at alpha=0.05).
    """
    t_stat, p_val = stats.ttest_rel(a, b)
    return {"t_stat": float(t_stat), "p_value": float(p_val),
            "significant": p_val < 0.05}


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test (non-parametric).
    Returns: dict with statistic, p_value, significant.
    """
    diff = a - b
    # Remove zeros (ties)
    diff = diff[diff != 0]
    if len(diff) < 10:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "significant": False}
    stat, p_val = stats.wilcoxon(diff)
    return {"statistic": float(stat), "p_value": float(p_val),
            "significant": p_val < 0.05}


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Bootstrap 95% confidence interval.
    Returns: dict with mean, ci_lower, ci_upper.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return {"mean": float(np.mean(values)),
            "ci_lower": float(lower), "ci_upper": float(upper)}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size for paired samples."""
    diff = a - b
    return float(diff.mean() / max(diff.std(ddof=1), 1e-12))


def significance_marker(p_value: float) -> str:
    """Return significance marker: ***, **, *, or ns."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "ns"


def run_all_tests(per_frame_results: dict, metrics: list = None) -> dict:
    """Run all statistical tests for all variant pairs.

    Args:
        per_frame_results: {variant: [per_frame_metrics_list]}
        metrics: list of metric keys to test (default: mae, chamfer_distance, iou, f1)

    Returns:
        Nested dict: {metric: {pair_label: {test_name: result}}}
    """
    if metrics is None:
        metrics = ["mae", "chamfer_distance", "iou", "f1"]

    variants = list(per_frame_results.keys())
    all_results = {}

    for m in metrics:
        all_results[m] = {}

        # Extract per-frame values
        variant_vals = {}
        for v in variants:
            vals = np.array([f[m] for f in per_frame_results[v]
                            if np.isfinite(f[m])])
            variant_vals[v] = vals

        # Compare proposed vs each other
        if "proposed" in variant_vals:
            for v in variants:
                if v == "proposed":
                    continue
                a = variant_vals["proposed"]
                b = variant_vals[v]
                n = min(len(a), len(b))
                if n < 5:
                    continue
                a, b = a[:n], b[:n]

                pair_label = f"proposed_vs_{v}"
                all_results[m][pair_label] = {
                    "paired_t": paired_t_test(a, b),
                    "wilcoxon": wilcoxon_test(a, b),
                    "cohens_d": cohens_d(a, b),
                    "bootstrap_proposed": bootstrap_ci(a),
                    "bootstrap_other": bootstrap_ci(b),
                }

    return all_results


def print_test_results(test_results: dict):
    """Pretty-print statistical test results."""
    for metric, pairs in test_results.items():
        print(f"\n{'='*60}")
        print(f"  Metric: {metric}")
        print(f"{'='*60}")

        for pair_label, tests in pairs.items():
            t_test = tests["paired_t"]
            wilcox = tests["wilcoxon"]
            d = tests["cohens_d"]
            ci_p = tests["bootstrap_proposed"]
            ci_o = tests["bootstrap_other"]

            marker = significance_marker(t_test["p_value"])
            print(f"\n  {pair_label}:")
            print(f"    Paired t-test: t={t_test['t_stat']:.3f}, "
                  f"p={t_test['p_value']:.2e} {marker}")
            print(f"    Wilcoxon:      W={wilcox['statistic']:.1f}, "
                  f"p={wilcox['p_value']:.2e} "
                  f"{significance_marker(wilcox['p_value'])}")
            print(f"    Cohen's d:     {d:.3f}")
            print(f"    95% CI proposed: [{ci_p['ci_lower']:.4f}, {ci_p['ci_upper']:.4f}]")
            print(f"    95% CI other:    [{ci_o['ci_lower']:.4f}, {ci_o['ci_upper']:.4f}]")
