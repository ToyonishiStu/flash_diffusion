"""Ablation experiment runner for FLASH+ variants."""

import os
import sys
import subprocess
import argparse
import json


VARIANTS = ["baseline", "rafk", "mkdisc", "proposed"]


def run_command(cmd: list, desc: str):
    """Run a command and stream output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with return code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run FLASH+ ablation experiments")
    parser.add_argument("--dev", action="store_true", help="Dev mode (small data, few epochs)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip_train", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--base_dir", type=str, default="experiments",
                        help="Base directory for all experiment outputs")
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)

    # Train all variants
    if not args.skip_train:
        for variant in VARIANTS:
            ckpt_dir = os.path.join(args.base_dir, variant, "checkpoints")
            log_dir = os.path.join(args.base_dir, variant, "runs")

            cmd = [
                sys.executable, "train.py",
                "--variant", variant,
                "--checkpoint_dir", ckpt_dir,
                "--log_dir", log_dir,
            ]
            if args.dev:
                cmd.append("--dev")
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])

            run_command(cmd, f"Training {variant}")

    # Evaluate all variants
    if not args.skip_eval:
        results = {}
        for variant in VARIANTS:
            ckpt_path = os.path.join(args.base_dir, variant, "checkpoints", "best.pt")
            output_path = os.path.join(args.base_dir, variant, "eval_results.npz")

            if not os.path.exists(ckpt_path):
                print(f"WARNING: No checkpoint found for {variant} at {ckpt_path}, skipping")
                continue

            cmd = [
                sys.executable, "evaluate.py",
                "--variant", variant,
                "--checkpoint", ckpt_path,
                "--output", output_path,
            ]
            if args.dev:
                cmd.append("--dev")

            run_command(cmd, f"Evaluating {variant}")
            results[variant] = output_path

        # Save summary
        summary_path = os.path.join(args.base_dir, "eval_paths.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nAll evaluations complete. Paths saved to {summary_path}")


if __name__ == "__main__":
    main()
