#!/usr/bin/env bash
# FLASH+ ablation experiment runner
# Usage:
#   bash run_ablation.sh --dev      # Quick dev test (few epochs, small data)
#   bash run_ablation.sh --full     # Full training run
set -euo pipefail

MODE="${1:---dev}"
BASE_DIR="experiments"

if [ "$MODE" = "--dev" ]; then
    echo "=== DEV MODE: quick ablation test ==="
    python run_ablation.py --dev --base_dir "$BASE_DIR"
elif [ "$MODE" = "--full" ]; then
    echo "=== FULL MODE: complete ablation experiment ==="
    python run_ablation.py --base_dir "$BASE_DIR"
else
    echo "Usage: $0 [--dev|--full]"
    exit 1
fi

# Compare results
echo ""
echo "=== Comparing results ==="
python compare_results.py --base_dir "$BASE_DIR"

echo ""
echo "=== Done ==="
