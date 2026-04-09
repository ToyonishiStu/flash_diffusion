#!/bin/bash
# =============================================================================
# FLASH LiDAR Super-Resolution — 論文再現パイプライン一括実行
# Usage:
#   bash run_flash.sh --dev          # 開発デバイス (10エポック, 1ドライブ)
#   bash run_flash.sh --full         # 学習デバイス (600エポック, 全データ)
#   bash run_flash.sh --full --resume checkpoints/epoch_0100.pt
# =============================================================================
set -e

# --- デフォルト設定 ---
MODE="dev"
EPOCHS=""
BATCH_SIZE=""
RESUME=""
SKIP_PREPROCESS=false
SKIP_TRAIN=false
NUM_VIS_FRAMES=5

# --- 引数パース ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)        MODE="dev";  shift ;;
        --full)       MODE="full"; shift ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --resume)     RESUME="$2"; shift 2 ;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        --skip-train)      SKIP_TRAIN=true; shift ;;
        --vis-frames) NUM_VIS_FRAMES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- モード別デフォルト ---
if [ "$MODE" = "dev" ]; then
    EPOCHS="${EPOCHS:-10}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    DEV_FLAG="--dev"
else
    EPOCHS="${EPOCHS:-600}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    DEV_FLAG=""
fi

# --- ヘルパー ---
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
section() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "  $(timestamp)"
    echo "================================================================"
}

cd "$(dirname "$0")"
echo "FLASH Pipeline — mode=$MODE, epochs=$EPOCHS, batch_size=$BATCH_SIZE"
echo "Working directory: $(pwd)"
START_TIME=$SECONDS

# =============================================================================
# Step 1: データ前処理 (点群 → range image)
# =============================================================================
if [ "$SKIP_PREPROCESS" = false ]; then
    section "Step 1/5: データ前処理"
    python data/preprocess.py
else
    echo ">>> Step 1 skipped (--skip-preprocess)"
fi

# =============================================================================
# Step 2: 学習
# =============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    section "Step 2/5: 学習 (${EPOCHS} epochs)"
    TRAIN_CMD="python train.py $DEV_FLAG --epochs $EPOCHS --batch_size $BATCH_SIZE"
    if [ -n "$RESUME" ]; then
        TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
    fi
    echo ">>> $TRAIN_CMD"
    $TRAIN_CMD
else
    echo ">>> Step 2 skipped (--skip-train)"
fi

# =============================================================================
# Step 3: 評価
# =============================================================================
section "Step 3/5: 評価"
python evaluate.py --checkpoint checkpoints/best.pt $DEV_FLAG

# =============================================================================
# Step 4: 可視化
# =============================================================================
section "Step 4/5: 可視化"
python visualize.py --checkpoint checkpoints/best.pt $DEV_FLAG \
    --num_frames "$NUM_VIS_FRAMES" --output_dir vis_output

# =============================================================================
# Step 5: 推論
# =============================================================================
section "Step 5/5: 推論"
python infer.py --checkpoint checkpoints/best.pt $DEV_FLAG --output_dir infer_output

# =============================================================================
# 完了
# =============================================================================
ELAPSED=$(( SECONDS - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))
section "パイプライン完了 (${MINS}分${SECS}秒)"
echo "  チェックポイント: checkpoints/"
echo "  評価結果:         eval_results.npz"
echo "  可視化:           vis_output/"
echo "  推論出力:         infer_output/"
echo "FINISHED"