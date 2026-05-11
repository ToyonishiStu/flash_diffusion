#!/bin/bash
# =============================================================================
# FLASH vs FLASH+RAFK 研究パイプライン
#
# FLASH（baseline）と提案モデル（FLASH+RAFK）の学習・評価・統計検定・可視化を
# 一括実行する。run_research.sh とは出力ディレクトリを完全に分離する。
#
#   run_research.sh の出力: experiments/,  vis_output/
#   本スクリプトの出力:     experiments_rafk/, vis_rafk/
#
# Usage:
#   bash run_rafk_research.sh --dev                   # 開発デバイス（10エポック）
#   bash run_rafk_research.sh --full                  # 学習デバイス（600エポック）
#   bash run_rafk_research.sh --full --skip-train     # 学習済みの場合（評価・可視化のみ）
#   bash run_rafk_research.sh --full --preprocess     # データ前処理も実行する場合
#   bash run_rafk_research.sh --full --vis-frames 10
#
# Outputs:
#   experiments_rafk/                 実験結果（チェックポイント・評価・比較表）
#     baseline/checkpoints/best.pt
#     rafk/checkpoints/best.pt
#     {baseline,rafk}/eval_results.npz
#     ablation_table.tex              FLASH vs FLASH+RAFK 比較表 (LaTeX)
#     ablation_comparison.png         比較グラフ
#   vis_rafk/flash/                   FLASH 単体の可視化
#   vis_rafk/rafk/                    FLASH+RAFK 単体の可視化
#   vis_rafk/comparison/              FLASH vs FLASH+RAFK 比較可視化・論文用LaTeX表
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト設定
# ---------------------------------------------------------------------------
MODE="dev"
PREPROCESS=false        # デフォルトはスキップ（run_research.sh と共有データを使う）
SKIP_TRAIN=false
VIS_FRAMES=""

BASE_DIR="experiments_rafk"
VIS_DIR="vis_rafk"

# ---------------------------------------------------------------------------
# 引数パース
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)          MODE="dev";  shift ;;
        --full)         MODE="full"; shift ;;
        --preprocess)   PREPROCESS=true; shift ;;
        --skip-train)   SKIP_TRAIN=true; shift ;;
        --vis-frames)   VIS_FRAMES="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dev|--full] [--preprocess] [--skip-train] [--vis-frames N]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# モード別デフォルト
# ---------------------------------------------------------------------------
if [ "$MODE" = "dev" ]; then
    DEV_FLAG="--dev"
    NUM_EPOCHS=10
    VIS_FRAMES="${VIS_FRAMES:-3}"
    echo "=== DEV MODE (10 epochs, batch=1, vis_frames=${VIS_FRAMES}) ==="
else
    DEV_FLAG=""
    NUM_EPOCHS=600
    VIS_FRAMES="${VIS_FRAMES:-5}"
    echo "=== FULL MODE (600 epochs, batch=8, vis_frames=${VIS_FRAMES}) ==="
fi

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# チェックポイントディレクトリを検査し、train.py に渡す引数を返す
# 引数: $1=ckpt_dir, $2=num_epochs
# 標準出力:
#   "skip"           → 学習完了済み（スキップ）
#   ""（空文字）      → 新規学習
#   "--resume PATH"  → 途中から再開
get_train_resume_arg() {
    local ckpt_dir="$1"
    local num_epochs="$2"

    [ ! -d "$ckpt_dir" ] && echo "" && return

    local latest_ckpt
    latest_ckpt=$(find "$ckpt_dir" -name "epoch_*.pt" 2>/dev/null | sort | tail -1)

    [ -z "$latest_ckpt" ] && echo "" && return

    local epoch_num
    epoch_num=$((10#$(basename "$latest_ckpt" | sed 's/epoch_\([0-9]*\)\.pt/\1/')))

    if [ "$epoch_num" -ge "$num_epochs" ]; then
        echo "skip"
    else
        echo "--resume $latest_ckpt"
    fi
}

section() {
    echo ""
    echo "======================================================================="
    echo "  $1"
    echo "  $(timestamp)"
    echo "======================================================================="
}

cd "$(dirname "$0")"
echo "Working directory: $(pwd)"
echo "Output dirs: ${BASE_DIR}/, ${VIS_DIR}/"
START_TIME=$SECONDS

# ===========================================================================
# Step 0: データ前処理（デフォルトはスキップ）
#   run_research.sh 実行済みの場合、前処理済みデータを共有して使う。
#   初回実行時など前処理が必要な場合は --preprocess フラグを指定する。
# ===========================================================================
if [ "$PREPROCESS" = true ]; then
    section "Step 0/6: データ前処理 (KITTI raw → range image)"
    python data/preprocess.py
else
    section "Step 0/6: データ前処理 スキップ"
    echo ">>> 前処理スキップ（run_research.sh と共有データを使用）"
    echo ">>> 初回実行時は --preprocess フラグを指定してください"
fi

# ===========================================================================
# Step 1: 学習 — baseline (FLASH) と rafk (FLASH+RAFK) のみ
#   → experiments_rafk/baseline/checkpoints/best.pt
#   → experiments_rafk/rafk/checkpoints/best.pt
# ===========================================================================
if [ "$SKIP_TRAIN" = false ]; then
    section "Step 1/6: 学習 (baseline + rafk)"
    echo "  対象バリアント: FLASH (baseline), FLASH+RAFK (rafk)"

    for VARIANT in baseline rafk; do
        CKPT_DIR="${BASE_DIR}/${VARIANT}/checkpoints"
        LOG_DIR="${BASE_DIR}/${VARIANT}/runs"

        RESUME_ARG=$(get_train_resume_arg "$CKPT_DIR" "$NUM_EPOCHS")

        if [ "$RESUME_ARG" = "skip" ]; then
            echo ">>> ${VARIANT}: 学習完了済み (${NUM_EPOCHS}エポック) → スキップ"
            continue
        elif [ -n "$RESUME_ARG" ]; then
            RESUME_EPOCH=$((10#$(basename "$(echo "$RESUME_ARG" | awk '{print $2}')" | sed 's/epoch_\([0-9]*\)\.pt/\1/')))
            echo ">>> ${VARIANT}: エポック ${RESUME_EPOCH} から再開..."
        else
            echo ">>> ${VARIANT}: 新規学習..."
        fi

        python train.py \
            --variant "$VARIANT" \
            --checkpoint_dir "$CKPT_DIR" \
            --log_dir "$LOG_DIR" \
            $DEV_FLAG \
            $RESUME_ARG
    done
else
    section "Step 1/6: 学習 スキップ"
    echo ">>> --skip-train が指定されたため学習をスキップします"
    echo ">>> ${BASE_DIR}/{baseline,rafk}/checkpoints/ に学習済みチェックポイントが必要です"
fi

# ===========================================================================
# Step 2: 評価 — baseline と rafk
#   → experiments_rafk/baseline/eval_results.npz
#   → experiments_rafk/rafk/eval_results.npz
# ===========================================================================
section "Step 2/6: 評価 (baseline + rafk)"

for VARIANT in baseline rafk; do
    CKPT_PATH="${BASE_DIR}/${VARIANT}/checkpoints/best.pt"
    OUTPUT_PATH="${BASE_DIR}/${VARIANT}/eval_results.npz"

    if [ -f "$CKPT_PATH" ]; then
        echo ">>> Evaluating ${VARIANT}..."
        python evaluate.py \
            --variant "$VARIANT" \
            --checkpoint "$CKPT_PATH" \
            --output "$OUTPUT_PATH" \
            $DEV_FLAG
    else
        echo "WARNING: ${CKPT_PATH} が見つかりません。${VARIANT} の評価をスキップします。"
    fi
done

# ===========================================================================
# Step 3: 比較表・統計検定・グラフ生成
#   compare_results.py → experiments_rafk/ablation_table.tex
#                      → experiments_rafk/ablation_comparison.png
#   統計検定: rafk vs baseline
# ===========================================================================
section "Step 3/6: 比較表・統計検定・グラフ (FLASH vs FLASH+RAFK)"

python compare_results.py \
    --base_dir "$BASE_DIR" \
    --output_dir "$BASE_DIR" \
    --reference_variant rafk

echo ""
echo "  Generated:"
echo "    ${BASE_DIR}/ablation_table.tex"
echo "    ${BASE_DIR}/ablation_comparison.png"

# ===========================================================================
# Step 4: FLASH vs FLASH+RAFK 比較可視化（論文形式）
#   compare_models.py → vis_rafk/comparison/
# ===========================================================================
section "Step 4/6: FLASH vs FLASH+RAFK 比較可視化 (論文形式)"

mkdir -p "${VIS_DIR}/comparison"
python compare_models.py \
    --base_dir "$BASE_DIR" \
    --output_dir "${VIS_DIR}/comparison" \
    --baseline_variant baseline \
    --proposed_variant rafk \
    --num_frames "$VIS_FRAMES" \
    $DEV_FLAG

echo ""
echo "  Generated:"
echo "    ${VIS_DIR}/comparison/range_compare_*.png   (Input | FLASH | FLASH+RAFK | GT)"
echo "    ${VIS_DIR}/comparison/bev_compare_*.png     (GT / FLASH / FLASH+RAFK BEV)"
echo "    ${VIS_DIR}/comparison/error_hist_overlay_*.png"
echo "    ${VIS_DIR}/comparison/paper_comparison_table.tex"

# ===========================================================================
# Step 5: 各モデル単体の可視化
# ===========================================================================
section "Step 5/6: 各モデル単体の可視化"

# FLASH (baseline)
FLASH_CKPT="${BASE_DIR}/baseline/checkpoints/best.pt"
if [ -f "$FLASH_CKPT" ]; then
    echo ">>> Visualizing FLASH (baseline)..."
    mkdir -p "${VIS_DIR}/flash"
    python visualize.py \
        --checkpoint "$FLASH_CKPT" \
        --variant baseline \
        --num_frames "$VIS_FRAMES" \
        --output_dir "${VIS_DIR}/flash" \
        $DEV_FLAG
else
    echo "WARNING: ${FLASH_CKPT} が見つかりません。FLASH の可視化をスキップします。"
fi

# FLASH+RAFK (rafk)
RAFK_CKPT="${BASE_DIR}/rafk/checkpoints/best.pt"
if [ -f "$RAFK_CKPT" ]; then
    echo ">>> Visualizing FLASH+RAFK (rafk)..."
    mkdir -p "${VIS_DIR}/rafk"
    python visualize.py \
        --checkpoint "$RAFK_CKPT" \
        --variant rafk \
        --num_frames "$VIS_FRAMES" \
        --output_dir "${VIS_DIR}/rafk" \
        $DEV_FLAG
else
    echo "WARNING: ${RAFK_CKPT} が見つかりません。FLASH+RAFK の可視化をスキップします。"
fi

# ===========================================================================
# Step 6: FPS ベンチマーク (FLASH vs FLASH+RAFK)
# ===========================================================================
section "Step 6/6: FPS ベンチマーク (FLASH vs FLASH+RAFK)"

echo ">>> FLASH (baseline) FPS:"
python visualize.py --benchmark --variant baseline $DEV_FLAG

echo ""
echo ">>> FLASH+RAFK (rafk) FPS:"
python visualize.py --benchmark --variant rafk $DEV_FLAG

# ===========================================================================
# 完了サマリ
# ===========================================================================
ELAPSED=$(( SECONDS - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

section "パイプライン完了 (${MINS}分${SECS}秒)"
echo ""
echo "  【FLASH vs FLASH+RAFK 比較実験】"
echo "    チェックポイント:    ${BASE_DIR}/{baseline,rafk}/checkpoints/"
echo "    評価結果 (npz):     ${BASE_DIR}/{baseline,rafk}/eval_results.npz"
echo "    比較グラフ:         ${BASE_DIR}/ablation_comparison.png"
echo "    比較表 (LaTeX):     ${BASE_DIR}/ablation_table.tex"
echo ""
echo "  【論文用 FLASH vs FLASH+RAFK 比較】"
echo "    Range image比較:    ${VIS_DIR}/comparison/range_compare_*.png"
echo "    BEV比較:            ${VIS_DIR}/comparison/bev_compare_*.png"
echo "    エラーヒストグラム: ${VIS_DIR}/comparison/error_hist_overlay_*.png"
echo "    論文用表 (LaTeX):   ${VIS_DIR}/comparison/paper_comparison_table.tex"
echo ""
echo "  【単体可視化】"
echo "    FLASH:              ${VIS_DIR}/flash/"
echo "    FLASH+RAFK:         ${VIS_DIR}/rafk/"
echo ""
echo "FINISHED"
