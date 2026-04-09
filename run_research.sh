#!/bin/bash
# =============================================================================
# FLASH+ 研究統合パイプライン
#
# FLASH（baseline）と提案モデル（FLASH+）の学習・アブレーション・比較・可視化を
# 一括実行する。FLASH論文 Section IV の実験構成に準拠し、以下の対応で実験を行う：
#   論文の「既存モデル」 → FLASH (baseline)
#   論文の「提案モデル」 → FLASH+ (proposed: RAFK + MKDisc)
#
# Usage:
#   bash run_research.sh --dev                  # 開発デバイス（10エポック）
#   bash run_research.sh --full                 # 学習デバイス（600エポック）
#   bash run_research.sh --full --skip-preprocess  # 前処理済みの場合
#   bash run_research.sh --full --skip-train       # 学習済みの場合（評価・可視化のみ）
#   bash run_research.sh --full --skip-preprocess --skip-train --vis-frames 10
#
# Outputs:
#   experiments/               アブレーション結果（チェックポイント・評価・表）
#   vis_output/flash/          FLASH単体の可視化
#   vis_output/proposed/       FLASH+単体の可視化
#   vis_output/comparison/     FLASH vs FLASH+ 比較可視化・論文用LaTeX表
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト設定
# ---------------------------------------------------------------------------
MODE="dev"
SKIP_PREPROCESS=false
SKIP_TRAIN=false
VIS_FRAMES=""

# ---------------------------------------------------------------------------
# 引数パース
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)             MODE="dev";  shift ;;
        --full)            MODE="full"; shift ;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        --skip-train)      SKIP_TRAIN=true; shift ;;
        --vis-frames)      VIS_FRAMES="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dev|--full] [--skip-preprocess] [--skip-train] [--vis-frames N]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# モード別デフォルト
# ---------------------------------------------------------------------------
if [ "$MODE" = "dev" ]; then
    DEV_FLAG="--dev"
    VIS_FRAMES="${VIS_FRAMES:-3}"
    echo "=== DEV MODE (10 epochs, batch=1, vis_frames=${VIS_FRAMES}) ==="
else
    DEV_FLAG=""
    VIS_FRAMES="${VIS_FRAMES:-5}"
    echo "=== FULL MODE (600 epochs, batch=8, vis_frames=${VIS_FRAMES}) ==="
fi

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

section() {
    echo ""
    echo "======================================================================="
    echo "  $1"
    echo "  $(timestamp)"
    echo "======================================================================="
}

cd "$(dirname "$0")"
echo "Working directory: $(pwd)"
START_TIME=$SECONDS

# ===========================================================================
# Step 0: データ前処理
# ===========================================================================
if [ "$SKIP_PREPROCESS" = false ]; then
    section "Step 0/6: データ前処理 (KITTI raw → range image)"
    python data/preprocess.py
else
    section "Step 0/6: データ前処理 スキップ"
    echo ">>> --skip-preprocess が指定されたため前処理をスキップします"
fi

# ===========================================================================
# Step 1: アブレーション学習
#   baseline (FLASH), +RAFK, +MKDisc, proposed (FLASH+) の4バリアントを学習
# ===========================================================================
if [ "$SKIP_TRAIN" = false ]; then
    section "Step 1/6: アブレーション学習 (4バリアント)"
    echo "  baseline (FLASH), +RAFK, +MKDisc, proposed (FLASH+)"
    python run_ablation.py $DEV_FLAG --skip_eval --base_dir experiments
else
    section "Step 1/6: アブレーション学習 スキップ"
    echo ">>> --skip-train が指定されたため学習をスキップします"
    echo ">>> experiments/ 以下に学習済みチェックポイントが必要です"
fi

# ===========================================================================
# Step 2: アブレーション評価
#   全バリアントについて evaluate.py を実行し metrics を生成
# ===========================================================================
section "Step 2/6: アブレーション評価"
python run_ablation.py $DEV_FLAG --skip_train --base_dir experiments

# ===========================================================================
# Step 3: アブレーション比較表・グラフ生成
#   compare_results.py → experiments/ablation_table.tex, ablation_comparison.png
# ===========================================================================
section "Step 3/6: アブレーション比較 (表・グラフ)"
python compare_results.py --base_dir experiments --output_dir experiments

echo ""
echo "  Generated:"
echo "    experiments/ablation_table.tex"
echo "    experiments/ablation_comparison.png"

# ===========================================================================
# Step 4: FLASH vs FLASH+ クロスモデル比較可視化
#   compare_models.py → vis_output/comparison/ (range image, BEV, histogram, LaTeX)
#   FLASH論文 Table III / Figure 形式に対応
# ===========================================================================
section "Step 4/6: FLASH vs FLASH+ 比較可視化 (論文形式)"
mkdir -p vis_output/comparison
python compare_models.py \
    --base_dir experiments \
    --output_dir vis_output/comparison \
    --num_frames "$VIS_FRAMES" \
    $DEV_FLAG

echo ""
echo "  Generated:"
echo "    vis_output/comparison/range_compare_*.png   (Input | FLASH | FLASH+ | GT)"
echo "    vis_output/comparison/bev_compare_*.png     (GT / FLASH / FLASH+ BEV)"
echo "    vis_output/comparison/error_hist_overlay_*.png"
echo "    vis_output/comparison/paper_comparison_table.tex"

# ===========================================================================
# Step 5: 各モデル単体の可視化
#   visualize.py で FLASH / FLASH+ それぞれの range image・BEV を個別出力
# ===========================================================================
section "Step 5/6: 各モデル単体の可視化"

# FLASH (baseline)
FLASH_CKPT="experiments/baseline/checkpoints/best.pt"
if [ -f "$FLASH_CKPT" ]; then
    echo ">>> Visualizing FLASH (baseline)..."
    mkdir -p vis_output/flash
    python visualize.py \
        --checkpoint "$FLASH_CKPT" \
        --variant baseline \
        --num_frames "$VIS_FRAMES" \
        --output_dir vis_output/flash \
        $DEV_FLAG
else
    echo "WARNING: $FLASH_CKPT not found, skipping FLASH visualization."
fi

# FLASH+ (proposed)
PROPOSED_CKPT="experiments/proposed/checkpoints/best.pt"
if [ -f "$PROPOSED_CKPT" ]; then
    echo ">>> Visualizing FLASH+ (proposed)..."
    mkdir -p vis_output/proposed
    python visualize.py \
        --checkpoint "$PROPOSED_CKPT" \
        --variant proposed \
        --num_frames "$VIS_FRAMES" \
        --output_dir vis_output/proposed \
        $DEV_FLAG
else
    echo "WARNING: $PROPOSED_CKPT not found, skipping FLASH+ visualization."
fi

# ===========================================================================
# Step 6: FPS ベンチマーク
#   FLASH と FLASH+ の推論速度を計測し比較する
#   提案モデルが FLASH と同等速度（≒66 FPS）を維持することを確認
# ===========================================================================
section "Step 6/6: FPS ベンチマーク (FLASH vs FLASH+)"

echo ">>> FLASH (baseline) FPS:"
python visualize.py --benchmark --variant baseline $DEV_FLAG

echo ""
echo ">>> FLASH+ (proposed) FPS:"
python visualize.py --benchmark --variant proposed $DEV_FLAG

# ===========================================================================
# 完了サマリ
# ===========================================================================
ELAPSED=$(( SECONDS - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

section "パイプライン完了 (${MINS}分${SECS}秒)"
echo ""
echo "  【アブレーション実験】"
echo "    チェックポイント:    experiments/{baseline,rafk,mkdisc,proposed}/checkpoints/"
echo "    評価結果 (npz):     experiments/{baseline,rafk,mkdisc,proposed}/eval_results.npz"
echo "    比較グラフ:         experiments/ablation_comparison.png"
echo "    比較表 (LaTeX):     experiments/ablation_table.tex"
echo ""
echo "  【論文用 FLASH vs FLASH+ 比較】"
echo "    Range image比較:    vis_output/comparison/range_compare_*.png"
echo "    BEV比較:            vis_output/comparison/bev_compare_*.png"
echo "    エラーヒストグラム: vis_output/comparison/error_hist_overlay_*.png"
echo "    論文用表 (LaTeX):   vis_output/comparison/paper_comparison_table.tex"
echo ""
echo "  【単体可視化】"
echo "    FLASH:              vis_output/flash/"
echo "    FLASH+:             vis_output/proposed/"
echo ""
echo "FINISHED"
