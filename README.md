# FLASH+ LiDAR Super-Resolution

FLASHモデル（arXiv:2511.07377）の完全再現実装に、遠距離精度向上のための2つのモジュールを追加した提案モデル（FLASH+）の実装リポジトリ。

**問題設定：** KITTIデータセットを使用した LiDAR range image の超解像（16行→64行、4倍アップサンプリング）。FLASHは近距離（0–30m）でMAE=0.239mを達成するが、遠距離（30–60m）ではMAE=2.045m（約8.6倍劣化）と課題がある。FLASH+はその遠距離精度の改善を目的とする。

**提案モジュール（FLASH+ = FLASH + 以下の2モジュール）：**
- **RAFK**（Range-Adaptive Frequency Kernel）：距離に応じた動的周波数フィルタリング
- **MKDisc**（Meta-Kernel Discriminator）：球面幾何制約付き敵対的損失（学習時のみ）

---

## ディレクトリ構成

```
flash_diffusion/
├── model/                  # コアモデル実装
├── data/                   # データ読み込み・前処理
├── config/                 # ハイパーパラメータ設定
├── utils/                  # ユーティリティ
├── docs/                   # 設計文書
├── experiments/            # アブレーション実験結果（学習後に生成）
├── vis_output/             # 可視化出力（実行後に生成）
├── checkpoints/            # チェックポイント（学習後に生成）
├── kitti_raw/              # KITTI生データ
├── kitti_processed/        # 前処理済みデータ（前処理後に生成）
└── runs/                   # TensorBoardログ（学習後に生成）
```

---

## 実行スクリプト

### `run_research.sh` — 統合研究パイプライン ⭐

研究に必要な全ステップを一括実行するメインスクリプト。

```bash
bash run_research.sh --dev     # 開発デバイス（10エポック、動作確認用）
bash run_research.sh --full    # 学習デバイス（600エポック、本番）

# オプション
bash run_research.sh --full --skip-preprocess   # KITTIデータ前処理スキップ
bash run_research.sh --full --skip-train        # 学習スキップ（評価・可視化のみ）
bash run_research.sh --full --vis-frames 10     # 可視化フレーム数を指定
```

**実行ステップ：**

| ステップ | 内容 | 使用スクリプト |
|---------|------|--------------|
| 0 | KITTIデータ前処理 | `data/preprocess.py` |
| 1 | アブレーション学習（4バリアント） | `run_ablation.py` |
| 2 | アブレーション評価 | `run_ablation.py --skip_train` |
| 3 | アブレーション比較表・グラフ生成 | `compare_results.py` |
| 4 | FLASH vs FLASH+ 比較可視化（論文形式） | `compare_models.py` |
| 5 | 各モデル単体の可視化 | `visualize.py` × 2 |
| 6 | FPS ベンチマーク | `visualize.py --benchmark` × 2 |

**出力ディレクトリ：**
```
experiments/                         # アブレーション結果（表・グラフ・チェックポイント）
vis_output/flash/                    # FLASH単体の可視化
vis_output/proposed/                 # FLASH+単体の可視化
vis_output/comparison/               # FLASH vs FLASH+ 比較（論文用図・LaTeX表）
```

---

### `run_flash.sh` — FLASH 単体パイプライン

FLASHモデル（baseline）の学習から評価・可視化までを実行する。

```bash
bash run_flash.sh --dev                          # 開発モード
bash run_flash.sh --full                         # 学習モード
bash run_flash.sh --full --resume checkpoints/epoch_0100.pt
bash run_flash.sh --full --skip-preprocess --skip-train  # 評価・可視化のみ
```

### `run_ablation.sh` — アブレーション実験

4バリアント（baseline / +RAFK / +MKDisc / proposed）を順次学習・評価する。

```bash
bash run_ablation.sh --dev    # 開発モード
bash run_ablation.sh --full   # 学習モード
```

### `download_kitti_velodyne.sh` — KITTIデータダウンロード（学習デバイス用）
### `download_kitti_velodyne2.sh` — KITTIデータダウンロード（開発デバイス用、一部のみ）

---

## Python スクリプト

### `train.py` — 学習ループ

FLASHおよびFLASH+の学習を実行する。バリアントは `--variant` で指定。

```bash
python train.py --dev                          # FLASH (baseline) 開発モード
python train.py --variant proposed             # FLASH+ 学習モード
python train.py --variant rafk --dev           # +RAFKのみ 開発モード
python train.py --resume checkpoints/best.pt   # チェックポイントから再開
python train.py --checkpoint_dir experiments/baseline/checkpoints \
                --log_dir experiments/baseline/runs
```

**主な機能：**
- 線形ウォームアップ + コサインアニーリング（ウォームリスタート）学習率スケジュール
- Mixed Precision（float16）+ Gradient Checkpointing
- Generator（FlashUNet）と Discriminator（MKDisc）の同時学習
- TensorBoardログ記録
- バリアント別チェックポイント管理

---

### `evaluate.py` — 評価パイプライン

バリデーションセット全体でメトリクスを計算する。

```bash
python evaluate.py --checkpoint checkpoints/best.pt
python evaluate.py --checkpoint experiments/proposed/checkpoints/best.pt \
                   --variant proposed --output experiments/proposed/eval_results.npz
```

**出力メトリクス（距離帯別）：**
- MAE（m）：近距離 0–30m / 遠距離 30–60m
- Chamfer Distance（CD）
- IoU、Precision、Recall、F1スコア

---

### `visualize.py` — 可視化

単一モデルの定性的可視化とFPS計測を行う。

```bash
python visualize.py --checkpoint experiments/baseline/checkpoints/best.pt \
                    --variant baseline --num_frames 5 --output_dir vis_output/flash
python visualize.py --benchmark --variant proposed   # FPS計測のみ
```

**生成物：**
- `range_compare_*.png`：Input / Prediction / Ground Truth / 誤差マップの4行比較
- `bev_*.png`：鳥瞰図（BEV）散布図
- `error_hist_*.png`：距離帯別エラーヒストグラム

**内部関数（`compare_models.py` からも再利用）：**
- `plot_range_image_comparison_multi(images, mask, save_path)` — 複数モデルのrange image比較
- `plot_error_histogram_overlay(models_pts, gt_pts, save_path)` — 複数モデルのエラーヒストグラム重ね表示
- `benchmark_fps(config, num_warmup, num_iters, device)` — 推論FPS計測

---

### `compare_models.py` — FLASH vs FLASH+ クロスモデル比較

FLASH論文 Section IV の実験構成に対応した比較可視化を生成する。  
（論文の「既存モデル」枠→FLASH、「提案モデル」枠→FLASH+）

```bash
python compare_models.py --base_dir experiments --num_frames 5 \
                         --output_dir vis_output/comparison
python compare_models.py --dev    # 開発モード
```

**生成物（`vis_output/comparison/`）：**
- `range_compare_*.png`：Input | FLASH | FLASH+ | GT の4行レイアウト
- `bev_compare_*.png`：GT / FLASH / FLASH+ の3列BEV比較
- `error_hist_overlay_*.png`：距離帯別エラーヒストグラム重ね（FLASH vs FLASH+）
- `paper_comparison_table.tex`：FLASH論文 Table III 形式のLaTeX比較表

---

### `run_ablation.py` — アブレーション実験オーケストレーター

4バリアントの学習・評価を自動管理するPythonスクリプト（`run_ablation.sh` から呼ばれる）。

```bash
python run_ablation.py --dev                   # 4バリアント全学習+評価（開発）
python run_ablation.py --skip_train            # 評価のみ（チェックポイントが存在する場合）
python run_ablation.py --skip_eval             # 学習のみ
python run_ablation.py --base_dir experiments  # 出力ディレクトリ指定
```

---

### `compare_results.py` — アブレーション比較

4バリアントの評価結果を読み込み、比較表とグラフを生成する。

```bash
python compare_results.py --base_dir experiments
```

**生成物：**
- `experiments/ablation_table.tex`：LaTeX形式のアブレーション表
- `experiments/ablation_comparison.png`：バリアント別棒グラフ

---

### `infer.py` — 単一フレーム推論

学習済みモデルで新しいrange imageを超解像する。

```bash
python infer.py --checkpoint checkpoints/best.pt --output_dir infer_output
```

---

## モジュール（`model/`）

| ファイル | クラス | 説明 |
|---------|-------|------|
| `unet.py` | `FlashUNet` | メインモデル。Swin Transformerエンコーダ＋デコーダ。FAA・AMSF・RAFK（オプション）を統合 |
| `faa.py` | `FrequencyAwareAttention` | 周波数認識注意機構。空間ウィンドウMSA + FFTデュアルブランチ。RAFK有効時は距離適応型周波数フィルタに切り替わる |
| `amsf.py` | `AdaptiveMultiScaleFusion` | スキップ接続のアダプティブマルチスケール融合 |
| `swin_block.py` | `SwinStage`, `SwinBlock` | Swin Transformerブロック（シフトウィンドウ注意）|
| `patch_embed.py` | `PatchEmbed`, `PatchMerge`, `PatchExpand` | パッチ埋め込み・縮小（エンコーダ）・拡張（デコーダ）|
| `mkdisc.py` | `MetaKernelDiscriminator` | Meta-Kernel Discriminator。3D球面座標から畳み込み重みを動的生成するPatchGAN形式の識別器。**学習時のみ使用** |
| `loss.py` | — | `masked_l1_loss`, `hinge_loss_disc`, `hinge_loss_gen`, `distance_weighted_adv_loss`, `freq_consistency_loss` |

### FLASH+ モジュールの詳細

**RAFK（Range-Adaptive Frequency Kernel）** — `faa.py` 内の `FrequencyAwareAttention`

FAモジュールの周波数ブランチ（論文式7）を距離適応型に置き換える。`use_rafk=True` 時に有効。
- `Conv_near`：高周波通過フィルタ（密な近距離向け）
- `Conv_far`：低周波通過フィルタ（疎な遠距離向け）
- `MLP_α`：行インデックス・平均距離・有効点率の3特徴量から混合係数を生成
- 推論時にも使用。FPS変化なし。

**MKDisc（Meta-Kernel Discriminator）** — `mkdisc.py`

RangeLDM（arXiv:2403.10094）の式(5)(6)に基づく幾何制約付き識別器。`use_mkdisc=True` 時に有効。
- 球面座標間の相対距離ベクトルからカーネル重みを動的生成
- 距離重み付き敵対的損失 `w(r) = 1 + β·max(0, r − r_near) / (r_far − r_near)` で遠距離を強調
- **推論時には使用しない。**チェックポイントのgenerator側のみを `infer.py`/`evaluate.py` でロード

---

## データ（`data/`）

| ファイル | 説明 |
|---------|------|
| `preprocess.py` | KITTIの生点群（.bin）をrange imageに変換し `kitti_processed/` に保存 |
| `dataset.py` | `RangeImageDataset`：前処理済みnpyファイルを読み込みLR/HRペアを生成。`create_dataloaders(config)` で train/val DataLoaderを返す |
| `projection.py` | 点群↔range imageの相互変換（球面射影） |

---

## 設定（`config/`）

| ファイル | 説明 |
|---------|------|
| `default.py` | `Config` データクラス。全ハイパーパラメータを一元管理 |

**主要パラメータ：**

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `H, W` | 64, 1024 | 出力range imageサイズ |
| `sr_factor` | 4 | アップサンプリング倍率（16→64行） |
| `num_epochs` | 600 | 学習エポック数（dev: 10） |
| `batch_size` | 8 | バッチサイズ（dev: 1） |
| `lr` | 5e-4 | 初期学習率 |
| `warmup_epochs` | 60 | 線形ウォームアップエポック数 |
| `use_rafk` | False | RAFK有効化フラグ |
| `use_mkdisc` | False | MKDisc有効化フラグ |
| `lambda_adv` | 0.1 | 敵対的損失の重み λ₁ |
| `lambda_freq` | 0.01 | 周波数一貫性損失の重み λ₂ |
| `beta_dist_weight` | 2.0 | 距離重みの傾き β（0–30m: 1.0、30–60m: 1.0→3.0） |
| `r_near, r_far` | 30m, 60m | 近距離・遠距離の境界 |

**ファクトリメソッド：**
```python
Config()                              # FLASH (baseline) 学習設定
Config.dev()                          # 開発モード（10エポック、バッチ1）
Config.ablation("baseline")           # アブレーション: FLASH
Config.ablation("rafk")               # アブレーション: FLASH + RAFK
Config.ablation("mkdisc")             # アブレーション: FLASH + MKDisc
Config.ablation("proposed")           # アブレーション: FLASH+（両方）
Config.ablation("proposed", dev=True) # 開発モードと組み合わせ可
```

---

## ユーティリティ（`utils/`）

| ファイル | 主要関数 | 説明 |
|---------|---------|------|
| `metrics.py` | `compute_all_metrics`, `compute_metrics_by_distance`, `compute_mae_by_distance` | MAE・CD・IoU・F1を距離帯別に計算 |
| `reprojection.py` | `range_image_to_points(range_img, mask, config)` | range image → 3D点群への変換 |
| `misc.py` | `get_device()`, `set_seed(seed)` | デバイス選択・乱数固定 |
| `stats.py` | `run_all_tests`, `print_test_results` | アブレーション結果の統計検定 |

---

## アブレーション実験設定

| 設定名 | RAFK | MKDisc | 役割 |
|--------|:----:|:------:|------|
| **baseline** | ✗ | ✗ | FLASH再現（論文Table IIIとの一致確認） |
| **+RAFK** | ✓ | ✗ | RAFKの遠距離スパース適応効果を分離 |
| **+MKDisc** | ✗ | ✓ | 幾何的一貫性損失の効果を分離 |
| **proposed** | ✓ | ✓ | FLASH+（両モジュールの相乗効果） |

---

## 環境・依存ライブラリ

`requirements.txt` の主要ライブラリ：

| ライブラリ | 用途 |
|-----------|------|
| PyTorch + CUDA | モデル学習・推論 |
| `spconv-cu120` | スパース畳み込み |
| `open3d` | 点群処理・可視化 |
| `scipy` | KDTreeによる最近傍探索（CD計算） |
| `kornia` | 画像処理 |
| `tensorboard` | 学習曲線のログ |
| `matplotlib` | 可視化 |
| `numba` | CUDA/JITアクセラレーション |

DevContainerの設定は `/workspaces/toyot/.devcontainer/` を参照。

---

## 設計文書（`docs/`）

| ファイル | 内容 |
|---------|------|
| `Plan.md` | 研究の背景・目的・開発方針（日本語） |
| `Suggestive_model_Plan.md` | FLASH+の詳細設計仕様：RAFK・MKDisc・損失関数・アブレーション設計（日本語） |
