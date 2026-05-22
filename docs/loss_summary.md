# 損失関数仕様書：FLASH / FLASH+RAFK

対象ファイル：`model/loss.py`、`train.py`、`model/faa.py`、`model/swin_block.py`

---

## 1. FLASH（Baseline）

### 1.1 概要

FLASHの学習は **マスク付きL1損失のみ** を使用する。

$$
\mathcal{L} = \mathcal{L}_{L1}
$$

### 1.2 マスク付きL1損失（`masked_l1_loss`）

**定義：**

$$
\mathcal{L}_{L1} = \frac{\sum_{i} \left| \hat{I}_h^{(i)} - I_h^{(i)} \right| \cdot m^{(i)}}{\max\!\left(\sum_{i} m^{(i)},\ 1\right)}
$$

| 記号 | 内容 |
|------|------|
| $\hat{I}_h$ | 予測range image `(B, 1, H, W)` |
| $I_h$ | GT range image `(B, 1, H, W)` |
| $m$ | 有効ピクセルマスク `(B, 1, H, W)`（有効点=1、無効点=0） |
| $i$ | 全ピクセルにわたるインデックス |

**処理の流れ：**

```
pred   (B, 1, H, W)  ← FlashUNet出力
target (B, 1, H, W)  ← GT range image（log(r+1)圧縮済み）
mask   (B, 1, H, W)  ← 有効点マスク

差分 = |pred - target| * mask       # 無効点は損失に寄与しない
loss = 差分の合計 / max(有効ピクセル数, 1)
```

実装参照：`model/loss.py:7-18`

**マスクの意味：**

- KITTIのrange imageは返点がない画素が`0`で埋まっている。
- `mask = (target > 0).float()` として構築され、これらの補間アーティファクトを損失から除外する。

### 1.3 オプティマイザ設定

| 設定 | 値 |
|------|----|
| オプティマイザ | AdamW |
| 学習率 | 5e-4（ウォームアップ+コサインアニーリング with warm restart） |
| weight_decay | 0.01 |
| 勾配クリッピング | max_norm = 1.0 |
| 混合精度 | float16（AMP） |

**学習率スケジュール：**

- ウォームアップ期（0 〜 `warmup_epochs=30` epoch）：線形増加
- 以降：コサインアニーリング（周期 `restart_period=85`、周期ごとにピーク学習率を `lr_decay_per_cycle=0.7` 倍に減衰）

実装参照：`train.py:22-37`

---

## 2. FLASH + RAFK

### 2.1 概要

RAFKを有効にすると、L1損失に加えて **周波数一貫性正則化** が追加される。

$$
\mathcal{L} = \mathcal{L}_{L1} + \lambda_{\text{freq}} \cdot \mathcal{L}_{\text{freq\_consistency}}
$$

| パラメータ | デフォルト値 |
|-----------|------------|
| $\lambda_{\text{freq}}$ | 0.01 |

実装参照：`train.py:88`

### 2.2 RAFK モジュールの構造（損失の前提）

`L_freq_consistency` はRAFKの `Conv_near` と `Conv_far` のカーネル重みを対象とするため、まずRAFKの動作を整理する。

#### FAモジュール（FLASH論文 式(6)〜(9)）

FLASHオリジナルのFFTブランチ（`use_rafk=False`）：

```
式(6): X_mean   = mean(X, dim=C)                    # (B_, wh, ww)    チャネル平均
式(6): X_fft    = rfft2(X_mean, norm="ortho")        # (B_, wh, ww//2+1) 複素数
式(7): amp      = |X_fft|                            # 振幅スペクトル (B_, 1, wh, ww//2+1)
式(7): F_attn   = σ(Conv(amp))                       # 周波数マスク (B_, wh, ww//2+1)
式(8): X_fft_out = X_fft ⊙ F_attn                   # マスク適用
式(8): X_spatial = irfft2(X_fft_out, s=(wh,ww))     # 逆変換 (B_, wh, ww)
式(9): Output   = α·X_freq_out + (1-α)·X_spatial_attn  # 空間ブランチと融合
```

#### RAFKによる式(7)の置き換え（`use_rafk=True`）

固定フィルタの`Conv`を**近距離用/遠距離用の2つのConvのソフト混合**に置き換える（`faa.py:113-118`）：

$$
F_{\text{attn}} = (1 - \alpha_{\text{blend}}) \cdot \sigma(\text{Conv}_{\text{near}}(|X_{\text{freq}}^w|))
               + \alpha_{\text{blend}} \cdot \sigma(\text{Conv}_{\text{far}}(|X_{\text{freq}}^w|))
$$

**混合係数 $\alpha_{\text{blend}}$ の生成（MLP_α）：**

各ウィンドウ $w$ の3次元特徴量をMLPに入力する（`swin_block.py:115-158`）：

$$
\mathbf{f}_w = \left[\ \frac{v_w}{H},\quad \frac{\bar{r}_w}{r_{\max}},\quad \frac{n_{\text{valid}}}{n_{\text{total}}}\ \right]
$$

| 特徴量 | 式 | 意味 |
|--------|-----|------|
| $v_w / H$ | `center_row` | ウィンドウ中心行インデックス（安定した距離代理変数） |
| $\bar{r}_w / r_{\max}$ | `mean_range` | ウィンドウ内の加重平均距離（正規化）。`log1p(80) ≈ 4.4` で割る |
| $n_{\text{valid}} / n_{\text{total}}$ | `valid_ratio` | 有効点率（上2特徴の信頼度） |

$$
\bar{r}_w = \frac{\sum_i r_i \cdot m_i}{\max(\sum_i m_i, 1)}, \quad
\alpha_{\text{blend}} = \text{MLP}_\alpha(\mathbf{f}_w) = \text{Sigmoid}(W_2 \cdot \text{ReLU}(W_1 \mathbf{f}_w))
$$

- $W_1$：Linear(3→16)、$W_2$：Linear(16→1)
- $\alpha_{\text{blend}} \to 0$：`Conv_near` 支配（高周波通過・近距離向け）
- $\alpha_{\text{blend}} \to 1$：`Conv_far` 支配（低周波通過・遠距離向け）

実装参照：`faa.py:50-58`（アーキテクチャ定義）、`faa.py:113-118`（forward計算）

> **注意：** コード上では `alpha_b=0` が `Conv_near` 支配（`faa.py:118`：`(1-alpha_b)*F_near + alpha_b*F_far`）。プランドキュメント（近距離で $\alpha=1$）と符号が逆転しているが、MLPが学習で自己調整するため機能的には同等。

### 2.3 周波数一貫性正則化（`freq_consistency_loss`）

**目的：** `Conv_near` と `Conv_far` が同一フィルタに収束することを防ぎ、実際に異なる周波数特性（高周波通過・低周波通過）を学習することを保証する。

**定義：**

$$
\mathcal{L}_{\text{freq\_consistency}} = -\frac{1}{N_{\text{pairs}}} \sum_{k=1}^{N_{\text{pairs}}} \left\| W_{\text{near}}^{(k)} - W_{\text{far}}^{(k)} \right\|_F
$$

| 記号 | 内容 |
|------|------|
| $W_{\text{near}}^{(k)}$ | $k$ 番目のFAモジュールの `Conv_near` の重み `(1, 1, 3, 3)` |
| $W_{\text{far}}^{(k)}$ | $k$ 番目のFAモジュールの `Conv_far` の重み `(1, 1, 3, 3)` |
| $\|\cdot\|_F$ | フロベニウスノルム |
| $N_{\text{pairs}}$ | 全SwinブロックのRAFK層のペアの総数 |

**負符号の意味：**

$\mathcal{L}_{\text{freq\_consistency}}$ を最小化すること（勾配降下）は、$\|W_{\text{near}} - W_{\text{far}}\|_F$ を**最大化**することに等しい。
つまり、2つのカーネルが互いに**異なる**方向へと学習を促進される。

**処理の流れ：**

```python
# train.py:84-86
pairs = model.get_rafk_weight_pairs()          # 全SwinブロックのConv重みペアを収集
loss_freq = freq_consistency_loss(pairs)        # -mean(||W_near - W_far||_F)

# モデル構造上のペア数
# encoder_stages: depths=(2,2,6,2) → 各stageのblockごとに1ペア
# decoder_stages: depths=(2,2,2) → 同様
# 合計 N_pairs = (2+2+6+2) + (2+2+2) = 12 + 6 = 18 ペア
```

実装参照：`model/loss.py:66-82`、`model/faa.py:149-153`、`model/unet.py:140-148`

**安定性の注意：**

- `weight_pairs` が空（RAFK無効）の場合、`torch.tensor(0.0)` を返す（デバイスがCPU固定になる点に注意）。
- `L_freq_consistency` がNaNになる場合、`Conv_near`・`Conv_far` にグラジェントクリッピング `max_norm=1.0` を適用すること（仕様書 §5.1 参照）。

### 2.4 全体損失の計算フロー（FLASH+RAFK）

```
1. FlashUNet.forward(inp) → pred (B, 1, 64, 1024)

2. loss_l1 = masked_l1_loss(pred, target, mask)
   ↑ 有効ピクセルのみL1誤差を平均

3. pairs = model.get_rafk_weight_pairs()
   ↑ 全FAモジュールの (conv_near.weight, conv_far.weight) を収集

4. loss_freq = freq_consistency_loss(pairs)
   ↑ -mean(||W_near - W_far||_F) を計算

5. loss_total = loss_l1 + lambda_freq * loss_freq
             = loss_l1 + 0.01 * loss_freq

6. loss_total.backward() → optimizer.step()
   ↑ Generator(FlashUNet)の全パラメータを一括更新
```

### 2.5 ハイパーパラメータ一覧

| パラメータ | デフォルト値 | 調整の目安 |
|-----------|------------|-----------|
| `lambda_freq` ($\lambda_{\text{freq}}$) | 0.01 | `Conv_near`と`Conv_far`が収束する場合は0.05に増やす |
| `rafk_mlp_hidden` | 16 | MLP_αの中間層ユニット数（追加パラメータ65個） |
| `window_size` | (2, 8) | ウィンドウごとの `n_total = 2×8 = 16` ピクセル |
| `r_max` | 80.0 m（config）| 特徴量正規化に使用（`log1p(80)≈4.4`） |

---

## 3. 両モデルの比較

| 損失項 | FLASH（baseline） | FLASH+RAFK |
|--------|:-----------------:|:----------:|
| $\mathcal{L}_{L1}$ | ✓ | ✓ |
| $\mathcal{L}_{\text{freq\_consistency}}$ | ✗ | ✓（$\lambda=0.01$） |
| $\mathcal{L}_{\text{adv}}$（MKDisc） | ✗ | ✗ |
| Discriminator | なし | なし |
| オプティマイザ数 | 1（Generator） | 1（Generator） |

---

## 4. 参照コード一覧

| 処理 | ファイル | 行番号 |
|------|---------|--------|
| `masked_l1_loss` 定義 | `model/loss.py` | 7–18 |
| `freq_consistency_loss` 定義 | `model/loss.py` | 66–82 |
| 損失の合算・backward | `train.py` | 83–95 |
| RAFKの `fft_branch` 実装 | `model/faa.py` | 95–128 |
| ウィンドウ特徴量 `_compute_window_feats` | `model/swin_block.py` | 115–158 |
| 重みペア収集 `get_rafk_weight_pairs` | `model/unet.py` | 140–148 |
| Config デフォルト値 | `config/default.py` | 1–135 |
