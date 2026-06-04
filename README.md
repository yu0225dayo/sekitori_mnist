# Sekitori MNIST — 席取り損失による VAE 学習

MNIST を題材に、**席取り（Optimal Assignment）損失**を用いた Variational Autoencoder (VAE) の実装です。

## 概要

通常の VAE では各入力に対して1つの再構成を学習しますが、このプロジェクトでは：

1. **複数サンプリング**: 1つの入力から N 個の潜在ベクトルを reparameterize でサンプリング
2. **関連教師**: 各数字 `d` に対して `d-1`, `d`, `d+1` の3クラスを教師として用意
3. **席取り割り当て**: N 個の再構成を教師 T 個へ重複なく最適割り当て（Hungarian 的な貪欲法）
4. **損失最小化**: 割り当て後の MSE 平均を最小化

この「席取り」方式により、潜在空間がマルチモーダルな構造を自然に学習することを目指しています。

## ファイル構成

```
.
├── config.py          # ハイパーパラメータ・データローダー設定
├── model.py           # VAE モデル定義 (CNN-based & FC-based)
├── loss.py            # 席取り損失の実装 (sekitori_loss_sum)
├── data_utils.py      # 関連数字サンプリングユーティリティ
├── train_random.py    # 学習スクリプト（教師バッチごとランダムサンプリング版）
├── train_fixed.py     # 学習スクリプト（固定教師版）
├── visualize.py       # 潜在空間・再構成の可視化
├── vis.py             # 補助可視化スクリプト
└── check_teachers.py  # 教師サンプル確認スクリプト
```

## モデルアーキテクチャ

### VAE (CNN)
- **Encoder**: Conv2d(1→32) → Conv2d(32→64) → Linear → μ/logvar
- **Decoder**: Linear → ConvTranspose2d(64→32) → ConvTranspose2d(32→1)
- 潜在次元: `z_dim=4`（デフォルト）

### VAE_fc (全結合)
- **Encoder**: Linear(784→400→200) → μ/logvar
- **Decoder**: Linear(z→200→400→784)

## 損失関数

```
Total Loss = SekitoriMSE + β × KLD
```

- **SekitoriMSE**: N 個の再構成を T 個の教師へ席取り割り当てした後の平均 MSE
- **KLD**: KL ダイバージェンス正則化
- `β = 0.1`（デフォルト）

## セットアップ

```bash
pip install torch torchvision tensorboard numpy matplotlib
```

## 実行方法

```bash
# ランダム教師版（バッチごとに教師画像をランダムサンプリング）
python train_random.py

# 固定教師版（数字ごとに1枚固定）
python train_fixed.py

# TensorBoard で学習曲線を確認
tensorboard --logdir runs
```

## ハイパーパラメータ（config.py）

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `latent_dim` | 4 | 潜在空間の次元数 |
| `epochs` | 1 | 学習エポック数 |
| `lr` | 1e-4 | Adam 学習率 |
| `num_samples` | 99 | reparameterize サンプル数（T=3 の倍数） |
| `beta` | 0.1 | KLD の重み |
| `TRAIN_PER_DIGIT` | 50 | 数字ごとの学習サンプル数 |
| `VAL_PER_DIGIT` | 10 | 数字ごとの検証サンプル数 |
| `TRACK_DIGIT` | 5 | 潜在空間追跡ターゲット数字 |

## 依存ライブラリ

- PyTorch
- torchvision
- TensorBoard
- NumPy
- Matplotlib
