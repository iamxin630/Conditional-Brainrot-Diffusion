# HW6 — Brainrot Image Generation

## 模型架構

Conditional DDPM with UNet backbone，從頭訓練（no pre-trained weights）。

### 核心設計
- **UNet**: Encoder-Decoder with skip connections, AdaGN (Adaptive Group Norm with scale-shift)
- **Condition**: Animal embedding (11 classes) + Object embedding (11 classes) + Pair embedding (101 classes)
- **Noise Schedule**: Cosine schedule (Improved DDPM)
- **Training**: Epsilon prediction with MSE loss
- **Sampling**: DDIM (deterministic) with Classifier-Free Guidance (CFG)
- **EMA**: Exponential Moving Average (decay=0.9999) for stable generation

### 參數量
約 62M 參數（與 baseline 對齊）。

## 環境設定

### 建立 Conda 環境
```bash
conda create -n AI python=3.10
conda activate AI
pip install -r requirements.txt
```

### 必要套件
見 `requirements.txt`。

## 訓練

### 基本訓練指令
```bash
conda activate AI
python scripts/train.py --config configs/default.json
```

### 從 checkpoint 恢復訓練
```bash
python scripts/train.py --config configs/default.json --resume checkpoints/latest.pth
```

### 訓練超參數
| 參數 | 值 |
|------|------|
| max_steps | 150,000 |
| batch_size | 128 |
| learning_rate | 2e-4 |
| optimizer | AdamW |
| EMA decay | 0.9999 |
| mixed precision | bf16 |
| gradient clip | 1.0 |
| warmup steps | 5,000 |
| CFG dropout | 10% |

### 訓練輸出
- `checkpoints/`: 定期存儲的 checkpoint
- `samples/`: 訓練過程預覽圖（10×10 grid, 每個 pair 一張）
- `runs/hw6/`: 訓練日誌（JSON Lines 格式）
- `model.pth`: 最終模型權重

## 生成圖片

### 基本生成指令
```bash
python scripts/generate.py --ckpt model.pth --output_dir generated_images
```

### 調整 CFG 和步數
```bash
# 較低 CFG（可能 FID 較好）
python scripts/generate.py --ckpt model.pth --steps 50 --cfg_scale 1.5

# 平衡設定
python scripts/generate.py --ckpt model.pth --steps 100 --cfg_scale 2.0

# 較高 CFG（可能 CLIP-T 較好）
python scripts/generate.py --ckpt model.pth --steps 100 --cfg_scale 3.0
```

### 生成輸出
- `generated_images/`: 2,000 張 64×64 PNG 圖片
- 檔名格式: `000001.png` ~ `002000.png`
- 每個 animal-object pair 各 20 張

## 本地評分

```bash
# 將生成圖片複製到評分目錄
cp generated_images/*.png scoring_program/input/res/

# 執行評分
cd scoring_program
python score.py \
    --input_dir ./input \
    --output_dir ./ \
    --image_size 64 \
    --num_images 2000 \
    --test_json test.json \
    --scores fid clip_t \
    --verbose
```

## 檔案結構

```
HW6/
├── configs/
│   └── default.json         # 模型與訓練設定
├── data/
│   ├── train.csv            # 訓練標籤
│   ├── generate.csv         # 生成目標
│   └── images/              # 訓練圖片
├── scripts/
│   ├── dataset.py           # 資料集讀取
│   ├── diffusion.py         # DDPM/DDIM 實作
│   ├── train.py             # 訓練腳本
│   ├── generate.py          # 圖片生成
│   ├── utils.py             # EMA、checkpoint、日誌
│   └── models/
│       └── unet.py          # Conditional UNet
├── generated_images/        # 生成結果
├── model.pth                # 模型權重
├── requirements.txt         # 套件清單
└── README.md                # 本文件
```
