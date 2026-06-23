# Enhancing Image Super-Resolution

A modular 4× image super-resolution system built on top of [RealESRGAN](https://github.com/xinntao/Real-ESRGAN), extended with **Channel Attention (CA)** modules and a choice of two discriminator architectures. The project includes a full training pipeline, a batch inference script, and a React + FastAPI web demo for interactive comparison of model variants.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Web Demo](#web-demo)

---

## Overview

This project fine-tunes a pretrained RealESRGAN generator (RRDBNet) with the following enhancements:

- **Squeeze-and-Excitation Channel Attention** injected after the trunk and each upsampling stage of the generator, allowing the network to selectively weight feature channels.
- **Two swappable discriminators**: a PatchGAN critic with spectral normalization, and a U-Net discriminator with skip connections and spectral normalization.
- A **two-phase training schedule**: pixel + perceptual warm-up followed by full GAN fine-tuning.
- A **React frontend + FastAPI backend** for uploading low-resolution images and comparing the output of different trained checkpoints side-by-side.

---

## Features

- 4× upscaling (LR → HR)
- Channel Attention on generator trunk, upsampling stage 1, and upsampling stage 2 (each individually toggleable)
- Relativistic GAN loss (RaGAN)
- VGG19 perceptual loss (feature layer 35)
- EMA (Exponential Moving Average) of generator weights
- Mixed-precision training (AMP) on CUDA
- Gradient accumulation
- TensorBoard logging of losses, metrics, and sample images
- Validation metrics: PSNR, SSIM, LPIPS
- Checkpoint resume support
- Interactive web demo with slider / side-by-side comparison view and real-time inference metrics

---

## Project Structure

```
Enhancing-Image-Super-Resolution/
│
├── RealESRGAN/                    # Core ML package
│   ├── __init__.py
│   ├── config.py                  # Dataclass-based config (all hyperparameters)
│   ├── data.py                    # Paired HR/LR dataset + dataloaders
│   ├── losses.py                  # L1, Perceptual (VGG19), Relativistic GAN losses
│   ├── metrics.py                 # PSNR, SSIM, LPIPS utilities
│   ├── trainer.py                 # RealESRGANFineTunerCA – main training loop
│   ├── train.py                   # Entry-point: assembles and runs training pipeline
│   ├── infer.py                   # Batch inference over a folder of LR images
│   ├── utils.py                   # Device detection, seed, AMP context
│   └── models/
│       ├── __init__.py
│       ├── attention.py           # ChannelAttention (SE-style)
│       ├── blocks.py              # ResidualDenseBlock, RRDB, CA variants
│       ├── generator.py           # RRDBNetCA – CA-enhanced generator
│       └── discriminator.py       # PatchGANDiscriminator, UNetDiscriminatorSN
│
├── Website/                       # Web demo
│   ├── main.py                    # FastAPI backend (model loading, /api/enhance)
│   ├── config.py                  # Backend model path config
│   ├── requirements-backend.txt
│   └── src/
│       ├── App.tsx                # Main React app
│       ├── api/enhanceApi.ts      # API client (enhance, health, model select)
│       ├── components/
│       │   ├── ComparisonViewer.tsx
│       │   ├── ControlsPanel.tsx
│       │   ├── MetricsPanel.tsx
│       │   └── StageTimeline.tsx
│       └── types/sr.ts
│
├── Results/
│   ├── Low Resolution Image/      # Input LR images used for evaluation
│   └── Enhanced Image/
│       ├── PatchGAN as Discriminator with Channel Attention enhanced Generator/
│       ├── PatchGAN as Discriminator without Channel Attention enhanced Generator/
│       └── UNet as Discriminator with Channel Attention enhanced Generator/
│
├── Single training code.ipynb     # Standalone Jupyter notebook for training
└── requirements.txt
```

---

## Architecture

### Generator — `RRDBNetCA`

Builds on the standard RRDBNet backbone:

```
Input (LR)
  └─ conv_first
      └─ 23× RRDB trunk  →  conv_body  →  residual add
          └─ [Optional] ChannelAttention (after trunk)
              └─ 2× Nearest-neighbour upsample + conv
                  └─ [Optional] ChannelAttention (after each upsample stage)
                      └─ conv_hr → conv_last → Output (SR ×4)
```

Each **RRDB** block stacks three **Residual Dense Blocks (RDB)**, each with a five-layer dense connection and a 0.2 residual scaling factor. The optional CA variant (`ResidualDenseBlockCA`) applies Squeeze-and-Excitation attention inside each RDB.

### Discriminators

| Variant | Architecture | Normalization |
|---|---|---|
| `patchgan` | 4-layer strided conv → patch logits | Spectral Norm |
| `unet` | Encoder–decoder with skip connections, full-resolution logits | Spectral Norm |

Select via `Config.discriminator_type = "patchgan"` or `"unet"`.

### Loss Function

```
L_total = λ_pixel × L1(SR, HR)
        + λ_perceptual × L1(VGG(SR), VGG(HR))
        + λ_gan × RaGAN(D_real, D_fake)      # activated after gan_start_epoch
```

Default weights: `λ_pixel=1.0`, `λ_perceptual=0.8`, `λ_gan=0.1`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Enhancing-Image-Super-Resolution.git
cd Enhancing-Image-Super-Resolution
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> Requires Python ≥ 3.9 and PyTorch ≥ 2.0 with CUDA for training.

### 3. Download a pretrained RealESRGAN checkpoint

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

Set `Config.pretrained_model_path` to this path before training.

---

## Usage

### Training

Edit `Config` in `RealESRGAN/config.py` (or pass a config object) then call `run_training`:

```python
from RealESRGAN.config import Config
from RealESRGAN.train import run_training

cfg = Config(
    pretrained_model_path="RealESRGAN_x4plus.pth",
    train_hr_dir="data/train/HR",
    train_lr_dir="data/train/LR",
    val_hr_dir="data/val/HR",
    val_lr_dir="data/val/LR",
    discriminator_type="patchgan",   # or "unet"
    use_ca_after_trunk=True,
    use_ca_after_up1=True,
    use_ca_after_up2=True,
    total_epochs=18,
    gan_start_epoch=10,
    batch_size=8,
)

trainer = run_training(cfg)
```

Or run the provided **Jupyter notebook**: `Single training code.ipynb`.

TensorBoard logs are written to `cfg.log_dir` (`experiments/logs` by default):

```bash
tensorboard --logdir experiments/logs
```

**Resuming from a checkpoint:**

```python
cfg.resume_checkpoint_path = "experiments/checkpoints/checkpoint_epoch_10.pth"
run_training(cfg)
```

---

### Inference

```python
from RealESRGAN.config import Config
from RealESRGAN.infer import run_inference_folder

cfg = Config(
    num_feat=64,
    num_block=23,
    scale_factor=4,
    use_ca_after_trunk=True,
    use_ca_after_up1=True,
    use_ca_after_up2=True,
)

saved = run_inference_folder(
    cfg=cfg,
    weight_path="experiments/checkpoints/best_perceptual_model.pth",
    input_dir="path/to/lr_images",
    output_dir="path/to/output",
)
print(f"Saved {len(saved)} images")
```

Supported input formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

---

### Web Demo

The web demo consists of a **FastAPI backend** (`Website/main.py`) and a **React + Vite frontend** (`Website/src/`).

#### 1. Configure model paths

The backend loads models from `Website/main.py` via the `MODEL_OPTIONS` list. Paths use `Path(__file__).resolve().parents[N]` — where `parents[2]` is two levels above `Website/` (i.e. the grandparent of the repo root) and `parents[1]` is one level above `Website/` (the repo root itself).

Place your checkpoints to match this layout, or edit the paths directly in `main.py`:

```
<repo-root>/
├── Website/
│   └── main.py
├── RealESRGAN_x4plus.pth               ← parents[1] / RealESRGAN_x4plus.pth
└── Model/
    ├── PatchGAN with CA/
    │   └── best_perceptual_model.pth   ← parents[2] / Model/PatchGAN with CA/...
    ├── PatchGAN without CA/
    │   └── best_perceptual_model.pth
    └── Unet with CA/
        └── best_perceptual_model.pth
```

Any entry in `MODEL_OPTIONS` whose file does not exist on disk is silently skipped at startup.

#### 2. Start the backend

```bash
cd Website
pip install -r requirements-backend.txt
uvicorn main:app --reload --port 8000
```

#### 3. Start the frontend

```bash
cd Website
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

**Features of the web demo:**
- Upload any image for 4× super-resolution
- Switch between multiple trained model checkpoints at runtime
- Three comparison modes: single, slider, and side-by-side
- Displays inference time and image resolution
- Workflow stage timeline
