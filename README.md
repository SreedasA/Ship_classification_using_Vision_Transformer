# Ship Classification using Vision Transformer (ViT)

Vision Transformer (ViT)-based deep learning pipeline for maritime ship classification using PyTorch and Hugging Face Transformers.

[![CI](./.github/workflows/ci.yml)](./.github/workflows/ci.yml)

## Author

- **Name**: Sreedas A / Sreedas Anjankudy
- **Email**: `asreedas@gmail.com`
- **GitHub**: https://github.com/SreedasA
- **LinkedIn**: https://www.linkedin.com/in/sreedasa/

## Why this project

Maritime surveillance often requires fast, robust recognition of ship types from RGB imagery. This project explores **Vision Transformers (ViT)** for ship classification, focusing on a clean, reproducible training/evaluation pipeline that can be understood quickly by a reviewer.

## Problem

Given a single RGB maritime ship image, predict one of:

- Cargo
- Military
- Carrier
- Cruise
- Tankers

## Dataset

This repository uses the Analytics Vidhya / Kaggle dataset:
- **Game of Deep Learning: Ship Dataset** (5 classes, optical RGB) — [Kaggle dataset page](https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets)

The dataset is **not included in this repository** due to licensing/ownership.

### Expected folder layout

```text
dataset/
├── train/
│   ├── images/
│   └── train.csv
└── test/
    └── images/
```

Update the dataset root in `configs/default.yaml` if your paths differ.

## Results

On the Kaggle dataset above, the final ViT model achieved approximately:

- **Accuracy**: ~98.25%
- **F1 (macro)**: ~0.9826

Hardware used for the run:
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4GB)
- CUDA: enabled
- cuDNN: 8700

## What’s inside

- **Training**: Hugging Face `Trainer` + ViT image classification head
- **Evaluation**: metrics + confusion matrix export
- **Inference**: single-image inference with confidence scores
- **Reproducibility**: config-driven paths in `configs/default.yaml` (no hardcoded local paths)

## Quickstart

### 1) Create environment + install

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Download dataset

Download the dataset from Kaggle and place it in the `dataset/` folder as shown above.

### 3) Train

```bash
python -m ship_vit.train --config configs/default.yaml
```

Artifacts are written to `runs/altered_vision_transformer/` (configurable).

### 4) Evaluate (confusion matrix)

```bash
python -m ship_vit.evaluate --config configs/default.yaml --model-dir runs/altered_vision_transformer
```

### 5) Inference on one image

```bash
python -m ship_vit.predict --model-dir runs/altered_vision_transformer --image path/to/image.jpg
```

Or run the minimal example script:

```bash
python scripts/inference_example.py --model-dir runs/altered_vision_transformer --image path/to/image.jpg --top-k 3
```

## Recruiter notes

- **What’s intentionally excluded from git**: dataset, logs, runs, checkpoints, large weights (see `.gitignore`)
- **Notebook**: the original notebook is kept under `notebooks/` for transparency; the runnable pipeline is in `src/`

## Repository structure

```text
configs/                 # YAML configs (paths + hyperparameters)
notebooks/               # Original exploration notebook
scripts/                 # Small runnable examples
src/ship_vit/            # Reusable training / eval / inference code
```

## Notes

- This is **classification**, not object detection.
- All code uses **relative paths** (config-driven) so it runs on any OS.

## Acknowledgements

- Hugging Face Transformers / Datasets / Evaluate
- Analytics Vidhya “Game of Deep Learning: Ship” dataset
