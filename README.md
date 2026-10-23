# Ship Classification using Vision Transformer

Fine-tuning a Vision Transformer (ViT-Base/16) to classify ship types in optical maritime images, built with PyTorch and Hugging Face Transformers.

**Published work:** this classifier is part of the research behind the IEEE paper [Deep Learning for Image Recognition and Classification: A Comparative Study on Optical and ISAR Datasets for Maritime and Automotive Applications](https://ieeexplore.ieee.org/abstract/document/11035465) (ICKECS 2025, DOI 10.1109/ICKECS65700.2025.11035465). This repository covers the optical ship classification part only.

[![CI](https://github.com/SreedasA/Ship_classification_using_Vision_Transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/SreedasA/Ship_classification_using_Vision_Transformer/actions/workflows/ci.yml)

## Background

I built this as part of my BCA at Bengaluru City University and my research internship at DRDO CAIR (Bengaluru), working with Smita Srivastava. The original experiment is the notebook `Altered_Vision.ipynb` (kept with its outputs); the `src/ship_vit/` package is a cleaned-up, config-driven version of the same pipeline.

## Problem

Given one RGB ship image, predict one of five classes: Cargo, Carrier, Cruise, Military, Tankers.

## Dataset

[Game of Deep Learning: Ship Dataset](https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets) (Analytics Vidhya / Kaggle), 5 classes, optical RGB. It is not included in this repo.

- The notebook found 8,932 image files, of which 6,252 have labels in `train.csv` and were used.
- Images are small (the samples shown in the notebook are about 210 px wide) and are resized to 224 x 224 for the model.
- The classes are imbalanced, so the notebook applies random oversampling (see the caveat below).

Expected layout:

```text
dataset/
├── train/
│   ├── images/
│   └── train.csv
└── test/
    └── images/
```

Change the paths in `configs/default.yaml` if yours differ.

## Method

| Item | Value |
|---|---|
| Model | `google/vit-base-patch16-224-in21k`, 85.8M trainable parameters |
| Input | 224 x 224, ImageNet-style normalisation |
| Augmentation (train) | random rotation (45 degrees), random sharpness, horizontal flip |
| Optimiser setup | learning rate 1e-4, weight decay 0.01, 10 warm-up steps, linear schedule |
| Batch | 4 per step with 4 gradient-accumulation steps (effective 16), fp16 |
| Epochs | 5 |
| Split | stratified 80 / 20 |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB |

## Results (from the notebook run)

On the 20 percent held-out split (2,120 images, 424 per class):

| Metric | Value |
|---|---|
| Accuracy | 0.9825 |
| Macro F1 | 0.9826 |
| Loss | 0.0653 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Cargo | 0.9598 | 0.9575 | 0.9587 |
| Carrier | 1.0000 | 0.9976 | 0.9988 |
| Cruise | 0.9953 | 1.0000 | 0.9976 |
| Military | 1.0000 | 0.9953 | 0.9976 |
| Tankers | 0.9577 | 0.9623 | 0.9600 |

Cargo and Tankers are the weakest classes; the other three are above 0.99 F1.

### How to read these numbers

I would rather flag the limits of this result myself than have a reviewer find them:

- **Oversampling happened before the train/test split.** The notebook oversamples the 6,252 labelled images to 10,600 (2,120 per class) and then splits. Duplicated images can therefore appear in both the training and test sets, which makes the score optimistic. A leak-free setup would split first and oversample only the training set.
- **The run started from an already fine-tuned checkpoint.** The notebook loads a local model directory, and evaluating it before the training cell already gave 0.9887 accuracy. The five epochs recorded here ended at 0.9825.
- **The held-out split was also used to select the best checkpoint** (`load_best_model_at_end`), so there is no separate untouched test set.
- **The headline figures come from the notebook, not from re-running `src/`.** The package does not apply oversampling, so its numbers will differ.

## Repository layout

| Path | Contents |
|---|---|
| `Altered_Vision.ipynb` | Original experiment with outputs: data prep, oversampling, training, evaluation, single-image inference |
| `src/ship_vit/` | Config-driven training, evaluation (confusion matrix) and inference code |
| `configs/default.yaml` | Dataset paths, label map and hyperparameters |
| `scripts/inference_example.py` | Minimal single-image example |
| `CITATION.cff` | How to cite the paper |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m ship_vit.train --config configs/default.yaml
python -m ship_vit.evaluate --config configs/default.yaml --model-dir runs/altered_vision_transformer
python -m ship_vit.predict --model-dir runs/altered_vision_transformer --image path/to/image.jpg --top-k 3
```

The code uses the `evaluation_strategy` argument, so it needs a Transformers release older than 4.46 (pinned in `pyproject.toml`).

## Possible next steps

- Split before oversampling and report results on an untouched test set
- Use class weights or augmentation instead of duplicating images
- Add the ISAR experiments from the paper as a second pipeline
- Publish the trained weights as a release

## Citation

Anjankudy, S., Upparapalli, R., Srivastava, S. (2025). Deep Learning for Image Recognition and Classification: A Comparative Study on Optical and ISAR Datasets for Maritime and Automotive Applications. 2025 International Conference on Knowledge Engineering and Communication Systems (ICKECS). DOI: 10.1109/ICKECS65700.2025.11035465

## Author

Sreedas Anjankudy, [GitHub](https://github.com/SreedasA), [LinkedIn](https://www.linkedin.com/in/sreedasa/)

## Acknowledgements

Hugging Face Transformers, Datasets and Evaluate; the Analytics Vidhya "Game of Deep Learning: Ship" dataset.
