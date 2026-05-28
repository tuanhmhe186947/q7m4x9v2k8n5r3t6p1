# Pig Behavior Classification

Research pipeline for pig behavior classification from cropped video frames,
bounding boxes, and tabular context features. The production path is packaged
under `src/pig_behavior`; exploratory notebooks are kept separately under
`notebooks/`.

## Repository Layout

```text
.
├── data/
│   ├── annotations/        # Scene masks and object annotations
│   ├── processed/          # Tabular labels and engineered features
│   └── raw/                # Local raw images, ignored by Git
├── docs/                   # Reproducibility and release notes
├── models/                 # Local model weights, ignored by Git
├── notebooks/              # Archived experiment notebooks
├── src/pig_behavior/       # Installable Python package
├── tests/                  # Lightweight tests
├── tools/                  # Maintenance utilities
├── main.py                 # Compatibility wrapper
└── pyproject.toml
```

Large image folders and model weights are intentionally ignored. Keep them in
the documented local paths, or publish them through a dataset registry, Git LFS,
DVC, Zenodo, OSF, Kaggle, Hugging Face Datasets, or a GitHub Release.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

For notebook, tracking, and development tools:

```bash
pip install -r requirements-dev.txt
```

## Data Contract

The default training CSV is:

```text
data/processed/behavior_with_feats_rectROI.csv
```

Training images are expected at:

```text
data/raw/images_clean/
```

The CSV must include:

- image and bounding box columns: `img_name`, `x1`, `y1`, `x2`, `y2`
- labels: `behavior`, `behavior_coarse`
- split and filtering columns: `group_id`, `hidden`
- tabular features: `in_feeder`, `in_drinker`, `in_toy`, `speed_feat`,
  `min_dist_other`, `num_close_other`

Splits are grouped by `group_id` to reduce leakage from the same frame burst.

## Usage

Quick smoke test:

```bash
pig-behavior --mode train --dry-run
```

Or without installing the console script:

```bash
python main.py --mode train --dry-run
```

Full training:

```bash
pig-behavior --mode train
```

Train with custom data paths:

```bash
pig-behavior --mode train ^
  --csv-path path\to\behavior_with_feats_rectROI.csv ^
  --images-dir path\to\images
```

Export trained Keras checkpoints to TFLite:

```bash
pig-behavior --mode export
```

Hybrid inference:

```bash
pig-behavior --mode infer ^
  --image data\raw\images_clean\example.jpg ^
  --bbox 10 20 200 240 ^
  --tabular 1 0 0 0.1 0.2 1
```

Image-only inference:

```bash
pig-behavior --mode infer --image data\raw\images_clean\example.jpg --image-only
```

## Training Flow

```text
pig_behavior.cli
  -> pig_behavior.config.TrainConfig
  -> pig_behavior.data_loader.build_datasets()
  -> pig_behavior.train.train()
  -> pig_behavior.model.build_model()
```

Data preparation before training:

1. Load the processed CSV.
2. Validate required columns.
3. Remove hidden pigs where `hidden` is `Yes`.
4. Encode fine or coarse behavior labels.
5. Split train, validation, and test sets by `group_id`.
6. Validate referenced image files exist.
7. Crop each pig using `x1`, `y1`, `x2`, `y2`.
8. Resize crops to `224x224`.
9. Normalize pixels to `[0, 1]`.
10. Apply lightweight augmentation to training samples only.

## Notebooks

Notebooks are retained as experiment history, not the primary production API.
See `notebooks/README.md` for the workflow index.

## Quality Checks

```bash
ruff check src main.py tools tests
pytest -q
```

## Release Notes

Before publishing publicly, choose explicit licenses for code and data, and add
a citation file with the correct author metadata. See
`docs/reproducibility.md`, `docs/DATASET_CARD.md`, and `docs/MODEL_CARD.md`.
