# Data Directory

This directory separates publishable metadata from large local research data.

- `raw/`: raw frames or extracted images. Ignored by Git.
- `processed/`: tabular labels, bounding boxes, and engineered features.
- `annotations/`: scene-level masks and annotation exports.

The default classifier expects `processed/behavior_with_feats_rectROI.csv` and
matching image files under `raw/images_clean/`.
