# Data Directory

This directory separates publishable metadata from large local research data.

- `raw/`: raw frames or extracted images. Ignored by Git.
- `processed/`: tabular labels, bounding boxes, and engineered features.
- `annotations/`: scene-level masks and annotation exports.
- `videos/`: local demo or research videos. Ignored by Git except README files.

The default classifier expects `processed/behavior_with_feats_rectROI.csv` and
matching image files under `raw/images_clean/`.
