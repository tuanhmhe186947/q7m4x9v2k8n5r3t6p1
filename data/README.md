# Data Directory

This directory separates publishable metadata from large local research data.

- `data/`: native CVAT task exports used to build classification CSVs.
- `raw/`: extracted images consumed by training. Ignored by Git.
- `processed/`: generated tabular datasets grouped by workflow and run time.
- `annotations/`: source annotation assets grouped by purpose.
- `videos/`: local demo or research videos. Ignored by Git except README files.

The maintained classification flow writes:

```text
processed/classification/<YYYYMMDD_HHMMSS>/behavior_clean_merged.csv
processed/classification/<YYYYMMDD_HHMMSS>/behavior_with_feats_rectROI.csv
```

`pig_behavior.config.TrainConfig` resolves the newest
`behavior_with_feats_rectROI.csv` run by default. Matching image files live
under `raw/images_clean/`.
