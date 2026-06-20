# Processed Data

Generated datasets are grouped by workflow and run timestamp.

```text
classification/<YYYYMMDD_HHMMSS>/behavior_clean_merged.csv
classification/<YYYYMMDD_HHMMSS>/behavior_with_feats_rectROI.csv
```

`behavior_clean_merged.csv` is the cleaned CVAT annotation table.
`behavior_with_feats_rectROI.csv` adds ROI, motion, and social-distance
features consumed by `pig_behavior.data_loader`.

`TrainConfig` resolves the newest classification run by default. Preserve the
column contract documented in the main README when regenerating these files.
