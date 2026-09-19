# Annotation Assets

This directory contains source annotations grouped by their role in the
research pipeline. Generated outputs should go under `data/processed/` or
`outputs/`, not in this folder.

- `classification/`: COCO annotations used as classification label assets.
- `detection/`: detector training annotations or exports.
- `roi/`: static feeder/drinker/toy ROI annotations, currently
  `ROI_annotations.coco.json`.
- `scene/`: scene-level support assets such as `background.png` and `mask.png`.
- `schemas/`: CVAT label schemas such as `cvat_pig_8_labels.json`.
- `tracking/`: CVAT video XML ground truth used for tracking evaluation.

Keep these files stable and versionable when licensing permits. Timestamped
run outputs belong in workflow-specific output folders.
