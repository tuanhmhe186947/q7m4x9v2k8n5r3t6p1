"""Build classification training CSVs from CVAT native task exports.

The maintained input flow is:

``data/data/task_*`` + ``data/annotations/roi/ROI_annotations.coco.json``
-> ``data/raw/images_clean``
-> ``data/processed/classification/<run_id>/*.csv``.
"""

from __future__ import annotations

from pig_behavior.data.classification_dataset import (
    BEHAVIORS,
    main,
)

BEHAVIOR_SET = set(BEHAVIORS)

if __name__ == "__main__":
    main()
