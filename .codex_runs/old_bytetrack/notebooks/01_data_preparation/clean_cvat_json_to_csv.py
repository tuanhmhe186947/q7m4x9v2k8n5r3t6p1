"""Compatibility wrapper for classification dataset preparation.

The implementation lives in ``pig_behavior.data_preparation.classification_dataset``.
Run the src entry point in scripts/automation and keep notebooks for exploration.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    start = start if start.is_dir() else start.parent
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pig_behavior.data_preparation.classification_dataset import (  # noqa: E402
    BEHAVIOR_SET,
    BEHAVIOR_TO_COARSE,
    BEHAVIORS,
    CLASSIFICATION_PROCESSED_DIR,
    DATA_EXPORT_ROOT,
    OUT_CLEAN_CSV,
    OUT_FEATS_CSV,
    RAW_IMAGE_DIR,
    ROI_COCO_JSON,
    RUN_ID,
    RUN_OUTPUT_DIR,
    add_training_features,
    clean_merged_annotations,
    copy_annotated_images,
    load_all_cvat_tasks,
    load_cvat_task,
    load_roi_boxes,
    main,
    validate_project_outputs,
)

__all__ = [
    "BEHAVIORS",
    "BEHAVIOR_SET",
    "BEHAVIOR_TO_COARSE",
    "CLASSIFICATION_PROCESSED_DIR",
    "DATA_EXPORT_ROOT",
    "OUT_CLEAN_CSV",
    "OUT_FEATS_CSV",
    "RAW_IMAGE_DIR",
    "ROI_COCO_JSON",
    "RUN_ID",
    "RUN_OUTPUT_DIR",
    "add_training_features",
    "clean_merged_annotations",
    "copy_annotated_images",
    "load_all_cvat_tasks",
    "load_cvat_task",
    "load_roi_boxes",
    "main",
    "validate_project_outputs",
]


if __name__ == "__main__":
    main()
