"""Build classification training CSVs from CVAT native task exports.

The maintained input flow is:

``data/data/task_*`` + ``data/annotations/roi/ROI_annotations.coco.json``
-> ``data/raw/images_clean``
-> ``data/processed/classification/<run_id>/*.csv``.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pig_behavior.data.classification_dataset import (
    ANNOTATION_DIR,
    BEHAVIORS,
    CLASSIFICATION_PROCESSED_DIR,
    DATA_EXPORT_ROOT,
    DROP_HIDDEN,
    OUT_CLEAN_CSV,
    OUT_FEATS_CSV,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_IMAGE_DIR,
    ROI_COCO_JSON,
    RUN_ID,
    RUN_OUTPUT_DIR,
    copy_annotated_images,
    find_project_root,
    main,
)
from pig_behavior.data.classification_features import (
    BEHAVIOR_TO_COARSE,
    add_training_features,
    clean_merged_annotations,
    load_roi_boxes,
    majority_fix_in_burst,
    rect_intersect_area,
)
from pig_behavior.data.cvat_native import (
    DEFAULT_BEHAVIORS,
    PIG_LABEL_PREFIX,
    frame_file_name,
    load_all_cvat_tasks,
    load_behaviors_from_project,
    load_cvat_task,
    load_manifest,
    parse_attrs,
    parse_burst_from_filename,
)
from pig_behavior.data.validation import (
    PROJECT_REQUIRED_COLUMNS,
    validate_project_outputs,
)

import numpy as np
import pandas as pd

BEHAVIOR_SET = set(BEHAVIORS)

if __name__ == "__main__":
    main()
