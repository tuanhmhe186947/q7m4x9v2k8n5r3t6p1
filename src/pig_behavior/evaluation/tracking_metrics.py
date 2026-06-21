"""Evaluate pig ID tracking metrics against CVAT video XML ground truth.

Ground truth XML files are expected in ``data/annotations/tracking`` and are
matched to videos in ``data/videos`` by video stem, for example:

``Tracking_annotation_Pigs291119_000263_30fps.xml`` ->
``Pigs291119_000263_30fps.mp4``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


from .tracking.assets import (
    DATA_DIR,
    DETECTOR_WEIGHTS,
    DETECTOR_WEIGHTS_V26,
    DETECTOR_WEIGHTS_V8,
    EVAL_OUTPUT_ROOT,
    PREDICTION_ROOT,
    PROJECT_ROOT,
    TRACKING_GT_DIR,
    VIDEO_DIR,
    TrackingPair,
    find_prediction_xml,
    find_project_root,
    list_tracking_pairs,
    normalize_key,
    resolve_mask_path,
    video_metadata,
)
from .tracking.cvat_io import (
    TrackingObject,
    box_hidden,
    box_id,
    id_from_label,
    is_outside,
    parse_cvat_video_xml,
    read_cvat_task_size,
    read_task_name,
)
from .tracking.matching import iou_xyxy, match_frame
from .tracking.diagnostics import (
    continuity_gaps_for_pair,
    continuity_gaps_to_dataframe,
    identity_events_for_pair,
    identity_events_to_dataframe,
    identity_mapping_for_pair,
    identity_mapping_to_dataframe,
)
from .tracking.metrics import (
    TrackingMetrics,
    aggregate_metrics,
    attach_remapped_metrics,
    best_id_mapping,
    compute_association_accuracy,
    compute_id_metrics,
    continuity_stats_from_matches,
    matched_identity_counts,
    remap_prediction_ids,
)

from .tracking.evaluator import (
    evaluate_dataset,
    evaluate_pair,
    evaluate_tracking,
    metrics_to_dataframe,
    pairs_to_dataframe,
    run_tracker_for_pair,
)

__all__ = [
    "DATA_DIR",
    "DETECTOR_WEIGHTS",
    "DETECTOR_WEIGHTS_V8",
    "DETECTOR_WEIGHTS_V26",
    "EVAL_OUTPUT_ROOT",
    "PREDICTION_ROOT",
    "PROJECT_ROOT",
    "TRACKING_GT_DIR",
    "VIDEO_DIR",
    "TrackingMetrics",
    "TrackingObject",
    "TrackingPair",
    "aggregate_metrics",
    "compute_association_accuracy",
    "compute_id_metrics",
    "evaluate_dataset",
    "evaluate_pair",
    "evaluate_tracking",
    "find_prediction_xml",
    "identity_events_for_pair",
    "identity_events_to_dataframe",
    "identity_mapping_for_pair",
    "identity_mapping_to_dataframe",
    "continuity_gaps_for_pair",
    "continuity_gaps_to_dataframe",
    "find_project_root",
    "iou_xyxy",
    "list_tracking_pairs",
    "match_frame",
    "metrics_to_dataframe",
    "normalize_key",
    "pairs_to_dataframe",
    "parse_cvat_video_xml",
    "read_cvat_task_size",
    "read_task_name",
    "remap_prediction_ids",
    "resolve_mask_path",
    "run_tracker_for_pair",
    "video_metadata",
]








if __name__ == "__main__":
    assets, metrics, output_dir = evaluate_dataset()
    print("[assets]")
    print(assets.to_string(index=False))
    print("[metrics]")
    print(metrics.to_string(index=False))
    print("[output]", output_dir)
