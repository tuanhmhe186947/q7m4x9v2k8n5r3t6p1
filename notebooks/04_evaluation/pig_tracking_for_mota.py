"""Notebook compatibility wrapper for tracking evaluation.

The maintained implementation lives in ``pig_behavior.evaluation`` so it can be
used by notebooks, CLI scripts, and future pipeline code without duplication.
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pig_behavior.evaluation.tracking_metrics import (  # noqa: E402
    DATA_DIR,
    DETECTOR_WEIGHTS,
    EVAL_OUTPUT_ROOT,
    PREDICTION_ROOT,
    TRACKING_GT_DIR,
    VIDEO_DIR,
    TrackingMetrics,
    TrackingObject,
    TrackingPair,
    aggregate_metrics,
    compute_association_accuracy,
    compute_id_metrics,
    evaluate_dataset,
    evaluate_pair,
    evaluate_tracking,
    find_prediction_xml,
    find_project_root,
    iou_xyxy,
    list_tracking_pairs,
    match_frame,
    metrics_to_dataframe,
    normalize_key,
    pairs_to_dataframe,
    parse_cvat_video_xml,
    read_cvat_task_size,
    read_task_name,
    resolve_mask_path,
    run_tracker_for_pair,
    video_metadata,
)

__all__ = [
    "DATA_DIR",
    "DETECTOR_WEIGHTS",
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
    "resolve_mask_path",
    "run_tracker_for_pair",
    "video_metadata",
]


if __name__ == "__main__":
    from pig_behavior.evaluation.tracking_metrics import evaluate_dataset

    assets, metrics, output_dir = evaluate_dataset()
    print("[assets]")
    print(assets.to_string(index=False))
    print("[metrics]")
    print(metrics.to_string(index=False))
    print("[output]", output_dir)
