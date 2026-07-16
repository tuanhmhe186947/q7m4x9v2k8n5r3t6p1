"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assets import (
    DETECTOR_WEIGHTS,
    DETECTOR_WEIGHTS_V26,
    EVAL_OUTPUT_ROOT,
    PREDICTION_ROOT,
    TRACKING_GT_DIR,
    VIDEO_DIR,
)


@dataclass(slots=True)
class TrackingEvaluationPipelineConfig:
    """Configuration for the tracking -> evaluation pipeline."""

    video_path: Path | None = None
    video_paths: list[Path] | None = None
    gt_xml: Path | None = None
    gt_dir: Path = TRACKING_GT_DIR
    video_dir: Path = VIDEO_DIR
    prediction_root: Path = PREDICTION_ROOT
    output_root: Path = EVAL_OUTPUT_ROOT
    weights_path: Path = DETECTOR_WEIGHTS
    weights_v26_path: Path = DETECTOR_WEIGHTS_V26
    detector_name: str = "yolov8"
    mask_path: Path | None = None
    iou_threshold: float = 0.5
    include_hidden: bool = False
    gap_tolerance_frames: int = 15
    # Bật tắt việc chạy tracker cho các video chưa có prediction XML
    run_missing_tracker: bool = True
    force_track: bool = False
    max_frames: int | None = None
    expected_video_count: int | None = None
    device: str | int | None = None
    half: bool = False
    USE_IOU_FALLBACK: bool = False
    USE_AREA_OCCLUSION_FREEZE: bool = False
    USE_CONDITIONAL_AREA_OCCLUSION_FREEZE: bool = False
    USE_MERGED_BOX_SPLIT: bool = False
    tracking_mode: str = "hybrid_bytetrack"
    profile_overrides: dict[str, Any] | None = None
