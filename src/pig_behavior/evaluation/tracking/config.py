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
from .contracts import (
    EVALUATOR_CONTRACT_ID,
    LEGACY_EVALUATOR_CONTRACT_ID,
)
from .frame_window import validate_frame_bounds


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
    include_hidden: bool = True
    gap_tolerance_frames: int = 15
    # Bật tắt việc chạy tracker cho các video chưa có prediction XML
    run_missing_tracker: bool = True
    force_track: bool = False
    max_frames: int | None = None
    evaluation_start_frame: int | None = None
    evaluation_end_frame: int | None = None
    expected_video_count: int | None = None
    device: str | int | None = None
    half: bool = False
    USE_IOU_FALLBACK: bool = False
    USE_AREA_OCCLUSION_FREEZE: bool = False
    USE_CONDITIONAL_AREA_OCCLUSION_FREEZE: bool = False
    USE_MERGED_BOX_SPLIT: bool = False
    tracking_mode: str = "hybrid_bytetrack"
    profile_overrides: dict[str, Any] | None = None
    evaluator_contract_id: str = EVALUATOR_CONTRACT_ID

    def __post_init__(self) -> None:
        """Fail closed on invalid inclusive score bounds."""

        validate_frame_bounds(
            self.evaluation_start_frame,
            self.evaluation_end_frame,
        )
        if self.evaluator_contract_id == LEGACY_EVALUATOR_CONTRACT_ID:
            raise ValueError(
                "TRACKING_EVALUATOR_LEGACY_V1 is historical read-only behavior"
            )
        if self.evaluator_contract_id != EVALUATOR_CONTRACT_ID:
            raise ValueError(
                f"Unsupported evaluator contract: {self.evaluator_contract_id}"
            )
