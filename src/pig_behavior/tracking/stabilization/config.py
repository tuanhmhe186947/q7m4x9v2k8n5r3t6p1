"""Configuration for the stable annotation tracking pipeline.

Designed to produce clean, stable CVAT XML outputs for manual annotators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig


@dataclass(slots=True)
class AnnotationStableConfig:
    """Config dataclass for the Annotation Stable tracking pipeline."""

    # --- Embedded base configs ---
    tracking_config: TrackingConfig = field(default_factory=TrackingConfig)
    rgbd_config: RGBDTrackingConfig | None = None  # None = pure 2D mode

    # --- Hybrid cost weights (when depth/BEV is valid) ---
    w_bev: float = 0.30
    w_iou_2d: float = 0.30
    w_area: float = 0.10
    w_hist: float = 0.20
    w_conf: float = 0.05
    w_depth_ambiguous: float = 0.05

    # --- Fallback weights (when depth/BEV is invalid) ---
    w_iou_2d_fallback: float = 0.50
    w_area_fallback: float = 0.15
    w_hist_fallback: float = 0.35

    # --- Conservative online ambiguity gate ---
    prefer_gap_over_bad_match: bool = True
    min_assignment_margin: float = 0.10
    max_center_jump_norm: float = 0.10
    max_area_log_ratio: float = 0.50
    min_iou_2d_for_match: float = 0.05
    allow_ambiguous_match: bool = False

    # --- Tracklet filtering ---
    min_tracklet_length: int = 5
    remove_short_tracks_under: int = 10
    max_predict_only_streak: int = 15  # break tracklet after N predict-only frames

    # --- Offline stitching ---
    stitch_max_gap: int = 60
    stitch_min_score: float = 0.65
    stitch_use_bev: bool = True
    stitch_max_center_distance_norm: float = 0.15
    stitch_max_area_log_ratio: float = 0.40
    stitch_max_hist_distance: float = 0.60

    # --- Swap detection ---
    detect_candidate_swaps: bool = True
    auto_fix_high_confidence_swaps: bool = False
    swap_confidence_threshold: float = 0.85
    swap_proximity_frames: int = 10
    swap_min_overlap_iou: float = 0.10

    # --- Bbox smoothing ---
    smooth_bbox: bool = True
    smooth_bbox_window: int = 5
    smooth_method: str = "median"  # "median" | "ema"
    smooth_skip_ambiguous: bool = True
    max_smoothing_shift_px: int = 20

    # --- Output ---
    export_debug_video: bool = True
    export_diagnostics: bool = True
    debug_video_fps: float = 30.0
