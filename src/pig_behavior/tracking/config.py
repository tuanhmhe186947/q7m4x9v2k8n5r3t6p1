"""Configuration and output path helpers for offline pig tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pig_behavior.output_layout import mode_scoped_video_dir
from pig_behavior.tracking.constants import (
    BEHAVIOR_VALUES,
    DEFAULT_DET_CONF_THRESHOLD,
    DEFAULT_DETECT_EVERY_N_FRAMES,
    DEFAULT_DUP_AREA_RATIO_THRESHOLD,
    DEFAULT_DUP_CENTER_THRESHOLD,
    DEFAULT_DUP_CONTAINMENT_THRESHOLD,
    DEFAULT_DUP_IOU_THRESHOLD,
    DEFAULT_ENABLE_OFFLINE_SMOOTHING,
    DEFAULT_EXPECTED_PIGS,
    DEFAULT_HARD_OCCLUSION_IOU_THRESHOLD,
    DEFAULT_MARK_INTERPOLATED_REVIEW,
    DEFAULT_MASK_PATH,
    DEFAULT_MAX_INTERPOLATION_GAP,
    DEFAULT_MAX_LOST_FRAMES,
    DEFAULT_MAX_RAW_DETECTIONS,
    DEFAULT_NMS_IOU_THRESHOLD,
    DEFAULT_OCCLUSION_IOU_THRESHOLD,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REVIEW_CONF_THRESHOLD,
    DEFAULT_SPLIT_RECOVERY_FRAMES,
    DEFAULT_TARGET_FPS,
    DEFAULT_TRACK_HIGH_CONF_THRESHOLD,
    DEFAULT_TRACK_MATCH_IOU_THRESHOLD,
    DEFAULT_VIDEO_PATH,
    DEFAULT_VISUAL_OPACITY,
    DEFAULT_WEIGHTS_PATH,
    ID_VALUES,
    TRACKING_TELEMETRY_KEYS,
)

logger = logging.getLogger(__name__)

TRACKING_MODE_CHOICES = (
    "realtime",
    "bytetrack_raw",
    "hybrid_bytetrack",
    "gt_export",
)
CANONICAL_TRACKING_MODES = {"realtime", "bytetrack_raw", "hybrid_bytetrack"}


@dataclass(slots=True)
class TrackingConfig:
    """Config for YOLOv8 detection, ID stabilization, JSON, and video export."""

    video_path: Path = DEFAULT_VIDEO_PATH
    weights_path: Path = DEFAULT_WEIGHTS_PATH
    mask_path: Path | None = DEFAULT_MASK_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_video: Path | None = None
    annotations_json: Path | None = None
    coco_annotations_json: Path | None = None
    clean_coco_annotations_json: Path | None = None
    cvat_video_xml: Path | None = None
    labels_json: Path | None = None
    tracker_yaml: Path | None = None
    quality_report_json: Path | None = None
    quality_report_csv: Path | None = None
    device: int | str | None = None
    half: bool = False
    tracker_type: str = "bytetrack"

    expected_pigs: int = DEFAULT_EXPECTED_PIGS
    output_fps: float = 30.0
    start_frame: int = 0
    det_conf: float = DEFAULT_DET_CONF_THRESHOLD
    track_high_conf: float = DEFAULT_TRACK_HIGH_CONF_THRESHOLD
    review_conf: float = DEFAULT_REVIEW_CONF_THRESHOLD
    adaptive_conf_step: float = 0.05
    conf: float | None = None
    nms_iou: float = DEFAULT_NMS_IOU_THRESHOLD
    iou: float | None = None  # Legacy alias/override for nms_iou
    track_match_iou: float = DEFAULT_TRACK_MATCH_IOU_THRESHOLD
    class_id: int | None = None
    allowed_class_name: str | None = None

    # Pipeline Mode & Realtime parameters
    mode: str = "realtime"  # realtime, bytetrack_raw, hybrid_bytetrack, or gt_export
    imgsz: int = 640
    detect_every_n_frames: int = DEFAULT_DETECT_EVERY_N_FRAMES
    max_raw_detections: int = DEFAULT_MAX_RAW_DETECTIONS
    target_fps: float = DEFAULT_TARGET_FPS
    enable_offline_smoothing: bool = DEFAULT_ENABLE_OFFLINE_SMOOTHING

    use_mask: bool = True
    mask_input_frame: bool = True
    roi_mode: str = "center"
    roi_min_cover: float = 0.10
    roi_dilate_px: int = 8

    max_missing_frames: int = DEFAULT_MAX_LOST_FRAMES
    max_lost_frames: int = DEFAULT_MAX_LOST_FRAMES  # Alias for max_missing_frames
    hidden_missed_frames: int = 5
    hidden_score_threshold: float = 0.15
    hidden_overlap_iou_threshold: float = 0.65
    hidden_overlap_window_frames: int = 2
    emit_hidden_tracks: bool = True
    use_mask_iou: bool = True
    mask_iou_max_missed: int = 10
    mask_iou_min_area: int = 64
    match_cost_threshold: float = 0.78
    unseen_track_cost_threshold: float = 1.10
    lost_track_cost_threshold: float = 0.95
    lost_track_reid_appearance_threshold: float = 0.25
    lost_track_reacquire_guard: bool = True
    lost_track_reacquire_same_raw_distance_guard: bool = True
    lost_track_reacquire_raw_owner_guard: bool = True
    lost_track_reacquire_same_raw_owner_guard: bool = True
    lost_track_reacquire_different_raw_owner_guard: bool = True
    # Current hybrid_bytetrack base: disabling this distance-only reject preserved
    # the strong 20260703_194929 9-video result while keeping raw-owner guards.
    lost_track_reacquire_non_same_raw_distance_guard: bool = False
    lost_track_reacquire_max_center_jump: float = 0.22
    lost_track_reacquire_same_raw_max_center_jump: float = 0.34
    lost_track_reacquire_same_raw_appearance_bypass: bool = True
    lost_track_reacquire_same_raw_appearance_threshold: float = 0.25
    lost_track_fast_motion_owner_grace: int = 3
    lost_track_fast_motion_min_center_jump: float = 0.06
    lost_track_fast_motion_max_center_jump: float = 0.42
    lost_track_fast_motion_owner_max_gap: float = 0.14
    lost_track_fast_motion_appearance_threshold: float = 0.30
    lost_track_raw_owner_transfer_min_center_gain: float = 0.04
    lost_track_raw_owner_transfer_min_appearance_gain: float = 0.05
    lost_track_raw_owner_transfer_appearance_threshold: float = 0.25
    lost_track_different_raw_hidden_owner_bypass: bool = True
    lost_track_different_raw_hidden_owner_min_missed: int = 2
    lost_track_different_raw_hidden_owner_min_center_gain: float = 0.03
    lost_track_different_raw_hidden_owner_appearance_threshold: float = 0.25
    dup_iou_threshold: float = DEFAULT_DUP_IOU_THRESHOLD
    dup_containment_threshold: float = DEFAULT_DUP_CONTAINMENT_THRESHOLD
    dup_center_threshold: float = DEFAULT_DUP_CENTER_THRESHOLD
    dup_area_ratio_threshold: float = DEFAULT_DUP_AREA_RATIO_THRESHOLD
    initial_track_conf: float = DEFAULT_TRACK_HIGH_CONF_THRESHOLD
    low_conf_motion_gate: bool = True
    motion_gate_confidence: float = DEFAULT_TRACK_HIGH_CONF_THRESHOLD
    low_conf_max_center_jump: float = 0.08
    low_conf_max_box_jump_scale: float = 1.75
    low_conf_min_iou: float = 0.01
    occlusion_aware_matching: bool = True
    occlusion_track_iom_threshold: float = DEFAULT_OCCLUSION_IOU_THRESHOLD
    occlusion_detection_iom_threshold: float = 0.30
    occlusion_stationary_speed: float = 0.006
    occlusion_stationary_max_center_jump: float = 0.045
    occlusion_switch_penalty: float = 0.45
    occlusion_competitor_margin: float = 0.12
    occlusion_appearance_penalty: float = 0.30
    occlusion_appearance_margin: float = 0.08
    directional_y_prior: bool = True
    directional_y_penalty_weight: float = 0.12
    directional_y_velocity_epsilon_px: float = 3.0
    directional_y_margin_px: float = 5.0
    occlusion_stationary_lock: bool = True
    freeze_identity_in_occlusion: bool = True
    hold_occluded_box: bool = True
    realtime_visible_close_competitor_guard: bool = False
    realtime_visible_close_competitor_margin: float = 0.012
    realtime_visible_close_competitor_max_cost: float = 0.35
    realtime_visible_close_competitor_min_hits: int = 3
    realtime_visible_better_competitor_reject: bool = False
    realtime_visible_better_competitor_prefer: bool = False
    realtime_visible_better_competitor_min_cost: float = 0.50
    realtime_visible_better_competitor_min_gain: float = 0.20
    realtime_low_conf_recovery_guard: bool = False
    realtime_low_conf_recovery_min_score: float = 0.50
    realtime_low_conf_recovery_min_missed: int = 3
    realtime_low_conf_recovery_max_missed: int = 20
    realtime_motion_pair_stabilizer: bool = False
    realtime_motion_pair_max_jump: float = 0.10
    realtime_motion_pair_min_gain: float = 0.01
    realtime_motion_pair_memory_frames: int = 30
    realtime_motion_pair_max_component_size: int = 2
    occlusion_hold_max_frames: int = 30
    occlusion_hold_hidden_frames: int = 2
    USE_IOU_FALLBACK: bool = False
    USE_AREA_OCCLUSION_FREEZE: bool = False
    USE_CONDITIONAL_AREA_OCCLUSION_FREEZE: bool = False
    USE_MERGED_BOX_SPLIT: bool = False
    iou_fallback_threshold: float = 0.45
    area_occlusion_shrink_ratio: float = 0.60
    area_occlusion_freeze_frames: int = 15
    merged_box_growth_ratio: float = 1.50
    merged_box_neighbor_distance: float = 0.12
    merged_box_split_max_tracks: int = 2
    hard_occlusion_track_iom_threshold: float = 0.35
    hard_occlusion_detection_iom_threshold: float = DEFAULT_HARD_OCCLUSION_IOU_THRESHOLD
    hard_occlusion_min_frames: int = 2
    hard_occlusion_recovery_frames: int = DEFAULT_SPLIT_RECOVERY_FRAMES
    hard_occlusion_score_threshold: float = 0.65
    identity_swap_guard: bool = True
    identity_swap_min_gain: float = 0.015
    identity_swap_iom_threshold: float = 0.10
    # Experimental: frame-local motion repair did not affect the 20260703_221004
    # / 20260703_222520 hard-case evals, so keep it opt-in until a stronger
    # episode-level repair is validated.
    local_pair_swap_repair: bool = False
    local_pair_swap_window_frames: int = 8
    local_pair_swap_max_gap_frames: int = 2
    local_pair_swap_min_overlap_iou: float = 0.20
    local_pair_swap_min_motion_gain: float = 0.08
    episode_pair_swap_repair: bool = False
    episode_pair_swap_max_frames: int = 12
    episode_pair_swap_anchor_window_frames: int = 8
    episode_pair_swap_min_overlap_iou: float = 0.20
    episode_pair_swap_min_motion_gain: float = 0.10
    long_pair_swap_repair: bool = False
    long_pair_swap_min_frames: int = 24
    long_pair_swap_max_gap_frames: int = 2
    long_pair_swap_min_start_gain: float = 0.08
    long_pair_swap_min_median_separation: float = 0.04
    suffix_pair_swap_repair: bool = False
    suffix_pair_swap_min_overlap_iou: float = 0.45
    suffix_pair_swap_max_overlap_frames: int = 8
    suffix_pair_swap_min_suffix_frames: int = 1500
    suffix_pair_swap_max_suffix_overlap_iou: float = 0.30
    overlap_small_box_suppression: bool = False
    overlap_small_box_min_iou: float = 0.40
    overlap_small_box_max_area_ratio: float = 0.65
    overlap_small_box_max_score: float = 0.75
    hidden_suffix_id_swap_repair: bool = False
    hidden_suffix_id_swap_min_hidden_frames: int = 8
    hidden_suffix_id_swap_max_hidden_frames: int = 15
    hidden_suffix_id_swap_min_overlap_iou: float = 0.70
    hidden_suffix_id_swap_max_hidden_median_score: float = 0.50
    hidden_suffix_id_swap_start_back_frames: int = 7
    hidden_suffix_id_swap_min_suffix_frames: int = 600
    association_debug: bool = False
    ambiguity_owner_guard: bool = False
    ambiguity_owner_guard_cost_margin: float = 0.04
    hidden_owner_guard: bool = False
    hidden_owner_guard_min_missed: int = 2
    hidden_owner_guard_cost_margin: float = 0.08
    hidden_owner_guard_hold_assignment: bool = False
    reentry_ambiguous_hold: bool = False
    reentry_ambiguous_hold_min_missed: int = 2
    reentry_ambiguous_hold_min_hits: int = 3
    reentry_ambiguous_hold_frame_windows: str = ""
    reentry_ambiguous_hold_video_stems: str = ""
    reentry_ambiguous_hold_raw_evidence_only: bool = False
    reentry_ambiguous_hold_max_missed: int = 0
    reentry_ambiguous_hold_min_cost: float = 0.0
    reentry_ambiguous_hold_max_cost: float = 1.0
    reentry_unowned_raw_mismatch_reject: bool = False
    reentry_unowned_raw_mismatch_min_missed: int = 1
    reentry_unowned_raw_mismatch_max_missed: int = 5
    reentry_unowned_raw_mismatch_max_cost: float = 0.30
    reentry_unowned_raw_mismatch_quarantine_frames: int = 0
    reentry_unowned_raw_mismatch_quarantine_min_seed_cost: float = 0.0
    reentry_unowned_raw_mismatch_quarantine_max_cost: float = 0.35
    reentry_unowned_raw_mismatch_episode_reject: bool = False
    reentry_unowned_raw_mismatch_episode_window_frames: int = 24
    reentry_unowned_raw_mismatch_episode_min_events: int = 3
    reentry_unowned_raw_mismatch_episode_max_events: int = 8
    reentry_unowned_raw_mismatch_episode_min_missed: int = 1
    reentry_unowned_raw_mismatch_episode_max_missed: int = 20
    reentry_unowned_raw_mismatch_episode_min_cost: float = 0.0
    reentry_unowned_raw_mismatch_episode_max_cost: float = 0.36
    reentry_unowned_raw_mismatch_episode_phases: str = "reid"
    reentry_unowned_raw_mismatch_episode_action: str = "reject"
    occlusion_reid_prefer_gap_over_bad_match: bool = False
    occlusion_reid_bad_match_min_cost: float = 0.60
    occlusion_reid_bad_match_max_cost: float = 1.0
    occlusion_reid_bad_match_min_missed: int = 0
    occlusion_reid_bad_match_max_missed: int = 3
    occlusion_reid_bad_match_same_raw_only: bool = True
    occlusion_reid_bad_match_raw_mismatch_only: bool = False
    occlusion_reid_bad_match_unowned_raw_only: bool = False
    occlusion_reid_bad_match_occlusion_hold_only: bool = False
    occlusion_reid_bad_match_once_per_episode: bool = False
    occlusion_reid_bad_match_include_recent_visible: bool = False
    occlusion_reid_bad_match_visible_min_cost: float = 0.70
    occlusion_reid_bad_match_action: str = "hold"
    reid_unowned_competing_candidate_hold: bool = False
    reid_unowned_competing_candidate_min_cost: float = 0.55
    reid_unowned_competing_candidate_min_gap: float = 0.15
    reid_unowned_competing_candidate_min_missed: int = 1
    reid_unowned_competing_candidate_occlusion_hold_only: bool = True
    visible_raw_owner_transfer_min_gain: float = 0.04
    hidden_motion_model: bool = True
    hidden_velocity_alpha: float = 0.65
    hidden_acceleration_alpha: float = 0.35
    hidden_stationary_speed: float = 0.006
    hidden_motion_history: int = 8
    hidden_min_motion_history: int = 4
    hidden_stationary_displacement: float = 0.015
    hidden_moving_displacement: float = 0.035
    hidden_motion_consistency: float = 0.55
    hidden_stationary_lock_frames: int = 8
    hidden_max_motion_step_box_scale: float = 1.50
    default_behavior: str = "lying"
    smooth_boxes: bool = True
    refine_boxes: bool = True
    refine_max_gap_frames: int = 15
    refine_size_jump_threshold: float = 0.45
    max_box_scale_change_per_frame: float = 0.25
    max_box_scale_change_after_gap: float = 0.75
    high_conf_smooth_alpha: float = 0.75
    mid_conf_smooth_alpha: float = 0.55
    low_conf_smooth_alpha: float = 0.35

    # Interpolation parameters (GT export only)
    max_interpolation_gap: int = DEFAULT_MAX_INTERPOLATION_GAP
    mark_interpolated_review: bool = DEFAULT_MARK_INTERPOLATED_REVIEW

    max_frames: int | None = None
    draw_mask_outline: bool = True
    shade_outside_mask: bool = True
    visual_opacity: float = DEFAULT_VISUAL_OPACITY
    show: bool = False
    display_inline: bool = False
    overrides: set[str] = field(default_factory=set)


def tracking_rule_flags_enabled(cfg: TrackingConfig) -> bool:
    return (
        cfg.USE_IOU_FALLBACK
        or cfg.USE_AREA_OCCLUSION_FREEZE
        or cfg.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE
        or cfg.USE_MERGED_BOX_SPLIT
    )


def get_telemetry_summary(source: Any) -> dict[str, int]:
    """Return telemetry counters in a stable schema for benchmark comparison."""
    telemetry = source.telemetry
    return {key: int(telemetry.get(key, 0)) for key in TRACKING_TELEMETRY_KEYS}


def validate_config(cfg: TrackingConfig) -> None:
    # 1. Map legacy iou to nms_iou if specified
    if cfg.iou is not None:
        cfg.nms_iou = cfg.iou
    else:
        cfg.iou = cfg.nms_iou

    if cfg.mode not in TRACKING_MODE_CHOICES:
        raise ValueError(
            "mode must be one of: realtime, bytetrack_raw, hybrid_bytetrack, "
            "gt_export."
        )
    if cfg.occlusion_reid_bad_match_action not in {"hold", "reject"}:
        raise ValueError("occlusion_reid_bad_match_action must be 'hold' or 'reject'.")

    requested_mode = cfg.mode
    if requested_mode == "gt_export":
        logger.warning(
            "[DEPRECATED] mode=gt_export is not a tracking algorithm. "
            "Use --mode hybrid_bytetrack --cvat-video-xml output.xml instead."
        )
        cfg.mode = "hybrid_bytetrack"

    # 2. Mode-based dynamic defaults overrides
    if cfg.mode == "bytetrack_raw":
        # Raw ByteTrack baseline: keep Ultralytics ByteTrack association, but
        # disable project-specific post-processing unless explicitly requested.
        raw_defaults = {
            "detect_every_n_frames": 1,
            "enable_offline_smoothing": False,
            "smooth_boxes": False,
            "refine_boxes": False,
            "occlusion_aware_matching": False,
            "identity_swap_guard": False,
            "hidden_motion_model": False,
            "USE_IOU_FALLBACK": False,
            "USE_AREA_OCCLUSION_FREEZE": False,
            "USE_CONDITIONAL_AREA_OCCLUSION_FREEZE": False,
            "USE_MERGED_BOX_SPLIT": False,
        }
        for name, value in raw_defaults.items():
            if name not in cfg.overrides:
                setattr(cfg, name, value)
    elif cfg.mode == "hybrid_bytetrack":
        # Preserve the ByteTrack-based improved behavior under a clear name.
        bytetrack_defaults = {
            "det_conf": 0.25,
            "track_high_conf": 0.50,
            "review_conf": 0.75,
            "nms_iou": 0.80,
            "track_match_iou": 0.80,
            "dup_iou_threshold": 0.80,
            "initial_track_conf": 0.50,
            "motion_gate_confidence": 0.50,
            "USE_IOU_FALLBACK": False,
        }
        for name, value in bytetrack_defaults.items():
            explicitly_overridden = name in cfg.overrides
            if name == "nms_iou" and "iou" in cfg.overrides:
                explicitly_overridden = True
            if not explicitly_overridden:
                setattr(cfg, name, value)
        cfg.iou = cfg.nms_iou
        # The original path ran model.track() on every source frame.
        cfg.detect_every_n_frames = 1
        if "max_missing_frames" not in cfg.overrides:
            cfg.max_missing_frames = 90
            cfg.max_lost_frames = 90
        if requested_mode == "gt_export":
            if "det_conf" not in cfg.overrides and cfg.det_conf == 0.25:
                cfg.det_conf = 0.15
            if "max_raw_detections" not in cfg.overrides and cfg.max_raw_detections == 20:
                cfg.max_raw_detections = 30
            if "max_missing_frames" not in cfg.overrides and cfg.max_missing_frames == 90:
                cfg.max_missing_frames = 60
                cfg.max_lost_frames = 60
    else:
        # In realtime mode, explicitly turn off offline smoothing / post-processing
        if "enable_offline_smoothing" not in cfg.overrides:
            cfg.enable_offline_smoothing = False
            cfg.smooth_boxes = False
            cfg.refine_boxes = False

    if cfg.conf is not None:
        cfg.review_conf = cfg.conf
    if cfg.start_frame < 0:
        raise ValueError("start_frame must be >= 0.")
    if isinstance(cfg.device, str) and not cfg.device.strip():
        raise ValueError("device must not be empty.")
    if cfg.expected_pigs != len(ID_VALUES):
        raise ValueError("The CVAT label schema is fixed to exactly 8 pig IDs.")
    if cfg.default_behavior not in BEHAVIOR_VALUES:
        raise ValueError(f"default_behavior must be one of: {BEHAVIOR_VALUES}")
    if cfg.roi_mode not in {"center", "cover"}:
        raise ValueError("roi_mode must be either 'center' or 'cover'.")
    confidence_values = {
        "det_conf": cfg.det_conf,
        "track_high_conf": cfg.track_high_conf,
        "review_conf": cfg.review_conf,
    }
    for name, value in confidence_values.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    gate_confidence_values = {
        "initial_track_conf": cfg.initial_track_conf,
        "motion_gate_confidence": cfg.motion_gate_confidence,
    }
    for name, value in gate_confidence_values.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if cfg.det_conf > cfg.track_high_conf:
        raise ValueError("det_conf should be <= track_high_conf.")
    if cfg.track_high_conf > cfg.review_conf:
        raise ValueError("track_high_conf should be <= review_conf.")
    if cfg.initial_track_conf < cfg.det_conf:
        raise ValueError("initial_track_conf should be >= det_conf.")
    if cfg.motion_gate_confidence < cfg.det_conf:
        raise ValueError("motion_gate_confidence should be >= det_conf.")
    if not 0.0 < cfg.adaptive_conf_step <= 0.50:
        raise ValueError("adaptive_conf_step must be between 0 and 0.50.")
    if not 0.0 < cfg.nms_iou < 1.0:
        raise ValueError("nms_iou must be between 0 and 1.")
    if not 0.0 < cfg.track_match_iou < 1.0:
        raise ValueError("track_match_iou must be between 0 and 1.")
    if not 0.0 <= cfg.visual_opacity <= 1.0:
        raise ValueError("visual_opacity must be between 0 and 1.")
    if cfg.hidden_missed_frames < 1:
        raise ValueError("hidden_missed_frames must be >= 1.")
    if not 0.0 <= cfg.hidden_score_threshold <= 1.0:
        raise ValueError("hidden_score_threshold must be between 0 and 1.")
    if cfg.mask_iou_max_missed < 0:
        raise ValueError("mask_iou_max_missed must be >= 0.")
    if cfg.mask_iou_min_area < 1:
        raise ValueError("mask_iou_min_area must be >= 1.")
    cost_thresholds = {
        "match_cost_threshold": cfg.match_cost_threshold,
        "unseen_track_cost_threshold": cfg.unseen_track_cost_threshold,
        "lost_track_cost_threshold": cfg.lost_track_cost_threshold,
    }
    for name, value in cost_thresholds.items():
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0.")
    if not 0.0 <= cfg.lost_track_reid_appearance_threshold <= 1.0:
        raise ValueError(
            "lost_track_reid_appearance_threshold must be between 0 and 1."
        )
    duplicate_values = {
        "dup_iou_threshold": cfg.dup_iou_threshold,
        "dup_containment_threshold": cfg.dup_containment_threshold,
        "dup_center_threshold": cfg.dup_center_threshold,
        "dup_area_ratio_threshold": cfg.dup_area_ratio_threshold,
    }
    for name, value in duplicate_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if not 0.0 <= cfg.low_conf_max_center_jump <= 1.0:
        raise ValueError("low_conf_max_center_jump must be between 0 and 1.")
    if not 0.0 <= cfg.low_conf_min_iou <= 1.0:
        raise ValueError("low_conf_min_iou must be between 0 and 1.")
    if cfg.low_conf_max_box_jump_scale < 0.0:
        raise ValueError("low_conf_max_box_jump_scale must be >= 0.")
    occlusion_values = {
        "occlusion_track_iom_threshold": cfg.occlusion_track_iom_threshold,
        "occlusion_detection_iom_threshold": cfg.occlusion_detection_iom_threshold,
        "occlusion_stationary_speed": cfg.occlusion_stationary_speed,
        "occlusion_stationary_max_center_jump": (
            cfg.occlusion_stationary_max_center_jump
        ),
        "occlusion_switch_penalty": cfg.occlusion_switch_penalty,
        "occlusion_competitor_margin": cfg.occlusion_competitor_margin,
        "occlusion_appearance_penalty": cfg.occlusion_appearance_penalty,
        "occlusion_appearance_margin": cfg.occlusion_appearance_margin,
    }
    for name, value in occlusion_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if cfg.occlusion_hold_max_frames < 0:
        raise ValueError("occlusion_hold_max_frames must be >= 0.")
    if cfg.occlusion_hold_hidden_frames < 1:
        raise ValueError("occlusion_hold_hidden_frames must be >= 1.")
    if cfg.USE_IOU_FALLBACK and not 0.0 <= cfg.iou_fallback_threshold <= 1.0:
        raise ValueError("iou_fallback_threshold must be between 0 and 1.")
    if cfg.USE_AREA_OCCLUSION_FREEZE or cfg.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE:
        if not 0.0 < cfg.area_occlusion_shrink_ratio <= 1.0:
            raise ValueError("area_occlusion_shrink_ratio must be in (0, 1].")
        if cfg.area_occlusion_freeze_frames < 1:
            raise ValueError("area_occlusion_freeze_frames must be >= 1.")
    if cfg.USE_MERGED_BOX_SPLIT:
        if cfg.merged_box_growth_ratio <= 1.0:
            raise ValueError("merged_box_growth_ratio must be > 1.")
        if not 0.0 < cfg.merged_box_neighbor_distance <= 1.0:
            raise ValueError("merged_box_neighbor_distance must be in (0, 1].")
        if cfg.merged_box_split_max_tracks < 2:
            raise ValueError("merged_box_split_max_tracks must be >= 2.")
        hard_occlusion_values = {
            "hard_occlusion_track_iom_threshold": (
                cfg.hard_occlusion_track_iom_threshold
            ),
            "hard_occlusion_detection_iom_threshold": (
                cfg.hard_occlusion_detection_iom_threshold
            ),
            "hard_occlusion_score_threshold": cfg.hard_occlusion_score_threshold,
        }
        for name, value in hard_occlusion_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if cfg.hard_occlusion_min_frames < 1:
            raise ValueError("hard_occlusion_min_frames must be >= 1.")
        if cfg.hard_occlusion_recovery_frames < 1:
            raise ValueError("hard_occlusion_recovery_frames must be >= 1.")
    identity_swap_values = {
        "identity_swap_min_gain": cfg.identity_swap_min_gain,
        "identity_swap_iom_threshold": cfg.identity_swap_iom_threshold,
    }
    for name, value in identity_swap_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    hidden_motion_values = {
        "hidden_velocity_alpha": cfg.hidden_velocity_alpha,
        "hidden_acceleration_alpha": cfg.hidden_acceleration_alpha,
        "hidden_stationary_speed": cfg.hidden_stationary_speed,
        "hidden_stationary_displacement": cfg.hidden_stationary_displacement,
        "hidden_moving_displacement": cfg.hidden_moving_displacement,
        "hidden_motion_consistency": cfg.hidden_motion_consistency,
    }
    for name, value in hidden_motion_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if cfg.hidden_motion_history < 2:
        raise ValueError("hidden_motion_history must be >= 2.")
    if cfg.hidden_min_motion_history < 2:
        raise ValueError("hidden_min_motion_history must be >= 2.")
    if cfg.hidden_min_motion_history > cfg.hidden_motion_history:
        raise ValueError(
            "hidden_min_motion_history must be <= hidden_motion_history."
        )
    if cfg.hidden_stationary_lock_frames < 1:
        raise ValueError("hidden_stationary_lock_frames must be >= 1.")
    if cfg.hidden_max_motion_step_box_scale < 0.0:
        raise ValueError("hidden_max_motion_step_box_scale must be >= 0.")
    if not 0.0 <= cfg.hidden_overlap_iou_threshold <= 1.0:
        raise ValueError("hidden_overlap_iou_threshold must be between 0 and 1.")
    if cfg.hidden_overlap_window_frames < 1:
        raise ValueError("hidden_overlap_window_frames must be >= 1.")
    scale_values = {
        "max_box_scale_change_per_frame": cfg.max_box_scale_change_per_frame,
        "max_box_scale_change_after_gap": cfg.max_box_scale_change_after_gap,
    }
    for name, value in scale_values.items():
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"{name} must be between 0 and 2.")
    if cfg.refine_max_gap_frames < 1:
        raise ValueError("refine_max_gap_frames must be >= 1.")
    if not 0.0 <= cfg.refine_size_jump_threshold <= 2.0:
        raise ValueError("refine_size_jump_threshold must be between 0 and 2.")
    alpha_values = {
        "high_conf_smooth_alpha": cfg.high_conf_smooth_alpha,
        "mid_conf_smooth_alpha": cfg.mid_conf_smooth_alpha,
        "low_conf_smooth_alpha": cfg.low_conf_smooth_alpha,
    }
    for name, value in alpha_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if not cfg.video_path.exists():
        raise FileNotFoundError(f"Video not found: {cfg.video_path}")
    if not cfg.weights_path.exists():
        raise FileNotFoundError(f"YOLOv8 weights not found: {cfg.weights_path}")
    if cfg.use_mask and cfg.mask_path is not None and not cfg.mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {cfg.mask_path}")


def resolve_output_paths(
    cfg: TrackingConfig,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    video_stem = cfg.video_path.stem
    run_output_dir = mode_scoped_video_dir(cfg.output_dir, cfg.mode, video_stem)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    output_video = cfg.output_video or (
        run_output_dir / "tracked_pigs_with_ids.mp4"
    )
    annotations_json = cfg.annotations_json or (
        run_output_dir / "annotations_cvat_shapes.json"
    )
    coco_annotations_json = cfg.coco_annotations_json or (
        run_output_dir / "annotations_coco.json"
    )
    clean_coco_annotations_json = cfg.clean_coco_annotations_json or (
        run_output_dir / "annotations_coco_clean_train.json"
    )
    cvat_video_xml = cfg.cvat_video_xml or (
        run_output_dir / "annotations_cvat_video_1_1.xml"
    )
    labels_json = cfg.labels_json or run_output_dir / "labels.json"
    tracker_yaml = cfg.tracker_yaml or (
        run_output_dir / "bytetrack_pig_8.yaml"
    )
    quality_report_json = cfg.quality_report_json or (
        run_output_dir / "tracking_quality_report.json"
    )
    quality_report_csv = cfg.quality_report_csv or (
        run_output_dir / "tracking_quality_report.csv"
    )
    output_video.parent.mkdir(parents=True, exist_ok=True)
    annotations_json.parent.mkdir(parents=True, exist_ok=True)
    coco_annotations_json.parent.mkdir(parents=True, exist_ok=True)
    clean_coco_annotations_json.parent.mkdir(parents=True, exist_ok=True)
    cvat_video_xml.parent.mkdir(parents=True, exist_ok=True)
    labels_json.parent.mkdir(parents=True, exist_ok=True)
    tracker_yaml.parent.mkdir(parents=True, exist_ok=True)
    quality_report_json.parent.mkdir(parents=True, exist_ok=True)
    quality_report_csv.parent.mkdir(parents=True, exist_ok=True)
    return (
        output_video,
        annotations_json,
        coco_annotations_json,
        clean_coco_annotations_json,
        cvat_video_xml,
        labels_json,
        tracker_yaml,
        quality_report_json,
        quality_report_csv,
    )


def write_tracker_yaml(path: Path, cfg: TrackingConfig) -> None:
    """Write a tracker config (ByteTrack or BoT-SORT) for pig videos."""
    track_low_thresh = min(cfg.det_conf, cfg.track_high_conf)
    if cfg.mode == "hybrid_bytetrack":
        lines = [
            "tracker_type: bytetrack",
            f"track_high_thresh: {cfg.track_high_conf:.2f}",
            f"track_low_thresh: {track_low_thresh:.2f}",
            f"new_track_thresh: {cfg.track_high_conf:.2f}",
            f"track_thresh: {cfg.track_high_conf:.2f}",
            f"match_thresh: {cfg.track_match_iou:.2f}",
            "track_buffer: 90",
            "min_box_area: 10",
            "mot20: false",
            "fuse_score: true",
            "proximity_thresh: 0.5",
            "appearance_thresh: 0.25",
            "max_age: 90",
            "n_init: 3",
            "with_reid: true",
            "",
        ]
    elif cfg.tracker_type == "botsort":
        lines = [
            "tracker_type: botsort",
            f"track_high_thresh: {cfg.track_high_conf:.2f}",
            f"track_low_thresh: {track_low_thresh:.2f}",
            f"new_track_thresh: {cfg.track_high_conf:.2f}",
            f"track_thresh: {cfg.track_high_conf:.2f}",
            f"match_thresh: {cfg.track_match_iou:.2f}",
            "track_buffer: 90",
            "min_box_area: 10",
            "mot20: false",
            "fuse_score: true",
            "gmc_method: sparseOptFlow",
            "proximity_thresh: 0.5",
            "appearance_thresh: 0.25",
            "with_reid: true",
            "model: auto",
            "",
        ]
    else:
        lines = [
            "tracker_type: bytetrack",
            f"track_high_thresh: {cfg.track_high_conf:.2f}",
            f"track_low_thresh: {track_low_thresh:.2f}",
            f"new_track_thresh: {cfg.track_high_conf:.2f}",
            f"track_thresh: {cfg.track_high_conf:.2f}",
            f"match_thresh: {cfg.track_match_iou:.2f}",
            "track_buffer: 90",
            "min_box_area: 10",
            "mot20: false",
            "fuse_score: true",
            "proximity_thresh: 0.5",
            "appearance_thresh: 0.25",
            "max_age: 90",
            "n_init: 3",
            "with_reid: false",
            "model: auto",
            "",
        ]
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "CANONICAL_TRACKING_MODES",
    "TrackingConfig",
    "TRACKING_MODE_CHOICES",
    "get_telemetry_summary",
    "resolve_output_paths",
    "tracking_rule_flags_enabled",
    "validate_config",
    "write_tracker_yaml",
]
