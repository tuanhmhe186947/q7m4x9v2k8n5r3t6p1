"""Constants and CVAT label schema for fixed-ID pig tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _find_project_root(start: Path) -> Path:
    """Return the nearest parent containing the project metadata."""
    start = start if start.is_dir() else start.parent
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return start


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "Pigs281119_000085_30fps.mp4"
DEFAULT_WEIGHTS_PATH = (
    PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov8.pt"
)
DEFAULT_MASK_PATH = PROJECT_ROOT / "data" / "annotations" / "scene" / "mask.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pred"

ID_VALUES = [f"ID_{idx}" for idx in range(1, 9)]
BEHAVIOR_VALUES = [
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
]
TRACK_COLORS_BGR = {
    1: (65, 105, 225),
    2: (50, 205, 50),
    3: (255, 140, 0),
    4: (220, 20, 60),
    5: (148, 0, 211),
    6: (0, 206, 209),
    7: (255, 20, 147),
    8: (154, 205, 50),
}
# Detection confidence thresholds
DEFAULT_DET_CONF_THRESHOLD = 0.20
DEFAULT_TRACK_HIGH_CONF_THRESHOLD = 0.45
DEFAULT_REVIEW_CONF_THRESHOLD = 0.50
DEFAULT_CONF_THRESHOLD = DEFAULT_DET_CONF_THRESHOLD

# YOLO inference / NMS
DEFAULT_NMS_IOU_THRESHOLD = 0.50
DEFAULT_TRACK_MATCH_IOU_THRESHOLD = 0.75
DEFAULT_MAX_RAW_DETECTIONS = 20

# Expected animal count
DEFAULT_EXPECTED_PIGS = 8
DEFAULT_MAX_EXPORT_VISIBLE_BOXES = 8

# Duplicate detection filtering
DEFAULT_DUP_IOU_THRESHOLD = 0.60
DEFAULT_DUP_CONTAINMENT_THRESHOLD = 0.85
DEFAULT_DUP_CENTER_THRESHOLD = 0.35
DEFAULT_DUP_AREA_RATIO_THRESHOLD = 0.80

# Occlusion state machine
DEFAULT_OCCLUSION_IOU_THRESHOLD = 0.30
DEFAULT_HARD_OCCLUSION_IOU_THRESHOLD = 0.45
DEFAULT_MERGE_DEFICIT_FRAMES = 3
DEFAULT_SPLIT_RECOVERY_FRAMES = 10

# Track lifecycle
DEFAULT_MAX_LOST_FRAMES = 30

# Realtime performance
DEFAULT_DETECT_EVERY_N_FRAMES = 1
DEFAULT_TARGET_FPS = 15
DEFAULT_ENABLE_OFFLINE_SMOOTHING = False

# GT export only
DEFAULT_MAX_INTERPOLATION_GAP = 30
DEFAULT_MARK_INTERPOLATED_REVIEW = True

# Visualization
DEFAULT_VISUAL_OPACITY = 0.75

# Backward compatibility legacy constant
DEFAULT_OVERLAP_THRESHOLD = DEFAULT_DUP_IOU_THRESHOLD

# Track states
TRACK_STATE_VISIBLE = "VISIBLE"
TRACK_STATE_OCCLUDED = "OCCLUDED"
TRACK_STATE_MISSING = "MISSING"
TRACK_STATE_LOST = "LOST"

SCENE_CLEAR = "CLEAR"
SCENE_SOFT_PROXIMITY = "SOFT_PROXIMITY"
SCENE_HARD_OCCLUSION_ARMED = "HARD_OCCLUSION_ARMED"
SCENE_HARD_MERGED = "HARD_MERGED"
SCENE_SPLIT_RECOVERY = "SPLIT_RECOVERY"

TRACKING_RULE_TELEMETRY_KEYS = (
    "hard_merges_triggered",
    "detections_intentionally_ignored",
    "recovery_frames_applied",
)
H1_R2_TELEMETRY_KEYS = (
    "h1_r2_stage_calls",
    "h1_r2_hidden_tracks_offered",
    "h1_r2_competitors_scored",
    "h1_r2_valid_score_pairs",
    "h1_r2_abstained_missing_evidence",
    "h1_r2_abstained_below_threshold",
    "h1_r2_abstained_tie_or_margin",
    "h1_r2_owner_preference_applied",
    "h1_r2_score_invalid",
    "h1_r2_reacquisition_observed",
)

H1_R3_SHADOW_TELEMETRY_KEYS = (
    "h1_r3_shadow_stage_calls",
    "h1_r3_shadow_hidden_tracks_offered",
    "h1_r3_shadow_pair_candidates",
    "h1_r3_shadow_core_eligible_pairs",
    "h1_r3_shadow_optional_appearance_available",
    "h1_r3_shadow_optional_motion_available",
    "h1_r3_shadow_score_pairs",
    "h1_r3_shadow_below_threshold",
    "h1_r3_shadow_margin_failed",
    "h1_r3_shadow_would_activate",
    "h1_r3_shadow_invalid_numeric",
    "h1_r3_shadow_missing_core_overlap",
    "h1_r3_shadow_missing_core_freshness",
)
H2_CDSP_SHADOW_TELEMETRY_KEYS = (
    "h2_shadow_stage_calls",
    "h2_shadow_visible_confirmed_tracks",
    "h2_shadow_dropout_entries",
    "h2_shadow_baseline_state_loss_points",
    "h2_shadow_preservation_candidates",
    "h2_shadow_preservable_states",
    "h2_shadow_unpreservable_missing_core",
    "h2_shadow_unpreservable_low_initial_quality",
    "h2_shadow_states_expired",
    "h2_shadow_states_invalidated",
    "h2_shadow_states_surviving_to_reentry",
    "h2_shadow_reentry_opportunities",
    "h2_shadow_extra_usable_state_at_reentry",
    "h2_shadow_control_preservation_events",
    "h2_shadow_control_overpreservation",
    "h2_shadow_invalid_numeric",
    "h2_shadow_terminal_revival_blocked",
)
TRACKING_ASSOCIATION_PHASES = (
    "visible",
    "visible_high_conf",
    "reid",
    "low_conf_recovery",
)
TRACKING_TIMING_STAGES = (
    "frame",
    "detector",
    "association",
)
TRACKING_INTEGER_TELEMETRY_KEYS = (
    *TRACKING_RULE_TELEMETRY_KEYS,
    *H1_R2_TELEMETRY_KEYS,
    *H1_R3_SHADOW_TELEMETRY_KEYS,
    *H2_CDSP_SHADOW_TELEMETRY_KEYS,
    "frames_processed",
    "detection_frames",
    "skipped_detection_frames",
    "association_calls",
    *(
        f"association_phase_{phase_name}_calls"
        for phase_name in TRACKING_ASSOCIATION_PHASES
    ),
    "association_assignments_accepted",
    "association_assignments_rejected",
    "association_assignments_held",
    "association_assignments_preferred",
    "declared_delay_frames",
    "peak_process_rss_bytes",
    "peak_cuda_memory_allocated_bytes",
    "peak_cuda_memory_reserved_bytes",
    "frame_deadline_miss_count",
    "output_age_deadline_miss_count",
    "max_backlog_frames",
    "final_backlog_frames",
)
TRACKING_FLOAT_TELEMETRY_KEYS = (
    *(
        f"{stage}_time_ms_{statistic}"
        for stage in TRACKING_TIMING_STAGES
        for statistic in ("total", "mean", "p50", "p95")
    ),
    "postprocess_time_ms_total",
    "tracking_loop_effective_fps",
    "effective_fps",
    "source_fps",
    "declared_delay_ms",
    "realtime_factor",
    "backlog_growth_frames_per_second",
    "frame_deadline_ms",
    "frame_deadline_miss_rate",
    "output_age_deadline_ms",
    "output_age_deadline_miss_rate",
    "output_age_ms_mean",
    "output_age_ms_p50",
    "output_age_ms_p95",
    "output_age_ms_max",
    "output_age_ms_final",
)
TRACKING_TEXT_TELEMETRY_KEYS = ("output_timing_contract",)
TRACKING_TELEMETRY_KEYS = (
    *TRACKING_INTEGER_TELEMETRY_KEYS,
    *TRACKING_FLOAT_TELEMETRY_KEYS,
    *TRACKING_TEXT_TELEMETRY_KEYS,
)


def _bgr_to_hex(color: tuple[int, int, int]) -> str:
    blue, green, red = color
    return f"#{red:02X}{green:02X}{blue:02X}"


def build_pig_label_schema() -> list[dict[str, Any]]:
    """Build CVAT labels where Pig_N and attribute ID_N are locked together."""
    labels: list[dict[str, Any]] = []
    for idx, id_value in enumerate(ID_VALUES, start=1):
        attr_base_id = 2522600 + idx * 10
        labels.append(
            {
                "name": f"Pig_{idx}",
                "id": 7872368 + idx - 1,
                "color": _bgr_to_hex(TRACK_COLORS_BGR[idx]),
                "type": "any",
                "attributes": [
                    {
                        "id": attr_base_id + 1,
                        "name": "ID",
                        "input_type": "select",
                        "mutable": False,
                        "values": [id_value],
                        "default_value": id_value,
                    },
                    {
                        "id": attr_base_id + 2,
                        "name": "Behavior",
                        "input_type": "select",
                        "mutable": True,
                        "values": BEHAVIOR_VALUES,
                        "default_value": "lying",
                    },
                    {
                        "id": attr_base_id + 3,
                        "name": "Hidden",
                        "input_type": "select",
                        "mutable": True,
                        "values": ["No", "Yes"],
                        "default_value": "No",
                    },
                ],
            }
        )
    return labels


PIG_LABEL_SCHEMA = build_pig_label_schema()

__all__ = [
    "BEHAVIOR_VALUES",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_DET_CONF_THRESHOLD",
    "DEFAULT_MASK_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_OVERLAP_THRESHOLD",
    "DEFAULT_REVIEW_CONF_THRESHOLD",
    "DEFAULT_TRACK_HIGH_CONF_THRESHOLD",
    "DEFAULT_VIDEO_PATH",
    "DEFAULT_VISUAL_OPACITY",
    "H1_R2_TELEMETRY_KEYS",
    "H1_R3_SHADOW_TELEMETRY_KEYS",
    "DEFAULT_WEIGHTS_PATH",
    "ID_VALUES",
    "PIG_LABEL_SCHEMA",
    "PROJECT_ROOT",
    "SCENE_CLEAR",
    "SCENE_HARD_MERGED",
    "SCENE_HARD_OCCLUSION_ARMED",
    "SCENE_SOFT_PROXIMITY",
    "SCENE_SPLIT_RECOVERY",
    "TRACKING_ASSOCIATION_PHASES",
    "TRACKING_FLOAT_TELEMETRY_KEYS",
    "TRACKING_INTEGER_TELEMETRY_KEYS",
    "TRACKING_RULE_TELEMETRY_KEYS",
    "TRACKING_TELEMETRY_KEYS",
    "TRACKING_TEXT_TELEMETRY_KEYS",
    "TRACKING_TIMING_STAGES",
    "TRACK_COLORS_BGR",
    "build_pig_label_schema",
    "DEFAULT_NMS_IOU_THRESHOLD",
    "DEFAULT_TRACK_MATCH_IOU_THRESHOLD",
    "DEFAULT_MAX_RAW_DETECTIONS",
    "DEFAULT_EXPECTED_PIGS",
    "DEFAULT_MAX_EXPORT_VISIBLE_BOXES",
    "DEFAULT_DUP_IOU_THRESHOLD",
    "DEFAULT_DUP_CONTAINMENT_THRESHOLD",
    "DEFAULT_DUP_CENTER_THRESHOLD",
    "DEFAULT_DUP_AREA_RATIO_THRESHOLD",
    "DEFAULT_OCCLUSION_IOU_THRESHOLD",
    "DEFAULT_HARD_OCCLUSION_IOU_THRESHOLD",
    "DEFAULT_MERGE_DEFICIT_FRAMES",
    "DEFAULT_SPLIT_RECOVERY_FRAMES",
    "DEFAULT_MAX_LOST_FRAMES",
    "DEFAULT_DETECT_EVERY_N_FRAMES",
    "DEFAULT_TARGET_FPS",
    "DEFAULT_ENABLE_OFFLINE_SMOOTHING",
    "DEFAULT_MAX_INTERPOLATION_GAP",
    "DEFAULT_MARK_INTERPOLATED_REVIEW",
    "TRACK_STATE_VISIBLE",
    "TRACK_STATE_OCCLUDED",
    "TRACK_STATE_MISSING",
    "TRACK_STATE_LOST",
]
