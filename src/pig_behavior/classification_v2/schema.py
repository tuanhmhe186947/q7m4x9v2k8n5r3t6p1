"""Shared schema and constants for classification dataset v2.

The main design rule is:
- Do not require every frame to contain all 8 pigs.
- Keep actor annotations if the actor bbox and behavior label are valid.
- Track global 8-pig context and local behavior-specific context separately.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------
# Behavior labels
# ---------------------------------------------------------------------

VALID_BEHAVIORS: Final[list[str]] = [
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

VALID_BEHAVIOR_SET: Final[set[str]] = set(VALID_BEHAVIORS)

BEHAVIOR_TO_COARSE: Final[dict[str, str]] = {
    "lying": "resting",
    "sitting": "resting",
    "eat": "feeding",
    "drink": "feeding",
    "move": "locomotion",
    "stand": "locomotion",
    "explore": "locomotion",
    "playwithtoy": "locomotion",
    "social-nose": "social",
    "fight": "social",
}

COARSE_BEHAVIORS: Final[list[str]] = [
    "resting",
    "feeding",
    "locomotion",
    "social",
]


# ---------------------------------------------------------------------
# Behavior-specific feature groups
# ---------------------------------------------------------------------

INTERACTION_BEHAVIORS: Final[set[str]] = {"fight", "social-nose"}

ROI_DOMINANT_BEHAVIORS: Final[set[str]] = {
    "eat",
    "drink",
    "playwithtoy",
}

MOTION_DOMINANT_BEHAVIORS: Final[set[str]] = {
    "move",
    "explore",
    "stand",
}

SHAPE_DOMINANT_BEHAVIORS: Final[set[str]] = {
    "lying",
    "sitting",
}


# ---------------------------------------------------------------------
# Pig IDs
# ---------------------------------------------------------------------

DEFAULT_PIG_IDS: Final[list[str]] = [
    "ID_1",
    "ID_2",
    "ID_3",
    "ID_4",
    "ID_5",
    "ID_6",
    "ID_7",
    "ID_8",
]

DEFAULT_PIG_ID_SET: Final[set[str]] = set(DEFAULT_PIG_IDS)


# ---------------------------------------------------------------------
# Source types
# ---------------------------------------------------------------------

SOURCE_TYPES: Final[set[str]] = {
    "legacy_recovered",
    "cvat_tracking_xml",
    "cvat_selected_native",
}

SOURCE_TYPE_LEGACY: Final[str] = "legacy_recovered"
SOURCE_TYPE_CVAT_TRACKING_XML: Final[str] = "cvat_tracking_xml"
SOURCE_TYPE_CVAT_SELECTED_NATIVE: Final[str] = "cvat_selected_native"


# ---------------------------------------------------------------------
# Annotation scope and context quality
# ---------------------------------------------------------------------

ANNOTATION_SCOPES: Final[set[str]] = {
    "actor_only",
    "selected_actor_group",
    "interaction_pair_or_group",
    "full_context",
    "overfull_context",
}

LOCAL_CONTEXT_QUALITIES: Final[set[str]] = {
    "unknown",
    "actor_only_ok",
    "selected_context_ok",
    "interaction_context_ok",
    "missing_interaction_partner",
    "full_context",
    "sufficient_actor_context",
    "sufficient_interaction_context",
    "needs_review_missing_partner",
}

SOCIAL_FEATURE_QUALITIES: Final[set[str]] = {
    "unknown",
    "not_required",
    "missing_context",
    "missing_partner",
    "partial_context",
    "usable_pair_or_group",
    "interaction_context",
    "full_context",
}

TRAINING_TIERS: Final[set[str]] = {
    "clean",
    "clean_full_context",
    "clean_interaction",
    "actor_only",
    "partial_context",
    "rare_actor_only",
    "legacy_recovered",
    "review",
    "rejected",
}

QA_STATUSES: Final[set[str]] = {
    "ok",
    "review",
    "review_interaction_missing_partner",
    "invalid_bbox",
    "invalid_behavior",
    "hidden",
    "missing_required_value",
    "rejected",
}


# ---------------------------------------------------------------------
# Label policies and sequence views
# ---------------------------------------------------------------------

LABEL_POLICIES: Final[set[str]] = {
    "anchor_label",
    "majority_window",
    "strict_constant_window",
}

SEQUENCE_VIEWS: Final[dict[str, list[int]]] = {
    # Legacy 13-frame temporary mode.
    "legacy_sparse_3_0_6_12": [0, 6, 12],
    "legacy_dense_6_same_span_0_12": [0, 2, 5, 7, 10, 12],
    "legacy_full_dense_0_to_12": list(range(13)),

    # Legacy 16-frame full old-burst mode.
    "legacy_gt_6_frames": [0, 3, 6, 9, 12, 15],
    "legacy_full_dense_0_to_15": list(range(16)),
}

CVAT_SEQUENCE_VIEWS: Final[set[str]] = {
    "cvat_window_6",
    "selected_contiguous_runs",
}


# ---------------------------------------------------------------------
# Canonical frame-object schema
# ---------------------------------------------------------------------

IDENTITY_COLUMNS: Final[list[str]] = [
    "source_type",
    "dataset_id",
    "video_key",
    "source_video_key",
    "clip_id",
    "task_id",
    "frame_uid",
    "image_key",
    "image_name",
    "object_id_in_image",
    "frame_index",
    "relative_frame_index",
    "sequence_frame_count",
    "legacy_sequence_mode",
    "legacy_expected_sequence_length",
    "legacy_anchor_relative_frames",
    "is_legacy_gt_anchor",
    "sequence_complete",
    "sequence_range_valid",
    "timestamp_sec",
    "timestamp_source",
    "image_width",
    "image_height",
]

PIG_COLUMNS: Final[list[str]] = [
    "pig_id",
    "track_id",
    "track_label",
]

BBOX_COLUMNS: Final[list[str]] = [
    "x1_raw",
    "y1_raw",
    "x2_raw",
    "y2_raw",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_valid",
    "bbox_was_clipped",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "cx",
    "cy",
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
    "box_diag",
    "box_diag_n",
    "box_compactness",
]

LABEL_COLUMNS: Final[list[str]] = [
    "behavior",
    "behavior_coarse",
    "hidden",
    "is_actor_label",
    "label_source",
    "bbox_source",
]

CONTEXT_COLUMNS: Final[list[str]] = [
    "global_context_pig_count",
    "global_context_complete_8",
    "missing_global_pig_ids",
    "duplicate_pig_id_in_frame",
    "context_overfull",
    "local_context_pig_count",
    "local_context_quality",
    "annotation_scope",
    "interaction_partner_count",
    "interaction_partner_ids",
    "context_quality",
    "social_feature_quality",
]

QUALITY_COLUMNS: Final[list[str]] = [
    "actor_bbox_valid",
    "actor_quality",
    "use_for_visual_training",
    "use_for_shape_training",
    "use_for_motion_training",
    "use_for_roi_training",
    "use_for_social_training",
    "use_for_main_eval",
    "include_in_training",
    "training_tier",
    "qa_status",
    "sample_weight",
]

PATH_COLUMNS: Final[list[str]] = [
    "crop_path",
    "source_video_path",
    "times_txt_path",
]

CANONICAL_FRAME_OBJECT_COLUMNS: Final[list[str]] = (
    IDENTITY_COLUMNS
    + PIG_COLUMNS
    + BBOX_COLUMNS
    + LABEL_COLUMNS
    + CONTEXT_COLUMNS
    + QUALITY_COLUMNS
    + PATH_COLUMNS
)


# ---------------------------------------------------------------------
# Feature schema placeholders for later steps
# ---------------------------------------------------------------------

GEOMETRY_FEATURE_COLUMNS: Final[list[str]] = [
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
    "box_diag_n",
    "box_compactness",
]

MOTION_FEATURE_COLUMNS: Final[list[str]] = [
    "prev_cx",
    "prev_cy",
    "dt",
    "vx",
    "vy",
    "speed_px_s",
    "speed_n",
    "accel",
    "direction_angle",
    "direction_change",
    "delta_area",
    "delta_aspect",
]

ROI_FEATURE_COLUMNS: Final[list[str]] = [
    "in_feeder",
    "in_drinker",
    "in_toy",
    "iou_feeder",
    "iou_drinker",
    "iou_toy",
    "overlap_ratio_feeder",
    "overlap_ratio_drinker",
    "overlap_ratio_toy",
    "dist_feeder",
    "dist_drinker",
    "dist_toy",
    "nearest_roi_type",
    "nearest_roi_dist",
]

SOCIAL_FEATURE_COLUMNS: Final[list[str]] = [
    "nearest_pig_id",
    "nearest_dist",
    "nearest_dist_n",
    "min_dist_other",
    "num_close_other_010",
    "num_close_other_015",
    "num_close_other_020",
    "social_density",
    "pair_iou_nearest",
    "max_pair_iou",
    "overlap_count",
    "social_missing_mask",
]


# ---------------------------------------------------------------------
# Legacy-compatible output schema
# ---------------------------------------------------------------------

LEGACY_COMPATIBLE_COLUMNS: Final[list[str]] = [
    "img_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "behavior",
    "behavior_coarse",
    "hidden",
    "group_id",
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "speed_feat",
    "min_dist_other",
    "num_close_other",
    "in_feeder",
    "in_drinker",
    "in_toy",
]

LABEL_STRENGTHS: Final[set[str]] = {
    "strong",
    "medium",
    "weak",
    "boundary",
    "unknown",
}

AMBIGUITY_GROUPS: Final[set[str]] = {
    "none",
    "roi_feeding_drinking_toy",
    "aggression_social",
    "motion_state",
    "posture",
    "unknown",
}

REVIEW_DECISIONS: Final[set[str]] = {
    "auto_accept",
    "accept",
    "corrected",
    "exclude",
    "pending",
    "not_required",
}

TRAINING_ACTIONS: Final[set[str]] = {
    "main_train",
    "low_weight_train",
    "robust_train_only",
    "exclude",
    "pending",
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def normalize_behavior(value: object) -> str | None:
    """Normalize behavior text to the canonical label set.

    Returns None when the value is missing or not a known behavior.
    """
    if value is None:
        return None

    behavior = str(value).strip().lower()
    behavior = behavior.replace("_", "-")

    aliases = {
        "social_nose": "social-nose",
        "socialnose": "social-nose",
        "play-with-toy": "playwithtoy",
        "play_with_toy": "playwithtoy",
        "play toy": "playwithtoy",
    }
    behavior = aliases.get(behavior, behavior)

    if behavior not in VALID_BEHAVIOR_SET:
        return None
    return behavior


def behavior_to_coarse(behavior: object) -> str | None:
    """Map a fine behavior label to a coarse behavior group."""
    normalized = normalize_behavior(behavior)
    if normalized is None:
        return None
    return BEHAVIOR_TO_COARSE.get(normalized)


def is_interaction_behavior(behavior: object) -> bool:
    """Return True for behaviors requiring local interaction context."""
    normalized = normalize_behavior(behavior)
    return normalized in INTERACTION_BEHAVIORS


def normalize_hidden(value: object) -> str:
    """Normalize CVAT Hidden attribute to 'Yes' or 'No'."""
    if value is None:
        return "No"

    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "y"}:
        return "Yes"
    return "No"


def normalize_pig_id(value: object) -> str | None:
    """Normalize pig ID to ID_1..ID_8 when possible."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    upper = text.upper().replace(" ", "")
    upper = upper.replace("-", "_")

    if upper.startswith("ID_"):
        candidate = upper
    elif upper.startswith("ID") and upper[2:].isdigit():
        candidate = f"ID_{upper[2:]}"
    elif upper.isdigit():
        candidate = f"ID_{upper}"
    else:
        candidate = upper

    if candidate in DEFAULT_PIG_ID_SET:
        return candidate
    return candidate


def required_columns_missing(
    columns: set[str] | list[str],
    required: list[str],
) -> list[str]:
    """Return required columns missing from an existing column collection."""
    available = set(columns)
    return sorted(set(required).difference(available))


def is_valid_source_type(source_type: object) -> bool:
    """Return True if source_type is one of the supported v2 sources."""
    return str(source_type) in SOURCE_TYPES


def is_valid_annotation_scope(scope: object) -> bool:
    """Return True if annotation scope is known."""
    return str(scope) in ANNOTATION_SCOPES


def is_valid_label_policy(policy: object) -> bool:
    """Return True if label policy is known."""
    return str(policy) in LABEL_POLICIES


def is_valid_sequence_view(view: object) -> bool:
    """Return True if a sequence view is supported."""
    text = str(view)
    return text in SEQUENCE_VIEWS or text in CVAT_SEQUENCE_VIEWS
