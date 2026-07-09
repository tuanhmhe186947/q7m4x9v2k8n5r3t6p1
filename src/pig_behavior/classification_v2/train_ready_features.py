"""Leakage-safe train-ready feature selection for classification_v2 windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

LABEL_COLUMNS = {
    "behavior",
    "behavior_before_review",
    "behavior_after_review",
    "behavior_original_frame",
    "behavior_temporal_final",
    "behavior_window_label",
    "dominant_behavior_in_interval",
    "dominant_behavior_in_unit",
    "raw_dominant_behavior_in_interval",
    "review_behavior_label",
    "review_corrected_behavior",
    "original_behavior",
}

ID_COLUMNS = {
    "dataset_id",
    "video_key",
    "source_video_key",
    "clip_id",
    "task_id",
    "frame_uid",
    "image_key",
    "image_name",
    "object_id_in_image",
    "pig_id",
    "track_id",
    "track_label",
    "object_track_key",
    "temporal_unit_key",
    "temporal_unit_keys_window",
    "review_unit_id",
    "review_unit_id_applied",
    "review_item_id",
    "window_id",
}

PATH_COLUMNS = {
    "crop_path",
    "source_video_path",
    "times_txt_path",
    "frame_path",
    "image_path",
}

AUDIT_PREFIXES = (
    "manual_",
    "review_",
    "raw_",
)

AUDIT_TEXT_COLUMNS = {
    "source_type",
    "source_window_type",
    "temporal_label_mode",
    "temporal_consistency_status",
    "sequence_label_status",
    "window_training_tier_recommendation",
    "window_exclusion_reason",
    "unique_behaviors_window",
    "unique_behaviors_in_interval",
    "raw_unique_behaviors_in_interval",
    "interaction_annotation_policy",
    "interaction_role_policy",
    "label_propagation_policy",
    "review_training_actions_window",
}

MASK_COLUMNS = {
    "window_valid_for_main_train",
    "review_include_in_training",
    "include_in_training_final",
    "use_for_main_train_final",
}

SAMPLE_WEIGHT_COLUMNS = {
    "window_sample_weight",
    "review_sample_weight",
    "training_weight_final",
    "sample_weight",
}

# Exact window-level numeric features that are safe by construction.
EXACT_FEATURE_COLUMNS = {
    "window_length_frames",
    "window_duration_sec",
    "effective_fps",
    "observed_row_count_window",
    "observed_frame_count_window",
    "num_temporal_units_window",
    "bbox_valid_ratio_window",
    "hidden_ratio_window",
    "visible_ratio_window",
    "spatiotemporal_feature_valid_ratio_window",
}

SAFE_FEATURE_PREFIXES = (
    "speed_",
    "path_length_",
    "motion_",
    "accel_",
    "direction_change_",
    "shape_transition_",
    "area_n_std_",
    "aspect_ratio_std_",
    "bbox_stability_",
    "displacement_",
    "nearest_",
    "social_density_",
    "pair_contact_",
    "approach_",
    "separation_",
    "aggression_score_proxy_",
)

# Excluded because these columns are currently computed against a label-derived
# target ROI class. Add class-specific feeder/drinker/toy aggregates before using
# ROI relation features as model inputs.
LEAKY_OR_POLICY_PREFIXES = (
    "target_roi_",
    "roi_target_",
)


@dataclass(slots=True)
class TrainReadyWindowTables:
    x: pd.DataFrame
    y: pd.Series
    mask: pd.Series
    sample_weight: pd.Series
    audit: dict[str, Any]


def select_window_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return deterministic model-input columns from a sequence-window table."""
    numeric_cols = set(df.select_dtypes(include=["number", "bool"]).columns)
    selected: list[str] = []
    for col in df.columns:
        if col not in numeric_cols:
            continue
        if _is_forbidden_feature_column(col):
            continue
        if col in EXACT_FEATURE_COLUMNS or col.startswith(SAFE_FEATURE_PREFIXES):
            selected.append(col)
    return selected


def build_train_ready_window_tables(
    windows: pd.DataFrame,
    *,
    label_col: str = "behavior_window_label",
    mask_col: str = "window_valid_for_main_train",
    sample_weight_col: str = "window_sample_weight",
) -> TrainReadyWindowTables:
    """Split reviewed sequence windows into X, y, mask, and sample_weight."""
    required = [label_col, mask_col]
    missing = [c for c in required if c not in windows.columns]
    if missing:
        raise ValueError(f"Missing train-ready window columns: {missing}")

    feature_cols = select_window_feature_columns(windows)
    leakage = sorted(c for c in feature_cols if _is_forbidden_feature_column(c))
    if leakage:
        raise ValueError(f"Leakage-prone columns selected for X: {leakage}")

    x = windows[feature_cols].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

    y = windows[label_col].fillna("").astype(str).copy()
    mask = _to_bool_series(windows[mask_col])

    if sample_weight_col in windows.columns:
        sample_weight = pd.to_numeric(windows[sample_weight_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    else:
        sample_weight = pd.Series(1.0, index=windows.index, name=sample_weight_col)
    sample_weight = sample_weight.where(mask, 0.0)

    audit = audit_train_ready_feature_selection(windows, feature_cols, label_col=label_col, mask_col=mask_col)
    return TrainReadyWindowTables(x=x, y=y, mask=mask, sample_weight=sample_weight, audit=audit)


def audit_train_ready_feature_selection(
    windows: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    label_col: str = "behavior_window_label",
    mask_col: str = "window_valid_for_main_train",
) -> dict[str, Any]:
    """Audit that model X is separated from label, ID, path, and review columns."""
    if feature_cols is None:
        feature_cols = select_window_feature_columns(windows)

    errors: list[str] = []
    warnings: list[str] = []
    forbidden_selected = sorted(c for c in feature_cols if _is_forbidden_feature_column(c))
    if forbidden_selected:
        errors.append(f"forbidden_columns_selected={forbidden_selected}")

    missing_label = label_col not in windows.columns
    missing_mask = mask_col not in windows.columns
    if missing_label:
        errors.append(f"missing_label_col={label_col}")
    if missing_mask:
        errors.append(f"missing_mask_col={mask_col}")

    if not feature_cols:
        errors.append("no_model_feature_columns_selected")

    numeric_cols = list(windows.select_dtypes(include=["number", "bool"]).columns)
    numeric_not_selected = sorted(set(numeric_cols).difference(feature_cols))

    return {
        "rows": int(len(windows)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "label_col": label_col,
        "mask_col": mask_col,
        "sample_weight_candidates": [c for c in SAMPLE_WEIGHT_COLUMNS if c in windows.columns],
        "forbidden_selected": forbidden_selected,
        "numeric_columns_not_selected_count": int(len(numeric_not_selected)),
        "numeric_columns_not_selected_sample": numeric_not_selected[:80],
        "errors": errors,
        "warnings": warnings,
    }


def _is_forbidden_feature_column(col: str) -> bool:
    return (
        col in LABEL_COLUMNS
        or col in ID_COLUMNS
        or col in PATH_COLUMNS
        or col in AUDIT_TEXT_COLUMNS
        or col in MASK_COLUMNS
        or col in SAMPLE_WEIGHT_COLUMNS
        or col.startswith(AUDIT_PREFIXES)
        or col.startswith(LEAKY_OR_POLICY_PREFIXES)
        or col.endswith("_id")
        or col.endswith("_path")
        or "behavior" in col
        or "label" in col
        or "review" in col
        or "temporal_unit" in col
    )


def _to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
