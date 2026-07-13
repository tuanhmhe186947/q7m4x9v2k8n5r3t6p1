"""Leakage-safe train-ready feature selection for classification_v2 windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.features.temporal_evidence import (
    WINDOW_TEMPORAL_EVIDENCE_COLUMNS,
)

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
    "identifier_schema_version",
    "scene_frame_uid",
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
    "spatiotemporal_feature_valid_ratio_window",
    *WINDOW_TEMPORAL_EVIDENCE_COLUMNS,
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


def select_window_feature_columns(
    df: pd.DataFrame,
    *,
    feature_whitelist: list[str] | None = None,
) -> list[str]:
    """Return deterministic, inference-safe model-input columns.

    An explicit whitelist is authoritative for exported artifacts. The legacy
    rule-based selection remains available for diagnostics and older callers.
    """
    if feature_whitelist is not None:
        return _validate_explicit_feature_whitelist(df, feature_whitelist)

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
    feature_whitelist: list[str] | None = None,
) -> TrainReadyWindowTables:
    """Split reviewed sequence windows into X, y, mask, and sample_weight."""
    required = [label_col, mask_col]
    missing = [c for c in required if c not in windows.columns]
    if missing:
        raise ValueError(f"Missing train-ready window columns: {missing}")

    feature_cols = select_window_feature_columns(
        windows,
        feature_whitelist=feature_whitelist,
    )
    leakage = sorted(c for c in feature_cols if _is_forbidden_feature_column(c))
    if leakage:
        raise ValueError(f"Leakage-prone columns selected for X: {leakage}")

    x = windows[feature_cols].copy()
    for col in x.columns:
        x[col] = (
            pd.to_numeric(x[col], errors="coerce")
            .replace([float("inf"), float("-inf")], pd.NA)
            .fillna(0.0)
        )

    y = windows[label_col].fillna("").astype(str).copy()
    mask = _to_bool_series(windows[mask_col])

    if sample_weight_col in windows.columns:
        sample_weight = (
            pd.to_numeric(windows[sample_weight_col], errors="coerce")
            .fillna(1.0)
            .clip(0.0, 1.0)
        )
    else:
        sample_weight = pd.Series(1.0, index=windows.index, name=sample_weight_col)
    sample_weight = sample_weight.where(mask, 0.0)

    audit = audit_train_ready_feature_selection(
        windows,
        feature_cols,
        label_col=label_col,
        mask_col=mask_col,
        expected_feature_columns=feature_whitelist,
    )
    return TrainReadyWindowTables(x=x, y=y, mask=mask, sample_weight=sample_weight, audit=audit)


def audit_train_ready_feature_selection(
    windows: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    label_col: str = "behavior_window_label",
    mask_col: str = "window_valid_for_main_train",
    expected_feature_columns: list[str] | None = None,
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
    expected = (
        list(expected_feature_columns)
        if expected_feature_columns is not None
        else None
    )
    missing_expected = (
        [column for column in expected if column not in feature_cols]
        if expected is not None
        else []
    )
    unexpected = (
        [column for column in feature_cols if column not in expected]
        if expected is not None
        else []
    )
    whitelist_match = feature_cols == expected if expected is not None else None
    if missing_expected:
        errors.append(f"missing_expected_features={missing_expected}")
    if unexpected:
        errors.append(f"unexpected_features={unexpected}")
    if expected is not None and not missing_expected and not unexpected:
        if not whitelist_match:
            errors.append("feature_whitelist_order_mismatch")

    return {
        "rows": int(len(windows)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "label_col": label_col,
        "mask_col": mask_col,
        "sample_weight_candidates": [c for c in SAMPLE_WEIGHT_COLUMNS if c in windows.columns],
        "forbidden_selected": forbidden_selected,
        "explicit_whitelist_used": expected is not None,
        "feature_whitelist_match": whitelist_match,
        "missing_expected_features": missing_expected,
        "unexpected_features": unexpected,
        "numeric_columns_not_selected_count": int(len(numeric_not_selected)),
        "numeric_columns_not_selected_sample": numeric_not_selected[:80],
        "errors": errors,
        "warnings": warnings,
    }


def _validate_explicit_feature_whitelist(
    df: pd.DataFrame,
    feature_whitelist: list[str],
) -> list[str]:
    """Validate and preserve the exact order of an explicit trainer whitelist."""
    whitelist = [str(column).strip() for column in feature_whitelist]
    if not whitelist or any(not column for column in whitelist):
        raise ValueError("Explicit feature whitelist must contain named columns")
    if len(whitelist) != len(set(whitelist)):
        raise ValueError("Explicit feature whitelist contains duplicate columns")

    duplicate_input = df.columns[df.columns.duplicated()].tolist()
    if duplicate_input:
        raise ValueError(f"Input contains duplicate columns: {duplicate_input}")

    missing = [column for column in whitelist if column not in df.columns]
    if missing:
        raise ValueError(f"Missing whitelisted feature columns: {missing}")

    forbidden = [
        column for column in whitelist if _is_forbidden_feature_column(column)
    ]
    if forbidden:
        raise ValueError(f"Whitelist contains forbidden feature columns: {forbidden}")

    non_numeric = [
        column
        for column in whitelist
        if not (
            pd.api.types.is_numeric_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
        )
    ]
    if non_numeric:
        raise ValueError(f"Whitelisted feature columns must be numeric: {non_numeric}")
    return whitelist


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
