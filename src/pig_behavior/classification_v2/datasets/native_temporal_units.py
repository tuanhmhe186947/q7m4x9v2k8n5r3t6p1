"""Native temporal-unit dataset builder for classification_v2.

This module creates the publication-facing primary prediction table: one row
per CVAT 6-frame anchor interval or legacy 16-frame burst. Sequence windows can
still be used as training augmentation, but confirmatory evaluation should
collapse to this table so overlapping windows are not treated as independent
observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

CVAT_SOURCE = "cvat_tracking_xml"
LEGACY_SOURCE = "legacy_recovered"


@dataclass(slots=True)
class NativeTemporalUnitTables:
    manifest: pd.DataFrame
    audit: dict[str, Any]


def build_native_temporal_units(
    intervals: pd.DataFrame,
    reviewed_frames: pd.DataFrame,
) -> NativeTemporalUnitTables:
    """Build one publication-facing row per temporal annotation unit."""
    interval_required = [
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "temporal_label_mode",
        "label_anchor_frame_index",
        "label_window_start",
        "label_window_end",
        "label_frame_count",
        "observed_frame_count",
        "expected_observed_frame_count",
        "temporal_interval_complete",
        "behavior_temporal_final",
        "temporal_consistency_status",
        "timestamp_start_sec",
        "timestamp_end_sec",
        "bbox_valid_ratio_interval",
        "hidden_ratio_interval",
        "visible_ratio_interval",
        "spatiotemporal_feature_valid_ratio_interval",
        "interaction_annotation_policy",
        "interaction_role_policy",
        "label_propagation_policy",
        "allow_label_propagation",
        "requires_partner_context",
        "social_nose_actor_only",
        "fight_group_label",
    ]
    frame_required = [
        "temporal_unit_key",
        "frame_index",
        "behavior_before_review",
        "behavior_after_review",
        "review_decision_applied",
        "review_manual_decision",
        "review_corrected_behavior",
        "review_training_action",
        "review_sample_weight",
        "review_include_in_training",
    ]
    missing_intervals = [c for c in interval_required if c not in intervals.columns]
    missing_frames = [c for c in frame_required if c not in reviewed_frames.columns]
    if missing_intervals:
        raise ValueError(f"Missing temporal interval columns: {missing_intervals}")
    if missing_frames:
        raise ValueError(f"Missing reviewed frame columns: {missing_frames}")

    base = intervals[interval_required].copy()
    duplicate_units = int(base["temporal_unit_key"].duplicated().sum())
    if duplicate_units:
        raise ValueError(f"Duplicate temporal_unit_key in intervals: {duplicate_units}")

    frame_audit = _aggregate_reviewed_frames(reviewed_frames[frame_required].copy())
    manifest = base.merge(frame_audit, on="temporal_unit_key", how="left", validate="one_to_one")
    missing_review_rows = int(manifest["reviewed_frame_count"].isna().sum())
    if missing_review_rows:
        raise ValueError(f"Temporal units without reviewed frames: {missing_review_rows}")

    manifest["behavior_label"] = manifest["behavior_temporal_final"].fillna("").astype(str)
    corrected_mask = manifest["unit_corrected_behavior"].fillna("").astype(str).ne("")
    manifest.loc[corrected_mask, "behavior_label"] = manifest.loc[corrected_mask, "unit_corrected_behavior"]
    manifest["behavior_before_review"] = manifest["behavior_temporal_final"].fillna("").astype(str)
    manifest["behavior_changed_by_review"] = (
        manifest["behavior_label"].fillna("").astype(str) != manifest["behavior_before_review"].fillna("").astype(str)
    )

    manifest["review_include_in_training"] = _as_bool(manifest["review_include_in_training"])
    manifest["native_unit_valid_for_main_eval"] = (
        _as_bool(manifest["temporal_interval_complete"])
        & manifest["review_include_in_training"]
        & manifest["behavior_label"].fillna("").astype(str).ne("")
    )
    manifest["native_unit_sample_weight"] = pd.to_numeric(
        manifest["review_sample_weight_mean"], errors="coerce"
    ).fillna(1.0)
    manifest.loc[~manifest["review_include_in_training"], "native_unit_sample_weight"] = 0.0

    manifest = manifest.sort_values(
        ["source_type", "dataset_id", "video_key", "object_track_key", "label_window_start"],
        kind="stable",
    ).reset_index(drop=True)
    manifest.insert(0, "native_temporal_unit_row_index", range(len(manifest)))

    audit = _audit_native_temporal_units(manifest)
    return NativeTemporalUnitTables(manifest=manifest, audit=audit)


def json_default(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _aggregate_reviewed_frames(frames: pd.DataFrame) -> pd.DataFrame:
    frames["review_include_in_training_bool"] = _as_bool(frames["review_include_in_training"])
    frames["review_sample_weight_num"] = pd.to_numeric(frames["review_sample_weight"], errors="coerce")
    frames["review_decision_applied_bool"] = _as_bool(frames["review_decision_applied"])

    rows: list[dict[str, Any]] = []
    for temporal_unit_key, group in frames.groupby("temporal_unit_key", sort=False):
        corrected = _unique_nonempty(group["review_corrected_behavior"])
        training_actions = _join_unique(group["review_training_action"])
        manual_decisions = _join_unique(group["review_manual_decision"])
        rows.append(
            {
                "temporal_unit_key": temporal_unit_key,
                "reviewed_frame_count": int(len(group)),
                "reviewed_observed_frame_count": int(group["frame_index"].nunique()),
                "frame_start_reviewed": int(pd.to_numeric(group["frame_index"], errors="coerce").min()),
                "frame_end_reviewed": int(pd.to_numeric(group["frame_index"], errors="coerce").max()),
                "frame_behavior_before_review_unique": _join_unique(group["behavior_before_review"]),
                "frame_behavior_after_review_unique": _join_unique(group["behavior_after_review"]),
                "review_decision_applied_any": bool(group["review_decision_applied_bool"].any()),
                "review_manual_decisions_unit": manual_decisions,
                "review_training_actions_unit": training_actions,
                "unit_corrected_behavior": corrected[0] if len(corrected) == 1 else "",
                "unit_corrected_behavior_conflict": bool(len(corrected) > 1),
                "review_include_in_training": bool(group["review_include_in_training_bool"].all()),
                "review_excluded_frame_count": int((~group["review_include_in_training_bool"]).sum()),
                "review_sample_weight_mean": float(group["review_sample_weight_num"].fillna(1.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _audit_native_temporal_units(manifest: pd.DataFrame) -> dict[str, Any]:
    duplicate_temporal_units = int(manifest["temporal_unit_key"].duplicated().sum())
    cvat = manifest["source_type"].eq(CVAT_SOURCE)
    legacy = manifest["source_type"].eq(LEGACY_SOURCE)
    cvat_bad_length = int((cvat & manifest["label_frame_count"].ne(6)).sum())
    legacy_bad_length = int((legacy & manifest["label_frame_count"].ne(16)).sum())
    excluded_units = int((~manifest["review_include_in_training"]).sum())
    corrected_units = int(manifest["behavior_changed_by_review"].sum())
    errors = []
    if duplicate_temporal_units:
        errors.append(f"duplicate_temporal_unit_key={duplicate_temporal_units}")
    if cvat_bad_length:
        errors.append(f"cvat_non_6f_units={cvat_bad_length}")
    if legacy_bad_length:
        errors.append(f"legacy_non_16f_units={legacy_bad_length}")
    if int(manifest["unit_corrected_behavior_conflict"].sum()):
        errors.append("conflicting_corrected_behavior_within_unit")

    return {
        "rows": int(len(manifest)),
        "duplicate_temporal_unit_key": duplicate_temporal_units,
        "source_type_counts": manifest["source_type"].value_counts(dropna=False).to_dict(),
        "behavior_label_counts": manifest["behavior_label"].value_counts(dropna=False).to_dict(),
        "temporal_consistency_status_counts": manifest["temporal_consistency_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "native_unit_valid_for_main_eval_counts": manifest["native_unit_valid_for_main_eval"]
        .value_counts(dropna=False)
        .to_dict(),
        "review_excluded_unit_count": excluded_units,
        "review_corrected_unit_count": corrected_units,
        "cvat_non_6f_units": cvat_bad_length,
        "legacy_non_16f_units": legacy_bad_length,
        "min_reviewed_frame_count": int(manifest["reviewed_frame_count"].min()),
        "max_reviewed_frame_count": int(manifest["reviewed_frame_count"].max()),
        "warnings": ["primary evaluation unit is temporal_unit_key, not overlapping sequence windows"],
        "errors": errors,
    }


def _join_unique(series: pd.Series) -> str:
    values = _unique_nonempty(series)
    return "|".join(values)


def _unique_nonempty(series: pd.Series) -> list[str]:
    values = []
    for value in series.dropna().astype(str):
        cleaned = value.strip()
        if cleaned and cleaned.lower() != "nan" and cleaned not in values:
            values.append(cleaned)
    return sorted(values)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
