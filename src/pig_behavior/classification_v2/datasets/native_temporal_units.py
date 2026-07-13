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

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

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

    intervals = intervals.copy()
    reviewed_frames = reviewed_frames.copy()
    input_alignment = _validate_input_alignment(intervals, reviewed_frames)

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
    reviewed_label = manifest["unit_behavior_after_review"].fillna("").astype(str)
    reviewed_label_mask = reviewed_label.ne("") & ~manifest[
        "unit_behavior_after_review_conflict"
    ]
    manifest.loc[reviewed_label_mask, "behavior_label"] = reviewed_label.loc[
        reviewed_label_mask
    ]
    manifest["behavior_before_review"] = manifest["behavior_temporal_final"].fillna("").astype(str)
    manifest["behavior_changed_by_review"] = (
        manifest["behavior_label"].fillna("").astype(str)
        != manifest["behavior_before_review"].fillna("").astype(str)
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
    audit["input_alignment"] = input_alignment
    return NativeTemporalUnitTables(manifest=manifest, audit=audit)


def _validate_input_alignment(
    intervals: pd.DataFrame,
    reviewed_frames: pd.DataFrame,
) -> dict[str, Any]:
    """Reject ambiguous keys, frame coverage, decisions, and review weights."""

    interval_keys = _clean_keys(intervals["temporal_unit_key"])
    frame_keys = _clean_keys(reviewed_frames["temporal_unit_key"])
    frame_indices = pd.to_numeric(reviewed_frames["frame_index"], errors="coerce")
    invalid_frame_index = (
        frame_indices.isna()
        | ~np.isfinite(frame_indices)
        | frame_indices.lt(0)
        | frame_indices.ne(np.floor(frame_indices))
    )
    valid_frame_keys = frame_keys.loc[~invalid_frame_index]
    valid_frame_indices = frame_indices.loc[~invalid_frame_index]
    duplicate_frame_rows = int(
        pd.DataFrame(
            {
                "temporal_unit_key": valid_frame_keys,
                "frame_index": valid_frame_indices,
            }
        ).duplicated(keep=False).sum()
    )
    interval_set = set(interval_keys)
    frame_set = set(frame_keys)
    missing_frame_units = interval_set.difference(frame_set)
    extra_frame_units = frame_set.difference(interval_set)

    include_invalid = _invalid_bool_count(
        reviewed_frames["review_include_in_training"]
    )
    decision_invalid = _invalid_bool_count(reviewed_frames["review_decision_applied"])
    weight_text = (
        reviewed_frames["review_sample_weight"].fillna("").astype(str).str.strip()
    )
    missing_weights = weight_text.eq("")
    weights = pd.to_numeric(reviewed_frames["review_sample_weight"], errors="coerce")
    invalid_weights = int(
        (
            ~missing_weights
            & (weights.isna() | ~np.isfinite(weights) | weights.lt(0))
        ).sum()
    )
    after_labels = reviewed_frames["behavior_after_review"].fillna("").astype(str)
    invalid_after_labels = sorted(
        set(after_labels).difference(VALID_BEHAVIORS)
    )

    counts = {
        "blank_interval_temporal_unit_key": int(interval_keys.eq("").sum()),
        "duplicate_interval_temporal_unit_key_rows": int(
            interval_keys.duplicated(keep=False).sum()
        ),
        "blank_reviewed_temporal_unit_key": int(frame_keys.eq("").sum()),
        "invalid_reviewed_frame_index_rows": int(invalid_frame_index.sum()),
        "duplicate_reviewed_unit_frame_rows": duplicate_frame_rows,
        "interval_units_without_reviewed_frames": len(missing_frame_units),
        "reviewed_frame_units_without_interval": len(extra_frame_units),
        "invalid_review_include_values": include_invalid,
        "invalid_review_decision_applied_values": decision_invalid,
        "invalid_review_sample_weight_rows": invalid_weights,
        "invalid_behavior_after_review_values": len(invalid_after_labels),
    }
    errors = [f"{name}={count}" for name, count in counts.items() if count]
    if invalid_after_labels:
        errors.append(
            f"invalid_behavior_after_review_examples={invalid_after_labels[:10]}"
        )
    if errors:
        raise ValueError("native temporal input alignment failed: " + "; ".join(errors))
    return {
        "interval_rows": int(len(intervals)),
        "reviewed_frame_rows": int(len(reviewed_frames)),
        "interval_temporal_units": int(len(interval_set)),
        "reviewed_frame_temporal_units": int(len(frame_set)),
        "defaulted_review_sample_weight_rows": int(missing_weights.sum()),
        **counts,
        "errors": [],
        "warnings": (
            ["blank_review_sample_weight_defaults_to_one"]
            if missing_weights.any()
            else []
        ),
    }


def _clean_keys(series: pd.Series) -> pd.Series:
    """Normalize keys for validation without changing merge values."""

    return series.fillna("").astype(str).str.strip()


def _invalid_bool_count(series: pd.Series) -> int:
    """Count values that cannot be interpreted as explicit booleans."""

    if pd.api.types.is_bool_dtype(series):
        return int(series.isna().sum())
    allowed = {"true", "1", "yes", "y", "t", "false", "0", "no", "n", "f"}
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return int((~normalized.isin(allowed)).sum())


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
    frames["review_include_in_training_bool"] = _as_bool(
        frames["review_include_in_training"]
    )
    frames["review_sample_weight_num"] = pd.to_numeric(
        frames["review_sample_weight"],
        errors="coerce",
    )
    frames["review_decision_applied_bool"] = _as_bool(frames["review_decision_applied"])
    frames["frame_index_num"] = pd.to_numeric(frames["frame_index"], errors="raise")

    rows: list[dict[str, Any]] = []
    for temporal_unit_key, group in frames.groupby("temporal_unit_key", sort=False):
        corrected = _unique_nonempty(group["review_corrected_behavior"])
        behavior_after = _unique_nonempty(group["behavior_after_review"])
        training_actions = _join_unique(group["review_training_action"])
        manual_decisions = _join_unique(group["review_manual_decision"])
        decision_any = bool(group["review_decision_applied_bool"].any())
        decision_all = bool(group["review_decision_applied_bool"].all())
        corrected_rows = int(
            group["review_corrected_behavior"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
        corrected_without_decision = bool(corrected_rows and not decision_all)
        corrected_after_mismatch = bool(
            len(corrected) == 1
            and len(behavior_after) == 1
            and corrected[0] != behavior_after[0]
        )
        rows.append(
            {
                "temporal_unit_key": temporal_unit_key,
                "reviewed_frame_count": int(len(group)),
                "reviewed_observed_frame_count": int(
                    group["frame_index_num"].nunique()
                ),
                "frame_start_reviewed": int(group["frame_index_num"].min()),
                "frame_end_reviewed": int(group["frame_index_num"].max()),
                "frame_behavior_before_review_unique": _join_unique(
                    group["behavior_before_review"]
                ),
                "frame_behavior_after_review_unique": _join_unique(group["behavior_after_review"]),
                "unit_behavior_after_review": (
                    behavior_after[0] if len(behavior_after) == 1 else ""
                ),
                "unit_behavior_after_review_conflict": bool(
                    len(behavior_after) > 1
                ),
                "review_decision_applied_any": decision_any,
                "review_decision_applied_all": decision_all,
                "review_decision_applied_partial": bool(
                    decision_any and not decision_all
                ),
                "review_manual_decisions_unit": manual_decisions,
                "review_training_actions_unit": training_actions,
                "unit_corrected_behavior": corrected[0] if len(corrected) == 1 else "",
                "unit_corrected_behavior_conflict": bool(len(corrected) > 1),
                "unit_corrected_behavior_row_count": corrected_rows,
                "unit_corrected_without_applied_decision": (
                    corrected_without_decision
                ),
                "unit_corrected_behavior_after_mismatch": corrected_after_mismatch,
                "review_include_in_training": bool(group["review_include_in_training_bool"].all()),
                "review_excluded_frame_count": int(
                    (~group["review_include_in_training_bool"]).sum()
                ),
                "review_sample_weight_mean": float(
                    group["review_sample_weight_num"].fillna(1.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _audit_native_temporal_units(manifest: pd.DataFrame) -> dict[str, Any]:
    duplicate_temporal_units = int(manifest["temporal_unit_key"].duplicated().sum())
    cvat = manifest["source_type"].eq(CVAT_SOURCE)
    legacy = manifest["source_type"].eq(LEGACY_SOURCE)
    cvat_bad_length = int((cvat & manifest["label_frame_count"].ne(6)).sum())
    legacy_bad_length = int((legacy & manifest["label_frame_count"].ne(16)).sum())
    reviewed_row_count_mismatch = int(
        manifest["reviewed_frame_count"].ne(manifest["label_frame_count"]).sum()
    )
    reviewed_unique_frame_mismatch = int(
        manifest["reviewed_observed_frame_count"]
        .ne(manifest["label_frame_count"])
        .sum()
    )
    reviewed_start_mismatch = int(
        manifest["frame_start_reviewed"]
        .ne(manifest["label_window_start"])
        .sum()
    )
    reviewed_end_mismatch = int(
        manifest["frame_end_reviewed"].ne(manifest["label_window_end"]).sum()
    )
    partial_review_scope = int(manifest["review_decision_applied_partial"].sum())
    corrected_without_decision = int(
        manifest["unit_corrected_without_applied_decision"].sum()
    )
    corrected_after_mismatch = int(
        manifest["unit_corrected_behavior_after_mismatch"].sum()
    )
    applied_after_conflict = int(
        (
            manifest["review_decision_applied_any"]
            & manifest["unit_behavior_after_review_conflict"]
        ).sum()
    )
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
    coverage_counts = {
        "reviewed_row_count_mismatch": reviewed_row_count_mismatch,
        "reviewed_unique_frame_mismatch": reviewed_unique_frame_mismatch,
        "reviewed_start_frame_mismatch": reviewed_start_mismatch,
        "reviewed_end_frame_mismatch": reviewed_end_mismatch,
        "partial_review_decision_scope": partial_review_scope,
        "corrected_without_applied_decision": corrected_without_decision,
        "corrected_behavior_after_review_mismatch": corrected_after_mismatch,
        "applied_review_with_behavior_after_conflict": applied_after_conflict,
    }
    errors.extend(
        f"{name}={count}" for name, count in coverage_counts.items() if count
    )

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
        **coverage_counts,
        "min_reviewed_frame_count": int(manifest["reviewed_frame_count"].min()),
        "max_reviewed_frame_count": int(manifest["reviewed_frame_count"].max()),
        "warnings": [
            "primary evaluation unit is temporal_unit_key, not overlapping "
            "sequence windows"
        ],
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
