"""Audited manifests for the isolated legacy 16-frame development lane.

This module never claims that legacy labels have completed the current human
review. It preserves every native burst, exposes fixed temporal tiers, and
keeps model-development evidence separate from the reviewed all-source lineage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    DEFAULT_TEMPORAL_TIERS,
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
    TEMPORAL_TIER_VIEWS,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIOR_SET, VALID_BEHAVIORS

LEGACY_SOURCE = "legacy_recovered"
LEGACY_NATIVE_LENGTH = 16
LEGACY_DEVELOPMENT_SCOPE = "legacy-only-unreviewed-development"
LEGACY_DEVELOPMENT_SCHEMA_VERSION = (
    "classification_v2.legacy_unreviewed_development.v2"
)
DEFAULT_LEGACY_WINDOW_STRIDE = 3


@dataclass(slots=True)
class LegacyUnreviewedDevelopmentTables:
    """All keyed tables required before a legacy-only model smoke."""

    source_units: pd.DataFrame
    native_units: pd.DataFrame
    all_sliding_windows: pd.DataFrame
    matched_windows: pd.DataFrame
    temporal_selection: pd.DataFrame
    temporal_slot_manifests: dict[str, pd.DataFrame]
    audit: dict[str, Any]


def build_legacy_unreviewed_development_manifests(
    source_frames: pd.DataFrame,
    harmonized_frames: pd.DataFrame,
    intervals: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    temporal_tiers: tuple[int, ...] = DEFAULT_TEMPORAL_TIERS,
    legacy_window_stride: int = DEFAULT_LEGACY_WINDOW_STRIDE,
) -> LegacyUnreviewedDevelopmentTables:
    """Build native-unit and temporal-tier manifests without review fiction."""

    _validate_tier_parameters(temporal_tiers, legacy_window_stride)
    source_units, source_audit = _build_source_unit_manifest(source_frames)
    frame_units, frame_audit = _build_harmonized_unit_summary(
        harmonized_frames
    )
    native_units, native_audit = _build_native_unit_manifest(
        intervals,
        source_units,
        frame_units,
    )
    all_sliding, matched, tier_audit = _build_temporal_tier_manifests(
        windows,
        native_units,
        temporal_tiers=temporal_tiers,
        legacy_window_stride=legacy_window_stride,
    )
    selection, slot_manifests, model_input_audit = (
        _build_temporal_model_input_manifests(
            harmonized_frames,
            all_sliding,
            matched,
            temporal_tiers=temporal_tiers,
        )
    )

    errors = [
        *source_audit["errors"],
        *frame_audit["errors"],
        *native_audit["errors"],
        *tier_audit["errors"],
        *model_input_audit["errors"],
    ]
    audit = {
        "schema_version": LEGACY_DEVELOPMENT_SCHEMA_VERSION,
        "lineage_scope": LEGACY_DEVELOPMENT_SCOPE,
        "source_scope": LEGACY_SOURCE,
        "human_review_complete": False,
        "training_evidence_status": "UNREVIEWED_DEVELOPMENT_ONLY",
        "q2_claim_allowed": False,
        "full_oof_authorized": False,
        "temporal_input_comparison_contract": {
            "native_unit_frames": LEGACY_NATIVE_LENGTH,
            "controlled_tiers_frames": list(temporal_tiers),
            "changed_scientific_family": "temporal_input_length_only",
            "views": list(TEMPORAL_TIER_VIEWS),
            "fold_grain": "recording_video_safe_native_burst",
            "evaluation_grain": "complete_16_frame_native_burst",
            "fixed_controls": [
                "fold_manifest",
                "backbone",
                "image_resolution",
                "temporal_encoder",
                "loss",
                "sampler",
                "seed",
            ],
        },
        "source_audit": source_audit,
        "harmonized_frame_audit": frame_audit,
        "native_unit_audit": native_audit,
        "temporal_tier_audit": tier_audit,
        "temporal_model_input_audit": model_input_audit,
        "errors": errors,
        "warnings": [
            "metrics must be labeled legacy-only-unreviewed-development",
            "results cannot replace reviewed all-source evaluation",
            "pig_id is annotation-local and not a cross-video identity",
        ],
        "valid_for_bounded_development": not errors,
    }
    if errors:
        raise ValueError(
            "Legacy unreviewed development contract failed: "
            + "; ".join(errors)
        )
    return LegacyUnreviewedDevelopmentTables(
        source_units=source_units,
        native_units=native_units,
        all_sliding_windows=all_sliding,
        matched_windows=matched,
        temporal_selection=selection,
        temporal_slot_manifests=slot_manifests,
        audit=audit,
    )


def _build_source_unit_manifest(
    source_frames: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reduce the reference export to one audited row per 16-frame tracklet."""

    required = {
        "source_type",
        "dataset_id",
        "video_key",
        "clip_id",
        "track_id",
        "pig_id",
        "relative_frame_index",
        "behavior",
        "bbox_valid",
        "include_in_training",
        "use_for_main_eval",
        "hidden",
    }
    _require_columns(source_frames, required, "source_frames")
    work = source_frames.copy()
    _require_legacy_only(work, "source_frames")
    _validate_text_keys(
        work,
        ("dataset_id", "video_key", "clip_id", "track_id", "pig_id"),
        "source_frames",
    )
    frame_index = pd.to_numeric(
        work["relative_frame_index"],
        errors="coerce",
    )
    work["_frame_index"] = frame_index
    key_columns = ["dataset_id", "video_key", "track_id", "pig_id"]
    duplicate_rows = int(
        work.duplicated(key_columns + ["_frame_index"], keep=False).sum()
    )
    grouped = work.groupby(key_columns, dropna=False, sort=True)
    units = grouped.agg(
        clip_id=("clip_id", "first"),
        source_row_count=("_frame_index", "size"),
        source_frame_count=("_frame_index", "nunique"),
        source_frame_start=("_frame_index", "min"),
        source_frame_end=("_frame_index", "max"),
        source_behavior=("behavior", "first"),
        source_behavior_count=("behavior", "nunique"),
        source_bbox_all_valid=("bbox_valid", _all_bool),
        source_include_all=("include_in_training", _all_bool),
        source_main_eval_all=("use_for_main_eval", _all_bool),
        source_hidden_yes_count=("hidden", _hidden_yes_count),
    ).reset_index()
    units["source_unit_complete"] = (
        units["source_row_count"].eq(LEGACY_NATIVE_LENGTH)
        & units["source_frame_count"].eq(LEGACY_NATIVE_LENGTH)
        & units["source_frame_start"].eq(0)
        & units["source_frame_end"].eq(LEGACY_NATIVE_LENGTH - 1)
        & units["source_behavior_count"].eq(1)
    )
    units["lineage_scope"] = LEGACY_DEVELOPMENT_SCOPE
    units["human_review_complete"] = False
    units = units.sort_values(key_columns, kind="mergesort").reset_index(
        drop=True
    )

    invalid_frame_index = int(
        (
            frame_index.isna()
            | frame_index.mod(1).ne(0)
            | frame_index.lt(0)
            | frame_index.ge(LEGACY_NATIVE_LENGTH)
        ).sum()
    )
    invalid_behaviors = sorted(
        set(work["behavior"].fillna("").astype(str))
        .difference(VALID_BEHAVIOR_SET)
    )
    incomplete_units = int((~units["source_unit_complete"]).sum())
    errors: list[str] = []
    if duplicate_rows:
        errors.append(f"duplicate_source_unit_frame_rows={duplicate_rows}")
    if invalid_frame_index:
        errors.append(f"invalid_source_relative_frame_rows={invalid_frame_index}")
    if invalid_behaviors:
        errors.append(f"invalid_source_behaviors={invalid_behaviors}")
    if incomplete_units:
        errors.append(f"incomplete_source_units={incomplete_units}")
    return units, {
        "rows": int(len(work)),
        "native_units": int(len(units)),
        "duplicate_source_unit_frame_rows": duplicate_rows,
        "invalid_relative_frame_rows": invalid_frame_index,
        "incomplete_native_units": incomplete_units,
        "behavior_unit_counts": _ordered_counts(
            units["source_behavior"],
            VALID_BEHAVIORS,
        ),
        "hidden_yes_rows": int(_hidden_yes_count(work["hidden"])),
        "include_false_rows": int((~_as_bool(work["include_in_training"])).sum()),
        "main_eval_false_rows": int((~_as_bool(work["use_for_main_eval"])).sum()),
        "errors": errors,
    }


def _build_harmonized_unit_summary(
    harmonized_frames: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit that harmonization preserved every source row and native burst."""

    required = {
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "track_id",
        "pig_id",
        "frame_index",
        "timestamp_sec",
        "temporal_unit_key",
        "behavior_temporal_final",
        "bbox_valid",
        "spatiotemporal_feature_valid",
        "include_in_training",
        "use_for_main_eval",
    }
    _require_columns(harmonized_frames, required, "harmonized_frames")
    work = harmonized_frames.copy()
    _require_legacy_only(work, "harmonized_frames")
    _validate_text_keys(
        work,
        (
            "dataset_id",
            "video_key",
            "object_track_key",
            "track_id",
            "pig_id",
            "temporal_unit_key",
        ),
        "harmonized_frames",
    )
    frame_index = pd.to_numeric(work["frame_index"], errors="coerce")
    work["_frame_index"] = frame_index
    duplicate_rows = int(
        work.duplicated(
            ["temporal_unit_key", "_frame_index"],
            keep=False,
        ).sum()
    )
    grouped = work.groupby("temporal_unit_key", dropna=False, sort=True)
    units = grouped.agg(
        dataset_id=("dataset_id", "first"),
        video_key=("video_key", "first"),
        object_track_key=("object_track_key", "first"),
        track_id=("track_id", "first"),
        pig_id=("pig_id", "first"),
        harmonized_row_count=("_frame_index", "size"),
        harmonized_frame_count=("_frame_index", "nunique"),
        harmonized_frame_start=("_frame_index", "min"),
        harmonized_frame_end=("_frame_index", "max"),
        harmonized_behavior=("behavior_temporal_final", "first"),
        harmonized_behavior_count=("behavior_temporal_final", "nunique"),
        harmonized_bbox_all_valid=("bbox_valid", _all_bool),
        harmonized_spatial_all_valid=(
            "spatiotemporal_feature_valid",
            _all_bool,
        ),
        harmonized_include_all=("include_in_training", _all_bool),
        harmonized_main_eval_all=("use_for_main_eval", _all_bool),
    ).reset_index()
    frame_span = units["harmonized_frame_end"].sub(
        units["harmonized_frame_start"]
    ).add(1)
    units["harmonized_unit_complete"] = (
        units["harmonized_row_count"].eq(LEGACY_NATIVE_LENGTH)
        & units["harmonized_frame_count"].eq(LEGACY_NATIVE_LENGTH)
        & frame_span.eq(LEGACY_NATIVE_LENGTH)
        & units["harmonized_behavior_count"].eq(1)
    )
    invalid_frame_index = int(
        (
            frame_index.isna()
            | frame_index.mod(1).ne(0)
            | frame_index.lt(0)
        ).sum()
    )
    incomplete_units = int((~units["harmonized_unit_complete"]).sum())
    errors: list[str] = []
    if duplicate_rows:
        errors.append(f"duplicate_harmonized_unit_frame_rows={duplicate_rows}")
    if invalid_frame_index:
        errors.append(
            f"invalid_harmonized_frame_index_rows={invalid_frame_index}"
        )
    if incomplete_units:
        errors.append(f"incomplete_harmonized_units={incomplete_units}")
    return units, {
        "rows": int(len(work)),
        "native_units": int(len(units)),
        "duplicate_temporal_unit_frame_rows": duplicate_rows,
        "invalid_frame_index_rows": invalid_frame_index,
        "incomplete_native_units": incomplete_units,
        "errors": errors,
    }


def _build_native_unit_manifest(
    intervals: pd.DataFrame,
    source_units: pd.DataFrame,
    frame_units: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one compatibility-safe, explicitly unreviewed row per burst."""

    required = {
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "pig_id",
        "track_id",
        "label_window_start",
        "label_window_end",
        "label_frame_count",
        "observed_frame_count",
        "expected_observed_frame_count",
        "temporal_interval_complete",
        "behavior_temporal_final",
        "behavior_consistency_in_interval",
        "bbox_valid_ratio_interval",
        "hidden_ratio_interval",
        "spatiotemporal_feature_valid_ratio_interval",
    }
    _require_columns(intervals, required, "intervals")
    work = intervals.copy()
    _require_legacy_only(work, "intervals")
    _validate_text_keys(
        work,
        (
            "temporal_unit_key",
            "dataset_id",
            "video_key",
            "object_track_key",
            "track_id",
            "pig_id",
        ),
        "intervals",
    )
    duplicate_units = int(
        work["temporal_unit_key"].duplicated(keep=False).sum()
    )
    if duplicate_units:
        raise ValueError(f"duplicate interval units={duplicate_units}")

    native = work.copy()
    native = native.merge(
        frame_units,
        on=[
            "temporal_unit_key",
            "dataset_id",
            "video_key",
            "object_track_key",
            "track_id",
            "pig_id",
        ],
        how="left",
        validate="one_to_one",
    )
    native = native.merge(
        source_units,
        on=["dataset_id", "video_key", "track_id", "pig_id"],
        how="left",
        validate="one_to_one",
    )
    missing_source = int(native["source_row_count"].isna().sum())
    missing_harmonized = int(native["harmonized_row_count"].isna().sum())
    behavior = native["behavior_temporal_final"].fillna("").astype(str)
    interval_complete = _as_bool(native["temporal_interval_complete"])
    consistency = _as_bool(native["behavior_consistency_in_interval"])
    label_count = pd.to_numeric(native["label_frame_count"], errors="coerce")
    label_start = pd.to_numeric(native["label_window_start"], errors="coerce")
    label_end = pd.to_numeric(native["label_window_end"], errors="coerce")
    observed = pd.to_numeric(native["observed_frame_count"], errors="coerce")
    expected_observed = pd.to_numeric(
        native["expected_observed_frame_count"],
        errors="coerce",
    )
    bbox_ratio = pd.to_numeric(
        native["bbox_valid_ratio_interval"],
        errors="coerce",
    )
    spatial_ratio = pd.to_numeric(
        native["spatiotemporal_feature_valid_ratio_interval"],
        errors="coerce",
    )
    source_label_match = behavior.eq(native["source_behavior"].astype(str))
    harmonized_label_match = behavior.eq(
        native["harmonized_behavior"].astype(str)
    )
    harmonized_start = pd.to_numeric(
        native["harmonized_frame_start"],
        errors="coerce",
    )
    harmonized_end = pd.to_numeric(
        native["harmonized_frame_end"],
        errors="coerce",
    )
    harmonized_interval_bounds_match = (
        harmonized_start.eq(label_start) & harmonized_end.eq(label_end)
    )
    interval_geometry_valid = (
        label_start.notna()
        & label_end.notna()
        & label_start.mod(1).eq(0)
        & label_end.mod(1).eq(0)
        & label_end.sub(label_start).add(1).eq(LEGACY_NATIVE_LENGTH)
    )
    technical_valid = (
        behavior.isin(VALID_BEHAVIOR_SET)
        & interval_complete
        & consistency
        & interval_geometry_valid
        & label_count.eq(LEGACY_NATIVE_LENGTH)
        & observed.eq(LEGACY_NATIVE_LENGTH)
        & expected_observed.eq(LEGACY_NATIVE_LENGTH)
        & np.isclose(bbox_ratio, 1.0, atol=1e-12)
        & np.isclose(spatial_ratio, 1.0, atol=1e-12)
        & native["source_unit_complete"].fillna(False).astype(bool)
        & native["harmonized_unit_complete"].fillna(False).astype(bool)
        & harmonized_interval_bounds_match
        & source_label_match
        & harmonized_label_match
        & native["source_include_all"].fillna(False).astype(bool)
        & native["source_main_eval_all"].fillna(False).astype(bool)
        & native["harmonized_include_all"].fillna(False).astype(bool)
        & native["harmonized_main_eval_all"].fillna(False).astype(bool)
    )
    native["behavior_label"] = behavior
    native["native_unit_valid_for_development"] = technical_valid
    native["native_unit_valid_for_main_eval"] = technical_valid
    native["harmonized_interval_bounds_match"] = (
        harmonized_interval_bounds_match
    )
    native["native_unit_validity_basis"] = "technical_unreviewed_v1"
    native["native_unit_exclusion_reason"] = [
        _native_exclusion_reason(row)
        for row in native.itertuples(index=False)
    ]
    native["lineage_scope"] = LEGACY_DEVELOPMENT_SCOPE
    native["human_review_complete"] = False
    native["review_status"] = "unreviewed_development"
    native = native.sort_values("temporal_unit_key", kind="mergesort")
    native = native.reset_index(drop=True)

    errors: list[str] = []
    if missing_source:
        errors.append(f"native_units_missing_source_rows={missing_source}")
    if missing_harmonized:
        errors.append(
            f"native_units_missing_harmonized_rows={missing_harmonized}"
        )
    bound_mismatch = int((~harmonized_interval_bounds_match).sum())
    if bound_mismatch:
        errors.append(
            f"harmonized_interval_bound_mismatch_units={bound_mismatch}"
        )
    if len(native) != len(source_units):
        errors.append(
            "native_source_unit_count_mismatch="
            f"{len(native)}:{len(source_units)}"
        )
    if len(native) != len(frame_units):
        errors.append(
            "native_harmonized_unit_count_mismatch="
            f"{len(native)}:{len(frame_units)}"
        )
    return native, {
        "rows": int(len(native)),
        "duplicate_temporal_unit_key": duplicate_units,
        "missing_source_units": missing_source,
        "missing_harmonized_units": missing_harmonized,
        "valid_development_units": int(technical_valid.sum()),
        "invalid_development_units": int((~technical_valid).sum()),
        "invalid_interval_geometry_units": int(
            (~interval_geometry_valid).sum()
        ),
        "harmonized_interval_bound_mismatch_units": bound_mismatch,
        "behavior_counts": _ordered_counts(behavior, VALID_BEHAVIORS),
        "human_review_complete": False,
        "hidden_used_as_exclusion": False,
        "errors": errors,
    }


def _build_temporal_tier_manifests(
    windows: pd.DataFrame,
    native_units: pd.DataFrame,
    *,
    temporal_tiers: tuple[int, ...],
    legacy_window_stride: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build natural and sample-matched views for T6/T8/T12/T16."""

    required = {
        "window_id",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "window_length_frames",
        "window_start_frame",
        "window_end_frame",
        "temporal_unit_keys_json",
        "num_temporal_units_window",
        "behavior_window_label",
        "window_valid_for_main_train",
    }
    _require_columns(windows, required, "windows")
    work = windows.copy()
    _require_legacy_only(work, "windows")
    _validate_text_keys(
        work,
        ("window_id", "dataset_id", "video_key", "object_track_key"),
        "windows",
    )
    duplicate_windows = int(work["window_id"].duplicated(keep=False).sum())
    if duplicate_windows:
        raise ValueError(f"duplicate window_id rows={duplicate_windows}")

    work["temporal_unit_key"] = work["temporal_unit_keys_json"].map(
        _parse_single_temporal_unit_key
    )
    lengths = pd.to_numeric(work["window_length_frames"], errors="coerce")
    starts = pd.to_numeric(work["window_start_frame"], errors="coerce")
    ends = pd.to_numeric(work["window_end_frame"], errors="coerce")
    work["window_length_frames"] = lengths
    work["window_start_frame"] = starts
    work["window_end_frame"] = ends
    invalid_numeric = (
        lengths.isna()
        | starts.isna()
        | ends.isna()
        | lengths.mod(1).ne(0)
        | starts.mod(1).ne(0)
        | ends.mod(1).ne(0)
        | ~ends.sub(starts).add(1).eq(lengths)
    )
    if invalid_numeric.any():
        raise ValueError(
            f"invalid temporal window geometry rows={int(invalid_numeric.sum())}"
        )
    unexpected_lengths = sorted(
        set(lengths.astype(int)).difference(temporal_tiers)
    )
    if unexpected_lengths:
        raise ValueError(f"unexpected temporal tiers={unexpected_lengths}")
    multi_unit = pd.to_numeric(
        work["num_temporal_units_window"],
        errors="coerce",
    ).ne(1)
    if multi_unit.any():
        raise ValueError(
            f"windows crossing native units={int(multi_unit.sum())}"
        )

    native_columns = [
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "behavior_label",
        "label_window_start",
        "label_window_end",
        "native_unit_valid_for_main_eval",
    ]
    joined = work.merge(
        native_units[native_columns],
        on="temporal_unit_key",
        how="left",
        suffixes=("", "_native"),
        validate="many_to_one",
    )
    missing_native = int(joined["behavior_label"].isna().sum())
    source_mismatch = _column_mismatch(joined, "source_type")
    dataset_mismatch = _column_mismatch(joined, "dataset_id")
    video_mismatch = _column_mismatch(joined, "video_key")
    track_mismatch = _column_mismatch(joined, "object_track_key")
    behavior_mismatch = int(
        joined["behavior_window_label"].fillna("").astype(str).ne(
            joined["behavior_label"].fillna("").astype(str)
        ).sum()
    )
    contained = (
        joined["window_start_frame"].ge(joined["label_window_start"])
        & joined["window_end_frame"].le(joined["label_window_end"])
    )
    outside_native = int((~contained.fillna(False)).sum())

    joined["temporal_tier"] = joined["window_length_frames"].astype(int).map(
        lambda value: f"T{value}"
    )
    window_valid = _as_bool(joined["window_valid_for_main_train"])
    native_valid = _as_bool(joined["native_unit_valid_for_main_eval"])
    joined["tier_window_valid"] = window_valid & native_valid & contained
    valid_counts = joined.groupby(
        ["temporal_unit_key", "temporal_tier"],
        sort=False,
    )["tier_window_valid"].transform("sum")
    joined["tier_event_mass_weight"] = np.where(
        joined["tier_window_valid"],
        1.0 / valid_counts.clip(lower=1),
        0.0,
    )
    joined["lineage_scope"] = LEGACY_DEVELOPMENT_SCOPE
    joined["human_review_complete"] = False
    joined["temporal_view"] = "all_sliding_event_balanced"

    expected_by_length = {
        length: ((LEGACY_NATIVE_LENGTH - length) // legacy_window_stride) + 1
        for length in temporal_tiers
    }
    support = joined.groupby(
        ["temporal_unit_key", "window_length_frames"],
        sort=True,
    ).size()
    support_errors = 0
    lattice_errors = 0
    for (_, length), count in support.items():
        if int(count) != expected_by_length[int(length)]:
            support_errors += 1
    for (_, length), group in joined.groupby(
        ["temporal_unit_key", "window_length_frames"],
        sort=True,
    ):
        native_starts = pd.to_numeric(
            group["label_window_start"],
            errors="coerce",
        ).unique()
        if len(native_starts) != 1 or not np.isfinite(native_starts[0]):
            lattice_errors += 1
            continue
        native_start = int(native_starts[0])
        expected_starts = [
            native_start + offset
            for offset in range(
                0,
                LEGACY_NATIVE_LENGTH - int(length) + 1,
                legacy_window_stride,
            )
        ]
        observed_starts = sorted(
            group["window_start_frame"].astype(int).tolist()
        )
        expected_ends = [start + int(length) - 1 for start in expected_starts]
        observed_ends = sorted(group["window_end_frame"].astype(int).tolist())
        if observed_starts != expected_starts or observed_ends != expected_ends:
            lattice_errors += 1
    expected_pairs = len(native_units) * len(temporal_tiers)
    missing_pairs = expected_pairs - len(support)
    mass = joined.loc[joined["tier_window_valid"]].groupby(
        ["temporal_unit_key", "temporal_tier"],
    )["tier_event_mass_weight"].sum()
    mass_error = float((mass - 1.0).abs().max()) if len(mass) else 0.0
    valid_pair_support = joined.groupby(
        ["temporal_unit_key", "temporal_tier"],
        sort=True,
    )["tier_window_valid"].sum()
    valid_native_keys = set(
        native_units.loc[
            _as_bool(native_units["native_unit_valid_for_main_eval"]),
            "temporal_unit_key",
        ].astype(str)
    )
    valid_pairs_without_windows = int(
        sum(
            str(unit_key) in valid_native_keys and int(count) == 0
            for (unit_key, _), count in valid_pair_support.items()
        )
    )

    native_center = (
        joined["label_window_start"] + joined["label_window_end"]
    ) / 2.0
    window_center = (
        joined["window_start_frame"] + joined["window_end_frame"]
    ) / 2.0
    joined["center_offset_frames"] = (window_center - native_center).abs()
    ranked = joined.sort_values(
        [
            "temporal_unit_key",
            "window_length_frames",
            "tier_window_valid",
            "center_offset_frames",
            "window_start_frame",
            "window_id",
        ],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
    )
    matched = ranked.drop_duplicates(
        ["temporal_unit_key", "window_length_frames"],
        keep="first",
    ).copy()
    matched["temporal_view"] = "one_centered_window_matched"
    matched["tier_event_mass_weight"] = matched["tier_window_valid"].astype(
        float
    )
    matched_duplicate_pairs = int(
        matched.duplicated(
            ["temporal_unit_key", "window_length_frames"],
            keep=False,
        ).sum()
    )

    output_columns = [
        "window_id",
        "temporal_unit_key",
        "temporal_tier",
        "window_length_frames",
        "window_start_frame",
        "window_end_frame",
        "behavior_label",
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "window_valid_for_main_train",
        "native_unit_valid_for_main_eval",
        "tier_window_valid",
        "tier_event_mass_weight",
        "center_offset_frames",
        "temporal_view",
        "lineage_scope",
        "human_review_complete",
    ]
    all_sliding = joined[output_columns].copy().reset_index(drop=True)
    matched = matched[output_columns].sort_values(
        ["window_length_frames", "temporal_unit_key"],
        kind="mergesort",
    ).reset_index(drop=True)

    errors: list[str] = []
    counts = {
        "missing_native_window_rows": missing_native,
        "source_mismatch_rows": source_mismatch,
        "dataset_mismatch_rows": dataset_mismatch,
        "video_mismatch_rows": video_mismatch,
        "track_mismatch_rows": track_mismatch,
        "behavior_mismatch_rows": behavior_mismatch,
        "outside_native_unit_rows": outside_native,
        "tier_support_count_mismatches": support_errors,
        "tier_window_lattice_mismatches": lattice_errors,
        "missing_native_tier_pairs": missing_pairs,
        "valid_native_tier_pairs_without_windows": (
            valid_pairs_without_windows
        ),
        "matched_duplicate_pairs": matched_duplicate_pairs,
    }
    errors.extend(f"{name}={count}" for name, count in counts.items() if count)
    if len(matched) != expected_pairs:
        errors.append(
            f"matched_row_count={len(matched)}:expected={expected_pairs}"
        )
    if mass_error > 1e-12:
        errors.append(f"tier_event_mass_error={mass_error}")
    return all_sliding, matched, {
        "all_sliding_rows": int(len(all_sliding)),
        "matched_rows": int(len(matched)),
        "native_units": int(len(native_units)),
        "temporal_tiers": [f"T{length}" for length in temporal_tiers],
        "expected_windows_per_unit_by_tier": {
            f"T{length}": count
            for length, count in expected_by_length.items()
        },
        "all_sliding_rows_by_tier": _ordered_counts(
            all_sliding["temporal_tier"],
            [f"T{length}" for length in temporal_tiers],
        ),
        "matched_rows_by_tier": _ordered_counts(
            matched["temporal_tier"],
            [f"T{length}" for length in temporal_tiers],
        ),
        "valid_rows_by_tier": _ordered_counts(
            all_sliding.loc[
                all_sliding["tier_window_valid"],
                "temporal_tier",
            ],
            [f"T{length}" for length in temporal_tiers],
        ),
        "tier_event_mass_max_abs_error": mass_error,
        "input_window_order_preserved": all_sliding["window_id"].tolist()
        == work["window_id"].astype(str).tolist(),
        "rows_dropped": 0,
        "labels_changed": 0,
        **counts,
        "errors": errors,
    }


def _build_temporal_model_input_manifests(
    harmonized_frames: pd.DataFrame,
    all_sliding: pd.DataFrame,
    matched: pd.DataFrame,
    *,
    temporal_tiers: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Create exact model-selection and observed-time slot contracts."""

    required = [
        "source_type",
        "object_track_key",
        "frame_index",
        "timestamp_sec",
    ]
    _require_columns(harmonized_frames, set(required), "harmonized_frames")
    frame_lookup = harmonized_frames[required].copy()
    _require_legacy_only(frame_lookup, "harmonized_frames")
    frame_lookup["frame_index"] = pd.to_numeric(
        frame_lookup["frame_index"],
        errors="coerce",
    )
    duplicate_frames = int(
        frame_lookup.duplicated(
            ["object_track_key", "frame_index"],
            keep=False,
        ).sum()
    )
    invalid_frame_index = int(
        (
            frame_lookup["frame_index"].isna()
            | frame_lookup["frame_index"].mod(1).ne(0)
        ).sum()
    )
    if not invalid_frame_index:
        frame_lookup["frame_index"] = frame_lookup["frame_index"].astype(int)

    windows = all_sliding.copy().reset_index(drop=True)
    windows["_source_window_order"] = np.arange(len(windows), dtype=np.int64)
    matched_ids = set(matched["window_id"].astype(str))
    selection = windows[
        [
            "window_id",
            "temporal_unit_key",
            "temporal_tier",
            "window_length_frames",
            "tier_window_valid",
            "lineage_scope",
            "human_review_complete",
        ]
    ].copy()

    slot_manifests: dict[str, pd.DataFrame] = {}
    view_audits: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if duplicate_frames:
        errors.append(f"duplicate_temporal_slot_source_frames={duplicate_frames}")
    if invalid_frame_index:
        errors.append(
            f"invalid_temporal_slot_source_frame_indices={invalid_frame_index}"
        )

    for view_name, spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.items():
        length = int(spec["sequence_length"])
        if length not in temporal_tiers:
            continue
        tier_mask = windows["temporal_tier"].eq(f"T{length}")
        if spec["sampling_view"] == "one_centered_window_matched":
            tier_mask &= windows["window_id"].astype(str).isin(matched_ids)
        selection[str(spec["selection_column"])] = tier_mask.to_numpy()
        selected = windows.loc[tier_mask].copy()
        slots, slot_audit = _build_temporal_slot_manifest(
            selected,
            frame_lookup,
            view_name=view_name,
            sequence_length=length,
        )
        slot_manifests[view_name] = slots
        view_audits[view_name] = slot_audit
        errors.extend(
            f"{view_name}:{error}" for error in slot_audit["errors"]
        )

    return selection, slot_manifests, {
        "selection_rows": int(len(selection)),
        "selection_order_matches_window_universe": selection[
            "window_id"
        ].astype(str).tolist()
        == all_sliding["window_id"].astype(str).tolist(),
        "selection_columns": [
            str(spec["selection_column"])
            for spec in LEGACY_TEMPORAL_MODEL_VIEW_SPECS.values()
        ],
        "view_specs": LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
        "view_audits": view_audits,
        "duplicate_temporal_slot_source_frames": duplicate_frames,
        "invalid_temporal_slot_source_frame_indices": invalid_frame_index,
        "rows_dropped": 0,
        "labels_changed": 0,
        "errors": errors,
    }


def _build_temporal_slot_manifest(
    windows: pd.DataFrame,
    frame_lookup: pd.DataFrame,
    *,
    view_name: str,
    sequence_length: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expand selected windows into exact, non-resampled observed-time slots."""

    records: list[dict[str, Any]] = []
    for item_order, row in enumerate(windows.itertuples(index=False)):
        start = int(row.window_start_frame)
        end = int(row.window_end_frame)
        if end - start + 1 != sequence_length:
            raise ValueError(
                f"{view_name} window length mismatch for {row.window_id}"
            )
        view_item_id = f"{view_name}|{row.window_id}"
        for slot_index, frame_index in enumerate(range(start, end + 1)):
            records.append(
                {
                    "temporal_view_name": view_name,
                    "view_item_id": view_item_id,
                    "parent_window_id": str(row.window_id),
                    "temporal_unit_key": str(row.temporal_unit_key),
                    "source_type": LEGACY_SOURCE,
                    "source_native_length_audit": LEGACY_NATIVE_LENGTH,
                    "item_order": item_order,
                    "slot_index": slot_index,
                    "slot_key": f"{view_item_id}|slot={slot_index}",
                    "declared_sequence_length": sequence_length,
                    "object_track_key_audit": str(row.object_track_key),
                    "frame_index_expected_audit": frame_index,
                }
            )
    slots = pd.DataFrame.from_records(records)
    if slots.empty:
        return slots, {
            "windows": 0,
            "slot_rows": 0,
            "missing_observed_slots": 0,
            "invalid_timing_slots": 0,
            "nonpositive_time_delta_slots": 0,
            "duplicate_slot_key_rows": 0,
            "errors": ["empty_temporal_slot_manifest"],
        }

    lookup = frame_lookup[
        ["object_track_key", "frame_index", "timestamp_sec"]
    ].rename(
        columns={
            "frame_index": "frame_index_expected_audit",
            "timestamp_sec": "timestamp_sec_audit",
        }
    )
    slots = slots.merge(
        lookup,
        left_on=["object_track_key_audit", "frame_index_expected_audit"],
        right_on=["object_track_key", "frame_index_expected_audit"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    slots["observed_mask"] = slots["_merge"].eq("both")
    timestamp = pd.to_numeric(slots["timestamp_sec_audit"], errors="coerce")
    slots["timing_valid_mask"] = slots["observed_mask"] & np.isfinite(
        timestamp
    )
    slots["timestamp_sec_audit"] = timestamp
    slots["time_coordinate_kind"] = "observed_seconds"
    slots["time_value"] = timestamp - slots.groupby(
        "view_item_id",
        sort=False,
    )["timestamp_sec_audit"].transform("first")
    slots["time_delta"] = slots.groupby("view_item_id", sort=False)[
        "timestamp_sec_audit"
    ].diff()
    slots.loc[slots["slot_index"].eq(0), "time_delta"] = 0.0
    slots["length_mask"] = True
    slots["padding_mask"] = False
    slots["lineage_scope"] = LEGACY_DEVELOPMENT_SCOPE
    slots["human_review_complete"] = False

    missing_observed = int((~slots["observed_mask"]).sum())
    invalid_timing = int((~slots["timing_valid_mask"]).sum())
    nonpositive_delta = int(
        (
            slots["slot_index"].gt(0)
            & pd.to_numeric(slots["time_delta"], errors="coerce").le(0.0)
        ).sum()
    )
    duplicate_slot_keys = int(slots["slot_key"].duplicated(keep=False).sum())
    errors = []
    counts = {
        "missing_observed_slots": missing_observed,
        "invalid_timing_slots": invalid_timing,
        "nonpositive_time_delta_slots": nonpositive_delta,
        "duplicate_slot_key_rows": duplicate_slot_keys,
    }
    errors.extend(f"{name}={count}" for name, count in counts.items() if count)
    expected_rows = len(windows) * sequence_length
    if len(slots) != expected_rows:
        errors.append(f"slot_rows={len(slots)}:expected={expected_rows}")

    output_columns = [
        "temporal_view_name",
        "view_item_id",
        "parent_window_id",
        "temporal_unit_key",
        "source_type",
        "source_native_length_audit",
        "item_order",
        "slot_index",
        "slot_key",
        "declared_sequence_length",
        "object_track_key_audit",
        "frame_index_expected_audit",
        "timestamp_sec_audit",
        "time_coordinate_kind",
        "time_value",
        "time_delta",
        "length_mask",
        "observed_mask",
        "timing_valid_mask",
        "padding_mask",
        "lineage_scope",
        "human_review_complete",
    ]
    slots = slots[output_columns].reset_index(drop=True)
    return slots, {
        "windows": int(len(windows)),
        "slot_rows": int(len(slots)),
        "sequence_length": sequence_length,
        **counts,
        "rows_dropped": 0,
        "errors": errors,
    }


def _native_exclusion_reason(row: Any) -> str:
    """Explain every technically invalid unit without deleting it."""

    if bool(row.native_unit_valid_for_development):
        return ""
    reasons: list[str] = []
    if str(row.behavior_temporal_final) not in VALID_BEHAVIOR_SET:
        reasons.append("invalid_behavior")
    if not bool(row.temporal_interval_complete):
        reasons.append("incomplete_interval")
    if not bool(row.behavior_consistency_in_interval):
        reasons.append("mixed_behavior")
    try:
        label_start = float(row.label_window_start)
        label_end = float(row.label_window_end)
        interval_geometry_valid = (
            np.isfinite(label_start)
            and np.isfinite(label_end)
            and label_start.is_integer()
            and label_end.is_integer()
            and label_end - label_start + 1 == LEGACY_NATIVE_LENGTH
        )
    except (TypeError, ValueError, OverflowError):
        interval_geometry_valid = False
    if not interval_geometry_valid:
        reasons.append("invalid_interval_geometry")
    if not bool(row.source_unit_complete):
        reasons.append("source_unit_incomplete")
    if not bool(row.harmonized_unit_complete):
        reasons.append("harmonized_unit_incomplete")
    if not bool(row.harmonized_interval_bounds_match):
        reasons.append("harmonized_interval_boundary_mismatch")
    if not bool(row.source_include_all) or not bool(row.harmonized_include_all):
        reasons.append("source_excluded")
    if not bool(row.source_main_eval_all) or not bool(
        row.harmonized_main_eval_all
    ):
        reasons.append("main_eval_excluded")
    if str(row.source_behavior) != str(row.behavior_temporal_final):
        reasons.append("source_label_mismatch")
    if str(row.harmonized_behavior) != str(row.behavior_temporal_final):
        reasons.append("harmonized_label_mismatch")
    return "|".join(reasons) or "technical_quality_invalid"


def _parse_single_temporal_unit_key(value: object) -> str:
    """Require the unambiguous JSON key encoding produced after harmonization."""

    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid temporal_unit_keys_json={value!r}") from exc
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError(
            "legacy temporal window must contain exactly one native unit: "
            f"{value!r}"
        )
    key = str(parsed[0]).strip()
    if not key:
        raise ValueError("legacy temporal window contains a blank native key")
    return key


def _validate_tier_parameters(
    temporal_tiers: tuple[int, ...],
    legacy_window_stride: int,
) -> None:
    if temporal_tiers != tuple(sorted(set(temporal_tiers))):
        raise ValueError("temporal_tiers must be unique and sorted")
    if temporal_tiers != DEFAULT_TEMPORAL_TIERS:
        raise ValueError(
            f"temporal_tiers must equal {DEFAULT_TEMPORAL_TIERS}"
        )
    if legacy_window_stride <= 0:
        raise ValueError("legacy_window_stride must be > 0")


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")


def _require_legacy_only(frame: pd.DataFrame, name: str) -> None:
    source = frame["source_type"].fillna("").astype(str).str.strip()
    observed = sorted(set(source))
    if observed != [LEGACY_SOURCE]:
        raise ValueError(f"{name} source scope must be legacy-only: {observed}")


def _validate_text_keys(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    name: str,
) -> None:
    errors: list[str] = []
    for column in columns:
        raw = frame[column].fillna("").astype(str)
        clean = raw.str.strip()
        blank = int(clean.eq("").sum())
        padded = int(raw.ne(clean).sum())
        if blank:
            errors.append(f"blank_{column}={blank}")
        if padded:
            errors.append(f"padded_{column}={padded}")
    if errors:
        raise ValueError(f"{name} key contract failed: {'; '.join(errors)}")


def _column_mismatch(frame: pd.DataFrame, column: str) -> int:
    return int(
        frame[column].fillna("").astype(str).ne(
            frame[f"{column}_native"].fillna("").astype(str)
        ).sum()
    )


def _ordered_counts(
    series: pd.Series,
    order: list[str] | tuple[str, ...],
) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    return {name: int(counts.get(name, 0)) for name in order}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    allowed = {"true", "1", "yes", "y", "t", "false", "0", "no", "n", "f"}
    invalid = ~normalized.isin(allowed)
    if invalid.any():
        raise ValueError(f"invalid boolean values={sorted(set(normalized[invalid]))}")
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _all_bool(series: pd.Series) -> bool:
    return bool(_as_bool(series).all())


def _hidden_yes_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().str.lower().eq("yes").sum())


__all__ = [
    "DEFAULT_TEMPORAL_TIERS",
    "LEGACY_DEVELOPMENT_SCHEMA_VERSION",
    "LEGACY_DEVELOPMENT_SCOPE",
    "LEGACY_TEMPORAL_MODEL_VIEW_SPECS",
    "TEMPORAL_TIER_VIEWS",
    "LegacyUnreviewedDevelopmentTables",
    "build_legacy_unreviewed_development_manifests",
]
