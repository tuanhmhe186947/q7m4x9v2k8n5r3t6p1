"""Sequence/window manifest builder for classification_v2.

This module consumes harmonized/enhanced frame-object features and creates a
long-format training-window table. Each output row is one candidate window for a
single tracked pig/object, with window-specific aggregate features. This avoids
using 16-frame or 6-frame unit means as if they represented every 6/8/12/16
training window.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.lineage_claims import (
    add_optional_lineage_claims_to_audit,
    attach_optional_lineage_claims,
    require_lineage_claims_preserved,
    resolve_optional_lineage_claims,
)
from pig_behavior.classification_v2.features.pen_context import (
    empty_pen_context_summary,
    recompute_pen_motion_for_view,
    summarize_pen_context,
)
from pig_behavior.classification_v2.features.temporal_evidence import (
    WINDOW_TEMPORAL_EVIDENCE_COLUMNS,
    TemporalEvidenceConfig,
    summarize_social_motion_dynamics,
    summarize_temporal_evidence,
)
from pig_behavior.classification_v2.features.temporal_harmonization import (
    CVAT_SOURCE_TYPES,
    LEGACY_SOURCE_TYPE,
    TemporalHarmonizationConfig,
    build_temporal_label_intervals,
    harmonize_temporal_labels,
)
from pig_behavior.classification_v2.sources.temporal_provenance import (
    audit_source_frame_clock,
)


@dataclass(slots=True)
class SequenceWindowConfig:
    """Configuration for sequence/window candidate generation."""

    window_lengths: tuple[int, ...] = (6, 8, 12, 16)
    legacy_window_stride: int = 3
    cvat_window_stride_intervals: int = 1
    cvat_label_stride: int = 6
    legacy_expected_sequence_length: int = 16
    default_fps: float | None = None
    min_bbox_valid_ratio: float = 1.0
    max_hidden_ratio_main: float = 0.25
    max_hidden_run_ratio_main: float = 0.20
    max_hidden_ratio_robust: float = 0.50
    max_hidden_run_ratio_robust: float = 0.40
    exclude_high_hidden_from_main: bool = True
    min_spatiotemporal_valid_ratio: float = 1.0
    include_mixed_windows: bool = True
    max_windows_per_track: int | None = None
    aggregate_observed_rows_only: bool = True
    stationary_speed_threshold: float = 0.002
    active_speed_threshold: float = 0.006
    stationary_speed_threshold_per_second: float = 0.06
    active_speed_threshold_per_second: float = 0.18
    turning_angle_threshold_rad: float = float(np.pi / 6.0)
    behavior_review_requirement: str = "optional_for_diagnostic"
    include_legacy_sparse_s6_at16: bool = False

    def validate(self) -> None:
        if not self.window_lengths:
            raise ValueError("window_lengths must not be empty")
        if any(w <= 0 for w in self.window_lengths):
            raise ValueError("all window_lengths must be > 0")
        if self.legacy_window_stride <= 0:
            raise ValueError("legacy_window_stride must be > 0")
        if self.cvat_window_stride_intervals <= 0:
            raise ValueError("cvat_window_stride_intervals must be > 0")
        if self.cvat_label_stride <= 0:
            raise ValueError("cvat_label_stride must be > 0")
        if self.default_fps is not None and self.default_fps <= 0:
            raise ValueError("default_fps must be None or > 0")
        if not (0 <= self.min_bbox_valid_ratio <= 1):
            raise ValueError("min_bbox_valid_ratio must be in [0, 1]")
        if not (0 <= self.max_hidden_ratio_main <= 1):
            raise ValueError("max_hidden_ratio_main must be in [0, 1]")
        if not (0 <= self.max_hidden_run_ratio_main <= 1):
            raise ValueError("max_hidden_run_ratio_main must be in [0, 1]")
        if not (0 <= self.max_hidden_ratio_robust <= 1):
            raise ValueError("max_hidden_ratio_robust must be in [0, 1]")
        if not (0 <= self.max_hidden_run_ratio_robust <= 1):
            raise ValueError("max_hidden_run_ratio_robust must be in [0, 1]")
        if self.max_hidden_ratio_main > self.max_hidden_ratio_robust:
            raise ValueError(
                "max_hidden_ratio_main must not exceed max_hidden_ratio_robust"
            )
        if self.max_hidden_run_ratio_main > self.max_hidden_run_ratio_robust:
            raise ValueError(
                "max_hidden_run_ratio_main must not exceed "
                "max_hidden_run_ratio_robust"
            )
        if not (0 <= self.min_spatiotemporal_valid_ratio <= 1):
            raise ValueError("min_spatiotemporal_valid_ratio must be in [0, 1]")
        if self.max_windows_per_track is not None and self.max_windows_per_track <= 0:
            raise ValueError("max_windows_per_track must be None or > 0")
        self.temporal_evidence_config().validate()
        if self.behavior_review_requirement not in {
            "optional_for_diagnostic",
            "full_native_unit_review_required",
        }:
            raise ValueError("invalid behavior_review_requirement")

    def temporal_evidence_config(self) -> TemporalEvidenceConfig:
        """Return thresholds shared by every generated window."""

        return TemporalEvidenceConfig(
            stationary_speed_threshold=self.stationary_speed_threshold,
            active_speed_threshold=self.active_speed_threshold,
            stationary_speed_threshold_per_second=(
                self.stationary_speed_threshold_per_second
            ),
            active_speed_threshold_per_second=(
                self.active_speed_threshold_per_second
            ),
            turning_angle_threshold_rad=self.turning_angle_threshold_rad,
        )


def build_sequence_windows(
    frame_features: pd.DataFrame,
    *,
    window_lengths: Sequence[int] = (6, 8, 12, 16),
    legacy_window_stride: int = 3,
    cvat_window_stride_intervals: int = 1,
    cvat_label_stride: int = 6,
    legacy_expected_sequence_length: int = 16,
    default_fps: float | None = None,
    min_bbox_valid_ratio: float = 1.0,
    max_hidden_ratio_main: float = 0.25,
    max_hidden_run_ratio_main: float = 0.20,
    max_hidden_ratio_robust: float = 0.50,
    max_hidden_run_ratio_robust: float = 0.40,
    exclude_high_hidden_from_main: bool = True,
    min_spatiotemporal_valid_ratio: float = 1.0,
    include_mixed_windows: bool = True,
    max_windows_per_track: int | None = None,
    stationary_speed_threshold: float = 0.002,
    active_speed_threshold: float = 0.006,
    stationary_speed_threshold_per_second: float = 0.06,
    active_speed_threshold_per_second: float = 0.18,
    turning_angle_threshold_rad: float = float(np.pi / 6.0),
    behavior_review_requirement: str = "optional_for_diagnostic",
    include_legacy_sparse_s6_at16: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build harmonized frame features, intervals, and window manifest.

    Returns
    -------
    harmonized_frames, temporal_intervals, sequence_windows
    """
    claims = resolve_optional_lineage_claims(
        frame_features,
        artifact_name="sequence window input",
    )
    config = SequenceWindowConfig(
        window_lengths=tuple(int(w) for w in window_lengths),
        legacy_window_stride=legacy_window_stride,
        cvat_window_stride_intervals=cvat_window_stride_intervals,
        cvat_label_stride=cvat_label_stride,
        legacy_expected_sequence_length=legacy_expected_sequence_length,
        default_fps=default_fps,
        min_bbox_valid_ratio=min_bbox_valid_ratio,
        max_hidden_ratio_main=max_hidden_ratio_main,
        max_hidden_run_ratio_main=max_hidden_run_ratio_main,
        max_hidden_ratio_robust=max_hidden_ratio_robust,
        max_hidden_run_ratio_robust=max_hidden_run_ratio_robust,
        exclude_high_hidden_from_main=exclude_high_hidden_from_main,
        min_spatiotemporal_valid_ratio=min_spatiotemporal_valid_ratio,
        include_mixed_windows=include_mixed_windows,
        max_windows_per_track=max_windows_per_track,
        stationary_speed_threshold=stationary_speed_threshold,
        active_speed_threshold=active_speed_threshold,
        stationary_speed_threshold_per_second=(
            stationary_speed_threshold_per_second
        ),
        active_speed_threshold_per_second=active_speed_threshold_per_second,
        turning_angle_threshold_rad=turning_angle_threshold_rad,
        behavior_review_requirement=behavior_review_requirement,
        include_legacy_sparse_s6_at16=include_legacy_sparse_s6_at16,
    )
    config.validate()

    harmonized = harmonize_temporal_labels(
        frame_features,
        cvat_label_stride=cvat_label_stride,
        legacy_expected_sequence_length=legacy_expected_sequence_length,
    )
    interval_config = TemporalHarmonizationConfig(
        cvat_label_stride=cvat_label_stride,
        legacy_expected_sequence_length=legacy_expected_sequence_length,
    )
    intervals = build_temporal_label_intervals(harmonized, config=interval_config)
    windows = _build_windows_from_harmonized(harmonized, intervals, config)
    windows = attach_optional_lineage_claims(windows, claims)
    require_lineage_claims_preserved(
        frame_features,
        windows,
        source_name="sequence window input",
        derived_name="sequence window output",
    )
    return harmonized, intervals, windows


def audit_sequence_windows(
    windows: pd.DataFrame, intervals: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Return an audit summary for generated sequence windows."""
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "window_id",
        "source_type",
        "object_track_key",
        "window_length_frames",
        "window_start_frame",
        "window_end_frame",
        "behavior_window_label",
        "sequence_label_status",
        "window_valid_for_main_train",
        "window_training_tier_recommendation",
        "window_exclusion_reason",
        "hidden_burden_ratio_window",
        "hidden_longest_run_ratio_window",
        "hidden_window_policy_tier",
        "annotation_consistency_status",
        "human_reviewed_behavior_consistency_status",
        "behavior_review_coverage_ratio_window",
        "behavior_review_label_resolution_ratio_window",
        "all_temporal_units_behavior_reviewed",
        "all_temporal_units_behavior_label_resolved",
        "all_temporal_units_behavior_train_eligible",
        "feature_computation_grain",
        "pair_scope_key",
        "view_type",
        "sampling_pattern",
        "selected_frame_offsets",
        "selected_frame_indices",
        "selected_timestamps_seconds",
        "pair_delta_frames",
        "pair_delta_seconds",
        "constituent_native_unit_keys",
        "primary_cross_source_eligible",
        "pair_recomputed_for_view",
        "aggregate_recomputed_for_view",
    ]
    missing = [c for c in required if c not in windows.columns]
    if missing:
        errors.append(f"missing_window_columns={missing}")

    if not windows.empty and {"window_end_frame", "window_start_frame"}.issubset(windows.columns):
        bad = int(
            (
                pd.to_numeric(windows["window_end_frame"], errors="coerce")
                < pd.to_numeric(windows["window_start_frame"], errors="coerce")
            ).sum()
        )
        if bad:
            errors.append(f"invalid_window_span_count={bad}")

    if not windows.empty:
        invalid_main = windows[
            windows.get("window_valid_for_main_train", False)
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
            & ~windows.get("sequence_label_status", "").astype(str).eq("stable")
        ]
        if len(invalid_main):
            errors.append(f"main_train_windows_not_stable={len(invalid_main)}")

        coverage = pd.to_numeric(
            windows.get("behavior_review_coverage_ratio_window", 0.0), errors="coerce"
        ).fillna(0.0)
        resolution = pd.to_numeric(
            windows.get("behavior_review_label_resolution_ratio_window", 0.0), errors="coerce"
        ).fillna(0.0)
        review_status = windows.get(
            "human_reviewed_behavior_consistency_status",
            pd.Series("unreviewed", index=windows.index),
        ).astype(str)
        fields_present = _to_bool_series(
            windows.get("behavior_review_fields_present", pd.Series(False, index=windows.index))
        )
        invalid_review = _to_bool_series(
            windows.get("window_valid_for_main_train", pd.Series(False, index=windows.index))
        ) & (
            fields_present
            & (
                (coverage < 1.0)
                | (resolution < 1.0)
                | ~review_status.eq("stable")
            )
        )
        if invalid_review.any():
            errors.append(
                "main_train_windows_fail_behavior_review_policy="
                f"{int(invalid_review.sum())}"
            )

        main_mask = _to_bool_series(
            windows.get(
                "window_valid_for_main_train",
                pd.Series(False, index=windows.index),
            )
        )
        grain = windows["feature_computation_grain"].fillna("").astype(str)
        invalid_grain = grain.ne("FINAL_VIEW_FEATURES")
        if invalid_grain.any():
            errors.append(
                "invalid_final_view_feature_grain="
                f"{int(invalid_grain.sum())}"
            )
        invalid_pair_scope = (
            windows["pair_scope_key"]
            .fillna("")
            .astype(str)
            .ne(windows["window_id"].fillna("").astype(str))
        )
        if invalid_pair_scope.any():
            errors.append(
                f"final_view_pair_scope_mismatch={int(invalid_pair_scope.sum())}"
            )
        pair_recomputed = _to_bool_series(windows["pair_recomputed_for_view"])
        aggregate_recomputed = _to_bool_series(
            windows["aggregate_recomputed_for_view"]
        )
        invalid_recompute_claim = main_mask & (
            ~pair_recomputed | ~aggregate_recomputed
        )
        if invalid_recompute_claim.any():
            errors.append(
                "main_train_view_not_recomputed="
                f"{int(invalid_recompute_claim.sum())}"
            )
        view_identity = windows["view_type"].fillna("").astype(str)
        sampling_identity = windows["sampling_pattern"].fillna("").astype(str)
        sparse_s6 = view_identity.eq("S6@16")
        expected_view_type = "T" + windows["window_length_frames"].astype(
            "int64"
        ).astype(str) + "_contiguous"
        invalid_view_identity = (~sparse_s6) & (
            view_identity.ne(expected_view_type)
            | sampling_identity.ne("contiguous")
        )
        invalid_sparse_identity = sparse_s6 & (
            ~windows["source_type"].fillna("").astype(str).eq(LEGACY_SOURCE_TYPE)
            | ~windows["window_length_frames"].eq(6)
            | ~sampling_identity.eq("uniform_sparse_offsets_0_3_6_9_12_15")
            | ~windows["selected_frame_offsets"]
            .fillna("")
            .astype(str)
            .eq("[0,3,6,9,12,15]")
            | ~windows["pair_delta_frames"]
            .fillna("")
            .astype(str)
            .eq("[3,3,3,3,3]")
            | _to_bool_series(windows["primary_cross_source_eligible"])
        )
        invalid_view_identity |= invalid_sparse_identity
        if invalid_view_identity.any():
            errors.append(
                f"invalid_contiguous_view_identity={int(invalid_view_identity.sum())}"
            )
        hidden_tier = windows.get(
            "hidden_window_policy_tier",
            pd.Series("", index=windows.index),
        ).astype(str)
        policy_enabled = _to_bool_series(
            windows.get(
                "hidden_exclusion_policy_enabled",
                pd.Series(False, index=windows.index),
            )
        )
        invalid_hidden_main = main_mask & policy_enabled & hidden_tier.ne("main_train")
        if invalid_hidden_main.any():
            errors.append(
                "main_train_windows_fail_hidden_policy="
                f"{int(invalid_hidden_main.sum())}"
            )
        training_tier = windows.get(
            "window_training_tier_recommendation",
            pd.Series("", index=windows.index),
        ).astype(str)
        invalid_hidden_exclude = hidden_tier.eq("exclude") & training_tier.ne(
            "exclude"
        )
        if invalid_hidden_exclude.any():
            errors.append(
                "hidden_exclude_windows_not_excluded="
                f"{int(invalid_hidden_exclude.sum())}"
            )
        sample_weight = pd.to_numeric(
            windows.get(
                "window_sample_weight",
                pd.Series(0.0, index=windows.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        nonzero_excluded_weight = training_tier.eq("exclude") & sample_weight.ne(0.0)
        if nonzero_excluded_weight.any():
            errors.append(
                "excluded_windows_have_nonzero_weight="
                f"{int(nonzero_excluded_weight.sum())}"
            )

        mixed = int(
            windows.get("sequence_label_status", pd.Series(dtype=str))
            .astype(str)
            .isin({"mixed", "transition"})
            .sum()
        )
        if mixed:
            warnings.append(f"mixed_or_transition_windows={mixed}")
    else:
        warnings.append("no_sequence_windows_generated")

    audit = {
        "window_rows": int(len(windows)),
        "temporal_intervals": int(len(intervals)) if intervals is not None else None,
        "sources": _value_counts_dict(windows, "source_type"),
        "window_length_frames": _value_counts_dict(windows, "window_length_frames"),
        "sequence_label_status": _value_counts_dict(windows, "sequence_label_status"),
        "window_valid_for_main_train": _value_counts_dict(windows, "window_valid_for_main_train"),
        "human_reviewed_behavior_consistency_status": _value_counts_dict(
            windows, "human_reviewed_behavior_consistency_status"
        ),
        "behavior_review_coverage_ratio_window": _numeric_summary(
            windows, "behavior_review_coverage_ratio_window"
        ),
        "behavior_review_label_resolution_ratio_window": _numeric_summary(
            windows, "behavior_review_label_resolution_ratio_window"
        ),
        "behavior_window_label": _value_counts_dict(windows, "behavior_window_label"),
        "label_propagation_policy": _value_counts_dict(windows, "label_propagation_policy"),
        "window_exclusion_reason_top": _value_counts_dict(windows, "window_exclusion_reason"),
        "review_excluded_frame_count_window": _value_counts_dict(
            windows, "review_excluded_frame_count_window"
        ),
        "window_sample_weight": _numeric_summary(windows, "window_sample_weight"),
        "speed_mean_window": _numeric_summary(windows, "speed_mean_window"),
        "target_roi_contact_ratio_window": _numeric_summary(
            windows, "target_roi_contact_ratio_window"
        ),
        "pair_contact_ratio_window": _numeric_summary(windows, "pair_contact_ratio_window"),
        "hidden_ratio_window": _numeric_summary(windows, "hidden_ratio_window"),
        "hidden_ratio_raw_window": _numeric_summary(
            windows,
            "hidden_ratio_raw_window",
        ),
        "hidden_review_coverage_ratio_window": _numeric_summary(
            windows,
            "hidden_review_coverage_ratio_window",
        ),
        "hidden_burden_ratio_window": _numeric_summary(
            windows,
            "hidden_burden_ratio_window",
        ),
        "hidden_longest_run_ratio_window": _numeric_summary(
            windows,
            "hidden_longest_run_ratio_window",
        ),
        "hidden_window_policy_tier": _value_counts_dict(
            windows,
            "hidden_window_policy_tier",
        ),
        "high_hidden_ratio_window": _value_counts_dict(
            windows,
            "high_hidden_ratio_window",
        ),
        "bbox_valid_ratio_window": _numeric_summary(windows, "bbox_valid_ratio_window"),
        "feature_computation_grain": _value_counts_dict(
            windows,
            "feature_computation_grain",
        ),
        "view_type": _value_counts_dict(windows, "view_type"),
        "sampling_pattern": _value_counts_dict(windows, "sampling_pattern"),
        "errors": errors,
        "warnings": warnings,
    }
    audit = add_optional_lineage_claims_to_audit(
        audit,
        windows,
        artifact_name="sequence window audit table",
    )
    if intervals is not None:
        try:
            require_lineage_claims_preserved(
                intervals,
                windows,
                source_name="sequence window audit intervals",
                derived_name="sequence window audit windows",
            )
        except ValueError as exc:
            audit["errors"].append(f"lineage_claim_contract={exc}")
    return audit


def _build_windows_from_harmonized(
    frames: pd.DataFrame,
    intervals: pd.DataFrame,
    config: SequenceWindowConfig,
) -> pd.DataFrame:
    frames = _prepare_frame_columns(frames)
    _validate_window_frame_contract(frames)
    rows: list[dict[str, Any]] = []

    # Build interval lookup for CVAT tracks.
    intervals_by_track: dict[str, pd.DataFrame] = {}
    if intervals is not None and not intervals.empty:
        for key, g in intervals.groupby("object_track_key", dropna=False, sort=False):
            intervals_by_track[str(key)] = g.sort_values(
                "label_window_start", kind="mergesort"
            ).reset_index(drop=True)

    for object_key, g in frames.groupby("object_track_key", dropna=False, sort=False):
        object_key = str(object_key)
        g = (
            g.sort_values("frame_index", kind="mergesort")
            .reset_index(drop=False)
            .rename(columns={"index": "_source_row_index"})
        )
        if g.empty:
            continue
        source_type = str(g.iloc[0].get("source_type", ""))
        if source_type in CVAT_SOURCE_TYPES:
            interval_g = intervals_by_track.get(object_key, pd.DataFrame())
            rows.extend(_generate_cvat_windows(g, interval_g, config))
        elif source_type == LEGACY_SOURCE_TYPE:
            rows.extend(_generate_legacy_windows(g, config))
        else:
            rows.extend(_generate_generic_windows(g, config))

    windows = pd.DataFrame(rows)
    if windows.empty:
        return windows

    windows = windows.sort_values(
        [
            "source_type",
            "dataset_id",
            "video_key",
            "object_track_key",
            "window_length_frames",
            "window_start_frame",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    windows.insert(0, "window_row_index", np.arange(len(windows), dtype="int64"))
    return windows


def _generate_legacy_windows(g: pd.DataFrame, config: SequenceWindowConfig) -> list[dict[str, Any]]:
    _validate_legacy_dense_source_mapping(g)
    rows: list[dict[str, Any]] = []
    frames_available = sorted(set(g["frame_index"].astype(int).tolist()))
    if not frames_available:
        return rows
    frame_set = set(frames_available)
    min_f, max_f = min(frames_available), max(frames_available)
    produced = 0
    for length in config.window_lengths:
        last_start = max_f - length + 1
        if last_start < min_f:
            rows.append(
                _empty_invalid_window(
                    g, length, min_f, min_f + length - 1, "not_enough_frames_for_window"
                )
            )
            continue
        for start in range(min_f, last_start + 1, config.legacy_window_stride):
            end = start + length - 1
            expected = set(range(start, end + 1))
            complete = expected.issubset(frame_set)
            wg = g[g["frame_index"].between(start, end, inclusive="both")]
            row = _summarize_window(
                wg,
                length,
                start,
                end,
                config,
                label_coverage_complete=complete,
                source_window_type="legacy_dense_frame_window",
            )
            rows.append(row)
            produced += 1
            if (
                config.max_windows_per_track is not None
                and produced >= config.max_windows_per_track
            ):
                return rows
    if config.include_legacy_sparse_s6_at16:
        sparse_rows = _generate_legacy_sparse_s6_at16_windows(g, config)
        for row in sparse_rows:
            rows.append(row)
            produced += 1
            if (
                config.max_windows_per_track is not None
                and produced >= config.max_windows_per_track
            ):
                return rows
    return rows


def _generate_legacy_sparse_s6_at16_windows(
    g: pd.DataFrame,
    config: SequenceWindowConfig,
) -> list[dict[str, Any]]:
    """Build one reviewed sparse six-slot ablation per exact legacy burst."""

    if "temporal_unit_key" not in g.columns:
        raise ValueError("legacy S6@16 requires temporal_unit_key")
    offsets = np.asarray([0, 3, 6, 9, 12, 15], dtype="int64")
    rows: list[dict[str, Any]] = []
    for _, unit in g.groupby("temporal_unit_key", dropna=False, sort=False):
        unit = unit.sort_values("frame_index", kind="mergesort")
        if unit.empty:
            continue
        declared_start = pd.to_numeric(
            unit.get(
                "label_window_start",
                pd.Series(np.nan, index=unit.index),
            ),
            errors="coerce",
        ).dropna()
        start = (
            int(declared_start.iloc[0])
            if not declared_start.empty
            else int(pd.to_numeric(unit["frame_index"]).min())
        )
        end = start + 15
        expected_full = set(range(start, end + 1))
        actual_full = set(
            pd.to_numeric(unit["frame_index"], errors="coerce")
            .dropna()
            .astype(int)
        )
        selected_indices = (start + offsets).tolist()
        selected = unit.loc[
            pd.to_numeric(unit["frame_index"], errors="coerce").isin(
                selected_indices
            )
        ]
        complete = expected_full.issubset(actual_full) and len(selected) == 6
        rows.append(
            _summarize_window(
                selected,
                6,
                start,
                end,
                config,
                label_coverage_complete=complete,
                source_window_type="legacy_sparse_s6_at16_ablation",
                authority_rows=unit,
                view_type="S6@16",
                sampling_pattern="uniform_sparse_offsets_0_3_6_9_12_15",
            )
        )
    return rows


def _generate_cvat_windows(
    g: pd.DataFrame, intervals: pd.DataFrame, config: SequenceWindowConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if intervals is None or intervals.empty:
        min_f = int(pd.to_numeric(g["frame_index"], errors="coerce").min())
        return [
            _empty_invalid_window(
                g,
                int(config.window_lengths[0]),
                min_f,
                min_f + int(config.window_lengths[0]) - 1,
                "no_cvat_intervals",
            )
        ]

    intervals = intervals.copy()
    intervals["label_window_start"] = pd.to_numeric(
        intervals["label_window_start"], errors="coerce"
    )
    intervals["label_window_end"] = pd.to_numeric(intervals["label_window_end"], errors="coerce")
    _validate_cvat_interval_contract(intervals)
    intervals = intervals.sort_values("label_window_start", kind="mergesort")
    if intervals.empty:
        return rows

    starts = intervals["label_window_start"].astype(int).tolist()
    starts_np = intervals["label_window_start"].astype(int).to_numpy()
    ends_np = intervals["label_window_end"].astype(int).to_numpy()
    produced = 0
    for length in config.window_lengths:
        for interval_pos in range(0, len(starts), config.cvat_window_stride_intervals):
            start = starts[interval_pos]
            end = start + length - 1
            left = int(np.searchsorted(ends_np, start, side="left"))
            right = int(np.searchsorted(starts_np, end, side="right"))
            overlap = intervals.iloc[left:right].copy()
            coverage_complete = _intervals_cover_span(overlap, start, end)
            interval_keys = (
                set(overlap["temporal_unit_key"].astype(str))
                if "temporal_unit_key" in overlap.columns
                else set()
            )
            if interval_keys:
                in_interval = g["temporal_unit_key"].astype(str).isin(interval_keys)
                in_window = g["frame_index"].between(start, end, inclusive="both")
                wg = g[in_interval & in_window]
            else:
                wg = g.iloc[0:0]
            row = _summarize_window(
                wg,
                length,
                start,
                end,
                config,
                label_coverage_complete=coverage_complete,
                source_window_type="cvat_anchor_interval_window",
                interval_subset=overlap,
            )
            rows.append(row)
            produced += 1
            if (
                config.max_windows_per_track is not None
                and produced >= config.max_windows_per_track
            ):
                return rows
    return rows


def _generate_generic_windows(
    g: pd.DataFrame, config: SequenceWindowConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frames_available = sorted(set(g["frame_index"].astype(int).tolist()))
    if not frames_available:
        return rows
    min_f, max_f = min(frames_available), max(frames_available)
    produced = 0
    for length in config.window_lengths:
        last_start = max_f - length + 1
        if last_start < min_f:
            rows.append(
                _empty_invalid_window(
                    g, length, min_f, min_f + length - 1, "not_enough_frames_for_window"
                )
            )
            continue
        for start in range(min_f, last_start + 1, config.legacy_window_stride):
            end = start + length - 1
            wg = g[g["frame_index"].between(start, end, inclusive="both")]
            complete = len(set(wg["frame_index"].astype(int))) >= length
            rows.append(
                _summarize_window(
                    wg,
                    length,
                    start,
                    end,
                    config,
                    label_coverage_complete=complete,
                    source_window_type="generic_frame_window",
                )
            )
            produced += 1
            if (
                config.max_windows_per_track is not None
                and produced >= config.max_windows_per_track
            ):
                return rows
    return rows


def _validate_window_frame_contract(frames: pd.DataFrame) -> None:
    """Reject frame rows that cannot map uniquely to temporal windows."""
    if frames.empty:
        return
    key = frames["object_track_key"].fillna("").astype(str).str.strip()
    frame_index = pd.to_numeric(frames["frame_index"], errors="coerce")
    invalid = (
        key.eq("")
        | frame_index.isna()
        | frame_index.mod(1).ne(0)
        | frame_index.lt(0)
    )
    duplicate = pd.DataFrame(
        {
            "object_track_key": key,
            "frame_index": frame_index,
        }
    ).duplicated(keep=False)
    duplicate &= ~invalid
    if invalid.any() or duplicate.any():
        affected = invalid | duplicate
        sample = [str(value) for value in frames.index[affected].tolist()[:10]]
        raise ValueError(
            "Sequence frame contract failed: "
            f"invalid_rows={int(invalid.sum())}, "
            f"duplicate_track_frame_rows={int(duplicate.sum())}, "
            f"sample_source_indices={sample}"
        )
    frames["frame_index"] = frame_index.astype(int)
    clock_columns = {
        "source_frame_index",
        "source_fps",
        "timestamp_sec",
        "timestamp_source",
    }
    provenance_declared = bool(
        {"source_frame_index", "source_fps"}.intersection(frames.columns)
    )
    if provenance_declared and not clock_columns.issubset(frames.columns):
        raise ValueError(
            "Incomplete source-frame clock contract: "
            f"missing={sorted(clock_columns.difference(frames.columns))}"
        )
    if clock_columns.issubset(frames.columns):
        fps = pd.to_numeric(frames["source_fps"], errors="coerce")
        if fps.notna().any():
            clock_audit = audit_source_frame_clock(frames)
            if clock_audit["errors"]:
                raise ValueError(
                    "Source-frame timestamp contract failed: "
                    f"{clock_audit['errors']}"
                )


def _validate_legacy_dense_source_mapping(frame: pd.DataFrame) -> None:
    """Prove consecutive legacy native offsets are decoded source frames."""
    offset_column = (
        "native_offset"
        if "native_offset" in frame
        else "relative_frame_index"
        if "relative_frame_index" in frame
        else None
    )
    source_column = (
        "source_frame_index"
        if "source_frame_index" in frame
        else "frame_index"
    )
    if offset_column is None:
        return
    grain = (
        ["temporal_unit_key"]
        if "temporal_unit_key" in frame
        else ["object_track_key"]
    )
    violations = 0
    for _, unit in frame.groupby(grain, dropna=False, sort=False):
        ordered = unit.assign(
            _native_offset=pd.to_numeric(
                unit[offset_column],
                errors="coerce",
            ),
            _source_frame=pd.to_numeric(
                unit[source_column],
                errors="coerce",
            ),
        ).sort_values("_native_offset", kind="mergesort")
        offset_delta = ordered["_native_offset"].diff()
        source_delta = ordered["_source_frame"].diff()
        violations += int(
            (offset_delta.eq(1) & source_delta.ne(1)).fillna(False).sum()
        )
    if violations:
        raise ValueError(
            "Legacy consecutive native offsets are not contiguous decoded "
            f"source frames: violating_pairs={violations}"
        )


def _validate_cvat_interval_contract(intervals: pd.DataFrame) -> None:
    """Reject malformed CVAT intervals rather than dropping them silently."""
    start = intervals["label_window_start"]
    end = intervals["label_window_end"]
    unit_key = intervals.get(
        "temporal_unit_key",
        pd.Series("", index=intervals.index),
    ).fillna("").astype(str).str.strip()
    invalid = (
        start.isna()
        | end.isna()
        | start.mod(1).ne(0)
        | end.mod(1).ne(0)
        | start.lt(0)
        | end.lt(start)
        | unit_key.eq("")
    )
    duplicate = unit_key.ne("") & unit_key.duplicated(keep=False)
    if invalid.any() or duplicate.any():
        affected = invalid | duplicate
        sample = [str(value) for value in intervals.index[affected].tolist()[:10]]
        raise ValueError(
            "CVAT interval contract failed: "
            f"invalid_rows={int(invalid.sum())}, "
            f"duplicate_temporal_unit_rows={int(duplicate.sum())}, "
            f"sample_source_indices={sample}"
        )


def _summarize_window(
    wg: pd.DataFrame,
    length: int,
    start: int,
    end: int,
    config: SequenceWindowConfig,
    *,
    label_coverage_complete: bool,
    source_window_type: str,
    interval_subset: pd.DataFrame | None = None,
    authority_rows: pd.DataFrame | None = None,
    view_type: str | None = None,
    sampling_pattern: str = "contiguous",
) -> dict[str, Any]:
    if wg.empty:
        return _empty_invalid_window(
            pd.DataFrame(),
            length,
            start,
            end,
            "no_observed_rows_in_window",
            source_window_type=source_window_type,
            view_type=view_type,
            sampling_pattern=sampling_pattern,
        )

    first = wg.iloc[0]
    authority = (
        authority_rows
        if authority_rows is not None and not authority_rows.empty
        else wg
    )
    if (
        interval_subset is not None
        and "behavior_temporal_final" in interval_subset.columns
    ):
        behavior_source = interval_subset["behavior_temporal_final"].fillna("").astype(str)
    else:
        behavior_source = authority.get(
            "behavior_temporal_final",
            authority["behavior"],
        ).fillna("").astype(str)
    behavior_values = [b for b in behavior_source.tolist() if b]
    unique_behaviors = sorted(set(behavior_values))
    dominant = pd.Series(behavior_values).value_counts().idxmax() if behavior_values else ""

    if not behavior_values:
        label_status = "uncertain"
    elif len(unique_behaviors) == 1 and label_coverage_complete:
        label_status = "stable"
    elif len(unique_behaviors) == 1 and not label_coverage_complete:
        label_status = "incomplete"
    else:
        label_status = "transition" if _looks_like_transition(interval_subset, wg) else "mixed"

    bbox_valid_ratio = _bool_mean(wg.get("bbox_valid", pd.Series(True, index=wg.index)))
    hidden_raw = _to_bool_series(wg.get("hidden", pd.Series(False, index=wg.index)))
    hidden_trust = _window_hidden_trust(wg)
    hidden_effective = hidden_raw & hidden_trust
    hidden_ratio_raw = float(hidden_raw.mean()) if len(wg) else 0.0
    hidden_ratio = float(hidden_effective.mean()) if len(wg) else 0.0
    hidden_untrusted_ratio = float((hidden_raw & ~hidden_trust).mean()) if len(wg) else 0.0
    hidden_review_coverage = float(hidden_trust.mean()) if len(wg) else 0.0
    hidden_longest_run_frames = _longest_hidden_run_frames(wg, hidden_raw)
    hidden_longest_run_ratio = (
        hidden_longest_run_frames / length if length > 0 else 0.0
    )
    hidden_trusted_longest_run_frames = _longest_hidden_run_frames(
        wg,
        hidden_effective,
    )
    hidden_trusted_longest_run_ratio = (
        hidden_trusted_longest_run_frames / length if length > 0 else 0.0
    )
    hidden_policy_tier, hidden_policy_reasons = _classify_hidden_window(
        hidden_ratio=hidden_ratio,
        longest_run_ratio=hidden_trusted_longest_run_ratio,
        config=config,
    )
    spatio_ratio = _bool_mean(
        wg.get("spatiotemporal_feature_valid", pd.Series(True, index=wg.index))
    )
    review_summary = _review_training_summary(wg)
    behavior_review = _behavior_review_summary(authority)
    reviewed_status = behavior_review[
        "human_reviewed_behavior_consistency_status"
    ]
    effective_status = (
        reviewed_status
        if behavior_review["behavior_review_fields_present"]
        and reviewed_status in {"stable", "transition"}
        else label_status
    )

    timing = _window_timing_summary(
        wg,
        start=start,
        end=end,
        expected_slot_count=length,
        default_fps=config.default_fps,
    )

    reasons: list[str] = []
    hard_exclusion = False
    if label_status != "stable":
        reasons.append(f"annotation_label_{label_status}")
        hard_exclusion |= label_status not in {"transition", "mixed"}
    if not label_coverage_complete:
        reasons.append("label_coverage_incomplete")
        hard_exclusion = True
    if bbox_valid_ratio < config.min_bbox_valid_ratio:
        reasons.append("bbox_valid_ratio_below_threshold")
        hard_exclusion = True
    reasons.extend(hidden_policy_reasons)
    hard_exclusion |= hidden_policy_tier == "exclude"
    if spatio_ratio < config.min_spatiotemporal_valid_ratio:
        reasons.append("spatiotemporal_valid_ratio_below_threshold")
        hard_exclusion = True
    if review_summary["review_excluded_frame_count_window"] > 0:
        reasons.append("review_excluded_rows_in_window")
        hard_exclusion = True
    review_required = (
        config.behavior_review_requirement
        == "full_native_unit_review_required"
    )
    if review_required and not behavior_review["behavior_review_fields_present"]:
        reasons.append("behavior_review_required_but_missing")
        hard_exclusion = True
    if behavior_review["behavior_review_fields_present"] and (
        behavior_review["human_reviewed_behavior_consistency_status"] != "stable"
    ):
        reasons.append(
            "behavior_review_"
            f"{behavior_review['human_reviewed_behavior_consistency_status']}"
        )
        hard_exclusion = True
    if effective_status in {"transition", "mixed"} and not config.include_mixed_windows:
        hard_exclusion = True

    valid_main = (
        effective_status == "stable"
        and not hard_exclusion
        and hidden_policy_tier in {"main_train", "audit_only"}
    )
    robust_eligible = not hard_exclusion and (
        hidden_policy_tier == "robust_train_only"
        or (
            effective_status in {"transition", "mixed"}
            and config.include_mixed_windows
        )
    )
    training_tier = (
        "main_train"
        if valid_main
        else "robust_train_only"
        if robust_eligible
        else "exclude"
    )
    if training_tier == "exclude":
        review_summary["window_sample_weight"] = 0.0

    constituent_rows = (
        interval_subset
        if interval_subset is not None and not interval_subset.empty
        else authority
    )
    temporal_unit_keys = _temporal_unit_keys(constituent_rows)
    selected_offsets = (
        [0, 3, 6, 9, 12, 15]
        if view_type == "S6@16"
        else list(range(length))
    )
    selected_indices = [int(start + value) for value in selected_offsets]
    selected_timestamps = _selected_view_coordinates(
        wg,
        selected_indices,
    )
    pair_delta_frames, pair_delta_seconds = _selected_view_pair_deltas(
        selected_indices,
        selected_timestamps,
    )
    window_id = _make_window_id(first, length, start, end)
    row: dict[str, Any] = {
        "window_id": window_id,
        "feature_computation_grain": "FINAL_VIEW_FEATURES",
        "pair_scope_key": window_id,
        "pair_recomputed_for_view": True,
        "aggregate_recomputed_for_view": True,
        "source_window_type": source_window_type,
        "view_type": view_type or f"T{length}_contiguous",
        "sampling_pattern": sampling_pattern,
        "primary_cross_source_eligible": (view_type != "S6@16"),
        "selected_frame_offsets": json.dumps(
            selected_offsets,
            separators=(",", ":"),
        ),
        "selected_frame_indices": json.dumps(
            selected_indices,
            separators=(",", ":"),
        ),
        "selected_timestamps_seconds": json.dumps(
            selected_timestamps,
            separators=(",", ":"),
        ),
        "pair_delta_frames": json.dumps(
            pair_delta_frames,
            separators=(",", ":"),
        ),
        "pair_delta_seconds": json.dumps(
            pair_delta_seconds,
            separators=(",", ":"),
        ),
        "source_type": str(first.get("source_type", "")),
        "dataset_id": str(first.get("dataset_id", "")),
        "video_key": str(first.get("video_key", "")),
        "object_track_key": str(first.get("object_track_key", "")),
        "pig_id": str(first.get("pig_id", "")),
        "track_id": str(first.get("track_id", "")),
        "window_length_frames": int(length),
        "window_start_frame": int(start),
        "window_end_frame": int(end),
        "window_duration_sec": timing["declared_window_duration_seconds"],
        "effective_fps": timing["effective_observation_rate_hz"],
        "timestamp_start_sec": timing["timestamp_start_sec"],
        "timestamp_end_sec": timing["timestamp_end_sec"],
        **timing,
        "observed_row_count_window": int(len(wg)),
        "observed_frame_count_window": int(wg["frame_index"].nunique(dropna=True))
        if "frame_index" in wg.columns
        else int(len(wg)),
        "label_coverage_complete": bool(label_coverage_complete),
        "temporal_unit_keys_window": "|".join(temporal_unit_keys),
        "temporal_unit_keys_json": json.dumps(
            temporal_unit_keys,
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "constituent_native_unit_keys": json.dumps(
            temporal_unit_keys,
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        "num_temporal_units_window": int(
            wg.get("temporal_unit_key", pd.Series(dtype=str)).nunique(dropna=True)
        )
        if "temporal_unit_key" in wg.columns
        else 0,
        "num_behaviors_window": int(len(unique_behaviors)),
        "unique_behaviors_window": "|".join(unique_behaviors),
        "behavior_window_label": str(
            behavior_review.get("behavior_reviewed_window_label") or dominant
        ),
        "sequence_label_status": effective_status,
        "annotation_consistency_status": label_status,
        "behavior_review_requirement": config.behavior_review_requirement,
        **behavior_review,
        "window_valid_for_main_train": bool(valid_main),
        "window_training_tier_recommendation": training_tier,
        "window_exclusion_reason": ";".join(reasons),
        "bbox_valid_ratio_window": bbox_valid_ratio,
        "hidden_ratio_window": hidden_ratio,
        "visible_ratio_window": 1.0 - hidden_ratio,
        "hidden_ratio_raw_window": hidden_ratio_raw,
        "hidden_ratio_trusted_window": hidden_ratio,
        "hidden_metadata_untrusted_ratio_window": hidden_untrusted_ratio,
        "hidden_review_coverage_ratio_window": hidden_review_coverage,
        "hidden_burden_ratio_window": hidden_ratio_raw,
        "hidden_longest_run_frames_window": hidden_longest_run_frames,
        "hidden_longest_run_ratio_window": hidden_longest_run_ratio,
        "hidden_trusted_longest_run_frames_window": (
            hidden_trusted_longest_run_frames
        ),
        "hidden_trusted_longest_run_ratio_window": (
            hidden_trusted_longest_run_ratio
        ),
        "hidden_window_policy_tier": hidden_policy_tier,
        "high_hidden_ratio_window": (
            hidden_ratio_raw > config.max_hidden_ratio_main
        ),
        "long_hidden_run_window": (
            hidden_longest_run_ratio > config.max_hidden_run_ratio_main
        ),
        "hidden_exclusion_policy_enabled": config.exclude_high_hidden_from_main,
        "spatiotemporal_feature_valid_ratio_window": spatio_ratio,
        **review_summary,
    }

    row.update(_interaction_policy_for_behavior(row["behavior_window_label"]))
    row.update(
        _aggregate_window_features(
            wg,
            timing["declared_window_duration_seconds"],
            expected_start=start,
            expected_end=end,
            evidence_config=config.temporal_evidence_config(),
        )
    )
    return row


def _empty_invalid_window(
    g: pd.DataFrame,
    length: int,
    start: int,
    end: int,
    reason: str,
    *,
    source_window_type: str = "unknown_window",
    view_type: str | None = None,
    sampling_pattern: str = "contiguous",
) -> dict[str, Any]:
    first = g.iloc[0] if g is not None and not g.empty else pd.Series(dtype=object)
    window_id = _make_window_id(first, length, start, end)
    sparse_s6 = view_type == "S6@16"
    row = {
        "window_id": window_id,
        "feature_computation_grain": "FINAL_VIEW_FEATURES",
        "pair_scope_key": window_id,
        "pair_recomputed_for_view": False,
        "aggregate_recomputed_for_view": False,
        "source_window_type": source_window_type,
        "view_type": view_type or f"T{length}_contiguous",
        "sampling_pattern": sampling_pattern,
        "primary_cross_source_eligible": (view_type != "S6@16"),
        "selected_frame_offsets": (
            "[0,3,6,9,12,15]" if sparse_s6 else "[]"
        ),
        "selected_frame_indices": "[]",
        "selected_timestamps_seconds": "[]",
        "pair_delta_frames": "[3,3,3,3,3]" if sparse_s6 else "[]",
        "pair_delta_seconds": "[]",
        "source_type": str(first.get("source_type", "")),
        "dataset_id": str(first.get("dataset_id", "")),
        "video_key": str(first.get("video_key", "")),
        "object_track_key": str(first.get("object_track_key", "")),
        "pig_id": str(first.get("pig_id", "")),
        "track_id": str(first.get("track_id", "")),
        "window_length_frames": int(length),
        "window_start_frame": int(start),
        "window_end_frame": int(end),
        "window_duration_sec": np.nan,
        "effective_fps": np.nan,
        "declared_window_duration_seconds": np.nan,
        "observed_timestamp_span_seconds": np.nan,
        "adjacent_observed_duration_seconds": 0.0,
        "physical_span_seconds": np.nan,
        "expected_slot_count": int(length),
        "observed_slot_count": 0,
        "effective_observation_rate_hz": np.nan,
        "adjacent_pair_coverage_ratio": 0.0,
        "declared_timeline_fps": np.nan,
        "timestamp_start_sec": np.nan,
        "timestamp_end_sec": np.nan,
        "observed_row_count_window": 0,
        "observed_frame_count_window": 0,
        "label_coverage_complete": False,
        "temporal_unit_keys_window": "",
        "temporal_unit_keys_json": "[]",
        "constituent_native_unit_keys": "[]",
        "num_temporal_units_window": 0,
        "num_behaviors_window": 0,
        "unique_behaviors_window": "",
        "behavior_window_label": "",
        "sequence_label_status": "incomplete",
        "annotation_consistency_status": "incomplete",
        "human_reviewed_behavior_consistency_status": "unreviewed",
        "behavior_reviewed_window_label": "",
        "behavior_review_fields_present": False,
        "behavior_review_coverage_ratio_window": 0.0,
        "behavior_review_label_resolution_ratio_window": 0.0,
        "behavior_review_train_eligibility_ratio_window": 0.0,
        "all_temporal_units_behavior_reviewed": False,
        "all_temporal_units_behavior_label_resolved": False,
        "all_temporal_units_behavior_train_eligible": False,
        "window_valid_for_main_train": False,
        "window_training_tier_recommendation": "exclude",
        "window_exclusion_reason": reason,
        "bbox_valid_ratio_window": 0.0,
        "hidden_ratio_window": 0.0,
        "visible_ratio_window": 0.0,
        "hidden_ratio_raw_window": 0.0,
        "hidden_ratio_trusted_window": 0.0,
        "hidden_metadata_untrusted_ratio_window": 0.0,
        "hidden_review_coverage_ratio_window": 0.0,
        "hidden_burden_ratio_window": 0.0,
        "hidden_longest_run_frames_window": 0,
        "hidden_longest_run_ratio_window": 0.0,
        "hidden_trusted_longest_run_frames_window": 0,
        "hidden_trusted_longest_run_ratio_window": 0.0,
        "hidden_window_policy_tier": "exclude",
        "high_hidden_ratio_window": False,
        "long_hidden_run_window": False,
        "hidden_exclusion_policy_enabled": False,
        "spatiotemporal_feature_valid_ratio_window": 0.0,
        "review_include_ratio_window": 1.0,
        "review_excluded_frame_count_window": 0,
        "review_training_actions_window": "",
        "review_sample_weight_mean_window": 1.0,
        "window_sample_weight": 0.0,
    }
    row.update(_interaction_policy_for_behavior(""))
    row.update(_empty_aggregate_features())
    return row


def _recompute_view_motion(window_rows: pd.DataFrame) -> dict[str, Any]:
    """Recompute pair-derived motion from exact selected view rows."""

    ordered = window_rows.sort_values("frame_index", kind="mergesort")
    row_count = len(ordered)
    zeros = np.zeros(row_count, dtype="float64")

    def values(column: str) -> np.ndarray:
        return pd.to_numeric(
            ordered.get(column, pd.Series(np.nan, index=ordered.index)),
            errors="coerce",
        ).to_numpy(dtype="float64")

    frames = values("frame_index")
    timestamps = values("timestamp_sec")
    cx = values("cx_n")
    cy = values("cy_n")
    row_valid = np.ones(row_count, dtype=bool)
    for column in [
        "bbox_valid",
        "geometry_feature_valid",
        "spatiotemporal_feature_valid",
    ]:
        if column in ordered.columns:
            row_valid &= _to_bool_series(ordered[column]).to_numpy(dtype=bool)

    frame_delta = np.diff(frames)
    time_delta = np.diff(timestamps)
    dx = np.diff(cx)
    dy = np.diff(cy)
    geometry_pair_valid = (
        np.isfinite(frame_delta)
        & (frame_delta > 0)
        & np.isfinite(dx)
        & np.isfinite(dy)
        & row_valid[:-1]
        & row_valid[1:]
    )
    time_valid = np.isfinite(time_delta) & (time_delta > 0)
    adjacent = geometry_pair_valid & np.isclose(frame_delta, 1.0) & time_valid
    sparse = geometry_pair_valid & (frame_delta > 1.0) & time_valid
    velocity_valid = adjacent | sparse
    distance = np.hypot(dx, dy)

    displacement = zeros.copy()
    displacement[1:] = np.where(adjacent, distance, 0.0)
    speed_per_frame = np.full(row_count, np.nan, dtype="float64")
    speed_per_frame[1:] = np.where(adjacent, distance, np.nan)
    speed_per_second = np.full(row_count, np.nan, dtype="float64")
    speed_per_second[1:] = np.where(
        velocity_valid,
        distance / time_delta,
        np.nan,
    )

    acceleration_per_frame = np.full(row_count, np.nan, dtype="float64")
    acceleration_per_second = np.full(row_count, np.nan, dtype="float64")
    if row_count >= 3:
        consecutive_adjacent = adjacent[:-1] & adjacent[1:]
        frame_acceleration = np.diff(speed_per_frame[1:])
        acceleration_per_frame[2:] = np.where(
            consecutive_adjacent,
            frame_acceleration,
            np.nan,
        )
        consecutive_velocity = velocity_valid[:-1] & velocity_valid[1:]
        acceleration_time = (time_delta[:-1] + time_delta[1:]) / 2.0
        acceleration_valid = (
            consecutive_velocity
            & np.isfinite(acceleration_time)
            & (acceleration_time > 0)
        )
        physical_acceleration = (
            np.diff(speed_per_second[1:]) / acceleration_time
        )
        acceleration_per_second[2:] = np.where(
            acceleration_valid,
            physical_acceleration,
            np.nan,
        )

    direction = np.full(max(0, row_count - 1), np.nan, dtype="float64")
    direction[adjacent] = np.arctan2(dy[adjacent], dx[adjacent])
    direction_change = np.full(row_count, np.nan, dtype="float64")
    if row_count >= 3:
        heading_valid = adjacent[:-1] & adjacent[1:]
        raw_change = (np.diff(direction) + np.pi) % (2 * np.pi) - np.pi
        direction_change[2:] = np.where(heading_valid, raw_change, np.nan)

    shape_change = np.full(row_count, np.nan, dtype="float64")
    shape_components: list[np.ndarray] = []
    for column, scale in [
        ("bw_n", 1.0),
        ("bh_n", 1.0),
        ("area_n", 1.0),
        ("aspect_ratio", 10.0),
    ]:
        component = np.diff(values(column)) / scale
        component = np.where(np.isfinite(component), component, 0.0)
        shape_components.append(component)
    if shape_components:
        shape_distance = np.sqrt(
            np.sum(np.square(np.vstack(shape_components)), axis=0)
        )
        shape_change[1:] = np.where(adjacent, shape_distance, np.nan)

    connected_displacement = _connected_view_displacement(
        cx,
        cy,
        adjacent,
    )
    index = ordered.index
    return {
        "speed_n_per_frame": pd.Series(speed_per_frame, index=index),
        "speed_n_per_second": pd.Series(speed_per_second, index=index),
        "displacement_n": pd.Series(displacement, index=index),
        "abs_acceleration_n_per_frame2": pd.Series(
            np.abs(acceleration_per_frame),
            index=index,
        ),
        "abs_tangential_acceleration_n_per_second2": pd.Series(
            np.abs(acceleration_per_second),
            index=index,
        ),
        "abs_direction_change_rad": pd.Series(
            np.abs(direction_change),
            index=index,
        ),
        "shape_change_score": pd.Series(shape_change, index=index),
        "connected_displacement_n": connected_displacement,
        "adjacent_pair_count": int(adjacent.sum()),
        "sparse_velocity_pair_count": int(sparse.sum()),
    }


def _connected_view_displacement(
    cx: np.ndarray,
    cy: np.ndarray,
    adjacent_pair_valid: np.ndarray,
) -> float:
    total = 0.0
    start: int | None = None
    for pair_index, valid in enumerate(adjacent_pair_valid):
        if valid and start is None:
            start = pair_index
        last_pair = pair_index == len(adjacent_pair_valid) - 1
        if start is not None and (not valid or last_pair):
            end = pair_index + 1 if valid and last_pair else pair_index
            total += float(np.hypot(cx[end] - cx[start], cy[end] - cy[start]))
            start = None
    return total


def _recompute_view_roi_transitions(
    window_rows: pd.DataFrame,
) -> dict[str, int]:
    ordered = window_rows.sort_values("frame_index", kind="mergesort")
    if len(ordered) < 2:
        return {
            "entry_count": 0,
            "exit_count": 0,
            "near_entry_count": 0,
            "near_exit_count": 0,
            "valid_pair_count": 0,
        }
    frames = pd.to_numeric(ordered["frame_index"], errors="coerce").to_numpy(
        dtype="float64"
    )
    timestamps = pd.to_numeric(
        ordered.get("timestamp_sec", pd.Series(np.nan, index=ordered.index)),
        errors="coerce",
    ).to_numpy(dtype="float64")
    available = _to_bool_series(
        ordered.get(
            "roi_target_available",
            pd.Series(False, index=ordered.index),
        )
    ).to_numpy(dtype=bool)
    contact = _to_bool_series(
        ordered.get("roi_target_contact", pd.Series(False, index=ordered.index))
    ).to_numpy(dtype=bool)
    near = _to_bool_series(
        ordered.get("roi_target_near", pd.Series(False, index=ordered.index))
    ).to_numpy(dtype=bool)
    pair_valid = (
        np.isclose(np.diff(frames), 1.0)
        & np.isfinite(np.diff(timestamps))
        & (np.diff(timestamps) > 0)
        & available[:-1]
        & available[1:]
    )
    return {
        "entry_count": int((pair_valid & ~contact[:-1] & contact[1:]).sum()),
        "exit_count": int((pair_valid & contact[:-1] & ~contact[1:]).sum()),
        "near_entry_count": int((pair_valid & ~near[:-1] & near[1:]).sum()),
        "near_exit_count": int((pair_valid & near[:-1] & ~near[1:]).sum()),
        "valid_pair_count": int(pair_valid.sum()),
    }


def _frame_local_primitives_for_view(rows: pd.DataFrame) -> pd.DataFrame:
    """Remove parent-grain pair/aggregate columns before final recomputation."""

    derived_prefixes = (
        "prev_",
        "delta_",
        "motion_",
        "displacement_",
        "sparse_displacement_",
        "vx_",
        "vy_",
        "speed_",
        "accel_",
        "abs_accel_",
        "acceleration_",
        "abs_acceleration_",
        "direction_",
        "abs_direction_",
        "shape_change_",
        "approach_speed_",
        "separation_speed_",
        "retreat_speed_",
        "aggression_score_proxy",
        "partner_distance_delta",
        "nearest_dist_delta",
        "pen_distance_delta_",
        "pen_normal_speed_",
        "pen_approach_speed_",
        "pen_retreat_speed_",
        "pen_parallel_speed_",
        "pen_motion_delta_",
        "pen_adjacent_motion_pair_",
        "pen_sparse_velocity_pair_",
    )
    derived_exact = {
        "adjacent_motion_pair_valid",
        "sparse_velocity_pair_valid",
        "motion_velocity_pair_valid",
        "roi_transition_pair_valid",
        "roi_target_entry_event",
        "roi_target_exit_event",
        "roi_target_near_entry_event",
        "roi_target_near_exit_event",
        "roi_motion_inside_score",
        "roi_motion_inside_score_per_second",
        "social_adjacent_pair_valid",
        "social_velocity_pair_valid",
    }
    drop = [
        column
        for column in rows.columns
        if column.endswith("_unit")
        or column in derived_exact
        or column.startswith(derived_prefixes)
    ]
    return rows.drop(columns=drop, errors="ignore").copy()


def _aggregate_window_features(
    wg: pd.DataFrame,
    window_duration_sec: float | None,
    *,
    expected_start: int,
    expected_end: int,
    evidence_config: TemporalEvidenceConfig,
) -> dict[str, Any]:
    if "feature_computation_grain" in wg.columns:
        grain_values = (
            wg["feature_computation_grain"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        input_grains = set(grain_values)
        invalid_grains = input_grains.difference(
            {"FRAME_LOCAL_PRIMITIVES", "NATIVE_UNIT_REVIEW_EVIDENCE"}
        )
        if invalid_grains:
            raise ValueError(
                "final view cannot consume another final-view artifact: "
                f"found={sorted(invalid_grains)}"
            )
        native_rows = grain_values.eq("NATIVE_UNIT_REVIEW_EVIDENCE")
        if native_rows.any():
            required_scope = {"pair_scope_key", "temporal_unit_key"}
            if not required_scope.issubset(wg.columns):
                raise ValueError(
                    "native review evidence lacks explicit pair scope"
                )
            invalid_native_scope = native_rows & (
                wg["pair_scope_key"]
                .fillna("")
                .astype(str)
                .ne(wg["temporal_unit_key"].fillna("").astype(str))
            )
            if invalid_native_scope.any():
                raise ValueError(
                    "native pair_scope_key does not match temporal_unit_key: "
                    f"count={int(invalid_native_scope.sum())}"
                )
    for claim_column in (
        "pair_recomputed_for_view",
        "aggregate_recomputed_for_view",
    ):
        if claim_column in wg.columns and _to_bool_series(
            wg[claim_column]
        ).any():
            raise ValueError(
                "final view cannot import a recomputed parent view: "
                f"column={claim_column}"
            )
    wg = _frame_local_primitives_for_view(wg)
    out: dict[str, Any] = {}

    def num(col: str) -> pd.Series:
        if col not in wg.columns:
            return pd.Series(dtype="float64")
        return pd.to_numeric(wg[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    motion = _recompute_view_motion(wg)
    speed = motion["speed_n_per_frame"]
    speed_sec = motion["speed_n_per_second"]
    disp = motion["displacement_n"]
    accel = motion["abs_acceleration_n_per_frame2"]
    acceleration_per_second2 = motion[
        "abs_tangential_acceleration_n_per_second2"
    ]
    direction = motion["abs_direction_change_rad"]
    shape = motion["shape_change_score"]
    area = num("area_n")
    aspect = num("aspect_ratio")

    out["speed_mean_window"] = _safe_mean(speed)
    out["speed_max_window"] = _safe_max(speed)
    out["speed_std_window"] = _safe_std(speed)
    out["speed_per_sec_mean_window"] = _safe_mean(speed_sec)
    out["speed_per_sec_max_window"] = _safe_max(speed_sec)
    out["speed_n_per_second_mean_window"] = _safe_mean(speed_sec)
    out["speed_n_per_second_max_window"] = _safe_max(speed_sec)
    out["speed_n_per_second_std_window"] = _safe_std(speed_sec)
    out["adjacent_motion_pair_count_window"] = motion[
        "adjacent_pair_count"
    ]
    out["sparse_velocity_pair_count_window"] = motion[
        "sparse_velocity_pair_count"
    ]
    out["path_length_n_window"] = _safe_sum(disp)
    out["path_length_n_per_sec_window"] = (
        out["path_length_n_window"] / window_duration_sec
        if window_duration_sec and window_duration_sec > 0
        else np.nan
    )
    out["motion_energy_window"] = (
        float(np.nansum(np.asarray(speed.dropna(), dtype="float64") ** 2))
        if not speed.dropna().empty
        else 0.0
    )
    out["motion_burstiness_window"] = (
        out["speed_std_window"] / (out["speed_mean_window"] + 1e-9)
        if np.isfinite(out["speed_std_window"])
        else 0.0
    )
    out["accel_abs_mean_window"] = _safe_mean(accel)
    out["accel_abs_max_window"] = _safe_max(accel)
    out["tangential_acceleration_n_per_second2_abs_mean_window"] = _safe_mean(
        acceleration_per_second2
    )
    out["tangential_acceleration_n_per_second2_abs_max_window"] = _safe_max(
        acceleration_per_second2
    )
    out["direction_change_abs_mean_window"] = _safe_mean(direction)
    out["direction_change_abs_max_window"] = _safe_max(direction)
    out["shape_transition_score_window"] = _safe_max(shape)
    out["area_n_std_window"] = _safe_std(area)
    out["aspect_ratio_std_window"] = _safe_std(aspect)
    out["bbox_stability_window"] = 1.0 / (
        1.0 + _nan_to_zero(out["area_n_std_window"]) + _nan_to_zero(out["aspect_ratio_std_window"])
    )

    displacement = float(motion["connected_displacement_n"])
    out["displacement_n_window"] = displacement
    out["displacement_ratio_window"] = (
        displacement / out["path_length_n_window"]
        if out["path_length_n_window"]
        and out["path_length_n_window"] > 0
        and np.isfinite(displacement)
        else np.nan
    )

    # ROI relation.
    for col, out_name in [
        ("roi_target_contact", "target_roi_contact_ratio_window"),
        ("roi_target_near", "target_roi_near_ratio_window"),
        ("roi_target_center_inside", "target_roi_center_inside_ratio_window"),
    ]:
        out[out_name] = _bool_mean(wg[col]) if col in wg.columns else 0.0
    out["target_roi_overlap_mean_window"] = _safe_mean(num("roi_target_max_overlap_ratio"))
    out["target_roi_overlap_max_window"] = _safe_max(num("roi_target_max_overlap_ratio"))
    out["target_roi_min_dist_n_mean_window"] = _safe_mean(num("roi_target_min_dist_n"))
    out["target_roi_min_dist_n_min_window"] = _safe_min(num("roi_target_min_dist_n"))
    roi_transitions = _recompute_view_roi_transitions(wg)
    out["target_roi_entry_count_window"] = roi_transitions["entry_count"]
    out["target_roi_exit_count_window"] = roi_transitions["exit_count"]
    out["target_roi_near_entry_count_window"] = roi_transitions[
        "near_entry_count"
    ]
    out["target_roi_near_exit_count_window"] = roi_transitions[
        "near_exit_count"
    ]
    out["roi_transition_valid_pair_count_window"] = roi_transitions[
        "valid_pair_count"
    ]

    # Social/interaction relation.
    out["nearest_dist_mean_window"] = _safe_mean(num("nearest_dist_n"))
    out["nearest_dist_min_window"] = _safe_min(num("nearest_dist_n"))
    out["nearest_pair_iou_max_window"] = _safe_max(num("nearest_pair_iou"))
    out["nearest_pair_overlap_max_window"] = _safe_max(num("nearest_pair_overlap_ratio"))
    out["social_density_mean_window"] = _safe_mean(num("social_density_near_count"))
    out["social_density_max_window"] = _safe_max(num("social_density_near_count"))
    out["pair_contact_ratio_window"] = (
        _bool_mean(wg["pair_contact_with_nearest"])
        if "pair_contact_with_nearest" in wg.columns
        else 0.0
    )
    social_motion = summarize_social_motion_dynamics(wg)
    out["approach_speed_max_window"] = social_motion["approach_speed_max"]
    out["separation_speed_max_window"] = social_motion[
        "separation_speed_max"
    ]
    out["approach_speed_n_per_second_max_window"] = social_motion[
        "approach_speed_n_per_second_max"
    ]
    out["retreat_speed_n_per_second_max_window"] = social_motion[
        "retreat_speed_n_per_second_max"
    ]
    out["aggression_score_proxy_mean_window"] = social_motion[
        "aggression_score_proxy_mean"
    ]
    out["aggression_score_proxy_max_window"] = social_motion[
        "aggression_score_proxy_max"
    ]
    out["aggression_score_proxy_n_per_second_mean_window"] = social_motion[
        "aggression_score_proxy_n_per_second_mean"
    ]
    out["aggression_score_proxy_n_per_second_max_window"] = social_motion[
        "aggression_score_proxy_n_per_second_max"
    ]
    pen_rows = wg
    pen_recompute_required = {
        "pen_context_available",
        "pen_center_inside",
        "pen_boundary_inward_normal_x",
        "pen_boundary_inward_normal_y",
        "image_width",
        "image_height",
        "timestamp_sec",
    }
    pen_available = _to_bool_series(
        wg.get(
            "pen_context_available",
            pd.Series(False, index=wg.index),
        )
    )
    if pen_available.any() and not pen_recompute_required.issubset(wg.columns):
        missing_pen = sorted(pen_recompute_required.difference(wg.columns))
        raise ValueError(
            "final view cannot recompute available pen context: "
            f"missing={missing_pen}"
        )
    if pen_recompute_required.issubset(wg.columns):
        pen_rows = recompute_pen_motion_for_view(wg)
    out.update(summarize_pen_context(pen_rows))
    temporal_summary = summarize_temporal_evidence(
        wg,
        expected_start=expected_start,
        expected_end=expected_end,
        suffix="_window",
        config=evidence_config,
    )
    out.update(temporal_summary)
    out["sparse_path_length_n_window"] = temporal_summary[
        "trajectory_sparse_path_length_n_window"
    ]

    return out


def _temporal_unit_keys(window_rows: pd.DataFrame) -> list[str]:
    """Return unique native-unit keys without ambiguous delimiter parsing."""

    if "temporal_unit_key" not in window_rows.columns:
        return []
    values = (
        window_rows["temporal_unit_key"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    return sorted(set(values.loc[values.ne("")]))


def _behavior_review_summary(wg: pd.DataFrame) -> dict[str, Any]:
    """Compute native-unit behavior review coverage without dropping rows."""
    fields_present = any(
        c in wg.columns
        for c in (
            "behavior_review_decision_present",
            "behavior_review_label_resolved",
            "behavior_reviewed_final",
        )
    )
    if not fields_present:
        return {
            "behavior_review_fields_present": False,
            "human_reviewed_behavior_consistency_status": "not_applicable",
            "behavior_reviewed_window_label": "",
            "behavior_review_coverage_ratio_window": np.nan,
            "behavior_review_label_resolution_ratio_window": np.nan,
            "behavior_review_train_eligibility_ratio_window": np.nan,
            "all_temporal_units_behavior_reviewed": False,
            "all_temporal_units_behavior_label_resolved": False,
            "all_temporal_units_behavior_train_eligible": False,
        }
    key = wg.get("temporal_unit_key", pd.Series(index=wg.index, dtype=str))
    key = key.fillna("").astype(str).str.strip()
    if key.eq("").all():
        key = pd.Series([f"frame:{i}" for i in wg.index], index=wg.index)
    present_col = (
        "behavior_review_decision_present"
        if "behavior_review_decision_present" in wg.columns
        else "review_decision_applied"
    )
    resolved_col = (
        "behavior_review_label_resolved"
        if "behavior_review_label_resolved" in wg.columns
        else None
    )
    eligible_col = (
        "behavior_review_include_in_training"
        if "behavior_review_include_in_training" in wg.columns
        else "review_include_in_training"
    )
    present = _to_bool_series(wg.get(present_col, pd.Series(False, index=wg.index)))
    resolved = (
        _to_bool_series(wg[resolved_col])
        if resolved_col is not None
        else present
        & wg.get("behavior_after_review", wg.get("behavior", ""))
        .fillna("")
        .astype(str)
        .ne("")
    )
    eligible = _to_bool_series(wg.get(eligible_col, pd.Series(False, index=wg.index)))
    final_label = wg.get(
        "behavior_reviewed_final",
        wg.get("behavior_after_review", wg.get("behavior", "")),
    ).fillna("").astype(str).str.strip()
    units = pd.DataFrame(
        {
            "present": present,
            "resolved": resolved,
            "eligible": eligible,
            "label": final_label,
            "unit": key,
        }
    ).groupby("unit", sort=False).agg(
        present=("present", "all"),
        resolved=("resolved", "all"),
        eligible=("eligible", "all"),
        label=("label", lambda s: "|".join(sorted(set(v for v in s if v)))),
    )
    total = len(units)
    reviewed = units["present"]
    resolved_all = units["resolved"]
    eligible_all = units["eligible"]
    coverage = float(reviewed.mean()) if total else 0.0
    resolution = float((reviewed & resolved_all).mean()) if total else 0.0
    eligibility = float((reviewed & resolved_all & eligible_all).mean()) if total else 0.0
    labels = units.loc[reviewed & resolved_all, "label"]
    final_values = final_label.loc[present & resolved & final_label.ne("")]
    final_dominant = (
        str(final_values.value_counts().idxmax()) if not final_values.empty else ""
    )
    if coverage < 1.0:
        status = "unreviewed" if coverage == 0.0 else "partial"
    elif resolution < 1.0:
        status = "unresolved"
    elif eligibility < 1.0:
        status = "excluded"
    elif len(set(labels[labels.ne("")])) == 1:
        status = "stable"
    else:
        status = "transition"
    return {
        "behavior_review_fields_present": True,
        "human_reviewed_behavior_consistency_status": status,
        "behavior_reviewed_window_label": final_dominant,
        "behavior_review_coverage_ratio_window": coverage,
        "behavior_review_label_resolution_ratio_window": resolution,
        "behavior_review_train_eligibility_ratio_window": eligibility,
        "all_temporal_units_behavior_reviewed": bool(total and reviewed.all()),
        "all_temporal_units_behavior_label_resolved": bool(
            total and (reviewed & resolved_all).all()
        ),
        "all_temporal_units_behavior_train_eligible": bool(
            total and (reviewed & resolved_all & eligible_all).all()
        ),
    }


def _review_training_summary(wg: pd.DataFrame) -> dict[str, Any]:
    """Summarize reviewed training masks without dropping any window row."""
    if wg.empty:
        return {
            "review_include_ratio_window": 1.0,
            "review_excluded_frame_count_window": 0,
            "review_training_actions_window": "",
            "review_sample_weight_mean_window": 1.0,
            "window_sample_weight": 0.0,
        }

    if "review_include_in_training" in wg.columns:
        include = _to_bool_series(wg["review_include_in_training"])
    else:
        include = pd.Series(True, index=wg.index)

    actions = ""
    if "review_training_action" in wg.columns:
        action_values = [
            str(v).strip()
            for v in wg["review_training_action"].dropna().astype(str).tolist()
            if str(v).strip() and str(v).strip().lower() != "nan"
        ]
        actions = "|".join(sorted(set(action_values)))
        exclude_action = pd.Series(
            [str(v).strip().lower() in {"exclude", "reject"} for v in wg["review_training_action"]],
            index=wg.index,
        )
        include = include & ~exclude_action

    if "review_sample_weight" in wg.columns:
        weights = pd.to_numeric(wg["review_sample_weight"], errors="coerce")
    else:
        weights = pd.Series(1.0, index=wg.index, dtype="float64")
    weights = weights.fillna(1.0).clip(lower=0.0, upper=1.0)

    excluded_count = int((~include).sum())
    include_ratio = float(include.mean()) if len(include) else 1.0
    weight_mean = float(weights.mean()) if len(weights) else 1.0
    window_weight = 0.0 if excluded_count else weight_mean

    return {
        "review_include_ratio_window": include_ratio,
        "review_excluded_frame_count_window": excluded_count,
        "review_training_actions_window": actions,
        "review_sample_weight_mean_window": weight_mean,
        "window_sample_weight": window_weight,
    }


def _empty_aggregate_features() -> dict[str, Any]:
    keys = [
        "speed_mean_window",
        "speed_max_window",
        "speed_std_window",
        "speed_per_sec_mean_window",
        "speed_per_sec_max_window",
        "speed_n_per_second_mean_window",
        "speed_n_per_second_max_window",
        "speed_n_per_second_std_window",
        "adjacent_motion_pair_count_window",
        "sparse_velocity_pair_count_window",
        "path_length_n_window",
        "sparse_path_length_n_window",
        "path_length_n_per_sec_window",
        "motion_energy_window",
        "motion_burstiness_window",
        "accel_abs_mean_window",
        "accel_abs_max_window",
        "tangential_acceleration_n_per_second2_abs_mean_window",
        "tangential_acceleration_n_per_second2_abs_max_window",
        "direction_change_abs_mean_window",
        "direction_change_abs_max_window",
        "shape_transition_score_window",
        "area_n_std_window",
        "aspect_ratio_std_window",
        "bbox_stability_window",
        "displacement_n_window",
        "displacement_ratio_window",
        "target_roi_contact_ratio_window",
        "target_roi_near_ratio_window",
        "target_roi_center_inside_ratio_window",
        "target_roi_overlap_mean_window",
        "target_roi_overlap_max_window",
        "target_roi_min_dist_n_mean_window",
        "target_roi_min_dist_n_min_window",
        "target_roi_entry_count_window",
        "target_roi_exit_count_window",
        "target_roi_near_entry_count_window",
        "target_roi_near_exit_count_window",
        "roi_transition_valid_pair_count_window",
        "nearest_dist_mean_window",
        "nearest_dist_min_window",
        "nearest_pair_iou_max_window",
        "nearest_pair_overlap_max_window",
        "social_density_mean_window",
        "social_density_max_window",
        "pair_contact_ratio_window",
        "approach_speed_max_window",
        "separation_speed_max_window",
        "approach_speed_n_per_second_max_window",
        "retreat_speed_n_per_second_max_window",
        "aggression_score_proxy_mean_window",
        "aggression_score_proxy_max_window",
        "aggression_score_proxy_n_per_second_mean_window",
        "aggression_score_proxy_n_per_second_max_window",
    ]
    out = {k: 0.0 if not k.endswith("count_window") else 0 for k in keys}
    out.update(empty_pen_context_summary())
    out.update(
        {
            key: 0 if key.endswith(("count_window", "valid_window")) else 0.0
            for key in WINDOW_TEMPORAL_EVIDENCE_COLUMNS
        }
    )
    return out


def _prepare_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "object_track_key" not in out.columns:
        track = out.get("track_id", pd.Series("", index=out.index)).fillna("").astype(str)
        pig = out.get("pig_id", pd.Series("", index=out.index)).fillna("").astype(str)
        out["object_track_key"] = (
            out.get("source_type", "").astype(str)
            + "|"
            + out.get("dataset_id", "").astype(str)
            + "|"
            + out.get("video_key", "").astype(str)
            + "|track="
            + track
            + "|pig="
            + pig
        )
    for col in ["frame_index", "timestamp_sec"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in [
        "bbox_valid",
        "hidden",
        "spatiotemporal_feature_valid",
        "roi_target_contact",
        "roi_target_near",
        "roi_target_center_inside",
        "pair_contact_with_nearest",
    ]:
        if col not in out.columns:
            out[col] = False if col == "hidden" else True
        out[col] = _to_bool_series(out[col])
    for col in [
        "source_type",
        "dataset_id",
        "video_key",
        "pig_id",
        "track_id",
        "behavior",
        "behavior_temporal_final",
        "temporal_unit_key",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


def _intervals_cover_span(intervals: pd.DataFrame, start: int, end: int) -> bool:
    if intervals is None or intervals.empty:
        return False
    spans = (
        intervals[["label_window_start", "label_window_end"]]
        .dropna()
        .astype(int)
        .sort_values("label_window_start")
        .to_numpy()
    )
    cursor = int(start)
    for s, e in spans:
        if e < cursor:
            continue
        if s > cursor:
            return False
        cursor = max(cursor, int(e) + 1)
        if cursor > end:
            return True
    return cursor > end


def _looks_like_transition(interval_subset: pd.DataFrame | None, wg: pd.DataFrame) -> bool:
    if (
        interval_subset is not None
        and not interval_subset.empty
        and "label_window_start" in interval_subset.columns
    ):
        ordered = (
            interval_subset.sort_values("label_window_start")["behavior_temporal_final"]
            .fillna("")
            .astype(str)
            .tolist()
        )
    else:
        ordered = (
            wg.sort_values("frame_index")
            .get("behavior_temporal_final", wg.sort_values("frame_index")["behavior"])
            .fillna("")
            .astype(str)
            .tolist()
        )
    ordered = [x for x in ordered if x]
    if len(set(ordered)) <= 1:
        return False
    # If behaviors appear in contiguous blocks rather than alternating, treat as transition.
    changes = sum(1 for a, b in zip(ordered, ordered[1:], strict=False) if a != b)
    return changes <= max(1, len(set(ordered)))


def _selected_view_coordinates(
    window_rows: pd.DataFrame,
    selected_indices: list[int],
) -> list[float | None]:
    ordered = window_rows.assign(
        _selected_frame_index=pd.to_numeric(
            window_rows["frame_index"],
            errors="coerce",
        ),
        _selected_timestamp=pd.to_numeric(
            window_rows.get(
                "timestamp_sec",
                pd.Series(np.nan, index=window_rows.index),
            ),
            errors="coerce",
        ),
    ).dropna(subset=["_selected_frame_index"])
    timestamp_by_frame = ordered.set_index("_selected_frame_index")[
        "_selected_timestamp"
    ]
    return [
        float(timestamp_by_frame.loc[index])
        if index in timestamp_by_frame.index
        and np.isfinite(timestamp_by_frame.loc[index])
        else None
        for index in selected_indices
    ]


def _selected_view_pair_deltas(
    selected_indices: list[int],
    selected_timestamps: list[float | None],
) -> tuple[list[int], list[float | None]]:
    frame_deltas = [int(value) for value in np.diff(selected_indices)]
    time_deltas: list[float | None] = []
    for previous, current in zip(
        selected_timestamps,
        selected_timestamps[1:],
        strict=False,
    ):
        time_deltas.append(
            float(current - previous)
            if previous is not None and current is not None
            else None
        )
    return frame_deltas, time_deltas


def _window_timing_summary(
    window_rows: pd.DataFrame,
    *,
    start: int,
    end: int,
    expected_slot_count: int,
    default_fps: float | None,
) -> dict[str, float | int]:
    """Separate declared timeline duration from sparse observation timing."""

    ordered = window_rows.sort_values("frame_index", kind="mergesort")
    frames = pd.to_numeric(ordered["frame_index"], errors="coerce").to_numpy(
        dtype="float64"
    )
    timestamps = pd.to_numeric(
        ordered.get("timestamp_sec", pd.Series(np.nan, index=ordered.index)),
        errors="coerce",
    ).to_numpy(dtype="float64")
    frame_delta = np.diff(frames)
    time_delta = np.diff(timestamps)
    valid_velocity_pair = (
        np.isfinite(frame_delta)
        & (frame_delta > 0)
        & np.isfinite(time_delta)
        & (time_delta > 0)
    )
    fps_samples = frame_delta[valid_velocity_pair] / time_delta[
        valid_velocity_pair
    ]
    declared_fps = (
        float(np.median(fps_samples))
        if fps_samples.size
        else float(default_fps)
        if default_fps is not None and default_fps > 0
        else np.nan
    )
    finite_timestamps = timestamps[np.isfinite(timestamps)]
    timestamp_start = (
        float(finite_timestamps.min()) if finite_timestamps.size else np.nan
    )
    timestamp_end = (
        float(finite_timestamps.max()) if finite_timestamps.size else np.nan
    )
    observed_span = (
        max(0.0, timestamp_end - timestamp_start)
        if finite_timestamps.size >= 2
        else np.nan
    )
    observed_count = int(np.unique(frames[np.isfinite(frames)]).size)
    observation_rate = (
        float((observed_count - 1) / observed_span)
        if observed_count >= 2
        and np.isfinite(observed_span)
        and observed_span > 0
        else np.nan
    )
    adjacent_valid = valid_velocity_pair & np.isclose(frame_delta, 1.0)
    expected_pairs = max(0, int(expected_slot_count) - 1)
    physical_span = (
        float((end - start) / declared_fps)
        if np.isfinite(declared_fps) and declared_fps > 0
        else np.nan
    )
    declared_timeline_slot_count = max(0, int(end) - int(start) + 1)
    declared_duration = (
        float(declared_timeline_slot_count / declared_fps)
        if np.isfinite(declared_fps) and declared_fps > 0
        else np.nan
    )
    return {
        "declared_window_duration_seconds": declared_duration,
        "observed_timestamp_span_seconds": observed_span,
        "adjacent_observed_duration_seconds": float(
            np.sum(time_delta[adjacent_valid])
        ),
        "physical_span_seconds": physical_span,
        "expected_slot_count": int(expected_slot_count),
        "observed_slot_count": observed_count,
        "effective_observation_rate_hz": observation_rate,
        "adjacent_pair_coverage_ratio": (
            float(adjacent_valid.sum() / expected_pairs)
            if expected_pairs
            else 0.0
        ),
        "declared_timeline_fps": declared_fps,
        "timestamp_start_sec": timestamp_start,
        "timestamp_end_sec": timestamp_end,
    }


def _interaction_policy_for_behavior(behavior: str) -> dict[str, Any]:
    if behavior == "fight":
        return {
            "interaction_annotation_policy": "fight_directly_involved_group",
            "interaction_role_policy": "attacker_or_target_reacting_or_directly_involved",
            "label_propagation_policy": "directly_involved_pigs",
            "allow_label_propagation": True,
            "requires_partner_context": True,
            "social_nose_actor_only": False,
            "fight_group_label": True,
        }
    if behavior == "social-nose":
        return {
            "interaction_annotation_policy": "social_nose_active_actor_only",
            "interaction_role_policy": "active_snout_actor_only",
            "label_propagation_policy": "actor_only",
            "allow_label_propagation": False,
            "requires_partner_context": True,
            "social_nose_actor_only": True,
            "fight_group_label": False,
        }
    return {
        "interaction_annotation_policy": "not_interaction",
        "interaction_role_policy": "none",
        "label_propagation_policy": "none",
        "allow_label_propagation": False,
        "requires_partner_context": False,
        "social_nose_actor_only": False,
        "fight_group_label": False,
    }


def _make_window_id(first: pd.Series, length: int, start: int, end: int) -> str:
    object_key = str(first.get("object_track_key", "")) if isinstance(first, pd.Series) else ""
    return f"{object_key}|win={length}|{start}-{end}"


def _bool_mean(s: pd.Series | Iterable[Any]) -> float:
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if len(s) == 0:
        return 0.0
    return float(_to_bool_series(s).mean())


def _window_hidden_trust(window_rows: pd.DataFrame) -> pd.Series:
    """Use explicit review trust, with a source-aware legacy fallback."""
    if "hidden_is_trusted" in window_rows.columns:
        return _to_bool_series(window_rows["hidden_is_trusted"])
    source = (
        window_rows.get(
            "source_type",
            pd.Series("", index=window_rows.index),
        )
        .fillna("")
        .astype(str)
    )
    return source.eq(LEGACY_SOURCE_TYPE)


def _longest_hidden_run_frames(
    window_rows: pd.DataFrame,
    hidden: pd.Series,
) -> int:
    """Count the longest run of Hidden rows on consecutive frame indices."""
    if window_rows.empty:
        return 0
    if "frame_index" in window_rows.columns:
        frame_index = pd.to_numeric(window_rows["frame_index"], errors="coerce")
    else:
        frame_index = pd.Series(
            np.arange(len(window_rows), dtype=int),
            index=window_rows.index,
        )
    ordered = pd.DataFrame(
        {
            "frame_index": frame_index,
            "hidden": hidden.reindex(window_rows.index).fillna(False).astype(bool),
        }
    ).sort_values("frame_index", kind="mergesort")

    longest = 0
    current = 0
    previous_frame: int | None = None
    for frame_value, is_hidden in ordered.itertuples(index=False, name=None):
        if pd.isna(frame_value):
            current = 0
            previous_frame = None
            continue
        frame = int(frame_value)
        if is_hidden:
            current = current + 1 if previous_frame == frame - 1 else 1
            longest = max(longest, current)
        else:
            current = 0
        previous_frame = frame
    return longest


def _classify_hidden_window(
    *,
    hidden_ratio: float,
    longest_run_ratio: float,
    config: SequenceWindowConfig,
) -> tuple[str, list[str]]:
    """Assign a visibility-evidence tier without using Hidden as model X."""
    if not config.exclude_high_hidden_from_main:
        return "audit_only", []

    robust_reasons: list[str] = []
    if hidden_ratio > config.max_hidden_ratio_robust:
        robust_reasons.append("hidden_ratio_above_robust_threshold")
    if longest_run_ratio > config.max_hidden_run_ratio_robust:
        robust_reasons.append("hidden_run_above_robust_threshold")
    if robust_reasons:
        return "exclude", robust_reasons

    main_reasons: list[str] = []
    if hidden_ratio > config.max_hidden_ratio_main:
        main_reasons.append("hidden_ratio_above_main_threshold")
    if longest_run_ratio > config.max_hidden_run_ratio_main:
        main_reasons.append("hidden_run_above_main_threshold")
    if main_reasons:
        return "robust_train_only", main_reasons
    return "main_train", []


def _to_bool_series(s: pd.Series | Iterable[Any]) -> pd.Series:
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if not s.empty else 0.0


def _safe_sum(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.sum()) if not s.empty else 0.0


def _safe_max(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.max()) if not s.empty else 0.0


def _safe_min(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.min()) if not s.empty else np.nan


def _safe_std(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.std(ddof=0)) if not s.empty else 0.0


def _nan_to_zero(x: float) -> float:
    try:
        return 0.0 if not np.isfinite(float(x)) else float(x)
    except Exception:
        return 0.0


def _float_or_nan(x: Any) -> float:
    try:
        val = float(x)
        return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def _value_counts_dict(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df is None or df.empty or column not in df.columns:
        return {}
    counts = df[column].fillna("<NA>").astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _numeric_summary(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    if df is None or df.empty or column not in df.columns:
        return {}
    s = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }
