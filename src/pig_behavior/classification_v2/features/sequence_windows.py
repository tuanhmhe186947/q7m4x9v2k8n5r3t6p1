"""Sequence/window manifest builder for classification_v2.

This module consumes harmonized/enhanced frame-object features and creates a
long-format training-window table. Each output row is one candidate window for a
single tracked pig/object, with window-specific aggregate features. This avoids
using 16-frame or 6-frame unit means as if they represented every 6/8/12/16
training window.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.temporal_harmonization import (
    CVAT_SOURCE_TYPES,
    LEGACY_SOURCE_TYPE,
    TemporalHarmonizationConfig,
    build_temporal_label_intervals,
    harmonize_temporal_labels,
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
    max_hidden_ratio_main: float = 0.5
    min_spatiotemporal_valid_ratio: float = 1.0
    include_mixed_windows: bool = True
    max_windows_per_track: int | None = None
    aggregate_observed_rows_only: bool = True

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
        if not (0 <= self.min_spatiotemporal_valid_ratio <= 1):
            raise ValueError("min_spatiotemporal_valid_ratio must be in [0, 1]")
        if self.max_windows_per_track is not None and self.max_windows_per_track <= 0:
            raise ValueError("max_windows_per_track must be None or > 0")


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
    max_hidden_ratio_main: float = 0.5,
    min_spatiotemporal_valid_ratio: float = 1.0,
    include_mixed_windows: bool = True,
    max_windows_per_track: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build harmonized frame features, intervals, and window manifest.

    Returns
    -------
    harmonized_frames, temporal_intervals, sequence_windows
    """
    config = SequenceWindowConfig(
        window_lengths=tuple(int(w) for w in window_lengths),
        legacy_window_stride=legacy_window_stride,
        cvat_window_stride_intervals=cvat_window_stride_intervals,
        cvat_label_stride=cvat_label_stride,
        legacy_expected_sequence_length=legacy_expected_sequence_length,
        default_fps=default_fps,
        min_bbox_valid_ratio=min_bbox_valid_ratio,
        max_hidden_ratio_main=max_hidden_ratio_main,
        min_spatiotemporal_valid_ratio=min_spatiotemporal_valid_ratio,
        include_mixed_windows=include_mixed_windows,
        max_windows_per_track=max_windows_per_track,
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
    return harmonized, intervals, windows


def audit_sequence_windows(windows: pd.DataFrame, intervals: pd.DataFrame | None = None) -> dict[str, Any]:
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
        "window_exclusion_reason",
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
            windows.get("window_valid_for_main_train", False).astype(str).str.lower().isin({"true", "1", "yes"})
            & ~windows.get("sequence_label_status", "").astype(str).eq("stable")
        ]
        if len(invalid_main):
            errors.append(f"main_train_windows_not_stable={len(invalid_main)}")

        mixed = int(
            windows.get("sequence_label_status", pd.Series(dtype=str)).astype(str).isin({"mixed", "transition"}).sum()
        )
        if mixed:
            warnings.append(f"mixed_or_transition_windows={mixed}")
    else:
        warnings.append("no_sequence_windows_generated")

    return {
        "window_rows": int(len(windows)),
        "temporal_intervals": int(len(intervals)) if intervals is not None else None,
        "sources": _value_counts_dict(windows, "source_type"),
        "window_length_frames": _value_counts_dict(windows, "window_length_frames"),
        "sequence_label_status": _value_counts_dict(windows, "sequence_label_status"),
        "window_valid_for_main_train": _value_counts_dict(windows, "window_valid_for_main_train"),
        "behavior_window_label": _value_counts_dict(windows, "behavior_window_label"),
        "label_propagation_policy": _value_counts_dict(windows, "label_propagation_policy"),
        "window_exclusion_reason_top": _value_counts_dict(windows, "window_exclusion_reason"),
        "review_excluded_frame_count_window": _value_counts_dict(windows, "review_excluded_frame_count_window"),
        "window_sample_weight": _numeric_summary(windows, "window_sample_weight"),
        "speed_mean_window": _numeric_summary(windows, "speed_mean_window"),
        "target_roi_contact_ratio_window": _numeric_summary(windows, "target_roi_contact_ratio_window"),
        "pair_contact_ratio_window": _numeric_summary(windows, "pair_contact_ratio_window"),
        "hidden_ratio_window": _numeric_summary(windows, "hidden_ratio_window"),
        "bbox_valid_ratio_window": _numeric_summary(windows, "bbox_valid_ratio_window"),
        "errors": errors,
        "warnings": warnings,
    }


def _build_windows_from_harmonized(
    frames: pd.DataFrame,
    intervals: pd.DataFrame,
    config: SequenceWindowConfig,
) -> pd.DataFrame:
    frames = _prepare_frame_columns(frames)
    rows: list[dict[str, Any]] = []

    # Build interval lookup for CVAT tracks.
    intervals_by_track: dict[str, pd.DataFrame] = {}
    if intervals is not None and not intervals.empty:
        for key, g in intervals.groupby("object_track_key", dropna=False, sort=False):
            intervals_by_track[str(key)] = g.sort_values("label_window_start", kind="mergesort").reset_index(drop=True)

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
        ["source_type", "dataset_id", "video_key", "object_track_key", "window_length_frames", "window_start_frame"],
        kind="mergesort",
    ).reset_index(drop=True)
    windows.insert(0, "window_row_index", np.arange(len(windows), dtype="int64"))
    return windows


def _generate_legacy_windows(g: pd.DataFrame, config: SequenceWindowConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frames_available = sorted(set(int(x) for x in pd.to_numeric(g["frame_index"], errors="coerce").dropna()))
    if not frames_available:
        return rows
    frame_set = set(frames_available)
    min_f, max_f = min(frames_available), max(frames_available)
    produced = 0
    for length in config.window_lengths:
        last_start = max_f - length + 1
        if last_start < min_f:
            rows.append(_empty_invalid_window(g, length, min_f, min_f + length - 1, "not_enough_frames_for_window"))
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
            if config.max_windows_per_track is not None and produced >= config.max_windows_per_track:
                return rows
    return rows


def _generate_cvat_windows(
    g: pd.DataFrame, intervals: pd.DataFrame, config: SequenceWindowConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if intervals is None or intervals.empty:
        min_f = int(pd.to_numeric(g["frame_index"], errors="coerce").min())
        return [
            _empty_invalid_window(
                g, int(config.window_lengths[0]), min_f, min_f + int(config.window_lengths[0]) - 1, "no_cvat_intervals"
            )
        ]

    intervals = intervals.copy()
    intervals["label_window_start"] = pd.to_numeric(intervals["label_window_start"], errors="coerce")
    intervals["label_window_end"] = pd.to_numeric(intervals["label_window_end"], errors="coerce")
    intervals = intervals.dropna(subset=["label_window_start", "label_window_end"]).sort_values(
        "label_window_start", kind="mergesort"
    )
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
                set(overlap["temporal_unit_key"].astype(str)) if "temporal_unit_key" in overlap.columns else set()
            )
            wg = g[g["temporal_unit_key"].astype(str).isin(interval_keys)] if interval_keys else g.iloc[0:0]
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
            if config.max_windows_per_track is not None and produced >= config.max_windows_per_track:
                return rows
    return rows


def _generate_generic_windows(g: pd.DataFrame, config: SequenceWindowConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frames_available = sorted(set(int(x) for x in pd.to_numeric(g["frame_index"], errors="coerce").dropna()))
    if not frames_available:
        return rows
    min_f, max_f = min(frames_available), max(frames_available)
    produced = 0
    for length in config.window_lengths:
        last_start = max_f - length + 1
        if last_start < min_f:
            rows.append(_empty_invalid_window(g, length, min_f, min_f + length - 1, "not_enough_frames_for_window"))
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
            if config.max_windows_per_track is not None and produced >= config.max_windows_per_track:
                return rows
    return rows


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
) -> dict[str, Any]:
    if wg.empty:
        return _empty_invalid_window(
            pd.DataFrame(), length, start, end, "no_observed_rows_in_window", source_window_type=source_window_type
        )

    first = wg.iloc[0]
    behavior_source = (
        interval_subset["behavior_temporal_final"].fillna("").astype(str)
        if interval_subset is not None and "behavior_temporal_final" in interval_subset.columns
        else wg.get("behavior_temporal_final", wg["behavior"]).fillna("").astype(str)
    )
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
    hidden_ratio = _bool_mean(wg.get("hidden", pd.Series(False, index=wg.index)))
    spatio_ratio = _bool_mean(wg.get("spatiotemporal_feature_valid", pd.Series(True, index=wg.index)))
    review_summary = _review_training_summary(wg)

    ts_start, ts_end, duration_from_ts = _timestamp_span(wg)
    effective_fps = _infer_effective_fps(wg, start, end, duration_from_ts, config.default_fps)
    window_duration_sec = (
        duration_from_ts
        if duration_from_ts and duration_from_ts > 0
        else (length / effective_fps if effective_fps and effective_fps > 0 else np.nan)
    )

    reasons: list[str] = []
    if label_status != "stable":
        reasons.append(f"label_{label_status}")
    if not label_coverage_complete:
        reasons.append("label_coverage_incomplete")
    if bbox_valid_ratio < config.min_bbox_valid_ratio:
        reasons.append("bbox_valid_ratio_below_threshold")
    if hidden_ratio > config.max_hidden_ratio_main:
        reasons.append("hidden_ratio_above_threshold")
    if spatio_ratio < config.min_spatiotemporal_valid_ratio:
        reasons.append("spatiotemporal_valid_ratio_below_threshold")
    if review_summary["review_excluded_frame_count_window"] > 0:
        reasons.append("review_excluded_rows_in_window")

    valid_main = not reasons and label_status == "stable"
    training_tier = (
        "main_train"
        if valid_main
        else "robust_train_only"
        if label_status in {"transition", "mixed"} and config.include_mixed_windows
        else "exclude"
    )

    row: dict[str, Any] = {
        "window_id": _make_window_id(first, length, start, end),
        "source_window_type": source_window_type,
        "source_type": str(first.get("source_type", "")),
        "dataset_id": str(first.get("dataset_id", "")),
        "video_key": str(first.get("video_key", "")),
        "object_track_key": str(first.get("object_track_key", "")),
        "pig_id": str(first.get("pig_id", "")),
        "track_id": str(first.get("track_id", "")),
        "window_length_frames": int(length),
        "window_start_frame": int(start),
        "window_end_frame": int(end),
        "window_duration_sec": _float_or_nan(window_duration_sec),
        "effective_fps": _float_or_nan(effective_fps),
        "timestamp_start_sec": _float_or_nan(ts_start),
        "timestamp_end_sec": _float_or_nan(ts_end),
        "observed_row_count_window": int(len(wg)),
        "observed_frame_count_window": int(wg["frame_index"].nunique(dropna=True))
        if "frame_index" in wg.columns
        else int(len(wg)),
        "label_coverage_complete": bool(label_coverage_complete),
        "temporal_unit_keys_window": "|".join(
            sorted(set(wg.get("temporal_unit_key", pd.Series(dtype=str)).fillna("").astype(str)))
        )
        if "temporal_unit_key" in wg.columns
        else "",
        "num_temporal_units_window": int(wg.get("temporal_unit_key", pd.Series(dtype=str)).nunique(dropna=True))
        if "temporal_unit_key" in wg.columns
        else 0,
        "num_behaviors_window": int(len(unique_behaviors)),
        "unique_behaviors_window": "|".join(unique_behaviors),
        "behavior_window_label": str(dominant),
        "sequence_label_status": label_status,
        "window_valid_for_main_train": bool(valid_main),
        "window_training_tier_recommendation": training_tier,
        "window_exclusion_reason": ";".join(reasons),
        "bbox_valid_ratio_window": bbox_valid_ratio,
        "hidden_ratio_window": hidden_ratio,
        "visible_ratio_window": 1.0 - hidden_ratio,
        "spatiotemporal_feature_valid_ratio_window": spatio_ratio,
        **review_summary,
    }

    row.update(_interaction_policy_for_behavior(row["behavior_window_label"]))
    row.update(_aggregate_window_features(wg, window_duration_sec))
    return row


def _empty_invalid_window(
    g: pd.DataFrame,
    length: int,
    start: int,
    end: int,
    reason: str,
    *,
    source_window_type: str = "unknown_window",
) -> dict[str, Any]:
    first = g.iloc[0] if g is not None and not g.empty else pd.Series(dtype=object)
    row = {
        "window_id": _make_window_id(first, length, start, end),
        "source_window_type": source_window_type,
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
        "observed_row_count_window": 0,
        "observed_frame_count_window": 0,
        "label_coverage_complete": False,
        "temporal_unit_keys_window": "",
        "num_temporal_units_window": 0,
        "num_behaviors_window": 0,
        "unique_behaviors_window": "",
        "behavior_window_label": "",
        "sequence_label_status": "incomplete",
        "window_valid_for_main_train": False,
        "window_training_tier_recommendation": "exclude",
        "window_exclusion_reason": reason,
        "bbox_valid_ratio_window": 0.0,
        "hidden_ratio_window": 0.0,
        "visible_ratio_window": 0.0,
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


def _aggregate_window_features(wg: pd.DataFrame, window_duration_sec: float | None) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def num(col: str) -> pd.Series:
        if col not in wg.columns:
            return pd.Series(dtype="float64")
        return pd.to_numeric(wg[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    speed = num("speed_n_per_frame")
    speed_sec = num("speed_n_per_sec")
    disp = num("displacement_n")
    accel = num("abs_accel_n_per_frame2")
    direction = num("abs_direction_change_rad")
    shape = num("shape_change_score")
    area = num("area_n")
    aspect = num("aspect_ratio")

    out["speed_mean_window"] = _safe_mean(speed)
    out["speed_max_window"] = _safe_max(speed)
    out["speed_std_window"] = _safe_std(speed)
    out["speed_per_sec_mean_window"] = _safe_mean(speed_sec)
    out["speed_per_sec_max_window"] = _safe_max(speed_sec)
    out["path_length_n_window"] = _safe_sum(disp)
    out["path_length_n_per_sec_window"] = (
        out["path_length_n_window"] / window_duration_sec if window_duration_sec and window_duration_sec > 0 else np.nan
    )
    out["motion_energy_window"] = (
        float(np.nansum(np.asarray(speed.dropna(), dtype="float64") ** 2)) if not speed.dropna().empty else 0.0
    )
    out["motion_burstiness_window"] = (
        out["speed_std_window"] / (out["speed_mean_window"] + 1e-9) if np.isfinite(out["speed_std_window"]) else 0.0
    )
    out["accel_abs_mean_window"] = _safe_mean(accel)
    out["accel_abs_max_window"] = _safe_max(accel)
    out["direction_change_abs_mean_window"] = _safe_mean(direction)
    out["direction_change_abs_max_window"] = _safe_max(direction)
    out["shape_transition_score_window"] = _safe_max(shape)
    out["area_n_std_window"] = _safe_std(area)
    out["aspect_ratio_std_window"] = _safe_std(aspect)
    out["bbox_stability_window"] = 1.0 / (
        1.0 + _nan_to_zero(out["area_n_std_window"]) + _nan_to_zero(out["aspect_ratio_std_window"])
    )

    # First-last displacement ratio.
    if {"cx_n", "cy_n"}.issubset(wg.columns) and len(wg) >= 2:
        coords = wg.sort_values("frame_index")[["cx_n", "cy_n"]].apply(pd.to_numeric, errors="coerce")
        first = coords.iloc[0]
        last = coords.iloc[-1]
        displacement = (
            float(np.sqrt((last["cx_n"] - first["cx_n"]) ** 2 + (last["cy_n"] - first["cy_n"]) ** 2))
            if coords.notna().all(axis=None)
            else np.nan
        )
    else:
        displacement = np.nan
    out["displacement_n_window"] = displacement
    out["displacement_ratio_window"] = (
        displacement / out["path_length_n_window"]
        if out["path_length_n_window"] and out["path_length_n_window"] > 0 and np.isfinite(displacement)
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
    out["target_roi_entry_count_window"] = (
        int(_safe_sum(num("roi_target_entry_event"))) if "roi_target_entry_event" in wg.columns else 0
    )
    out["target_roi_exit_count_window"] = (
        int(_safe_sum(num("roi_target_exit_event"))) if "roi_target_exit_event" in wg.columns else 0
    )

    # Social/interaction relation.
    out["nearest_dist_mean_window"] = _safe_mean(num("nearest_dist_n"))
    out["nearest_dist_min_window"] = _safe_min(num("nearest_dist_n"))
    out["nearest_pair_iou_max_window"] = _safe_max(num("nearest_pair_iou"))
    out["nearest_pair_overlap_max_window"] = _safe_max(num("nearest_pair_overlap_ratio"))
    out["social_density_mean_window"] = _safe_mean(num("social_density_near_count"))
    out["social_density_max_window"] = _safe_max(num("social_density_near_count"))
    out["pair_contact_ratio_window"] = (
        _bool_mean(wg["pair_contact_with_nearest"]) if "pair_contact_with_nearest" in wg.columns else 0.0
    )
    out["approach_speed_max_window"] = _safe_max(num("approach_speed_n_per_frame"))
    out["separation_speed_max_window"] = _safe_max(num("separation_speed_n_per_frame"))
    out["aggression_score_proxy_mean_window"] = _safe_mean(num("aggression_score_proxy"))
    out["aggression_score_proxy_max_window"] = _safe_max(num("aggression_score_proxy"))

    return out


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
        "path_length_n_window",
        "path_length_n_per_sec_window",
        "motion_energy_window",
        "motion_burstiness_window",
        "accel_abs_mean_window",
        "accel_abs_max_window",
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
        "nearest_dist_mean_window",
        "nearest_dist_min_window",
        "nearest_pair_iou_max_window",
        "nearest_pair_overlap_max_window",
        "social_density_mean_window",
        "social_density_max_window",
        "pair_contact_ratio_window",
        "approach_speed_max_window",
        "separation_speed_max_window",
        "aggression_score_proxy_mean_window",
        "aggression_score_proxy_max_window",
    ]
    return {k: 0.0 if not k.endswith("count_window") else 0 for k in keys}


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
    if interval_subset is not None and not interval_subset.empty and "label_window_start" in interval_subset.columns:
        ordered = (
            interval_subset.sort_values("label_window_start")["behavior_temporal_final"].fillna("").astype(str).tolist()
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


def _timestamp_span(wg: pd.DataFrame) -> tuple[float, float, float | None]:
    if "timestamp_sec" not in wg.columns:
        return np.nan, np.nan, None
    ts = pd.to_numeric(wg["timestamp_sec"], errors="coerce").dropna().sort_values()
    if ts.empty:
        return np.nan, np.nan, None
    start = float(ts.min())
    end = float(ts.max())
    if len(ts) >= 2:
        deltas = ts.diff().dropna()
        median_delta = float(deltas.median()) if not deltas.empty and deltas.median() > 0 else 0.0
        duration = max(0.0, end - start + median_delta)
    else:
        duration = None
    return start, end, duration


def _infer_effective_fps(
    wg: pd.DataFrame, start: int, end: int, duration: float | None, default_fps: float | None
) -> float:
    observed_frames = int(wg["frame_index"].nunique(dropna=True)) if "frame_index" in wg.columns else int(len(wg))
    if duration is not None and duration > 0 and observed_frames > 1:
        return float(observed_frames / duration)
    if default_fps is not None and default_fps > 0:
        return float(default_fps)
    frame_span = max(1, int(end - start + 1))
    return float(frame_span)


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
        return {"count": 0, "mean": None, "std": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p50": float(s.quantile(0.50)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }
