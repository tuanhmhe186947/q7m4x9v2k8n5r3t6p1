"""Leakage-safe per-frame spatial sequence export for classification_v2 windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FORBIDDEN_SUBSTRINGS = (
    "behavior",
    "label",
    "review",
    "manual",
    "temporal_unit",
    "window_id",
    "frame_uid",
    "video_key",
    "dataset_id",
    "pig_id",
    "track_id",
    "path",
)

SPATIAL_FRAME_FEATURES: dict[str, list[str]] = {
    "bbox_xywh_n": ["cx_n", "cy_n", "bw_n", "bh_n"],
    "bbox_shape_n": ["area_n", "aspect_ratio"],
    "motion_delta": [
        "delta_cx_n",
        "delta_cy_n",
        "delta_bw_n",
        "delta_bh_n",
        "delta_area_n",
        "delta_aspect_ratio",
        "speed_n_per_frame",
        "speed_n_per_sec",
        "abs_accel_n_per_frame2",
        "abs_direction_change_rad",
    ],
    "roi_class_relation": [
        "roi_feeder_min_dist_n",
        "roi_feeder_max_overlap_ratio",
        "roi_feeder_max_iou",
        "roi_feeder_center_inside",
        "roi_feeder_near",
        "roi_feeder_contact",
        "roi_drinker_min_dist_n",
        "roi_drinker_max_overlap_ratio",
        "roi_drinker_max_iou",
        "roi_drinker_center_inside",
        "roi_drinker_near",
        "roi_drinker_contact",
        "roi_toy_min_dist_n",
        "roi_toy_max_overlap_ratio",
        "roi_toy_max_iou",
        "roi_toy_center_inside",
        "roi_toy_near",
        "roi_toy_contact",
    ],
    "social_relation": [
        "nearest_dist_n",
        "nearest_pair_iou",
        "nearest_pair_overlap_ratio",
        "social_density_near_count",
        "social_contact_count",
        "nearest_dist_delta",
        "approach_speed_n_per_frame",
        "separation_speed_n_per_frame",
        "pair_contact_with_nearest",
        "aggression_score_proxy",
    ],
    "quality_mask": [
        "bbox_valid",
        "actor_bbox_valid",
        "geometry_feature_valid",
        "spatiotemporal_feature_valid",
    ],
}


@dataclass(slots=True)
class SpatialSequenceExport:
    arrays: dict[str, np.ndarray]
    feature_names: dict[str, list[str]]
    audit: dict[str, Any]


def export_spatial_sequences(
    windows: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    max_window_length: int | None = None,
) -> SpatialSequenceExport:
    """Build fixed-length spatial arrays aligned to sequence-window rows.

    The returned arrays are model inputs only. Label, ID, path, review, and
    policy columns are excluded; identifiers are retained only in the audit
    surface outside the arrays.
    """
    required_windows = [
        "window_id",
        "object_track_key",
        "window_start_frame",
        "window_end_frame",
        "window_length_frames",
    ]
    required_frames = ["object_track_key", "frame_index"]
    missing_windows = [c for c in required_windows if c not in windows.columns]
    missing_frames = [c for c in required_frames if c not in frames.columns]
    if missing_windows or missing_frames:
        raise ValueError(f"Missing columns: windows={missing_windows} frames={missing_frames}")

    feature_names = _available_feature_names(frames)
    selected_cols = [c for cols in feature_names.values() for c in cols]
    forbidden_selected = [c for c in selected_cols if _is_forbidden(c)]
    if forbidden_selected:
        raise ValueError(f"Forbidden spatial feature columns selected: {forbidden_selected}")

    work_windows = windows.reset_index(drop=True).copy()
    work_windows["window_start_frame"] = pd.to_numeric(work_windows["window_start_frame"], errors="coerce")
    work_windows["window_end_frame"] = pd.to_numeric(work_windows["window_end_frame"], errors="coerce")
    work_windows["window_length_frames"] = pd.to_numeric(work_windows["window_length_frames"], errors="coerce")
    if max_window_length is None:
        max_window_length = int(work_windows["window_length_frames"].max())

    work_frames = frames[["object_track_key", "frame_index", *selected_cols]].copy()
    work_frames["frame_index"] = pd.to_numeric(work_frames["frame_index"], errors="coerce")
    work_frames = work_frames.dropna(subset=["object_track_key", "frame_index"])
    work_frames["frame_index"] = work_frames["frame_index"].astype(int)
    for col in selected_cols:
        work_frames[col] = _numeric_feature(work_frames[col])
    flat_feature_names: list[str] = []
    group_slices: dict[str, slice] = {}
    start_col = 0
    for name, cols in feature_names.items():
        flat_feature_names.extend(cols)
        group_slices[name] = slice(start_col, start_col + len(cols))
        start_col += len(cols)

    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, group in work_frames.groupby("object_track_key", sort=False):
        group = group.sort_values("frame_index")
        grouped[str(key)] = (
            group["frame_index"].to_numpy(dtype=np.int32, copy=True),
            group[flat_feature_names].to_numpy(dtype=np.float32, copy=True),
        )

    arrays = {
        name: np.zeros((len(work_windows), max_window_length, len(cols)), dtype=np.float32)
        for name, cols in feature_names.items()
    }
    length_mask = np.zeros((len(work_windows), max_window_length), dtype=np.float32)
    observed_mask = np.zeros((len(work_windows), max_window_length), dtype=np.float32)
    frame_index_sequence = np.full((len(work_windows), max_window_length), -1, dtype=np.int32)

    missing_frame_slots = 0
    truncated_windows = 0
    for object_key, window_group in work_windows.groupby("object_track_key", sort=False):
        frame_data = grouped.get(str(object_key))
        for i, row in window_group.iterrows():
            start = row["window_start_frame"]
            end = row["window_end_frame"]
            if pd.isna(start) or pd.isna(end):
                missing_frame_slots += max_window_length
                continue
            wanted_frames = np.arange(int(start), int(end) + 1, dtype=np.int32)
            if len(wanted_frames) > max_window_length:
                wanted_frames = wanted_frames[:max_window_length]
                truncated_windows += 1
            length_mask[i, : len(wanted_frames)] = 1.0
            frame_index_sequence[i, : len(wanted_frames)] = wanted_frames

            if frame_data is None:
                missing_frame_slots += len(wanted_frames)
                continue
            frame_indices, feature_matrix = frame_data
            positions = np.searchsorted(frame_indices, wanted_frames)
            bounded_positions = np.minimum(positions, len(frame_indices) - 1)
            valid = (positions < len(frame_indices)) & (frame_indices[bounded_positions] == wanted_frames)
            if not valid.any():
                missing_frame_slots += len(wanted_frames)
                continue
            valid_positions = positions[valid]
            slot_positions = np.flatnonzero(valid)
            observed_mask[i, slot_positions] = 1.0
            values = feature_matrix[valid_positions]
            for name, col_slice in group_slices.items():
                arrays[name][i, slot_positions, :] = values[:, col_slice]
            missing_frame_slots += int((~valid).sum())

    arrays["length_mask"] = length_mask
    arrays["observed_mask"] = observed_mask
    arrays["frame_index_sequence"] = frame_index_sequence
    valid_length_slots = int(length_mask.sum())
    observed_frame_slots = int(observed_mask.sum())
    padding_slots = int(length_mask.size - valid_length_slots)
    missing_observed_slots = int(valid_length_slots - observed_frame_slots)

    audit = {
        "rows": int(len(work_windows)),
        "max_window_length": int(max_window_length),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "feature_names": feature_names,
        "forbidden_selected": forbidden_selected,
        "missing_frame_slots": int(missing_frame_slots),
        "valid_length_slots": valid_length_slots,
        "observed_frame_slots": observed_frame_slots,
        "padding_slots": padding_slots,
        "missing_observed_slots_within_length": missing_observed_slots,
        "total_frame_slots": int(observed_mask.size),
        "observed_ratio": float(observed_frame_slots / max(1, observed_mask.size)),
        "observed_within_length_ratio": float(observed_frame_slots / max(1, valid_length_slots)),
        "truncated_windows": int(truncated_windows),
        "errors": [],
        "warnings": [],
    }
    if missing_frame_slots:
        audit["warnings"].append(f"missing_frame_slots={missing_frame_slots}")
    return SpatialSequenceExport(arrays=arrays, feature_names=feature_names, audit=audit)


def _available_feature_names(frames: pd.DataFrame) -> dict[str, list[str]]:
    available: dict[str, list[str]] = {}
    for group_name, cols in SPATIAL_FRAME_FEATURES.items():
        present = [c for c in cols if c in frames.columns]
        if present:
            available[group_name] = present
    return available


def _numeric_feature(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    if series.dtype == object:
        lower = series.astype(str).str.strip().str.lower()
        bool_like = lower.isin(["true", "false", "yes", "no", "1", "0", "nan", "<na>", "none", ""])
        if bool_like.mean() > 0.95:
            mapped = lower.map({"true": 1.0, "yes": 1.0, "1": 1.0, "false": 0.0, "no": 0.0, "0": 0.0})
            return mapped.fillna(0.0).astype(float)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _is_forbidden(column: str) -> bool:
    lower = column.lower()
    return any(token in lower for token in FORBIDDEN_SUBSTRINGS) or lower.startswith(("target_roi_", "roi_target_"))
