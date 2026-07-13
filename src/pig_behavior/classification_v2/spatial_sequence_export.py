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
    for column in [
        "window_start_frame",
        "window_end_frame",
        "window_length_frames",
    ]:
        work_windows[column] = pd.to_numeric(
            work_windows[column],
            errors="coerce",
        )
    _validate_window_alignment_contract(work_windows)
    if max_window_length is None:
        max_window_length = int(work_windows["window_length_frames"].max())
    if max_window_length <= 0:
        raise ValueError("max_window_length must be greater than zero")
    if max_window_length < int(work_windows["window_length_frames"].max()):
        raise ValueError(
            "max_window_length is smaller than a declared window length: "
            f"max_window_length={max_window_length}"
        )

    work_frames = frames[["object_track_key", "frame_index", *selected_cols]].copy()
    work_frames["frame_index"] = pd.to_numeric(work_frames["frame_index"], errors="coerce")
    _validate_frame_alignment_contract(work_frames)
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
    motion_rebased_windows = 0
    motion_valid_pair_count = 0
    motion_reset_row_count = 0
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
            valid = (positions < len(frame_indices)) & (
                frame_indices[bounded_positions] == wanted_frames
            )
            if not valid.any():
                missing_frame_slots += len(wanted_frames)
                continue
            valid_positions = positions[valid]
            slot_positions = np.flatnonzero(valid)
            observed_mask[i, slot_positions] = 1.0
            values = feature_matrix[valid_positions]
            values, motion_audit = _rebase_window_motion(
                values,
                flat_feature_names,
                wanted_frames[valid],
            )
            motion_rebased_windows += int(motion_audit["rebased"])
            motion_valid_pair_count += int(motion_audit["valid_pairs"])
            motion_reset_row_count += int(motion_audit["reset_rows"])
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
        "input_window_rows": int(len(windows)),
        "aligned_window_rows": int(len(work_windows)),
        "input_frame_rows": int(len(frames)),
        "aligned_frame_rows": int(len(work_frames)),
        "invalid_window_alignment_rows": 0,
        "invalid_frame_alignment_rows": 0,
        "duplicate_window_id_rows": 0,
        "duplicate_frame_alignment_rows": 0,
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
        "motion_rebased_windows": int(motion_rebased_windows),
        "motion_valid_pair_count": int(motion_valid_pair_count),
        "motion_reset_row_count": int(motion_reset_row_count),
        "errors": [],
        "warnings": [],
    }
    if missing_frame_slots:
        audit["warnings"].append(f"missing_frame_slots={missing_frame_slots}")
    return SpatialSequenceExport(arrays=arrays, feature_names=feature_names, audit=audit)


def _rebase_window_motion(
    values: np.ndarray,
    feature_names: list[str],
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Recompute pair-derived motion without reading frames outside a window."""
    motion_columns = SPATIAL_FRAME_FEATURES["motion_delta"]
    present_motion = [column for column in motion_columns if column in feature_names]
    if not present_motion or len(values) == 0:
        return values, {"rebased": False, "valid_pairs": 0, "reset_rows": 0}

    out = values.copy()
    indices = {column: feature_names.index(column) for column in feature_names}
    for column in present_motion:
        out[:, indices[column]] = 0.0

    required_position = ["cx_n", "cy_n", "bw_n", "bh_n"]
    if not all(column in indices for column in required_position):
        return out, {
            "rebased": True,
            "valid_pairs": 0,
            "reset_rows": int(len(out)),
        }

    row_valid = np.ones(len(out), dtype=bool)
    for column in [
        "bbox_valid",
        "actor_bbox_valid",
        "geometry_feature_valid",
        "spatiotemporal_feature_valid",
    ]:
        if column in indices:
            row_valid &= out[:, indices[column]] > 0.5

    frame_delta = np.diff(frame_indices.astype("float64"))
    valid_pair = (
        np.isfinite(frame_delta)
        & (frame_delta > 0)
        & row_valid[:-1]
        & row_valid[1:]
    )
    pair_rows = np.flatnonzero(valid_pair) + 1
    previous_rows = pair_rows - 1

    raw_to_delta = {
        "cx_n": "delta_cx_n",
        "cy_n": "delta_cy_n",
        "bw_n": "delta_bw_n",
        "bh_n": "delta_bh_n",
        "area_n": "delta_area_n",
        "aspect_ratio": "delta_aspect_ratio",
    }
    for raw_column, delta_column in raw_to_delta.items():
        if raw_column not in indices or delta_column not in indices:
            continue
        delta = (
            out[pair_rows, indices[raw_column]]
            - out[previous_rows, indices[raw_column]]
        )
        finite = np.isfinite(delta)
        out[pair_rows[finite], indices[delta_column]] = delta[finite]

    dx = out[pair_rows, indices["cx_n"]] - out[
        previous_rows,
        indices["cx_n"],
    ]
    dy = out[pair_rows, indices["cy_n"]] - out[
        previous_rows,
        indices["cy_n"],
    ]
    speed = np.hypot(dx, dy) / frame_delta[valid_pair]
    finite_speed = np.isfinite(speed)
    if "speed_n_per_frame" in indices:
        out[pair_rows[finite_speed], indices["speed_n_per_frame"]] = speed[
            finite_speed
        ]

    if "speed_n_per_sec" in indices:
        original_speed_sec = values[:, indices["speed_n_per_sec"]]
        finite_speed_sec = np.isfinite(original_speed_sec[pair_rows])
        out[
            pair_rows[finite_speed_sec],
            indices["speed_n_per_sec"],
        ] = original_speed_sec[pair_rows[finite_speed_sec]]

    _recompute_higher_order_motion(
        out,
        indices,
        pair_rows,
        frame_delta,
        valid_pair,
    )
    return out, {
        "rebased": True,
        "valid_pairs": int(valid_pair.sum()),
        "reset_rows": int(len(out) - valid_pair.sum()),
    }


def _recompute_higher_order_motion(
    values: np.ndarray,
    indices: dict[str, int],
    pair_rows: np.ndarray,
    frame_delta: np.ndarray,
    valid_pair: np.ndarray,
) -> None:
    """Recompute acceleration and turning after window-local speeds exist."""
    if len(pair_rows) < 2:
        return
    speed_column = indices.get("speed_n_per_frame")
    if speed_column is None:
        return

    pair_is_consecutive = np.diff(pair_rows) == 1
    higher_rows = pair_rows[1:][pair_is_consecutive]
    previous_pair_rows = higher_rows - 1
    if "abs_accel_n_per_frame2" in indices:
        accel = np.abs(
            values[higher_rows, speed_column]
            - values[previous_pair_rows, speed_column]
        ) / frame_delta[valid_pair][1:][pair_is_consecutive]
        values[higher_rows, indices["abs_accel_n_per_frame2"]] = accel

    if {
        "abs_direction_change_rad",
        "delta_cx_n",
        "delta_cy_n",
    }.issubset(indices):
        dx = values[pair_rows, indices["delta_cx_n"]]
        dy = values[pair_rows, indices["delta_cy_n"]]
        direction = np.arctan2(dy, dx)
        change = np.abs(
            (direction[1:] - direction[:-1] + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )
        values[
            higher_rows,
            indices["abs_direction_change_rad"],
        ] = change[pair_is_consecutive]


def _validate_window_alignment_contract(windows: pd.DataFrame) -> None:
    """Reject ambiguous or malformed window rows before tensor alignment."""
    if windows.empty:
        raise ValueError("Window alignment contract failed: no window rows")

    key_text = windows["object_track_key"].fillna("").astype(str).str.strip()
    id_text = windows["window_id"].fillna("").astype(str).str.strip()
    start = windows["window_start_frame"]
    end = windows["window_end_frame"]
    length = windows["window_length_frames"]
    integer_fields = (
        start.notna()
        & end.notna()
        & length.notna()
        & start.mod(1).eq(0)
        & end.mod(1).eq(0)
        & length.mod(1).eq(0)
    )
    span_valid = start.le(end) & length.eq(end - start + 1) & length.gt(0)
    invalid = key_text.eq("") | id_text.eq("") | ~integer_fields | ~span_valid
    duplicate_id = id_text.ne("") & id_text.duplicated(keep=False)
    if invalid.any() or duplicate_id.any():
        _raise_alignment_error(
            "Window",
            windows,
            invalid,
            duplicate_id,
            duplicate_name="duplicate_window_id_rows",
        )


def _validate_frame_alignment_contract(frames: pd.DataFrame) -> None:
    """Reject frame rows that would otherwise be dropped or truncated."""
    key_text = frames["object_track_key"].fillna("").astype(str).str.strip()
    frame_index = frames["frame_index"]
    integer_index = frame_index.notna() & frame_index.mod(1).eq(0)
    invalid = key_text.eq("") | ~integer_index
    duplicate = pd.DataFrame(
        {
            "object_track_key": key_text,
            "frame_index": frame_index,
        }
    ).duplicated(keep=False)
    duplicate &= ~invalid
    if invalid.any() or duplicate.any():
        _raise_alignment_error(
            "Frame",
            frames,
            invalid,
            duplicate,
            duplicate_name="duplicate_frame_alignment_rows",
        )


def _raise_alignment_error(
    kind: str,
    rows: pd.DataFrame,
    invalid: pd.Series,
    duplicate: pd.Series,
    *,
    duplicate_name: str,
) -> None:
    """Raise a compact alignment error with counts and source-row samples."""
    affected = invalid | duplicate
    sample_indices = [str(value) for value in rows.index[affected].tolist()[:10]]
    raise ValueError(
        f"{kind} alignment contract failed: invalid_rows={int(invalid.sum())}, "
        f"{duplicate_name}={int(duplicate.sum())}, "
        f"sample_source_indices={sample_indices}"
    )


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
            mapped = lower.map(
                {
                    "true": 1.0,
                    "yes": 1.0,
                    "1": 1.0,
                    "false": 0.0,
                    "no": 0.0,
                    "0": 0.0,
                }
            )
            return mapped.fillna(0.0).astype(float)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _is_forbidden(column: str) -> bool:
    lower = column.lower()
    return any(
        token in lower for token in FORBIDDEN_SUBSTRINGS
    ) or lower.startswith(("target_roi_", "roi_target_"))
