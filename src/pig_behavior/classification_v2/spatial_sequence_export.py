"""Leakage-safe per-frame spatial sequence export for classification_v2 windows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.window_alignment import (
    require_ordered_window_ids,
)
from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_REQUIRED_MASKS,
    MOTION_SCHEMA_DIMENSION,
    MOTION_SCHEMA_DTYPE,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_ID,
    MOTION_SCHEMA_VERSION,
    require_motion_schema,
)
from pig_behavior.classification_v2.features.pen_context import (
    PEN_CONTEXT_LEGACY_MODEL_FEATURE_COLUMNS,
    PEN_CONTEXT_MODEL_FEATURE_COLUMNS,
)
from pig_behavior.classification_v2.features.spatial_semantics import (
    SOCIAL_IDENTITY_VERSION,
    SOCIAL_TIE_BREAK_VERSION,
    is_target_roi_model_forbidden,
)

FORBIDDEN_SUBSTRINGS = (
    "behavior",
    "label",
    "review",
    "manual",
    "temporal_unit",
    "window_id",
    "identifier_schema",
    "scene_frame_uid",
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
    "motion_delta": list(MOTION_FEATURE_NAMES),
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
        "partner_distance_delta_n",
        "approach_speed_n_per_second",
        "retreat_speed_n_per_second",
        "pair_contact_with_nearest",
        "aggression_score_proxy_per_second",
    ],
    "pen_boundary_context": list(PEN_CONTEXT_MODEL_FEATURE_COLUMNS),
}

LEGACY_SPATIAL_FRAME_FEATURES: dict[str, list[str]] = {
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
    "roi_class_relation": list(SPATIAL_FRAME_FEATURES["roi_class_relation"]),
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
    "pen_boundary_context": list(PEN_CONTEXT_LEGACY_MODEL_FEATURE_COLUMNS),
    "quality_mask": [
        "bbox_valid",
        "actor_bbox_valid",
        "geometry_feature_valid",
        "spatiotemporal_feature_valid",
        "roi_feeder_available",
        "roi_drinker_available",
        "roi_toy_available",
        "social_neighbor_available",
    ],
}

SPATIAL_QUALITY_COLUMNS: tuple[str, ...] = (
    "bbox_valid",
    "actor_bbox_valid",
    "geometry_feature_valid",
    "spatiotemporal_feature_valid",
)

DERIVATION_COLUMNS: tuple[str, ...] = (
    "timestamp_sec",
    "image_width",
    "image_height",
    "pen_boundary_inward_normal_x",
    "pen_boundary_inward_normal_y",
    "pen_context_available",
    "pen_center_inside",
    *SPATIAL_QUALITY_COLUMNS,
    "roi_feeder_available",
    "roi_drinker_available",
    "roi_toy_available",
    "social_neighbor_available",
    *MOTION_REQUIRED_MASKS,
    "velocity_sample_time_sec",
    "acceleration_delta_t_sec",
)


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
    feature_schema: dict[str, list[str]] | None = None,
    motion_schema_manifest: dict[str, Any] | None = None,
) -> SpatialSequenceExport:
    """Build fixed-length spatial arrays aligned to sequence-window rows.

    The returned arrays are model inputs only. Label, ID, path, review, and
    policy columns are excluded; identifiers are retained only in the audit
    surface outside the arrays.
    """
    require_final_view_contract = feature_schema is None
    selected_schema = feature_schema or SPATIAL_FRAME_FEATURES
    legacy_schema = selected_schema is LEGACY_SPATIAL_FRAME_FEATURES
    required_windows = [
        "window_id",
        "object_track_key",
        "window_start_frame",
        "window_end_frame",
        "window_length_frames",
    ]
    if require_final_view_contract:
        required_windows.extend(
            [
                "feature_computation_grain",
                "pair_scope_key",
                "view_type",
                "sampling_pattern",
                "selected_frame_offsets",
                "selected_frame_indices",
                "selected_timestamps_seconds",
                "pair_delta_frames",
                "pair_delta_seconds",
                "pair_recomputed_for_view",
                "aggregate_recomputed_for_view",
            ]
        )
    required_frames = ["object_track_key", "frame_index"]
    missing_windows = [c for c in required_windows if c not in windows.columns]
    missing_frames = [c for c in required_frames if c not in frames.columns]
    if missing_windows or missing_frames:
        raise ValueError(f"Missing columns: windows={missing_windows} frames={missing_frames}")

    feature_frames = frames.copy()
    if "nearest_partner_key" in feature_frames.columns:
        nearest_partner_key = (
            feature_frames["nearest_partner_key"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    elif legacy_schema:
        nearest_pig = feature_frames.get(
            "nearest_pig_id",
            pd.Series("", index=feature_frames.index),
        ).fillna("").astype(str).str.strip()
        nearest_track = feature_frames.get(
            "nearest_track_id",
            pd.Series("", index=feature_frames.index),
        ).fillna("").astype(str).str.strip()
        nearest_partner_key = nearest_pig.where(
            nearest_pig.ne(""),
            nearest_track,
        )
    else:
        nearest_partner_key = pd.Series("", index=feature_frames.index)
    feature_frames["_social_partner_key"] = nearest_partner_key
    feature_frames["social_neighbor_available"] = feature_frames[
        "_social_partner_key"
    ].ne("")

    alignment_windows = windows.reset_index(drop=True).copy()
    for column in (
        "window_start_frame",
        "window_end_frame",
        "window_length_frames",
    ):
        alignment_windows[column] = pd.to_numeric(
            alignment_windows[column],
            errors="coerce",
        )
    _validate_window_alignment_contract(
        alignment_windows,
        require_final_view_contract=require_final_view_contract,
    )
    alignment_frames = feature_frames[
        ["object_track_key", "frame_index"]
    ].copy()
    alignment_frames["frame_index"] = pd.to_numeric(
        alignment_frames["frame_index"],
        errors="coerce",
    )
    _validate_frame_alignment_contract(alignment_frames)

    social_source_present = any(
        column in frames.columns
        for column in selected_schema.get("social_relation", [])
    ) or "nearest_neighbor_available" in frames.columns
    if (
        "social_relation" in selected_schema
        and not legacy_schema
        and social_source_present
        and "nearest_partner_key" not in frames.columns
    ):
        raise ValueError(
            "Missing canonical social identity column: "
            "nearest_partner_key"
        )
    motion_preflight: dict[str, Any] | None = None
    if "motion_delta" in selected_schema and not legacy_schema:
        producer_metadata = (
            motion_schema_manifest
            if motion_schema_manifest is not None
            else _motion_metadata_from_frames(feature_frames)
        )
        motion_preflight = require_motion_schema(
            source_columns=list(feature_frames.columns),
            actual_feature_names=selected_schema["motion_delta"],
            actual_masks=[
                name for name in MOTION_REQUIRED_MASKS
                if name in feature_frames.columns
            ],
            metadata=producer_metadata,
        )
    declared_forbidden = [
        column
        for columns in selected_schema.values()
        for column in columns
        if _is_forbidden(column)
    ]
    if declared_forbidden:
        raise ValueError(
            "Forbidden spatial feature columns requested: "
            f"{declared_forbidden}"
        )
    feature_names = _available_feature_names(
        feature_frames,
        selected_schema,
        fail_closed_groups={"motion_delta"} if not legacy_schema else set(),
    )
    selected_cols = [c for cols in feature_names.values() for c in cols]
    derivation_cols = [
        column
        for column in DERIVATION_COLUMNS
        if column in feature_frames.columns and column not in selected_cols
    ]
    forbidden_selected = [c for c in selected_cols if _is_forbidden(c)]
    if forbidden_selected:
        raise ValueError(f"Forbidden spatial feature columns selected: {forbidden_selected}")

    work_windows = windows.reset_index(drop=True).copy()
    window_alignment = require_ordered_window_ids(
        "spatial_windows",
        work_windows["window_id"],
    )
    for column in [
        "window_start_frame",
        "window_end_frame",
        "window_length_frames",
    ]:
        work_windows[column] = pd.to_numeric(
            work_windows[column],
            errors="coerce",
        )
    _validate_window_alignment_contract(
        work_windows,
        require_final_view_contract=require_final_view_contract,
    )
    if max_window_length is None:
        max_window_length = int(work_windows["window_length_frames"].max())
    if max_window_length <= 0:
        raise ValueError("max_window_length must be greater than zero")
    if max_window_length < int(work_windows["window_length_frames"].max()):
        raise ValueError(
            "max_window_length is smaller than a declared window length: "
            f"max_window_length={max_window_length}"
        )

    work_frames = feature_frames[
        [
            "object_track_key",
            "frame_index",
            *selected_cols,
            *derivation_cols,
            "_social_partner_key",
        ]
    ].copy()
    work_frames["frame_index"] = pd.to_numeric(work_frames["frame_index"], errors="coerce")
    _validate_frame_alignment_contract(work_frames)
    work_frames["frame_index"] = work_frames["frame_index"].astype(int)
    for col in [*selected_cols, *derivation_cols]:
        work_frames[col] = _numeric_feature(work_frames[col])
    flat_feature_names: list[str] = []
    group_slices: dict[str, slice] = {}
    start_col = 0
    for name, cols in feature_names.items():
        flat_feature_names.extend(cols)
        group_slices[name] = slice(start_col, start_col + len(cols))
        start_col += len(cols)

    computational_names = [*flat_feature_names, *derivation_cols]
    grouped: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, group in work_frames.groupby("object_track_key", sort=False):
        group = group.sort_values("frame_index")
        grouped[str(key)] = (
            group["frame_index"].to_numpy(dtype=np.int32, copy=True),
                group[computational_names].to_numpy(dtype=np.float64, copy=True),
            group["_social_partner_key"].to_numpy(dtype=str, copy=True),
        )

    arrays = {
        name: np.zeros((len(work_windows), max_window_length, len(cols)), dtype=np.float32)
        for name, cols in feature_names.items()
    }
    length_mask = np.zeros((len(work_windows), max_window_length), dtype=np.float32)
    observed_mask = np.zeros((len(work_windows), max_window_length), dtype=np.float32)
    spatial_quality_mask = np.zeros(
        (len(work_windows), max_window_length),
        dtype=np.float32,
    )
    roi_validity_mask = np.zeros(
        (len(work_windows), max_window_length, 3),
        dtype=np.float32,
    )
    social_validity_mask = np.zeros(
        (len(work_windows), max_window_length),
        dtype=np.float32,
    )
    pen_validity_mask = np.zeros(
        (len(work_windows), max_window_length),
        dtype=np.float32,
    )
    adjacent_motion_pair_mask = np.zeros(
        (len(work_windows), max_window_length),
        dtype=np.float32,
    )
    sparse_velocity_pair_mask = np.zeros(
        (len(work_windows), max_window_length),
        dtype=np.float32,
    )
    derivative_masks = {
        name: np.zeros(
            (len(work_windows), max_window_length),
            dtype=np.float32,
        )
        for name in MOTION_REQUIRED_MASKS
    }
    frame_index_sequence = np.full((len(work_windows), max_window_length), -1, dtype=np.int32)

    missing_frame_slots = 0
    truncated_windows = 0
    motion_rebased_windows = 0
    motion_valid_pair_count = 0
    motion_adjacent_pair_count = 0
    motion_sparse_pair_count = 0
    motion_reset_row_count = 0
    social_rebased_windows = 0
    social_valid_pair_count = 0
    social_reset_row_count = 0
    pen_rebased_windows = 0
    pen_valid_pair_count = 0
    pen_reset_row_count = 0
    for object_key, window_group in work_windows.groupby("object_track_key", sort=False):
        frame_data = grouped.get(str(object_key))
        for i, row in window_group.iterrows():
            start = row["window_start_frame"]
            end = row["window_end_frame"]
            if pd.isna(start) or pd.isna(end):
                missing_frame_slots += max_window_length
                continue
            wanted_frames = _selected_window_frame_indices(
                row,
                require_final_view_contract=require_final_view_contract,
            )
            if len(wanted_frames) > max_window_length:
                wanted_frames = wanted_frames[:max_window_length]
                truncated_windows += 1
            length_mask[i, : len(wanted_frames)] = 1.0
            frame_index_sequence[i, : len(wanted_frames)] = wanted_frames

            if frame_data is None:
                missing_frame_slots += len(wanted_frames)
                continue
            frame_indices, feature_matrix, partner_keys = frame_data
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
            selected_timestamps = _column_or_nan(
                values,
                computational_names,
                "timestamp_sec",
            )
            if require_final_view_contract:
                declared_timestamps = _json_number_list(
                    row["selected_timestamps_seconds"],
                    field="selected_timestamps_seconds",
                    window_id=str(row["window_id"]),
                    expected_count=len(wanted_frames),
                    allow_null=True,
                )
                declared_observed = [
                    declared_timestamps[position]
                    for position in slot_positions
                ]
                timestamp_mismatch = [
                    position
                    for position, (declared, actual) in enumerate(
                        zip(
                            declared_observed,
                            selected_timestamps,
                            strict=True,
                        )
                    )
                    if declared is None
                    or not np.isfinite(actual)
                    or not np.isclose(
                        float(declared),
                        float(actual),
                        rtol=0.0,
                        atol=1e-9,
                    )
                ]
                if timestamp_mismatch:
                    raise ValueError(
                        "Final-view source timestamp mismatch for window_id="
                        f"{row['window_id']}: observed_positions="
                        f"{timestamp_mismatch[:10]}"
                    )
            values, motion_audit = _rebase_window_motion(
                values,
                computational_names,
                wanted_frames[valid],
                timestamps=selected_timestamps,
            )
            values, social_audit = _rebase_window_social_motion(
                values,
                computational_names,
                wanted_frames[valid],
                partner_keys[valid_positions],
                timestamps=_column_or_nan(
                    values,
                    computational_names,
                    "timestamp_sec",
                ),
            )
            values, pen_audit = _rebase_window_pen_motion(
                values,
                computational_names,
                wanted_frames[valid],
                timestamps=_column_or_nan(
                    values,
                    computational_names,
                    "timestamp_sec",
                ),
            )
            motion_rebased_windows += int(motion_audit["rebased"])
            motion_valid_pair_count += int(motion_audit["valid_pairs"])
            motion_adjacent_pair_count += int(
                motion_audit.get("adjacent_pairs", 0)
            )
            motion_sparse_pair_count += int(
                motion_audit.get("sparse_pairs", 0)
            )
            motion_reset_row_count += int(motion_audit["reset_rows"])
            social_rebased_windows += int(social_audit["rebased"])
            social_valid_pair_count += int(social_audit["valid_pairs"])
            social_reset_row_count += int(social_audit["reset_rows"])
            pen_rebased_windows += int(pen_audit["rebased"])
            pen_valid_pair_count += int(pen_audit["valid_pairs"])
            pen_reset_row_count += int(pen_audit["reset_rows"])
            masks = _view_quality_masks(values, computational_names)
            spatial_quality_mask[i, slot_positions] = masks["spatial"]
            roi_validity_mask[i, slot_positions, :] = masks["roi"]
            social_validity_mask[i, slot_positions] = masks["social"]
            pen_validity_mask[i, slot_positions] = masks["pen"]
            adjacent_pairs, sparse_pairs = _view_motion_pair_masks(
                wanted_frames[valid],
                _column_or_nan(
                    values,
                    computational_names,
                    "timestamp_sec",
                ),
                masks["spatial"] > 0.5,
            )
            adjacent_motion_pair_mask[i, slot_positions] = adjacent_pairs
            sparse_velocity_pair_mask[i, slot_positions] = sparse_pairs
            for mask_name, mask_array in derivative_masks.items():
                mask_array[i, slot_positions] = _column_or_zero(
                    values,
                    {
                        column: index
                        for index, column in enumerate(computational_names)
                    },
                    mask_name,
                )
            masked_values = _zero_invalid_feature_groups(
                values,
                group_slices,
                feature_names,
                masks,
            )
            for name, col_slice in group_slices.items():
                arrays[name][i, slot_positions, :] = masked_values[:, col_slice]
            missing_frame_slots += int((~valid).sum())

    arrays["length_mask"] = length_mask
    arrays["observed_mask"] = observed_mask
    arrays["spatial_quality_mask"] = spatial_quality_mask
    arrays["roi_validity_mask"] = roi_validity_mask
    arrays["social_validity_mask"] = social_validity_mask
    arrays["pen_validity_mask"] = pen_validity_mask
    arrays["adjacent_motion_pair_mask"] = adjacent_motion_pair_mask
    arrays["sparse_velocity_pair_mask"] = sparse_velocity_pair_mask
    for mask_name, mask_array in derivative_masks.items():
        arrays[f"{mask_name}_mask"] = mask_array
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
        "motion_schema_preflight": motion_preflight,
        "motion_schema_id": MOTION_SCHEMA_ID if motion_preflight else None,
        "motion_schema_version": (
            MOTION_SCHEMA_VERSION if motion_preflight else None
        ),
        "motion_schema_dimension": (
            MOTION_SCHEMA_DIMENSION if motion_preflight else None
        ),
        "motion_schema_feature_names": (
            list(MOTION_FEATURE_NAMES) if motion_preflight else None
        ),
        "motion_schema_hash": MOTION_SCHEMA_HASH if motion_preflight else None,
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
        "motion_adjacent_pair_count": int(motion_adjacent_pair_count),
        "motion_sparse_pair_count": int(motion_sparse_pair_count),
        "motion_reset_row_count": int(motion_reset_row_count),
        "social_rebased_windows": int(social_rebased_windows),
        "social_valid_pair_count": int(social_valid_pair_count),
        "social_reset_row_count": int(social_reset_row_count),
        "pen_rebased_windows": int(pen_rebased_windows),
        "pen_valid_pair_count": int(pen_valid_pair_count),
        "pen_reset_row_count": int(pen_reset_row_count),
        "social_partner_available_frame_rows": int(
            work_frames["_social_partner_key"].ne("").sum()
        ),
        "social_identity_version": SOCIAL_IDENTITY_VERSION,
        "social_tie_break_version": SOCIAL_TIE_BREAK_VERSION,
        "window_alignment": window_alignment,
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
    *,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Recompute pair-derived motion without reading frames outside a window."""
    motion_columns = list(
        dict.fromkeys(
            [
                *SPATIAL_FRAME_FEATURES["motion_delta"],
                *LEGACY_SPATIAL_FRAME_FEATURES["motion_delta"],
            ]
        )
    )
    present_motion = [column for column in motion_columns if column in feature_names]
    if not present_motion or len(values) == 0:
        return values, {"rebased": False, "valid_pairs": 0, "reset_rows": 0}

    out = values.copy()
    indices = {column: feature_names.index(column) for column in feature_names}
    for column in present_motion:
        out[:, indices[column]] = 0.0
    for column in (
        *MOTION_REQUIRED_MASKS,
        "velocity_sample_time_sec",
        "acceleration_delta_t_sec",
    ):
        if column in indices:
            out[:, indices[column]] = 0.0

    v2_motion_present = all(
        column in indices for column in MOTION_FEATURE_NAMES
    )
    required_position = ["cx_n", "cy_n", "bw_n", "bh_n"]
    if v2_motion_present:
        required_position.extend(["area_n", "aspect_ratio"])
    if not all(column in indices for column in required_position):
        if v2_motion_present:
            missing_base = [
                column for column in required_position
                if column not in indices
            ]
            raise ValueError(
                "Cannot recompute v2 motion without base geometry: "
                f"{missing_base}"
            )
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
    time_delta = np.diff(timestamps.astype("float64"))
    velocity_pair_valid = (
        np.isfinite(frame_delta)
        & (frame_delta > 0)
        & np.isfinite(time_delta)
        & (time_delta > 0)
        & row_valid[:-1]
        & row_valid[1:]
    )
    adjacent_pair_valid = velocity_pair_valid & np.isclose(frame_delta, 1.0)
    pair_rows = np.flatnonzero(velocity_pair_valid) + 1
    previous_rows = pair_rows - 1

    raw_to_rate = {
        "cx_n": "vx_n_per_second",
        "cy_n": "vy_n_per_second",
        "bw_n": "bw_rate_n_per_second",
        "bh_n": "bh_rate_n_per_second",
        "area_n": "area_rate_n_per_second",
        "aspect_ratio": "aspect_ratio_rate_per_second",
    }
    for raw_column, rate_column in raw_to_rate.items():
        if raw_column not in indices or rate_column not in indices:
            continue
        rate = (
            out[pair_rows, indices[raw_column]]
            - out[previous_rows, indices[raw_column]]
        ) / time_delta[velocity_pair_valid]
        finite = np.isfinite(rate)
        out[pair_rows[finite], indices[rate_column]] = rate[finite]

    dx = out[pair_rows, indices["cx_n"]] - out[
        previous_rows,
        indices["cx_n"],
    ]
    dy = out[pair_rows, indices["cy_n"]] - out[
        previous_rows,
        indices["cy_n"],
    ]
    speed = np.hypot(dx, dy) / time_delta[velocity_pair_valid]
    finite_speed = np.isfinite(speed)
    if "speed_n_per_second" in indices:
        out[pair_rows[finite_speed], indices["speed_n_per_second"]] = speed[
            finite_speed
        ]
    for mask_name in ("valid_motion_pair", "velocity_valid"):
        if mask_name in indices:
            out[pair_rows, indices[mask_name]] = 1.0
    if "motion_feature_available" in indices and pair_rows.size:
        out[:, indices["motion_feature_available"]] = 1.0
    if "velocity_sample_time_sec" in indices:
        midpoint = (
            timestamps[pair_rows].astype("float64")
            + timestamps[previous_rows].astype("float64")
        ) / 2.0
        out[pair_rows, indices["velocity_sample_time_sec"]] = midpoint

    bbox_rate_valid = velocity_pair_valid.copy()
    for rate_column in (
        "bw_rate_n_per_second",
        "bh_rate_n_per_second",
        "area_rate_n_per_second",
        "aspect_ratio_rate_per_second",
    ):
        if rate_column in indices:
            bbox_rate_valid[pair_rows - 1] &= np.isfinite(
                out[pair_rows, indices[rate_column]]
            )
    if "bbox_rate_valid" in indices:
        bbox_rows = np.flatnonzero(bbox_rate_valid) + 1
        out[bbox_rows, indices["bbox_rate_valid"]] = 1.0

    adjacent_rows = np.flatnonzero(adjacent_pair_valid) + 1
    adjacent_previous = adjacent_rows - 1
    legacy_raw_to_delta = {
        "cx_n": "delta_cx_n",
        "cy_n": "delta_cy_n",
        "bw_n": "delta_bw_n",
        "bh_n": "delta_bh_n",
        "area_n": "delta_area_n",
        "aspect_ratio": "delta_aspect_ratio",
    }
    for raw_column, delta_column in legacy_raw_to_delta.items():
        if raw_column not in indices or delta_column not in indices:
            continue
        delta = (
            out[adjacent_rows, indices[raw_column]]
            - out[adjacent_previous, indices[raw_column]]
        )
        finite = np.isfinite(delta)
        out[adjacent_rows[finite], indices[delta_column]] = delta[finite]
    if adjacent_rows.size:
        adjacent_dx = (
            out[adjacent_rows, indices["cx_n"]]
            - out[adjacent_previous, indices["cx_n"]]
        )
        adjacent_dy = (
            out[adjacent_rows, indices["cy_n"]]
            - out[adjacent_previous, indices["cy_n"]]
        )
        adjacent_distance = np.hypot(adjacent_dx, adjacent_dy)
        adjacent_frame_delta = frame_delta[adjacent_pair_valid]
        adjacent_time_delta = time_delta[adjacent_pair_valid]
        if "speed_n_per_frame" in indices:
            out[adjacent_rows, indices["speed_n_per_frame"]] = (
                adjacent_distance / adjacent_frame_delta
            )
        if "speed_n_per_sec" in indices:
            out[adjacent_rows, indices["speed_n_per_sec"]] = (
                adjacent_distance / adjacent_time_delta
            )

    _recompute_higher_order_motion(
        out,
        indices,
        timestamps,
        velocity_pair_valid,
        adjacent_pair_valid,
    )
    return out, {
        "rebased": True,
        "valid_pairs": int(velocity_pair_valid.sum()),
        "adjacent_pairs": int(adjacent_pair_valid.sum()),
        "sparse_pairs": int(
            (velocity_pair_valid & ~adjacent_pair_valid).sum()
        ),
        "reset_rows": int(len(out) - velocity_pair_valid.sum()),
    }


def _rebase_window_pen_motion(
    values: np.ndarray,
    feature_names: list[str],
    frame_indices: np.ndarray,
    *,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Recompute boundary approach/parallel motion inside each window."""

    derived = [
        "pen_distance_delta_n_per_frame",
        "pen_approach_speed_n_per_frame",
        "pen_retreat_speed_n_per_frame",
        "pen_parallel_speed_n_per_frame",
        "pen_distance_delta_n_per_second",
        "pen_normal_speed_n_per_second",
        "pen_approach_speed_n_per_second",
        "pen_retreat_speed_n_per_second",
        "pen_parallel_speed_n_per_second",
    ]
    present = [column for column in derived if column in feature_names]
    if not present or len(values) == 0:
        return values, {"rebased": False, "valid_pairs": 0, "reset_rows": 0}

    out = values.copy()
    indices = {column: feature_names.index(column) for column in feature_names}
    for column in present:
        out[:, indices[column]] = 0.0
    required = {
        "cx_n",
        "cy_n",
        "pen_center_signed_distance_n",
        "image_width",
        "image_height",
        "pen_boundary_inward_normal_x",
        "pen_boundary_inward_normal_y",
    }
    if not required.issubset(indices):
        return out, {
            "rebased": True,
            "valid_pairs": 0,
            "reset_rows": int(len(out)),
        }

    frame_delta = np.diff(frame_indices.astype("float64"))
    time_delta = np.diff(timestamps.astype("float64"))
    distance = out[:, indices["pen_center_signed_distance_n"]]
    distance_delta = np.diff(distance)
    # Signed distance is positive inside the calibrated pen. Deriving this
    # validity condition avoids exposing the highly camera-correlated binary
    # ``pen_center_inside`` flag as a model feature.
    inside = distance > 0.0
    row_valid = inside & np.isfinite(distance)
    valid_pair = (
        np.isfinite(frame_delta)
        & (frame_delta > 0)
        & np.isfinite(time_delta)
        & (time_delta > 0)
        & np.isfinite(distance_delta)
        & row_valid[:-1]
        & row_valid[1:]
    )
    pair_rows = np.flatnonzero(valid_pair) + 1
    if pair_rows.size:
        previous_rows = pair_rows - 1
        frame_denominator = frame_delta[valid_pair]
        time_denominator = time_delta[valid_pair]
        signed_delta_per_frame = (
            distance_delta[valid_pair] / frame_denominator
        )
        signed_delta_per_second = distance_delta[valid_pair] / time_denominator
        width = out[pair_rows, indices["image_width"]]
        height = out[pair_rows, indices["image_height"]]
        previous_width = out[previous_rows, indices["image_width"]]
        previous_height = out[previous_rows, indices["image_height"]]
        image_diag = np.hypot(width, height)
        metric_valid = (
            np.isfinite(image_diag)
            & (image_diag > 0)
            & np.isclose(width, previous_width)
            & np.isclose(height, previous_height)
        )
        dx_metric = (
            out[pair_rows, indices["cx_n"]] * width
            - out[previous_rows, indices["cx_n"]] * previous_width
        ) / image_diag
        dy_metric = (
            out[pair_rows, indices["cy_n"]] * height
            - out[previous_rows, indices["cy_n"]] * previous_height
        ) / image_diag
        normal_x = out[pair_rows, indices["pen_boundary_inward_normal_x"]]
        normal_y = out[pair_rows, indices["pen_boundary_inward_normal_y"]]
        normal_delta = dx_metric * normal_x + dy_metric * normal_y
        tangent_delta = -dx_metric * normal_y + dy_metric * normal_x
        metric_valid &= (
            np.isfinite(normal_delta) & np.isfinite(tangent_delta)
        )
        normal_per_frame = normal_delta / frame_denominator
        normal_per_second = normal_delta / time_denominator
        parallel_per_frame = np.abs(tangent_delta) / frame_denominator
        parallel_per_second = np.abs(tangent_delta) / time_denominator
        adjacent = np.isclose(frame_denominator, 1.0) & metric_valid
        if "pen_distance_delta_n_per_frame" in indices:
            out[pair_rows[adjacent], indices["pen_distance_delta_n_per_frame"]] = (
                signed_delta_per_frame[adjacent]
            )
        if "pen_approach_speed_n_per_frame" in indices:
            out[pair_rows[adjacent], indices["pen_approach_speed_n_per_frame"]] = np.clip(
                -normal_per_frame[adjacent],
                0.0,
                None,
            )
        if "pen_retreat_speed_n_per_frame" in indices:
            out[pair_rows[adjacent], indices["pen_retreat_speed_n_per_frame"]] = np.clip(
                normal_per_frame[adjacent],
                0.0,
                None,
            )
        if "pen_parallel_speed_n_per_frame" in indices:
            out[pair_rows[adjacent], indices["pen_parallel_speed_n_per_frame"]] = (
                parallel_per_frame[adjacent]
            )
        physical_rows = pair_rows[metric_valid]
        physical_values = {
            "pen_distance_delta_n_per_second": signed_delta_per_second,
            "pen_normal_speed_n_per_second": normal_per_second,
            "pen_approach_speed_n_per_second": np.clip(
                -normal_per_second,
                0.0,
                None,
            ),
            "pen_retreat_speed_n_per_second": np.clip(
                normal_per_second,
                0.0,
                None,
            ),
            "pen_parallel_speed_n_per_second": parallel_per_second,
        }
        for column, column_values in physical_values.items():
            if column in indices:
                out[physical_rows, indices[column]] = column_values[
                    metric_valid
                ]

    return out, {
        "rebased": True,
        "valid_pairs": int(valid_pair.sum()),
        "reset_rows": int(len(out) - valid_pair.sum()),
    }


def _rebase_window_social_motion(
    values: np.ndarray,
    feature_names: list[str],
    frame_indices: np.ndarray,
    partner_keys: np.ndarray,
    *,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Recompute pair-derived social signals within one requested window."""

    derived = [
        "nearest_dist_delta",
        "approach_speed_n_per_frame",
        "separation_speed_n_per_frame",
        "aggression_score_proxy",
        "partner_distance_delta_n",
        "approach_speed_n_per_second",
        "retreat_speed_n_per_second",
        "aggression_score_proxy_per_second",
    ]
    present = [column for column in derived if column in feature_names]
    if not present or len(values) == 0:
        return values, {"rebased": False, "valid_pairs": 0, "reset_rows": 0}

    out = values.copy()
    indices = {column: feature_names.index(column) for column in feature_names}
    for column in present:
        out[:, indices[column]] = 0.0

    required = {"nearest_dist_n", "cx_n", "cy_n"}
    if not required.issubset(indices):
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
    time_delta = np.diff(timestamps.astype("float64"))
    partner_text = np.asarray(partner_keys, dtype=str)
    same_partner = (
        (partner_text[:-1] != "")
        & (partner_text[1:] != "")
        & (partner_text[:-1] == partner_text[1:])
    )
    distance = out[:, indices["nearest_dist_n"]]
    distance_delta = np.diff(distance)
    valid_pair = (
        np.isfinite(frame_delta)
        & (frame_delta > 0)
        & np.isfinite(time_delta)
        & (time_delta > 0)
        & same_partner
        & np.isfinite(distance_delta)
        & row_valid[:-1]
        & row_valid[1:]
    )
    pair_rows = np.flatnonzero(valid_pair) + 1
    if pair_rows.size:
        delta = distance_delta[valid_pair]
        frame_denom = frame_delta[valid_pair]
        time_denom = time_delta[valid_pair]
        adjacent = np.isclose(frame_denom, 1.0)
        if "nearest_dist_delta" in indices:
            out[pair_rows[adjacent], indices["nearest_dist_delta"]] = delta[
                adjacent
            ]
        if "approach_speed_n_per_frame" in indices:
            out[pair_rows[adjacent], indices["approach_speed_n_per_frame"]] = np.clip(
                -delta[adjacent] / frame_denom[adjacent],
                0.0,
                None,
            )
        if "separation_speed_n_per_frame" in indices:
            out[pair_rows[adjacent], indices["separation_speed_n_per_frame"]] = np.clip(
                delta[adjacent] / frame_denom[adjacent],
                0.0,
                None,
            )
        if "partner_distance_delta_n" in indices:
            out[pair_rows, indices["partner_distance_delta_n"]] = delta
        if "approach_speed_n_per_second" in indices:
            out[pair_rows, indices["approach_speed_n_per_second"]] = np.clip(
                -delta / time_denom,
                0.0,
                None,
            )
        if "retreat_speed_n_per_second" in indices:
            out[pair_rows, indices["retreat_speed_n_per_second"]] = np.clip(
                delta / time_denom,
                0.0,
                None,
            )

    if "aggression_score_proxy" in indices:
        contact = _column_or_zero(out, indices, "pair_contact_with_nearest")
        speed = _column_or_zero(out, indices, "speed_n_per_frame")
        approach = _column_or_zero(
            out,
            indices,
            "approach_speed_n_per_frame",
        )
        density = np.clip(
            _column_or_zero(out, indices, "social_density_near_count"),
            0.0,
            None,
        )
        partner_available = (partner_text != "").astype(float)
        aggression = (
            (contact > 0.5).astype(float)
            * partner_available
            * (np.clip(speed, 0.0, None) + approach)
            * (1.0 + density)
        )
        aggression[~row_valid] = 0.0
        out[:, indices["aggression_score_proxy"]] = aggression
    if "aggression_score_proxy_per_second" in indices:
        contact = _column_or_zero(out, indices, "pair_contact_with_nearest")
        speed = _column_or_zero(out, indices, "speed_n_per_second")
        approach = _column_or_zero(
            out,
            indices,
            "approach_speed_n_per_second",
        )
        density = np.clip(
            _column_or_zero(out, indices, "social_density_near_count"),
            0.0,
            None,
        )
        partner_available = (partner_text != "").astype(float)
        aggression = (
            (contact > 0.5).astype(float)
            * partner_available
            * (np.clip(speed, 0.0, None) + approach)
            * (1.0 + density)
        )
        aggression[~row_valid] = 0.0
        out[:, indices["aggression_score_proxy_per_second"]] = aggression

    return out, {
        "rebased": True,
        "valid_pairs": int(valid_pair.sum()),
        "reset_rows": int(len(out) - valid_pair.sum()),
    }


def _column_or_nan(
    values: np.ndarray,
    feature_names: list[str],
    column: str,
) -> np.ndarray:
    if column not in feature_names:
        return np.full(len(values), np.nan, dtype="float64")
    return values[:, feature_names.index(column)].astype("float64", copy=True)


def _view_quality_masks(
    values: np.ndarray,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    indices = {column: index for index, column in enumerate(feature_names)}
    spatial = np.ones(len(values), dtype=bool)
    for column in SPATIAL_QUALITY_COLUMNS:
        if column in indices:
            spatial &= values[:, indices[column]] > 0.5
    roi = np.zeros((len(values), 3), dtype="float32")
    for roi_index, roi_class in enumerate(("feeder", "drinker", "toy")):
        column = f"roi_{roi_class}_available"
        if column in indices:
            roi[:, roi_index] = (
                spatial & (values[:, indices[column]] > 0.5)
            ).astype("float32")
    social = spatial.copy()
    if "social_neighbor_available" in indices:
        social &= values[:, indices["social_neighbor_available"]] > 0.5
    else:
        social[:] = False
    pen = spatial.copy()
    if "pen_context_available" in indices:
        pen &= values[:, indices["pen_context_available"]] > 0.5
    else:
        pen[:] = False
    return {
        "spatial": spatial.astype("float32"),
        "roi": roi,
        "social": social.astype("float32"),
        "pen": pen.astype("float32"),
    }


def _zero_invalid_feature_groups(
    values: np.ndarray,
    group_slices: dict[str, slice],
    feature_names: dict[str, list[str]],
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    out = np.where(np.isfinite(values), values, 0.0).copy()
    spatial = masks["spatial"][:, None]
    for group_name in ["bbox_xywh_n", "bbox_shape_n", "motion_delta"]:
        if group_name in group_slices:
            out[:, group_slices[group_name]] *= spatial
    if "social_relation" in group_slices:
        out[:, group_slices["social_relation"]] *= masks["social"][:, None]
    if "pen_boundary_context" in group_slices:
        out[:, group_slices["pen_boundary_context"]] *= masks["pen"][:, None]
    if "roi_class_relation" in group_slices:
        roi_slice = group_slices["roi_class_relation"]
        roi_names = feature_names["roi_class_relation"]
        for feature_offset, column in enumerate(roi_names):
            roi_index = next(
                index
                for index, roi_class in enumerate(("feeder", "drinker", "toy"))
                if column.startswith(f"roi_{roi_class}_")
            )
            column_index = roi_slice.start + feature_offset
            out[:, column_index] *= masks["roi"][:, roi_index]
    return out


def _view_motion_pair_masks(
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    spatial_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    adjacent = np.zeros(len(frame_indices), dtype="float32")
    sparse = np.zeros(len(frame_indices), dtype="float32")
    if len(frame_indices) < 2:
        return adjacent, sparse
    frame_delta = np.diff(frame_indices.astype("float64"))
    time_delta = np.diff(timestamps.astype("float64"))
    valid = (
        (frame_delta > 0)
        & np.isfinite(frame_delta)
        & (time_delta > 0)
        & np.isfinite(time_delta)
        & spatial_valid[:-1]
        & spatial_valid[1:]
    )
    adjacent[1:] = (valid & np.isclose(frame_delta, 1.0)).astype("float32")
    sparse[1:] = (valid & (frame_delta > 1.0)).astype("float32")
    return adjacent, sparse


def _column_or_zero(
    values: np.ndarray,
    indices: dict[str, int],
    column: str,
) -> np.ndarray:
    """Read one finite feature vector or return a zero vector."""

    if column not in indices:
        return np.zeros(len(values), dtype="float32")
    observed = values[:, indices[column]]
    return np.where(np.isfinite(observed), observed, 0.0)


def _recompute_higher_order_motion(
    values: np.ndarray,
    indices: dict[str, int],
    timestamps: np.ndarray,
    velocity_pair_valid: np.ndarray,
    adjacent_pair_valid: np.ndarray,
) -> None:
    """Recompute derivative-order-specific v2 motion inside one window."""
    if len(values) < 3:
        if len(values) >= 2 and {
            "velocity_valid",
            "speed_n_per_second",
        }.issubset(indices):
            speed = values[:, indices["speed_n_per_second"]]
            valid = np.zeros(len(values), dtype=bool)
            valid[1:] = velocity_pair_valid & np.isfinite(
                speed[1:]
            ) & (speed[1:] > 0)
            if "direction_valid" in indices:
                values[valid, indices["direction_valid"]] = 1.0
        return

    speed_column = indices.get("speed_n_per_second")
    time_delta = np.diff(timestamps.astype("float64"))
    acceleration_time = (time_delta[:-1] + time_delta[1:]) / 2.0
    acceleration_valid = (
        velocity_pair_valid[:-1]
        & velocity_pair_valid[1:]
        & np.isfinite(acceleration_time)
        & (acceleration_time > 0)
    )
    acceleration_rows = np.flatnonzero(acceleration_valid) + 2
    if "acceleration_delta_t_sec" in indices:
        values[
            acceleration_rows,
            indices["acceleration_delta_t_sec"],
        ] = acceleration_time[acceleration_valid]
    for mask_name in (
        "tangential_acceleration_valid",
        "vector_acceleration_valid",
    ):
        if mask_name in indices:
            values[acceleration_rows, indices[mask_name]] = 1.0

    tangential = np.zeros(len(values) - 2, dtype="float64")
    if speed_column is not None:
        tangential[acceleration_valid] = (
            np.diff(values[1:, speed_column])[acceleration_valid]
            / acceleration_time[acceleration_valid]
        )
    if "tangential_acceleration_n_per_second2" in indices:
        values[
            2:,
            indices["tangential_acceleration_n_per_second2"],
        ] = tangential
    if "acceleration_n_per_second2" in indices:
        values[2:, indices["acceleration_n_per_second2"]] = tangential
    if "abs_acceleration_n_per_second2" in indices:
        values[
            2:,
            indices["abs_acceleration_n_per_second2"],
        ] = np.abs(tangential)

    ax = np.zeros(len(values) - 2, dtype="float64")
    ay = np.zeros(len(values) - 2, dtype="float64")
    if {"vx_n_per_second", "vy_n_per_second"}.issubset(indices):
        ax[acceleration_valid] = (
            np.diff(values[1:, indices["vx_n_per_second"]])[
                acceleration_valid
            ]
            / acceleration_time[acceleration_valid]
        )
        ay[acceleration_valid] = (
            np.diff(values[1:, indices["vy_n_per_second"]])[
                acceleration_valid
            ]
            / acceleration_time[acceleration_valid]
        )
    if "ax_n_per_second2" in indices:
        values[2:, indices["ax_n_per_second2"]] = ax
    if "ay_n_per_second2" in indices:
        values[2:, indices["ay_n_per_second2"]] = ay
    if "acceleration_vector_magnitude_n_per_second2" in indices:
        values[
            2:,
            indices["acceleration_vector_magnitude_n_per_second2"],
        ] = np.hypot(ax, ay)

    legacy_speed_column = indices.get("speed_n_per_frame")
    legacy_acceleration_valid = (
        adjacent_pair_valid[:-1] & adjacent_pair_valid[1:]
    )
    if (
        legacy_speed_column is not None
        and "abs_accel_n_per_frame2" in indices
    ):
        legacy_acceleration = np.zeros(len(values) - 2, dtype="float64")
        legacy_acceleration[legacy_acceleration_valid] = np.abs(
            np.diff(values[1:, legacy_speed_column])[
                legacy_acceleration_valid
            ]
        )
        values[2:, indices["abs_accel_n_per_frame2"]] = legacy_acceleration

    if {
        "vx_n_per_second",
        "vy_n_per_second",
        "speed_n_per_second",
    }.issubset(indices):
        interval_direction = np.arctan2(
            values[1:, indices["vy_n_per_second"]],
            values[1:, indices["vx_n_per_second"]],
        )
        direction_valid = (
            velocity_pair_valid
            & np.isfinite(values[1:, speed_column])
            & (values[1:, speed_column] > 0)
        )
        if "direction_valid" in indices:
            direction_rows = np.flatnonzero(direction_valid) + 1
            values[direction_rows, indices["direction_valid"]] = 1.0
        heading_valid = direction_valid[:-1] & direction_valid[1:]
        raw_change = (
            np.diff(interval_direction) + np.pi
        ) % (2.0 * np.pi) - np.pi
        change = np.zeros(len(values) - 2, dtype="float64")
        change[heading_valid] = raw_change[heading_valid]
        if "direction_change_rad" in indices:
            values[2:, indices["direction_change_rad"]] = change
        if "abs_direction_change_rad" in indices:
            values[2:, indices["abs_direction_change_rad"]] = np.abs(change)
        if "direction_change_valid" in indices:
            direction_change_rows = np.flatnonzero(heading_valid) + 2
            values[
                direction_change_rows,
                indices["direction_change_valid"],
            ] = 1.0


def _selected_window_frame_indices(
    row: pd.Series,
    *,
    require_final_view_contract: bool,
) -> np.ndarray:
    if not require_final_view_contract:
        return np.arange(
            int(row["window_start_frame"]),
            int(row["window_end_frame"]) + 1,
            dtype=np.int32,
        )
    window_id = str(row["window_id"])
    indices = _json_int_list(
        row["selected_frame_indices"],
        field="selected_frame_indices",
        window_id=window_id,
    )
    offsets = _json_int_list(
        row["selected_frame_offsets"],
        field="selected_frame_offsets",
        window_id=window_id,
    )
    pair_deltas = _json_int_list(
        row["pair_delta_frames"],
        field="pair_delta_frames",
        window_id=window_id,
    )
    expected_count = int(row["window_length_frames"])
    start = int(row["window_start_frame"])
    end = int(row["window_end_frame"])
    if len(indices) != expected_count or len(offsets) != expected_count:
        raise ValueError(
            "Final-view selected-slot count mismatch for window_id="
            f"{window_id}: indices={len(indices)}, offsets={len(offsets)}, "
            f"expected={expected_count}"
        )
    if not indices or indices[0] != start or indices[-1] != end:
        raise ValueError(
            "Final-view selected frames do not bind declared span for "
            f"window_id={window_id}"
        )
    expected_offsets = [value - start for value in indices]
    if offsets != expected_offsets:
        raise ValueError(
            "Final-view selected offsets mismatch selected frames for "
            f"window_id={window_id}"
        )
    expected_pair_deltas = [
        current - previous
        for previous, current in zip(indices, indices[1:], strict=False)
    ]
    if pair_deltas != expected_pair_deltas or any(
        value <= 0 for value in expected_pair_deltas
    ):
        raise ValueError(
            "Final-view pair delta contract failed for "
            f"window_id={window_id}"
        )
    view_type = str(row["view_type"]).strip()
    sampling_pattern = str(row["sampling_pattern"]).strip()
    if view_type == "S6@16":
        valid_identity = (
            expected_count == 6
            and offsets == [0, 3, 6, 9, 12, 15]
            and sampling_pattern
            == "uniform_sparse_offsets_0_3_6_9_12_15"
        )
    else:
        valid_identity = (
            view_type == f"T{expected_count}_contiguous"
            and sampling_pattern == "contiguous"
            and expected_pair_deltas == [1] * max(0, expected_count - 1)
        )
    if not valid_identity:
        raise ValueError(
            "Final-view identity/sampling contract failed for "
            f"window_id={window_id}: view_type={view_type}, "
            f"sampling_pattern={sampling_pattern}"
        )
    if str(row["feature_computation_grain"]).strip() != "FINAL_VIEW_FEATURES":
        raise ValueError(
            "Spatial export requires FINAL_VIEW_FEATURES for "
            f"window_id={window_id}"
        )
    if str(row["pair_scope_key"]).strip() != window_id:
        raise ValueError(
            f"Spatial export pair scope mismatch for window_id={window_id}"
        )
    if not _bool_scalar(row["pair_recomputed_for_view"]) or not _bool_scalar(
        row["aggregate_recomputed_for_view"]
    ):
        raise ValueError(
            "Spatial export requires recomputed pair and aggregate claims for "
            f"window_id={window_id}"
        )
    selected_timestamps = _json_number_list(
        row["selected_timestamps_seconds"],
        field="selected_timestamps_seconds",
        window_id=window_id,
        expected_count=expected_count,
        allow_null=True,
    )
    pair_delta_seconds = _json_number_list(
        row["pair_delta_seconds"],
        field="pair_delta_seconds",
        window_id=window_id,
        expected_count=max(0, expected_count - 1),
        allow_null=True,
    )
    for position, declared_delta in enumerate(pair_delta_seconds):
        previous = selected_timestamps[position]
        current = selected_timestamps[position + 1]
        expected_delta = (
            current - previous
            if previous is not None and current is not None
            else None
        )
        if expected_delta is None:
            if declared_delta is not None:
                raise ValueError(
                    "Final-view pair delta seconds must be null when a "
                    f"timestamp is absent for window_id={window_id}"
                )
        elif (
            declared_delta is None
            or declared_delta <= 0
            or not np.isclose(
                declared_delta,
                expected_delta,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise ValueError(
                "Final-view pair delta seconds mismatch selected timestamps "
                f"for window_id={window_id}"
            )
    return np.asarray(indices, dtype=np.int32)


def _json_int_list(value: Any, *, field: str, window_id: str) -> list[int]:
    parsed = _json_list(value, field=field, window_id=window_id)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise ValueError(
            f"{field} must contain integers for window_id={window_id}"
        )
    return [int(item) for item in parsed]


def _json_number_list(
    value: Any,
    *,
    field: str,
    window_id: str,
    expected_count: int,
    allow_null: bool,
) -> list[float | None]:
    parsed = _json_list(value, field=field, window_id=window_id)
    if len(parsed) != expected_count:
        raise ValueError(
            f"{field} count mismatch for window_id={window_id}: "
            f"observed={len(parsed)}, expected={expected_count}"
        )
    result: list[float | None] = []
    for item in parsed:
        if item is None and allow_null:
            result.append(None)
        elif isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(
                f"{field} must contain finite numbers or null for "
                f"window_id={window_id}"
            )
        elif not np.isfinite(float(item)):
            raise ValueError(
                f"{field} contains non-finite value for window_id={window_id}"
            )
        else:
            result.append(float(item))
    return result


def _json_list(value: Any, *, field: str, window_id: str) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid {field} JSON for window_id={window_id}"
        ) from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a list for window_id={window_id}")
    return parsed


def _bool_scalar(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _validate_window_alignment_contract(
    windows: pd.DataFrame,
    *,
    require_final_view_contract: bool,
) -> None:
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
    span_valid = start.le(end) & length.gt(0)
    if not require_final_view_contract:
        span_valid &= length.eq(end - start + 1)
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
    if require_final_view_contract:
        for _, row in windows.iterrows():
            _selected_window_frame_indices(
                row,
                require_final_view_contract=True,
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


def _available_feature_names(
    frames: pd.DataFrame,
    feature_schema: dict[str, list[str]],
    *,
    fail_closed_groups: set[str] | None = None,
) -> dict[str, list[str]]:
    fail_closed = fail_closed_groups or set()
    available: dict[str, list[str]] = {}
    for group_name, cols in feature_schema.items():
        if group_name in fail_closed:
            missing = [column for column in cols if column not in frames]
            if missing:
                raise ValueError(
                    f"Missing required {group_name} features: {missing}"
                )
            available[group_name] = list(cols)
            continue
        present = [c for c in cols if c in frames.columns]
        if present:
            available[group_name] = present
    return available


def _motion_metadata_from_frames(frames: pd.DataFrame) -> dict[str, Any]:
    required = {
        "motion_schema_id": "schema_id",
        "motion_schema_version": "schema_version",
        "motion_schema_dimension": "dimension",
        "motion_schema_feature_names": "ordered_feature_names",
        "motion_schema_hash": "schema_hash",
    }
    metadata: dict[str, Any] = {}
    missing = [column for column in required if column not in frames]
    if missing:
        raise ValueError(
            "Missing producer motion schema metadata columns: "
            f"{missing}"
        )
    for column, field in required.items():
        values = frames[column].dropna().unique().tolist()
        if len(values) != 1:
            raise ValueError(
                "Producer motion schema metadata must be constant: "
                f"{column}={values[:5]}"
            )
        metadata[field] = values[0]
    metadata["dimension"] = int(metadata["dimension"])
    feature_names_value = metadata["ordered_feature_names"]
    if isinstance(feature_names_value, str):
        try:
            feature_names_value = json.loads(feature_names_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Producer motion_schema_feature_names is not valid JSON"
            ) from exc
    metadata["ordered_feature_names"] = list(feature_names_value)
    if metadata["ordered_feature_names"] != list(MOTION_FEATURE_NAMES):
        raise ValueError(
            "Producer motion_schema_feature_names does not match authority: "
            f"{metadata['ordered_feature_names']}"
        )
    metadata["dtype"] = MOTION_SCHEMA_DTYPE
    metadata["validity_masks"] = list(MOTION_REQUIRED_MASKS)
    return metadata


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
    ) or is_target_roi_model_forbidden(lower)
