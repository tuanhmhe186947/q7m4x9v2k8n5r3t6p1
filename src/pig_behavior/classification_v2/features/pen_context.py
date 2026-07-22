"""Label-independent pen-boundary context for classification_v2.

The scene mask is fixed camera calibration. It is never a target label and its
path/hash stay in the audit surface rather than model X. Per-row features keep
the actor's relation to the pen boundary after actor cropping removes that
context. Temporal derivatives reset at source/video/actor/native-unit boundaries.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.lineage_claims import (
    add_optional_lineage_claims_to_audit,
    require_lineage_claims_preserved,
    resolve_optional_lineage_claims,
)

DEFAULT_PEN_MASK_SHA256 = (
    "b59b998ef49335b730c5f117e7161f24ccd277d3b5130c0e640dab7bbb980658"
)

PEN_CONTEXT_FRAME_LOCAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "pen_center_signed_distance_n",
    "pen_center_clearance_box_ratio",
    "pen_bbox_inside_ratio",
)

PEN_CONTEXT_PER_FRAME_AUDIT_COLUMNS: tuple[str, ...] = (
    "pen_distance_delta_n_per_frame",
    "pen_approach_speed_n_per_frame",
    "pen_retreat_speed_n_per_frame",
    "pen_parallel_speed_n_per_frame",
)

PEN_CONTEXT_MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    *PEN_CONTEXT_FRAME_LOCAL_FEATURE_COLUMNS,
    "pen_approach_speed_n_per_second",
    "pen_retreat_speed_n_per_second",
    "pen_parallel_speed_n_per_second",
)

PEN_CONTEXT_LEGACY_MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    *PEN_CONTEXT_FRAME_LOCAL_FEATURE_COLUMNS,
    *PEN_CONTEXT_PER_FRAME_AUDIT_COLUMNS,
)

PEN_CONTEXT_ALL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *PEN_CONTEXT_MODEL_FEATURE_COLUMNS,
            *PEN_CONTEXT_PER_FRAME_AUDIT_COLUMNS,
        )
    )
)

PEN_CONTEXT_QUALITY_COLUMNS: tuple[str, ...] = (
    "pen_context_available",
    "pen_context_quality_valid",
    "pen_motion_context_valid",
    "pen_velocity_context_valid",
    "pen_adjacent_motion_pair_valid",
    "pen_sparse_velocity_pair_valid",
)

PEN_CONTEXT_DERIVATION_COLUMNS: tuple[str, ...] = (
    "pen_boundary_inward_normal_x",
    "pen_boundary_inward_normal_y",
)

REQUIRED_PEN_CONTEXT_INPUT_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_uid",
    "frame_index",
    "timestamp_sec",
    "object_track_key",
    "temporal_unit_key",
    "bbox_valid",
    "x1",
    "y1",
    "x2",
    "y2",
    "image_width",
    "image_height",
)


@dataclass(frozen=True, slots=True)
class PenContextConfig:
    """Thresholds defining the fixed-camera pen context contract."""

    mask_threshold: int = 127
    near_boundary_clearance_ratio: float = 1.0

    def validate(self) -> None:
        if not 0 <= self.mask_threshold <= 254:
            raise ValueError("mask_threshold must be in [0, 254]")
        if self.near_boundary_clearance_ratio <= 0:
            raise ValueError("near_boundary_clearance_ratio must be > 0")


@dataclass(frozen=True, slots=True)
class _PenMaskGeometry:
    binary: np.ndarray
    signed_distance_px: np.ndarray
    inward_normal_x: np.ndarray
    inward_normal_y: np.ndarray
    integral: np.ndarray


def build_pen_context_features(
    frame_features: pd.DataFrame,
    *,
    mask_path: str | Path,
    mask_threshold: int = 127,
    near_boundary_clearance_ratio: float = 1.0,
    expected_mask_sha256: str | None = None,
) -> pd.DataFrame:
    """Append row-preserving pen-boundary features and temporal derivatives."""

    resolve_optional_lineage_claims(
        frame_features,
        artifact_name="pen context input",
    )
    config = PenContextConfig(
        mask_threshold=int(mask_threshold),
        near_boundary_clearance_ratio=float(near_boundary_clearance_ratio),
    )
    config.validate()
    missing = [
        column
        for column in REQUIRED_PEN_CONTEXT_INPUT_COLUMNS
        if column not in frame_features.columns
    ]
    if missing:
        raise ValueError(f"Missing pen context input columns: {missing}")

    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pen mask does not exist: {path}")
    expected_hash = str(expected_mask_sha256 or "").strip().lower()
    if expected_hash:
        observed_hash = _sha256_file(path).lower()
        if observed_hash != expected_hash:
            raise ValueError(
                "Pen mask SHA-256 mismatch: "
                f"expected={expected_hash}, observed={observed_hash}"
            )
    raw_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise ValueError(f"Failed to read pen mask as grayscale: {path}")

    out = frame_features.copy().reset_index(drop=True)
    _initialize_pen_columns(out)
    numeric = _numeric_input_columns(out)
    bbox_valid = _to_bool_series(out["bbox_valid"])
    mask_cache: dict[tuple[int, int], _PenMaskGeometry] = {}

    size_frame = pd.DataFrame(
        {
            "width": numeric["image_width"],
            "height": numeric["image_height"],
        },
        index=out.index,
    )
    valid_size = (
        size_frame.notna().all(axis=1)
        & size_frame["width"].gt(0)
        & size_frame["height"].gt(0)
        & size_frame["width"].mod(1).eq(0)
        & size_frame["height"].mod(1).eq(0)
    )
    for (width_value, height_value), indices in size_frame.loc[valid_size].groupby(
        ["width", "height"],
        sort=False,
    ).groups.items():
        width = int(width_value)
        height = int(height_value)
        geometry = mask_cache.get((width, height))
        if geometry is None:
            geometry = _prepare_mask_geometry(
                raw_mask,
                width=width,
                height=height,
                threshold=config.mask_threshold,
            )
            mask_cache[(width, height)] = geometry
        for index in indices:
            if not bbox_valid.loc[index]:
                continue
            _assign_static_pen_context(
                out,
                index=index,
                numeric=numeric,
                geometry=geometry,
                width=width,
                height=height,
                config=config,
            )

    out = _add_pen_temporal_derivatives(out)
    require_lineage_claims_preserved(
        frame_features,
        out,
        source_name="pen context input",
        derived_name="pen context output",
    )
    return out


def audit_pen_context_features(
    frame_features: pd.DataFrame,
    *,
    mask_path: str | Path,
    mask_threshold: int = 127,
    near_boundary_clearance_ratio: float = 1.0,
    input_rows: int | None = None,
    expected_mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit mask lineage, row preservation, ranges, and finite values."""

    config = PenContextConfig(
        mask_threshold=int(mask_threshold),
        near_boundary_clearance_ratio=float(near_boundary_clearance_ratio),
    )
    config.validate()
    path = Path(mask_path)
    errors: list[str] = []
    warnings: list[str] = []
    required = set(PEN_CONTEXT_ALL_FEATURE_COLUMNS)
    required.update(PEN_CONTEXT_QUALITY_COLUMNS)
    required.update(PEN_CONTEXT_DERIVATION_COLUMNS)
    missing = sorted(required.difference(frame_features.columns))
    if missing:
        errors.append(f"missing_pen_context_columns={missing}")

    if input_rows is not None and len(frame_features) != input_rows:
        errors.append(
            "pen_context_row_count_changed="
            f"input:{input_rows},output:{len(frame_features)}"
        )

    available = _optional_bool(frame_features, "pen_context_available")
    quality = _optional_bool(frame_features, "pen_context_quality_valid")
    motion_valid = _optional_bool(frame_features, "pen_motion_context_valid")
    if available is not None and quality is not None:
        invalid_quality = quality & ~available
        if invalid_quality.any():
            errors.append(
                "pen_quality_true_outside_availability="
                f"{int(invalid_quality.sum())}"
            )
    if available is not None and motion_valid is not None:
        invalid_motion = motion_valid & ~available
        if invalid_motion.any():
            errors.append(
                "pen_motion_true_outside_availability="
                f"{int(invalid_motion.sum())}"
            )

    numeric_errors = _numeric_contract_errors(frame_features, available)
    errors.extend(numeric_errors)
    mask_metadata = _mask_metadata(path, config, errors)
    expected_hash = str(expected_mask_sha256 or "").strip().lower()
    observed_hash = str(mask_metadata.get("sha256", "")).lower()
    if expected_hash and observed_hash != expected_hash:
        errors.append(
            "pen_mask_sha256_mismatch="
            f"expected:{expected_hash},observed:{observed_hash}"
        )

    audit = {
        "rows": int(len(frame_features)),
        "input_rows": input_rows,
        "row_count_preserved": input_rows is None or len(frame_features) == input_rows,
        "mask": mask_metadata,
        "parameters": {
            "mask_threshold": config.mask_threshold,
            "near_boundary_clearance_ratio": (
                config.near_boundary_clearance_ratio
            ),
            "expected_mask_sha256": expected_hash or None,
        },
        "source_image_sizes": _image_size_counts(frame_features),
        "pen_context_available": _value_counts(frame_features, "pen_context_available"),
        "pen_context_quality_valid": _value_counts(
            frame_features,
            "pen_context_quality_valid",
        ),
        "pen_motion_context_valid": _value_counts(
            frame_features,
            "pen_motion_context_valid",
        ),
        "pen_center_inside": _value_counts(frame_features, "pen_center_inside"),
        "pen_near_boundary": _value_counts(frame_features, "pen_near_boundary"),
        "pen_bbox_inside_ratio": _numeric_summary(
            frame_features,
            "pen_bbox_inside_ratio",
        ),
        "pen_center_signed_distance_n": _numeric_summary(
            frame_features,
            "pen_center_signed_distance_n",
        ),
        "pen_approach_speed_n_per_frame": _numeric_summary(
            frame_features,
            "pen_approach_speed_n_per_frame",
        ),
        "pen_parallel_speed_n_per_frame": _numeric_summary(
            frame_features,
            "pen_parallel_speed_n_per_frame",
        ),
        "model_feature_columns": list(PEN_CONTEXT_MODEL_FEATURE_COLUMNS),
        "audit_only_columns": [
            "pen_center_inside",
            "pen_near_boundary",
            *PEN_CONTEXT_PER_FRAME_AUDIT_COLUMNS,
            *PEN_CONTEXT_QUALITY_COLUMNS,
            *PEN_CONTEXT_DERIVATION_COLUMNS,
        ],
        "errors": errors,
        "warnings": warnings,
    }
    return add_optional_lineage_claims_to_audit(
        audit,
        frame_features,
        artifact_name="pen context audit frame table",
    )


def summarize_pen_context(window_rows: pd.DataFrame) -> dict[str, Any]:
    """Summarize transient versus persistent boundary context in one window."""

    if window_rows.empty or "pen_context_available" not in window_rows.columns:
        return empty_pen_context_summary()
    ordered = window_rows.sort_values("frame_index", kind="mergesort")
    available = _to_bool_series(ordered["pen_context_available"])
    near = _optional_bool(ordered, "pen_near_boundary")
    inside = _optional_bool(ordered, "pen_center_inside")
    near = pd.Series(False, index=ordered.index) if near is None else near
    inside = pd.Series(False, index=ordered.index) if inside is None else inside
    near = near & available
    inside = inside & available
    frames = pd.to_numeric(ordered["frame_index"], errors="coerce")
    run = _frame_run_stats(near, available, frames)
    denominator = max(1, len(ordered))

    return {
        "pen_context_availability_ratio_window": float(available.mean()),
        "pen_center_inside_ratio_window": float(inside.mean()),
        "pen_near_boundary_ratio_window": float(near.mean()),
        "pen_near_boundary_longest_run_ratio_window": float(
            run["longest"] / denominator
        ),
        "pen_near_boundary_episode_count_window": int(run["episodes"]),
        "pen_bbox_inside_ratio_mean_window": _safe_numeric(
            ordered,
            "pen_bbox_inside_ratio",
            available,
            operation="mean",
        ),
        "pen_bbox_inside_ratio_min_window": _safe_numeric(
            ordered,
            "pen_bbox_inside_ratio",
            available,
            operation="min",
        ),
        "pen_center_signed_distance_mean_window": _safe_numeric(
            ordered,
            "pen_center_signed_distance_n",
            available,
            operation="mean",
        ),
        "pen_center_signed_distance_min_window": _safe_numeric(
            ordered,
            "pen_center_signed_distance_n",
            available,
            operation="min",
        ),
        "pen_approach_speed_mean_window": _safe_numeric(
            ordered,
            "pen_approach_speed_n_per_frame",
            available,
            operation="mean",
        ),
        "pen_approach_speed_max_window": _safe_numeric(
            ordered,
            "pen_approach_speed_n_per_frame",
            available,
            operation="max",
        ),
        "pen_parallel_speed_mean_window": _safe_numeric(
            ordered,
            "pen_parallel_speed_n_per_frame",
            available,
            operation="mean",
        ),
        "pen_parallel_speed_max_window": _safe_numeric(
            ordered,
            "pen_parallel_speed_n_per_frame",
            available,
            operation="max",
        ),
        "pen_approach_speed_n_per_second_mean_window": _safe_numeric(
            ordered,
            "pen_approach_speed_n_per_second",
            available,
            operation="mean",
        ),
        "pen_approach_speed_n_per_second_max_window": _safe_numeric(
            ordered,
            "pen_approach_speed_n_per_second",
            available,
            operation="max",
        ),
        "pen_parallel_speed_n_per_second_mean_window": _safe_numeric(
            ordered,
            "pen_parallel_speed_n_per_second",
            available,
            operation="mean",
        ),
        "pen_parallel_speed_n_per_second_max_window": _safe_numeric(
            ordered,
            "pen_parallel_speed_n_per_second",
            available,
            operation="max",
        ),
    }


def recompute_pen_motion_for_view(frame_rows: pd.DataFrame) -> pd.DataFrame:
    """Recompute pen motion only from selected rows in one temporal view."""

    if frame_rows.empty:
        return frame_rows.copy()
    required = set(REQUIRED_PEN_CONTEXT_INPUT_COLUMNS)
    required.update(PEN_CONTEXT_DERIVATION_COLUMNS)
    required.update({"pen_context_available", "pen_center_inside"})
    missing = sorted(required.difference(frame_rows.columns))
    if missing:
        raise ValueError(f"Missing pen view-recompute columns: {missing}")
    identity = [
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
    ]
    nonunique = [
        column
        for column in identity
        if frame_rows[column].fillna("").astype(str).nunique(dropna=False) != 1
    ]
    if nonunique:
        raise ValueError(f"Pen view spans multiple identities: {nonunique}")
    out = frame_rows.copy()
    out["temporal_unit_key"] = "__selected_view__"
    return _add_pen_temporal_derivatives(out)


def empty_pen_context_summary() -> dict[str, Any]:
    """Return stable zero-valued pen summary columns for empty windows."""

    return {
        "pen_context_availability_ratio_window": 0.0,
        "pen_center_inside_ratio_window": 0.0,
        "pen_near_boundary_ratio_window": 0.0,
        "pen_near_boundary_longest_run_ratio_window": 0.0,
        "pen_near_boundary_episode_count_window": 0,
        "pen_bbox_inside_ratio_mean_window": 0.0,
        "pen_bbox_inside_ratio_min_window": 0.0,
        "pen_center_signed_distance_mean_window": 0.0,
        "pen_center_signed_distance_min_window": 0.0,
        "pen_approach_speed_mean_window": 0.0,
        "pen_approach_speed_max_window": 0.0,
        "pen_parallel_speed_mean_window": 0.0,
        "pen_parallel_speed_max_window": 0.0,
        "pen_approach_speed_n_per_second_mean_window": 0.0,
        "pen_approach_speed_n_per_second_max_window": 0.0,
        "pen_parallel_speed_n_per_second_mean_window": 0.0,
        "pen_parallel_speed_n_per_second_max_window": 0.0,
    }


def _initialize_pen_columns(out: pd.DataFrame) -> None:
    float_columns = [
        "pen_center_signed_distance_n",
        "pen_center_clearance_box_ratio",
        "pen_bbox_inside_ratio",
        "pen_boundary_inward_normal_x",
        "pen_boundary_inward_normal_y",
    ]
    for column in float_columns:
        out[column] = np.nan
    for column in [
        "pen_distance_delta_n_per_frame",
        "pen_approach_speed_n_per_frame",
        "pen_retreat_speed_n_per_frame",
        "pen_parallel_speed_n_per_frame",
        "pen_distance_delta_n_per_second",
        "pen_normal_speed_n_per_second",
        "pen_approach_speed_n_per_second",
        "pen_retreat_speed_n_per_second",
        "pen_parallel_speed_n_per_second",
        "pen_motion_delta_frames",
        "pen_motion_delta_seconds",
    ]:
        out[column] = 0.0
    for column in [
        "pen_center_inside",
        "pen_near_boundary",
        *PEN_CONTEXT_QUALITY_COLUMNS,
    ]:
        out[column] = False


def _numeric_input_columns(out: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        column: pd.to_numeric(out[column], errors="coerce")
        for column in [
            "frame_index",
            "x1",
            "y1",
            "x2",
            "y2",
            "image_width",
            "image_height",
        ]
    }


def _prepare_mask_geometry(
    raw_mask: np.ndarray,
    *,
    width: int,
    height: int,
    threshold: int,
) -> _PenMaskGeometry:
    binary = (raw_mask > threshold).astype(np.uint8)
    if binary.shape != (height, width):
        binary = cv2.resize(
            binary,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        binary = (binary > 0).astype(np.uint8)
    if not binary.any() or binary.all():
        raise ValueError("Pen mask must contain both inside and outside pixels")

    inside = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(
        1 - binary,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    signed = inside.astype(np.float32) - outside.astype(np.float32)
    gradient_y, gradient_x = np.gradient(inside.astype(np.float32))
    magnitude = np.hypot(gradient_x, gradient_y)
    normal_x = np.divide(
        gradient_x,
        magnitude,
        out=np.zeros_like(gradient_x),
        where=magnitude > 1e-6,
    )
    normal_y = np.divide(
        gradient_y,
        magnitude,
        out=np.zeros_like(gradient_y),
        where=magnitude > 1e-6,
    )
    integral = cv2.integral(binary, sdepth=cv2.CV_32S)
    return _PenMaskGeometry(
        binary=binary,
        signed_distance_px=signed,
        inward_normal_x=normal_x,
        inward_normal_y=normal_y,
        integral=integral,
    )


def _assign_static_pen_context(
    out: pd.DataFrame,
    *,
    index: Any,
    numeric: dict[str, pd.Series],
    geometry: _PenMaskGeometry,
    width: int,
    height: int,
    config: PenContextConfig,
) -> None:
    x1 = numeric["x1"].loc[index]
    y1 = numeric["y1"].loc[index]
    x2 = numeric["x2"].loc[index]
    y2 = numeric["y2"].loc[index]
    values = np.asarray([x1, y1, x2, y2], dtype="float64")
    if not np.isfinite(values).all():
        return
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        return

    center_x = min(width - 1, max(0, int(np.floor((x1 + x2) / 2.0))))
    center_y = min(height - 1, max(0, int(np.floor((y1 + y2) / 2.0))))
    center_inside = bool(geometry.binary[center_y, center_x])
    signed_distance_px = float(geometry.signed_distance_px[center_y, center_x])
    image_diag = float(np.hypot(width, height))
    signed_distance_n = signed_distance_px / image_diag
    box_diag = float(np.hypot(x2 - x1, y2 - y1))
    clearance_box_ratio = signed_distance_px / max(0.5 * box_diag, 1.0)
    bbox_inside_ratio = _bbox_inside_ratio(
        geometry.integral,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        width=width,
        height=height,
    )
    boundary_overlap = bbox_inside_ratio < 1.0 - 1e-9
    near_boundary = center_inside and (
        clearance_box_ratio <= config.near_boundary_clearance_ratio
        or boundary_overlap
    )

    out.at[index, "pen_context_available"] = True
    out.at[index, "pen_context_quality_valid"] = center_inside
    out.at[index, "pen_center_inside"] = center_inside
    out.at[index, "pen_center_signed_distance_n"] = signed_distance_n
    out.at[index, "pen_center_clearance_box_ratio"] = clearance_box_ratio
    out.at[index, "pen_bbox_inside_ratio"] = bbox_inside_ratio
    out.at[index, "pen_near_boundary"] = near_boundary
    out.at[index, "pen_boundary_inward_normal_x"] = float(
        geometry.inward_normal_x[center_y, center_x]
    )
    out.at[index, "pen_boundary_inward_normal_y"] = float(
        geometry.inward_normal_y[center_y, center_x]
    )


def _bbox_inside_ratio(
    integral: np.ndarray,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> float:
    ix1 = max(0, min(width, int(np.floor(x1))))
    iy1 = max(0, min(height, int(np.floor(y1))))
    ix2 = max(0, min(width, int(np.ceil(x2))))
    iy2 = max(0, min(height, int(np.ceil(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inside = (
        int(integral[iy2, ix2])
        - int(integral[iy1, ix2])
        - int(integral[iy2, ix1])
        + int(integral[iy1, ix1])
    )
    return float(inside / ((ix2 - ix1) * (iy2 - iy1)))


def _add_pen_temporal_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    grain = [
        "source_type",
        "dataset_id",
        "video_key",
        "object_track_key",
        "temporal_unit_key",
    ]
    ordered = out.sort_values(
        [*grain, "frame_index"],
        kind="mergesort",
    )
    group = ordered.groupby(grain, dropna=False, sort=False)
    frame_index = pd.to_numeric(ordered["frame_index"], errors="coerce")
    previous_frame = group["frame_index"].shift(1)
    frame_delta = frame_index - pd.to_numeric(previous_frame, errors="coerce")
    timestamp = pd.to_numeric(ordered["timestamp_sec"], errors="coerce")
    previous_timestamp = group["timestamp_sec"].shift(1)
    time_delta = timestamp - pd.to_numeric(previous_timestamp, errors="coerce")
    available = _to_bool_series(ordered["pen_context_available"])
    inside = _to_bool_series(ordered["pen_center_inside"])
    previous_available = group["pen_context_available"].shift(1)
    previous_inside = group["pen_center_inside"].shift(1)
    geometry_pair_valid = (
        available
        & inside
        & _to_bool_series(previous_available)
        & _to_bool_series(previous_inside)
    )

    signed_distance = pd.to_numeric(
        ordered["pen_center_signed_distance_n"],
        errors="coerce",
    )
    previous_distance = group["pen_center_signed_distance_n"].shift(1)
    raw_distance_delta = (
        signed_distance - pd.to_numeric(previous_distance, errors="coerce")
    )
    ordered["_pen_center_x_px"] = (
        pd.to_numeric(ordered["x1"], errors="coerce")
        + pd.to_numeric(ordered["x2"], errors="coerce")
    ) / 2.0
    ordered["_pen_center_y_px"] = (
        pd.to_numeric(ordered["y1"], errors="coerce")
        + pd.to_numeric(ordered["y2"], errors="coerce")
    ) / 2.0
    group = ordered.groupby(grain, dropna=False, sort=False)
    previous_x_px = group["_pen_center_x_px"].shift(1)
    previous_y_px = group["_pen_center_y_px"].shift(1)
    width = pd.to_numeric(ordered["image_width"], errors="coerce")
    height = pd.to_numeric(ordered["image_height"], errors="coerce")
    previous_width = pd.to_numeric(
        group["image_width"].shift(1),
        errors="coerce",
    )
    previous_height = pd.to_numeric(
        group["image_height"].shift(1),
        errors="coerce",
    )
    image_diag = pd.Series(np.hypot(width, height), index=ordered.index)
    dx_metric = (ordered["_pen_center_x_px"] - previous_x_px) / image_diag
    dy_metric = (ordered["_pen_center_y_px"] - previous_y_px) / image_diag
    normal_x = pd.to_numeric(
        ordered["pen_boundary_inward_normal_x"],
        errors="coerce",
    )
    normal_y = pd.to_numeric(
        ordered["pen_boundary_inward_normal_y"],
        errors="coerce",
    )
    finite_projection = (
        np.isfinite(dx_metric)
        & np.isfinite(dy_metric)
        & np.isfinite(normal_x)
        & np.isfinite(normal_y)
        & image_diag.gt(0)
        & width.eq(previous_width)
        & height.eq(previous_height)
    )
    adjacent_pair_valid = (
        geometry_pair_valid
        & frame_delta.eq(1)
        & time_delta.gt(0)
        & finite_projection
    )
    sparse_pair_valid = (
        geometry_pair_valid
        & frame_delta.gt(1)
        & time_delta.gt(0)
        & finite_projection
    )
    velocity_pair_valid = adjacent_pair_valid | sparse_pair_valid
    normal_delta = dx_metric * normal_x + dy_metric * normal_y
    tangent_delta = -dx_metric * normal_y + dy_metric * normal_x
    normal_per_frame = normal_delta / frame_delta

    ordered["pen_motion_delta_frames"] = frame_delta
    ordered["pen_motion_delta_seconds"] = time_delta
    ordered["pen_adjacent_motion_pair_valid"] = adjacent_pair_valid
    ordered["pen_sparse_velocity_pair_valid"] = sparse_pair_valid
    ordered["pen_motion_context_valid"] = adjacent_pair_valid
    ordered["pen_velocity_context_valid"] = velocity_pair_valid
    ordered["pen_distance_delta_n_per_frame"] = (
        raw_distance_delta / frame_delta
    ).where(
        adjacent_pair_valid,
        0.0,
    )
    ordered["pen_approach_speed_n_per_frame"] = (-normal_per_frame).clip(
        lower=0.0
    ).where(adjacent_pair_valid, 0.0)
    ordered["pen_retreat_speed_n_per_frame"] = normal_per_frame.clip(
        lower=0.0
    ).where(adjacent_pair_valid, 0.0)
    ordered["pen_parallel_speed_n_per_frame"] = (
        tangent_delta.abs() / frame_delta
    ).where(adjacent_pair_valid, 0.0)

    normal_per_second = normal_delta / time_delta
    ordered["pen_distance_delta_n_per_second"] = (
        raw_distance_delta / time_delta
    ).where(velocity_pair_valid, 0.0)
    ordered["pen_normal_speed_n_per_second"] = normal_per_second.where(
        velocity_pair_valid,
        0.0,
    )
    ordered["pen_approach_speed_n_per_second"] = (
        -normal_per_second
    ).clip(lower=0.0).where(velocity_pair_valid, 0.0)
    ordered["pen_retreat_speed_n_per_second"] = normal_per_second.clip(
        lower=0.0
    ).where(velocity_pair_valid, 0.0)
    ordered["pen_parallel_speed_n_per_second"] = (
        tangent_delta.abs() / time_delta
    ).where(velocity_pair_valid, 0.0)
    ordered = ordered.drop(columns=["_pen_center_x_px", "_pen_center_y_px"])
    return ordered.sort_index(kind="mergesort")


def _numeric_contract_errors(
    frame: pd.DataFrame,
    available: pd.Series | None,
) -> list[str]:
    if available is None:
        return []
    errors: list[str] = []
    for column in PEN_CONTEXT_ALL_FEATURE_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame.loc[available, column], errors="coerce")
        invalid = ~np.isfinite(values.to_numpy(dtype="float64"))
        if invalid.any():
            errors.append(f"nonfinite_available_{column}={int(invalid.sum())}")
    for column in ["pen_bbox_inside_ratio"]:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame.loc[available, column], errors="coerce")
        invalid = values.lt(0) | values.gt(1)
        if invalid.any():
            errors.append(f"out_of_range_{column}={int(invalid.sum())}")
    for column in [
        "pen_approach_speed_n_per_frame",
        "pen_retreat_speed_n_per_frame",
        "pen_parallel_speed_n_per_frame",
    ]:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame.loc[available, column], errors="coerce")
        if values.lt(0).any():
            errors.append(f"negative_{column}={int(values.lt(0).sum())}")
    return errors


def _mask_metadata(
    path: Path,
    config: PenContextConfig,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing_pen_mask={path}")
        return {"path": str(path), "exists": False}
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        errors.append(f"unreadable_pen_mask={path}")
        return {"path": str(path), "exists": True, "readable": False}
    binary = raw > config.mask_threshold
    if not binary.any() or binary.all():
        errors.append("pen_mask_must_contain_inside_and_outside_pixels")
    return {
        "path": str(path),
        "exists": True,
        "readable": True,
        "sha256": _sha256_file(path),
        "width": int(raw.shape[1]),
        "height": int(raw.shape[0]),
        "threshold": config.mask_threshold,
        "inside_pixel_count": int(binary.sum()),
        "inside_pixel_ratio": float(binary.mean()),
    }


def _frame_run_stats(
    state: pd.Series,
    available: pd.Series,
    frames: pd.Series,
) -> dict[str, int]:
    longest = 0
    current = 0
    episodes = 0
    previous_frame: int | None = None
    active = False
    for value, is_available, frame_value in zip(
        state.tolist(),
        available.tolist(),
        frames.tolist(),
        strict=True,
    ):
        if not is_available or pd.isna(frame_value):
            current = 0
            active = False
            previous_frame = None
            continue
        frame_index = int(frame_value)
        contiguous = previous_frame is not None and frame_index == previous_frame + 1
        if bool(value):
            if not active or not contiguous:
                episodes += 1
                current = 1
            else:
                current += 1
            active = True
            longest = max(longest, current)
        else:
            current = 0
            active = False
        previous_frame = frame_index
    return {"longest": longest, "episodes": episodes}


def _safe_numeric(
    frame: pd.DataFrame,
    column: str,
    available: pd.Series,
    *,
    operation: str,
) -> float:
    if column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame.loc[available, column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0
    if operation == "mean":
        return float(values.mean())
    if operation == "min":
        return float(values.min())
    if operation == "max":
        return float(values.max())
    raise ValueError(f"Unsupported numeric summary operation: {operation}")


def _optional_bool(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    return _to_bool_series(frame[column])


def _to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def _image_size_counts(frame: pd.DataFrame) -> dict[str, int]:
    if not {"image_width", "image_height"}.issubset(frame.columns):
        return {}
    width = pd.to_numeric(frame["image_width"], errors="coerce")
    height = pd.to_numeric(frame["image_height"], errors="coerce")
    keys = width.astype("Int64").astype(str) + "x" + height.astype("Int64").astype(str)
    counts = keys.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    if column not in frame.columns:
        return {"count": 0}
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_PEN_MASK_SHA256",
    "PEN_CONTEXT_ALL_FEATURE_COLUMNS",
    "PEN_CONTEXT_DERIVATION_COLUMNS",
    "PEN_CONTEXT_FRAME_LOCAL_FEATURE_COLUMNS",
    "PEN_CONTEXT_LEGACY_MODEL_FEATURE_COLUMNS",
    "PEN_CONTEXT_MODEL_FEATURE_COLUMNS",
    "PEN_CONTEXT_PER_FRAME_AUDIT_COLUMNS",
    "PEN_CONTEXT_QUALITY_COLUMNS",
    "PenContextConfig",
    "audit_pen_context_features",
    "build_pen_context_features",
    "empty_pen_context_summary",
    "summarize_pen_context",
]
