"""Production frame-local primitive contract for Classification V2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.context_policy import (
    normalize_hidden_provenance,
)
from pig_behavior.classification_v2.features.geometry import (
    build_geometry_features,
    validate_geometry_features,
)
from pig_behavior.classification_v2.features.pen_context import (
    DEFAULT_PEN_MASK_SHA256,
    PEN_CONTEXT_DERIVATION_COLUMNS,
    PEN_CONTEXT_FRAME_LOCAL_FEATURE_COLUMNS,
    build_static_pen_context_features,
)
from pig_behavior.classification_v2.features.roi import (
    build_roi_features,
    validate_roi_features,
)
from pig_behavior.classification_v2.features.social import (
    STATIC_SOCIAL_COLUMNS,
    build_static_social_context_features,
)
from pig_behavior.classification_v2.features.temporal_harmonization import (
    attach_structural_temporal_unit_identity,
)
from pig_behavior.classification_v2.sources.temporal_provenance import (
    CANONICAL_TIMESTAMP_SOURCE,
    apply_source_frame_clock,
    audit_source_frame_clock,
)

FRAME_LOCAL_GRAIN = "FRAME_LOCAL_PRIMITIVES"
FRAME_LOCAL_SCHEMA_VERSION = "classification_v2.frame_local_primitives.v2"
ACTIVE_SOURCE_FPS = 30.0

FRAME_LOCAL_STRUCTURAL_IDENTITY_COLUMNS: frozenset[str] = frozenset(
    {
        "object_track_key",
        "temporal_unit_key",
    }
)

# This registry is deliberately explicit. Prefix checks supplement it only for
# unknown future columns and are not the scientific authority by themselves.
FORBIDDEN_FRAME_LOCAL_COLUMNS: frozenset[str] = frozenset(
    {
        "adjacent_motion_pair_valid",
        "sparse_velocity_pair_valid",
        "motion_velocity_pair_valid",
        "acceleration_pair_valid",
        "motion_delta_frames",
        "motion_delta_seconds",
        "delta_frame_prev",
        "delta_time_prev_sec",
        "displacement_n",
        "speed_n_per_frame",
        "speed_n_per_second",
        "accel_n_per_frame2",
        "acceleration_n_per_second2",
        "legacy_acceleration_alias_tangential_only",
        "direction_rad",
        "heading_rad",
        "heading_change_rad",
        "path_length_n_unit",
        "roi_transition_pair_valid",
        "roi_target_entry_event",
        "roi_target_exit_event",
        "nearest_dist_delta",
        "partner_distance_delta_n",
        "approach_speed_n_per_second",
        "retreat_speed_n_per_second",
        "pen_motion_context_valid",
        "pen_velocity_context_valid",
        "pen_adjacent_motion_pair_valid",
        "pen_sparse_velocity_pair_valid",
        "pen_distance_delta_n_per_frame",
        "pen_distance_delta_n_per_second",
        "pen_normal_speed_n_per_second",
        "pen_approach_speed_n_per_second",
        "pen_retreat_speed_n_per_second",
        "pen_parallel_speed_n_per_second",
        "pair_recomputed_for_view",
        "aggregate_recomputed_for_view",
    }
)

FORBIDDEN_FRAME_LOCAL_PREFIXES: tuple[str, ...] = (
    "prev_",
    "next_",
    "rolling_",
)

FORBIDDEN_FRAME_LOCAL_SEMANTIC_PATTERNS: tuple[str, ...] = (
    "*_pair_valid",
    "*_delta_*",
    "delta_*",
    "prev_*",
    "*_speed_*",
    "speed_*",
    "*accel*",
    "*path_length*",
    "*transition*",
    "*_entry_event",
    "*_exit_event",
    "*_unit",
    "*_window",
    "*_run",
    "turning_*",
    "motion_energy*",
    "motion_pair_invalid*",
    "aggression_score_proxy*",
    "roi_motion_inside_score*",
)

OBJECT_IDENTITY_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_uid",
    "object_id_in_image",
    "pig_id",
    "track_id",
    "frame_index",
)


def build_frame_local_primitives(
    frame_objects: pd.DataFrame,
    *,
    roi_coco_path: Path,
    pen_mask_path: Path,
    source_fps: float = ACTIVE_SOURCE_FPS,
    expected_pen_mask_sha256: str | None = DEFAULT_PEN_MASK_SHA256,
) -> pd.DataFrame:
    """Build row-preserving one-frame primitives from merged frame objects."""

    if not np.isclose(float(source_fps), ACTIVE_SOURCE_FPS, atol=1e-9):
        raise ValueError(
            "active Classification V2 data requires decoded-frame FPS 30"
        )
    original_rows = len(frame_objects)
    out = frame_objects.copy().reset_index(drop=True)
    out["source_row_ordinal"] = np.arange(len(out), dtype="int64")
    if {"source_type", "hidden"}.issubset(out.columns):
        out = normalize_hidden_provenance(out)
    out = apply_source_frame_clock(
        out,
        source_fps=float(source_fps),
        preserve_input_as_acquisition=True,
    )
    out = build_geometry_features(out)
    out = build_roi_features(out, roi_coco_path=roi_coco_path)
    out = attach_structural_temporal_unit_identity(out)
    out = build_static_social_context_features(out)
    out = build_static_pen_context_features(
        out,
        mask_path=pen_mask_path,
        expected_mask_sha256=expected_pen_mask_sha256,
    )
    out["feature_computation_grain"] = FRAME_LOCAL_GRAIN
    out["pair_scope_key"] = ""
    if len(out) != original_rows:
        raise RuntimeError("frame-local construction changed row count")
    semantic_errors = forbidden_frame_local_columns(out.columns)
    if semantic_errors:
        raise ValueError(f"forbidden frame-local semantics: {semantic_errors}")
    return out


def audit_frame_local_primitives(
    source: pd.DataFrame,
    output: pd.DataFrame,
    *,
    tolerance_seconds: float = 1e-9,
) -> dict[str, Any]:
    """Independently verify row identity, clock, grain, and static ranges."""

    errors: list[str] = []
    if len(source) != len(output):
        errors.append(f"row_count_mismatch={len(source)}:{len(output)}")
    identity_columns = [
        column for column in OBJECT_IDENTITY_COLUMNS if column in source.columns
    ]
    missing_identity = sorted(set(identity_columns).difference(output.columns))
    if missing_identity:
        errors.append(f"missing_identity_columns={missing_identity}")
    elif len(source) == len(output):
        for column in identity_columns:
            left = source[column].fillna("").astype(str).reset_index(drop=True)
            right = output[column].fillna("").astype(str).reset_index(drop=True)
            mismatch = int(left.ne(right).sum())
            if mismatch:
                errors.append(f"row_order_or_identity_mismatch={column}:{mismatch}")
    if "source_row_ordinal" not in output:
        errors.append("missing_source_row_ordinal")
    else:
        ordinal = pd.to_numeric(output["source_row_ordinal"], errors="coerce")
        expected = pd.Series(np.arange(len(output)), index=output.index)
        mismatch = int(ordinal.ne(expected).sum())
        if mismatch:
            errors.append(f"source_row_ordinal_mismatch={mismatch}")
    errors.extend(audit_source_frame_clock(
        output,
        tolerance_seconds=tolerance_seconds,
    )["errors"])
    errors.extend(_temporal_identity_errors(source, output))
    if "source_fps" in output:
        fps = pd.to_numeric(output["source_fps"], errors="coerce")
        wrong_fps = int(
            (~np.isclose(fps, ACTIVE_SOURCE_FPS, atol=1e-9)).sum()
        )
        if wrong_fps:
            errors.append(f"active_source_fps_not_30={wrong_fps}")
    grain = output.get("feature_computation_grain")
    if grain is None:
        errors.append("missing_feature_computation_grain")
    else:
        mismatch = int(grain.fillna("").astype(str).ne(FRAME_LOCAL_GRAIN).sum())
        if mismatch:
            errors.append(f"feature_computation_grain_mismatch={mismatch}")
    errors.extend(forbidden_frame_local_columns(output.columns))
    errors.extend(validate_geometry_features(output).get("errors", []))
    errors.extend(validate_roi_features(output).get("errors", []))
    errors.extend(_static_range_errors(output))
    return {
        "schema_version": FRAME_LOCAL_SCHEMA_VERSION,
        "rows": int(len(output)),
        "source_rows": int(len(source)),
        "identity_columns": identity_columns,
        "source_identity_sha256": dataframe_identity_sha256(
            source,
            identity_columns,
        ),
        "output_identity_sha256": dataframe_identity_sha256(
            output,
            identity_columns,
        ),
        "canonical_timestamp_formula": (
            "timestamp_sec=source_frame_index/source_fps"
        ),
        "canonical_timestamp_source": CANONICAL_TIMESTAMP_SOURCE,
        "columns": output.columns.astype(str).tolist(),
        "errors": errors,
        "valid": not errors,
    }


def forbidden_frame_local_columns(columns: Iterable[str]) -> list[str]:
    """Return explicit semantic-registry violations in a frame-local schema."""

    forbidden: list[str] = []
    for raw in columns:
        column = str(raw).strip()
        if column in FRAME_LOCAL_STRUCTURAL_IDENTITY_COLUMNS:
            continue
        if column in FORBIDDEN_FRAME_LOCAL_COLUMNS:
            forbidden.append(column)
            continue
        if column.startswith(FORBIDDEN_FRAME_LOCAL_PREFIXES):
            forbidden.append(column)
            continue
        if any(
            fnmatch(column, pattern)
            for pattern in FORBIDDEN_FRAME_LOCAL_SEMANTIC_PATTERNS
        ):
            forbidden.append(column)
    return sorted(set(forbidden))


def frame_local_schema_payload(frame: pd.DataFrame) -> dict[str, Any]:
    columns = frame.columns.astype(str).tolist()
    return {
        "schema_version": FRAME_LOCAL_SCHEMA_VERSION,
        "feature_computation_grain": FRAME_LOCAL_GRAIN,
        "columns": columns,
        "column_count": len(columns),
        "forbidden_semantic_registry": sorted(FORBIDDEN_FRAME_LOCAL_COLUMNS),
        "forbidden_columns_present": forbidden_frame_local_columns(columns),
        "frame_local_pen_columns": [
            *PEN_CONTEXT_FRAME_LOCAL_FEATURE_COLUMNS,
            *PEN_CONTEXT_DERIVATION_COLUMNS,
        ],
        "frame_local_social_columns": list(STATIC_SOCIAL_COLUMNS),
        "structural_identity_columns": sorted(
            FRAME_LOCAL_STRUCTURAL_IDENTITY_COLUMNS
        ),
    }


def dataframe_identity_sha256(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> str:
    selected = [column for column in columns if column in frame.columns]
    records = frame.loc[:, selected].fillna("").astype(str).to_dict("records")
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _temporal_identity_errors(
    source: pd.DataFrame,
    output: pd.DataFrame,
) -> list[str]:
    if "temporal_unit_key" not in output.columns:
        return ["missing_temporal_unit_key"]
    actual = output["temporal_unit_key"].fillna("").astype(str)
    blank = actual.str.strip().eq("")
    errors = [f"blank_temporal_unit_key={int(blank.sum())}"] if blank.any() else []
    try:
        expected = attach_structural_temporal_unit_identity(source)
    except ValueError as exc:
        return [*errors, f"structural_temporal_identity_invalid={exc}"]
    if len(expected) != len(output):
        return errors
    mismatch = actual.ne(
        expected["temporal_unit_key"].fillna("").astype(str)
    )
    if mismatch.any():
        errors.append(f"temporal_unit_key_mismatch={int(mismatch.sum())}")
    return errors


def _static_range_errors(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for column in (
        "nearest_pair_iou",
        "nearest_pair_overlap_ratio",
        "pen_bbox_inside_ratio",
    ):
        if column not in frame:
            errors.append(f"missing_static_validity_column={column}")
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.notna() & (~np.isfinite(values) | values.lt(0) | values.gt(1))
        if invalid.any():
            errors.append(f"static_range_violation={column}:{int(invalid.sum())}")
    return errors


__all__ = [
    "ACTIVE_SOURCE_FPS",
    "FORBIDDEN_FRAME_LOCAL_COLUMNS",
    "FRAME_LOCAL_GRAIN",
    "FRAME_LOCAL_SCHEMA_VERSION",
    "FRAME_LOCAL_STRUCTURAL_IDENTITY_COLUMNS",
    "audit_frame_local_primitives",
    "build_frame_local_primitives",
    "dataframe_identity_sha256",
    "forbidden_frame_local_columns",
    "frame_local_schema_payload",
]
