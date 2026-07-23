"""Complete semantic registry for behavior-review evidence columns."""

from __future__ import annotations

from fnmatch import fnmatch
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.model_io import (
    DEFAULT_FORBIDDEN_X_PATTERNS,
)
from pig_behavior.classification_v2.features.frame_local import (
    FORBIDDEN_FRAME_LOCAL_COLUMNS,
    FRAME_LOCAL_GRAIN,
)
from pig_behavior.classification_v2.features.native_evidence_contract import (
    NATIVE_EVIDENCE_SEMANTICS_VERSION,
    NATIVE_FEATURE_COMPUTATION_GRAIN,
    NATIVE_MOTION_SCHEMA_VERSION,
)

EVIDENCE_SEMANTICS_SCHEMA_VERSION = (
    "classification_v2.behavior_review_evidence_semantics.v2"
)
EVIDENCE_COLUMN_SEMANTIC_VERSION = "classification_v2.review_evidence.v2"
NATIVE_GRAIN = "NATIVE_UNIT_REVIEW_EVIDENCE"

GUI_FRAME_COLUMNS: frozenset[str] = frozenset(
    {
        "source_type",
        "dataset_id",
        "video_key",
        "source_video_path",
        "frame_index",
        "source_frame_index",
        "timestamp_sec",
        "pig_id",
        "track_id",
        "object_track_key",
        "temporal_unit_key",
        "x1",
        "y1",
        "x2",
        "y2",
        "crop_path",
        "actor_crop_path",
        "scene_image_path",
        "full_frame_path",
        "nearest_pig_id",
        "nearest_track_id",
    }
)

GUI_UNIT_EVIDENCE_COLUMNS: frozenset[str] = frozenset(
    {
        "review_evidence_available",
        "review_motion_evidence_available",
        "review_roi_evidence_available",
        "review_social_evidence_available",
        "review_posture_evidence_available",
        "review_relevant_evidence_available",
        "review_evidence_quality_score",
        "review_evidence_insufficiency_score",
        "review_evidence_conflict_score",
        "review_evidence_priority_auto",
        "review_evidence_reason_auto",
        "review_evidence_status_auto",
        "review_pig_history_available_ratio",
        "review_pig_history_display_frame_indices",
    }
)

EXTRA_MODEL_X_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "*label*",
    "hidden*",
    "review_pig_*",
    "*review*",
    "*policy*",
    "*training*",
    "include_in_training",
    "sample_weight",
    "target_*",
    "*_path",
    "*identity*",
)

EXTRA_MODEL_X_FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "frame_index",
        "native_offset",
        "relative_frame_index",
        "label_anchor_frame_index",
        "image_name",
        "object_id_in_image",
        "timestamp_source",
        "acquisition_timestamp_sec",
        "acquisition_timestamp_source",
        "pair_scope_key",
    }
)

PAIR_VALIDITY_COLUMNS: frozenset[str] = frozenset(
    {
        "previous_observation_available",
        "valid_delta_time",
        "valid_motion_pair",
        *(
            column
            for column in FORBIDDEN_FRAME_LOCAL_COLUMNS
            if column.endswith("pair_valid")
            or column.endswith("context_valid")
        ),
    }
)


def build_evidence_semantics(
    frame_local: pd.DataFrame,
    native_evidence: pd.DataFrame,
    *,
    lineage_id: str = "",
    code_authority_sha: str = "",
) -> dict[str, Any]:
    """Declare and validate every frame-local and native evidence column."""

    errors: list[str] = []
    errors.extend(_native_scope_errors(native_evidence))
    entries: dict[str, dict[str, Any]] = {}
    frame_columns = set(frame_local.columns.astype(str))
    native_columns = set(native_evidence.columns.astype(str))
    declared_columns = (
        frame_columns
        | native_columns
        | GUI_FRAME_COLUMNS
        | GUI_UNIT_EVIDENCE_COLUMNS
    )
    for column in sorted(declared_columns):
        grain = FRAME_LOCAL_GRAIN if column in frame_columns else NATIVE_GRAIN
        if column in GUI_FRAME_COLUMNS:
            grain = FRAME_LOCAL_GRAIN
        pair_derived = column in FORBIDDEN_FRAME_LOCAL_COLUMNS or _pair_name(column)
        if pair_derived:
            grain = NATIVE_GRAIN
        entries[column] = _entry(column, grain, pair_derived)
        values = (
            native_evidence[column]
            if column in native_evidence
            else frame_local[column] if column in frame_local else None
        )
        if values is not None:
            entries[column]["allowed_values"] = _observed_allowed_values(values)
    missing_gui = sorted(
        (GUI_FRAME_COLUMNS | GUI_UNIT_EVIDENCE_COLUMNS).difference(entries)
    )
    if missing_gui:
        errors.append(f"gui_columns_without_semantics={sorted(set(missing_gui))}")
    leakage = sorted(
        column
        for column, entry in entries.items()
        if _must_forbid_model_x(column)
        and entry["model_x_eligibility"] != "forbidden"
    )
    if leakage:
        errors.append(f"model_x_leakage_semantic_drift={leakage}")
    return {
        "schema_version": EVIDENCE_SEMANTICS_SCHEMA_VERSION,
        "lineage_id": str(lineage_id),
        "code_authority_sha": str(code_authority_sha).lower(),
        "evidence_column_semantic_version": EVIDENCE_COLUMN_SEMANTIC_VERSION,
        "pair_scope_contract": "temporal_unit_key",
        "mask_semantics": {
            "observed_mask": {
                "meaning": "selected source frame exists",
                "must_not_substitute": "spatial_quality_mask",
            },
            "spatial_quality_mask": {
                "meaning": "same-frame geometry is scientifically valid",
                "must_not_substitute": "observed_mask",
            },
            "roi_validity_mask": {
                "meaning": "same-frame ROI geometry is available and valid",
            },
            "partner_social_validity_mask": {
                "meaning": "same-frame partner geometry is valid",
            },
        },
        "gui_consumed_columns": sorted(
            GUI_FRAME_COLUMNS | GUI_UNIT_EVIDENCE_COLUMNS
        ),
        "fields": entries,
        "errors": errors,
        "valid": not errors,
    }


def _entry(column: str, grain: str, pair_derived: bool) -> dict[str, Any]:
    target_derived = _target_derived(column)
    return {
        "computation_grain": grain,
        "pair_scope": "temporal_unit_key" if pair_derived else None,
        "physical_unit": _physical_unit(column),
        "source_primitives": _source_primitives(column),
        "validity_mask": _validity_mask(column),
        "gui_display_role": (
            "consumed"
            if column in GUI_FRAME_COLUMNS | GUI_UNIT_EVIDENCE_COLUMNS
            else "available_not_directly_consumed"
        ),
        "audit_only": bool(_must_forbid_model_x(column) or target_derived),
        "model_x_eligibility": (
            "forbidden" if _must_forbid_model_x(column) else "not_authorized_here"
        ),
        "target_derived": target_derived,
        "allowed_values": _allowed_values(column),
    }


def _native_scope_errors(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = {
        "temporal_unit_key",
        "frame_index",
        "pair_scope_key",
        "feature_computation_grain",
        "evidence_semantics_version",
        "motion_schema_version",
        "valid_motion_pair",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        return [f"native_evidence_missing_scope_columns={missing}"]
    scope_mismatch = int(
        frame["pair_scope_key"].fillna("").astype(str).ne(
            frame["temporal_unit_key"].fillna("").astype(str)
        ).sum()
    )
    if scope_mismatch:
        errors.append(f"native_pair_scope_mismatch={scope_mismatch}")
    expected = {
        "feature_computation_grain": NATIVE_FEATURE_COMPUTATION_GRAIN,
        "evidence_semantics_version": NATIVE_EVIDENCE_SEMANTICS_VERSION,
        "motion_schema_version": NATIVE_MOTION_SCHEMA_VERSION,
    }
    for column, value in expected.items():
        mismatch = int(
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne(value)
            .sum()
        )
        if mismatch:
            errors.append(f"native_provenance_mismatch={column}:{mismatch}")
    starts = (
        frame.sort_values(
            ["temporal_unit_key", "frame_index"],
            kind="mergesort",
        )
        .groupby("temporal_unit_key", sort=False)
        .head(1)
    )
    for column in sorted(PAIR_VALIDITY_COLUMNS & set(starts.columns)):
        nonzero = int(_to_bool(starts[column]).sum())
        if nonzero:
            errors.append(f"native_first_frame_inherited_pair={column}:{nonzero}")
    first_row_values = {
        column
        for column in FORBIDDEN_FRAME_LOCAL_COLUMNS
        if not column.endswith(("_unit", "_window"))
    }
    for column in sorted(first_row_values & set(starts.columns)):
        if column in PAIR_VALIDITY_COLUMNS or not pd.api.types.is_numeric_dtype(
            starts[column]
        ):
            continue
        values = pd.to_numeric(starts[column], errors="coerce").fillna(0.0)
        nonzero = int((~np.isclose(values, 0.0, atol=1e-12)).sum())
        if nonzero:
            errors.append(f"native_first_frame_inherited_value={column}:{nonzero}")
    return errors


def _must_forbid_model_x(column: str) -> bool:
    if column in EXTRA_MODEL_X_FORBIDDEN_COLUMNS:
        return True
    patterns = (*DEFAULT_FORBIDDEN_X_PATTERNS, *EXTRA_MODEL_X_FORBIDDEN_PATTERNS)
    return any(fnmatch(column, pattern) for pattern in patterns)


def _target_derived(column: str) -> bool:
    lowered = column.casefold()
    if lowered.startswith("review_pig_"):
        return False
    return any(
        token in lowered
        for token in (
            "behavior",
            "label",
            "target_",
            "manual_",
            "decision",
            "policy",
            "review_feature",
            "review_evidence",
            "use_for_",
            "include_in_training",
            "sample_weight",
        )
    )


def _pair_name(column: str) -> bool:
    lowered = column.casefold()
    return any(
        token in lowered
        for token in (
            "delta_",
            "speed_",
            "acceleration",
            "transition",
            "path_length",
            "entry_event",
            "exit_event",
        )
    )


def _physical_unit(column: str) -> str:
    lowered = column.casefold()
    if lowered.endswith("_sec") or "seconds" in lowered:
        return "seconds"
    if "per_second2" in lowered:
        return "normalized_diagonal_per_second_squared"
    if "per_second" in lowered:
        return "normalized_metric_per_second"
    if "_px" in lowered or lowered in {"x1", "y1", "x2", "y2"}:
        return "pixels"
    if lowered.endswith("_ratio") or lowered.endswith("_iou"):
        return "unit_interval"
    if lowered.endswith("_n"):
        return "normalized_metric"
    if "frame_index" in lowered or lowered.endswith("_count"):
        return "count"
    if lowered.endswith("_valid") or lowered.endswith("_available"):
        return "boolean"
    return "categorical_or_dimensionless"


def _source_primitives(column: str) -> list[str]:
    if column.startswith("roi_"):
        return ["bbox_xyxy", "image_size", "static_roi_calibration"]
    if column.startswith("pen_"):
        return ["bbox_xyxy", "image_size", "static_pen_mask"]
    if column.startswith("nearest_") or column.startswith("social_"):
        return ["same_frame_actor_bboxes", "same_frame_actor_identity"]
    if _pair_name(column):
        return ["ordered_frame_local_primitives", "canonical_timestamp_sec"]
    return [column]


def _validity_mask(column: str) -> str | None:
    if column.startswith("roi_"):
        return "roi_feature_valid"
    if column.startswith("pen_"):
        return "pen_context_quality_valid"
    if column.startswith("nearest_") or column.startswith("social_"):
        return "social_context_valid"
    if _pair_name(column):
        return "motion_velocity_pair_valid"
    if any(token in column for token in ("bbox", "cx", "cy", "area")):
        return "geometry_feature_valid"
    return None


def _allowed_values(column: str) -> dict[str, Any]:
    unit = _physical_unit(column)
    if unit == "unit_interval":
        return {"minimum": 0.0, "maximum": 1.0, "nullable": True}
    if unit == "boolean":
        return {"categories": [False, True], "nullable": False}
    return {"nullable": True}


def _observed_allowed_values(series: pd.Series) -> dict[str, Any]:
    nullable = bool(series.isna().any())
    if pd.api.types.is_bool_dtype(series):
        return {"categories": [False, True], "nullable": nullable}
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        return {
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
            "nullable": nullable,
        }
    categories = sorted(series.dropna().astype(str).unique().tolist())
    if len(categories) <= 64:
        return {"categories": categories, "nullable": nullable}
    return {
        "category_contract": "open_identity_or_media_domain",
        "observed_category_count": len(categories),
        "nullable": nullable,
    }


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.casefold().isin(
        {"1", "true", "yes", "y"}
    )


__all__ = [
    "EVIDENCE_COLUMN_SEMANTIC_VERSION",
    "EVIDENCE_SEMANTICS_SCHEMA_VERSION",
    "build_evidence_semantics",
]
