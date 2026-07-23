"""Fail-closed contract checks for native-unit review evidence."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_AGGREGATION_OUTPUTS,
    MOTION_FEATURE_NAMES,
    MOTION_REQUIRED_MASKS,
    MOTION_SCHEMA_DIMENSION,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_ID,
    MOTION_SCHEMA_VERSION,
)

NATIVE_FEATURE_COMPUTATION_GRAIN = "NATIVE_UNIT_REVIEW_EVIDENCE"
NATIVE_PAIR_SCOPE_KEY = "temporal_unit_key"
NATIVE_EVIDENCE_SEMANTICS_VERSION = (
    "classification_v2.native_review_evidence.v2"
)
NATIVE_MOTION_SCHEMA_VERSION = MOTION_SCHEMA_VERSION

NATIVE_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "feature_computation_grain",
    "pair_scope_key",
    "evidence_semantics_version",
    "motion_schema_version",
)
PAIR_GRAIN_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "temporal_unit_key",
)
PAIR_COVERAGE_COLUMNS: tuple[str, ...] = (
    *MOTION_AGGREGATION_OUTPUTS,
    "motion_feature_available",
    "motion_feature_coverage_available",
)
INVALID_PAIR_MISSING_COLUMNS: tuple[str, ...] = (
    *MOTION_FEATURE_NAMES[:7],
)
MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "object_track_key",
    "temporal_unit_key",
    "frame_index",
    "frame_uid",
)


def check_native_review_evidence(
    source: pd.DataFrame,
    output: pd.DataFrame,
    *,
    producer_audit: dict[str, Any] | None,
    code_sha: str,
    input_sha256: str,
    contract_manifest_sha256: str,
) -> dict[str, Any]:
    """Independently validate native evidence without trusting its audit."""

    errors: list[str] = []
    warnings: list[str] = []
    _check_hash("code_sha", code_sha, 40, errors)
    _check_hash("input_sha256", input_sha256, 64, errors)
    _check_hash(
        "contract_manifest_sha256",
        contract_manifest_sha256,
        64,
        errors,
    )

    required = {
        *PAIR_GRAIN_COLUMNS,
        *NATIVE_PROVENANCE_COLUMNS,
        *PAIR_COVERAGE_COLUMNS,
        *MOTION_FEATURE_NAMES,
        *MOTION_REQUIRED_MASKS,
        "frame_index",
        "timestamp_sec",
        "bbox_valid",
        "previous_observation_available",
        "previous_temporal_unit_key",
        "previous_object_track_key",
        "same_temporal_unit_pair",
        "same_actor_trajectory_pair",
        "current_geometry_valid",
        "previous_geometry_valid",
        "valid_delta_time",
        "valid_motion_pair",
    }
    missing = sorted(required.difference(output.columns))
    if missing:
        errors.append(f"missing_native_contract_columns={missing}")

    _check_population(source, output, errors)
    _check_provenance(output, producer_audit, errors)

    cross_unit_pair_count = 0
    pair_reset_errors = 0
    invalid_pair_numeric_values = 0
    pair_count_mismatches = 0
    if not missing:
        (
            cross_unit_pair_count,
            pair_reset_errors,
            invalid_pair_numeric_values,
            pair_count_mismatches,
        ) = _check_pairs_and_coverage(output, errors)

    audit = {
        "schema_version": (
            "classification_v2.native_review_evidence_checker.v1"
        ),
        "feature_computation_grain": NATIVE_FEATURE_COMPUTATION_GRAIN,
        "pair_scope_key": NATIVE_PAIR_SCOPE_KEY,
        "evidence_semantics_version": NATIVE_EVIDENCE_SEMANTICS_VERSION,
        "motion_schema_version": NATIVE_MOTION_SCHEMA_VERSION,
        "motion_schema_id": MOTION_SCHEMA_ID,
        "motion_schema_dimension": MOTION_SCHEMA_DIMENSION,
        "motion_schema_feature_names": list(MOTION_FEATURE_NAMES),
        "motion_schema_hash": MOTION_SCHEMA_HASH,
        "code_sha": str(code_sha).lower(),
        "input_sha256": str(input_sha256).lower(),
        "contract_manifest_sha256": str(
            contract_manifest_sha256
        ).lower(),
        "input_rows": int(len(source)),
        "output_rows": int(len(output)),
        "input_temporal_units": _nunique(source, "temporal_unit_key"),
        "output_temporal_units": _nunique(output, "temporal_unit_key"),
        "population_preserved": len(source) == len(output),
        "cross_unit_pair_count": cross_unit_pair_count,
        "native_pair_reset_errors": pair_reset_errors,
        "invalid_pair_numeric_values": invalid_pair_numeric_values,
        "pair_count_mismatches": pair_count_mismatches,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    return audit


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash a dataframe deterministically without changing row order."""

    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check_hash(
    field: str,
    value: str,
    length: int,
    errors: list[str],
) -> None:
    normalized = str(value).strip().lower()
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", normalized):
        errors.append(f"invalid_{field}")


def _check_population(
    source: pd.DataFrame,
    output: pd.DataFrame,
    errors: list[str],
) -> None:
    if len(source) != len(output):
        errors.append(
            f"population_row_count_mismatch={len(source)}:{len(output)}"
        )
    columns = [
        column
        for column in MEMBERSHIP_COLUMNS
        if column in source.columns and column in output.columns
    ]
    if not columns:
        errors.append("population_membership_columns_unavailable")
        return
    source_digest = _membership_digest(source, columns)
    output_digest = _membership_digest(output, columns)
    if source_digest != output_digest:
        errors.append("population_or_temporal_membership_mismatch")


def _membership_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    normalized = frame[columns].copy()
    for column in columns:
        normalized[column] = normalized[column].fillna("").astype(str)
    records = sorted(map(tuple, normalized.to_numpy().tolist()))
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_provenance(
    output: pd.DataFrame,
    producer_audit: dict[str, Any] | None,
    errors: list[str],
) -> None:
    expected = {
        "feature_computation_grain": NATIVE_FEATURE_COMPUTATION_GRAIN,
        "evidence_semantics_version": NATIVE_EVIDENCE_SEMANTICS_VERSION,
        "motion_schema_version": NATIVE_MOTION_SCHEMA_VERSION,
        "motion_schema_id": MOTION_SCHEMA_ID,
        "motion_schema_dimension": str(MOTION_SCHEMA_DIMENSION),
        "motion_schema_feature_names": json.dumps(
            list(MOTION_FEATURE_NAMES),
            separators=(",", ":"),
        ),
        "motion_schema_hash": MOTION_SCHEMA_HASH,
    }
    for column, value in expected.items():
        if column not in output:
            continue
        observed = set(output[column].fillna("").astype(str).str.strip().unique())
        if observed != {str(value)}:
            errors.append(
                f"native_provenance_mismatch={column}:{sorted(observed)}"
            )
    if {"pair_scope_key", "temporal_unit_key"}.issubset(output.columns):
        scope_mismatch = int(
            output["pair_scope_key"].fillna("").astype(str).ne(
                output["temporal_unit_key"].fillna("").astype(str)
            ).sum()
        )
        if scope_mismatch:
            errors.append(f"native_pair_scope_mismatch={scope_mismatch}")

    if producer_audit is None:
        errors.append("producer_audit_missing")
        return
    audit_expected = {
        "feature_computation_grain": NATIVE_FEATURE_COMPUTATION_GRAIN,
        "evidence_semantics_version": NATIVE_EVIDENCE_SEMANTICS_VERSION,
        "motion_schema_version": NATIVE_MOTION_SCHEMA_VERSION,
        "motion_schema_id": MOTION_SCHEMA_ID,
        "motion_schema_dimension": MOTION_SCHEMA_DIMENSION,
        "motion_schema_hash": MOTION_SCHEMA_HASH,
        "pair_scope_key": NATIVE_PAIR_SCOPE_KEY,
    }
    for field, value in audit_expected.items():
        if str(producer_audit.get(field, "")).strip() != value:
            if str(producer_audit.get(field, "")).strip() != str(value):
                errors.append(f"producer_audit_provenance_mismatch={field}")
    if producer_audit.get("motion_schema_feature_names") != list(
        MOTION_FEATURE_NAMES
    ):
        errors.append(
            "producer_audit_provenance_mismatch=motion_schema_feature_names"
        )
    for field in ("code_sha", "input_sha256", "contract_manifest_sha256"):
        if not str(producer_audit.get(field, "")).strip():
            errors.append(f"producer_audit_missing={field}")
    if producer_audit.get("errors"):
        errors.append("producer_audit_has_errors")


def _check_pairs_and_coverage(
    output: pd.DataFrame,
    errors: list[str],
) -> tuple[int, int, int, int]:
    work = output.copy()
    work["_checker_order"] = np.arange(len(work), dtype="int64")
    sort_columns = [*PAIR_GRAIN_COLUMNS, "frame_index"]
    if "frame_uid" in work:
        sort_columns.append("frame_uid")
    work = work.sort_values(sort_columns, kind="mergesort")
    group = work.groupby(list(PAIR_GRAIN_COLUMNS), dropna=False, sort=False)

    observed = _observed_mask(work)
    previous_observed = observed.groupby(
        [work[column] for column in PAIR_GRAIN_COLUMNS],
        dropna=False,
        sort=False,
    ).shift(1).fillna(False)
    previous_frame = group["frame_index"].shift(1)
    previous_available = (
        previous_frame.notna() & observed & previous_observed
    )
    geometry = _geometry_valid(work)
    previous_geometry = geometry.groupby(
        [work[column] for column in PAIR_GRAIN_COLUMNS],
        dropna=False,
        sort=False,
    ).shift(1).fillna(False)
    previous_time = group["timestamp_sec"].shift(1)
    delta_time = pd.to_numeric(
        work["timestamp_sec"],
        errors="coerce",
    ) - pd.to_numeric(previous_time, errors="coerce")
    delta_frame = pd.to_numeric(
        work["frame_index"],
        errors="coerce",
    ) - pd.to_numeric(previous_frame, errors="coerce")
    expected_valid = (
        previous_available
        & geometry
        & previous_geometry
        & np.isfinite(delta_time)
        & delta_time.gt(0)
        & np.isfinite(delta_frame)
        & delta_frame.gt(0)
    )
    actual_valid = _bool_series(work["valid_motion_pair"])
    mismatch = int(actual_valid.ne(expected_valid).sum())
    if mismatch:
        errors.append(f"valid_motion_pair_mismatch={mismatch}")

    first = group.head(1)
    pair_reset_errors = int(
        _bool_series(first["previous_observation_available"]).sum()
        + _bool_series(first["valid_motion_pair"]).sum()
    )
    if pair_reset_errors:
        errors.append(f"native_pair_reset_errors={pair_reset_errors}")

    previous_unit = work["previous_temporal_unit_key"].fillna("").astype(str)
    cross_unit_pair_count = int(
        (
            actual_valid
            & previous_unit.ne("")
            & previous_unit.ne(
                work["temporal_unit_key"].fillna("").astype(str)
            )
        ).sum()
    )
    if cross_unit_pair_count:
        errors.append(f"cross_unit_pair_count={cross_unit_pair_count}")

    invalid_pair_numeric_values = 0
    for column in INVALID_PAIR_MISSING_COLUMNS:
        if column not in work:
            continue
        invalid_pair_numeric_values += int(
            pd.to_numeric(
                work.loc[~actual_valid, column],
                errors="coerce",
            ).notna().sum()
        )
    derivative_masks = {
        "direction_change_valid": ("direction_change_rad",),
        "tangential_acceleration_valid": (
            "tangential_acceleration_n_per_second2",
        ),
        "vector_acceleration_valid": (
            "ax_n_per_second2",
            "ay_n_per_second2",
            "acceleration_vector_magnitude_n_per_second2",
        ),
    }
    for mask_column, feature_columns in derivative_masks.items():
        valid = _bool_series(work[mask_column])
        for column in feature_columns:
            invalid_pair_numeric_values += int(
                pd.to_numeric(
                    work.loc[~valid, column],
                    errors="coerce",
                ).notna().sum()
            )
    if invalid_pair_numeric_values:
        errors.append(
            "invalid_pairs_have_numeric_motion_values="
            f"{invalid_pair_numeric_values}"
        )

    pair_count_mismatches = _check_coverage(work, errors)
    return (
        cross_unit_pair_count,
        pair_reset_errors,
        invalid_pair_numeric_values,
        pair_count_mismatches,
    )


def _check_coverage(
    work: pd.DataFrame,
    errors: list[str],
) -> int:
    observed = _observed_mask(work)
    coverage = pd.DataFrame(
        {
            "temporal_unit_key": work["temporal_unit_key"].astype(str),
            "observed": observed.astype("int64"),
            "valid": _bool_series(work["valid_motion_pair"]).astype("int64"),
            "velocity_valid": _bool_series(
                work["velocity_valid"]
            ).astype("int64"),
            "direction_change_valid": _bool_series(
                work["direction_change_valid"]
            ).astype("int64"),
            "acceleration_valid": _bool_series(
                work["vector_acceleration_valid"]
            ).astype("int64"),
        }
    ).groupby("temporal_unit_key", sort=False).sum()
    coverage["possible"] = np.maximum(coverage["observed"] - 1, 0)
    denominator = coverage["possible"].replace(0, np.nan)
    coverage["ratio"] = (
        coverage["valid"].div(denominator).fillna(0.0)
    )
    coverage["available"] = coverage["valid"].gt(0)
    coverage["coverage_available"] = coverage["possible"].gt(0)
    coverage["higher_order_possible"] = np.maximum(
        coverage["observed"] - 2,
        0,
    )
    higher_denominator = coverage["higher_order_possible"].replace(0, np.nan)
    coverage["velocity_coverage"] = (
        coverage["velocity_valid"].div(denominator).fillna(0.0)
    )
    coverage["direction_change_coverage"] = (
        coverage["direction_change_valid"]
        .div(higher_denominator)
        .fillna(0.0)
    )
    coverage["acceleration_coverage"] = (
        coverage["acceleration_valid"]
        .div(higher_denominator)
        .fillna(0.0)
    )

    expected_columns = {
        "observed_frame_count": "observed",
        "possible_pair_count": "possible",
        "valid_pair_count": "valid",
        "valid_pair_ratio": "ratio",
        "motion_feature_coverage": "ratio",
        "motion_feature_available": "available",
        "motion_feature_coverage_available": "coverage_available",
        "velocity_possible_count": "possible",
        "velocity_valid_count": "velocity_valid",
        "velocity_coverage": "velocity_coverage",
        "direction_change_possible_count": "higher_order_possible",
        "direction_change_valid_count": "direction_change_valid",
        "direction_change_coverage": "direction_change_coverage",
        "acceleration_possible_count": "higher_order_possible",
        "acceleration_valid_count": "acceleration_valid",
        "acceleration_coverage": "acceleration_coverage",
    }
    mismatches = 0
    keys = work["temporal_unit_key"].astype(str)
    for output_column, expected_column in expected_columns.items():
        expected = keys.map(coverage[expected_column])
        actual = work[output_column]
        if pd.api.types.is_bool_dtype(expected):
            mismatches += int(_bool_series(actual).ne(expected).sum())
        else:
            actual_numeric = pd.to_numeric(actual, errors="coerce")
            mismatches += int(
                (~np.isclose(
                    actual_numeric,
                    pd.to_numeric(expected, errors="coerce"),
                    equal_nan=True,
                )).sum()
            )
    invalid_range = int(
        (
            pd.to_numeric(
                work["motion_feature_coverage"],
                errors="coerce",
            ).lt(0)
            | pd.to_numeric(
                work["motion_feature_coverage"],
                errors="coerce",
            ).gt(1)
        ).sum()
    )
    if invalid_range:
        errors.append(f"motion_feature_coverage_out_of_range={invalid_range}")
    if mismatches:
        errors.append(f"pair_coverage_mismatch={mismatches}")
    return mismatches


def _observed_mask(frame: pd.DataFrame) -> pd.Series:
    if "observed_mask" not in frame:
        return pd.Series(True, index=frame.index, dtype=bool)
    return _bool_series(frame["observed_mask"])


def _geometry_valid(frame: pd.DataFrame) -> pd.Series:
    columns = ["cx_n", "cy_n", "bw_n", "bh_n", "area_n", "aspect_ratio"]
    finite = frame[columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return _bool_series(frame["bbox_valid"]) & finite


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .isin({"1", "true", "yes", "y", "t"})
    )


def _nunique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].nunique(dropna=False))


__all__ = [
    "NATIVE_EVIDENCE_SEMANTICS_VERSION",
    "NATIVE_FEATURE_COMPUTATION_GRAIN",
    "NATIVE_MOTION_SCHEMA_VERSION",
    "NATIVE_PAIR_SCOPE_KEY",
    "NATIVE_PROVENANCE_COLUMNS",
    "PAIR_COVERAGE_COLUMNS",
    "check_native_review_evidence",
    "dataframe_sha256",
]
