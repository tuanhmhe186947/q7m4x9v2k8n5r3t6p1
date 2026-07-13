"""Cross-artifact audit for temporal evidence and review-unit lineage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.temporal_evidence import (
    TEMPORAL_EVIDENCE_BASE_COLUMNS,
    UNIT_TEMPORAL_EVIDENCE_COLUMNS,
    WINDOW_TEMPORAL_EVIDENCE_COLUMNS,
)
from pig_behavior.classification_v2.review.behavior_evidence import (
    REVIEW_EVIDENCE_COLUMNS,
)


def audit_temporal_evidence_lineage(
    enhanced: pd.DataFrame,
    intervals: pd.DataFrame,
    windows: pd.DataFrame,
    review_units: pd.DataFrame | None,
    trainer_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate evidence semantics, keys, row lineage, and model separation."""

    errors: list[str] = []
    warnings: list[str] = []
    required_enhanced = ["temporal_unit_key", *UNIT_TEMPORAL_EVIDENCE_COLUMNS]
    required_intervals = [
        "temporal_unit_key",
        "source_type",
        "label_frame_count",
        *UNIT_TEMPORAL_EVIDENCE_COLUMNS,
    ]
    required_windows = ["window_id", *WINDOW_TEMPORAL_EVIDENCE_COLUMNS]
    _require_columns(enhanced, required_enhanced, "enhanced", errors)
    _require_columns(intervals, required_intervals, "intervals", errors)
    _require_columns(windows, required_windows, "windows", errors)

    enhanced_units = _unique_count(enhanced, "temporal_unit_key")
    interval_units = _unique_count(intervals, "temporal_unit_key")
    duplicate_intervals = _duplicate_count(intervals, "temporal_unit_key")
    duplicate_windows = _duplicate_count(windows, "window_id")
    if duplicate_intervals:
        errors.append(f"duplicate_temporal_unit_key={duplicate_intervals}")
    if duplicate_windows:
        errors.append(f"duplicate_window_id={duplicate_windows}")
    if enhanced_units != interval_units:
        errors.append(
            "enhanced_interval_unit_count_mismatch="
            f"{enhanced_units}!={interval_units}"
        )

    unit_nonconstant = _unit_nonconstant_counts(enhanced)
    if unit_nonconstant:
        errors.append(f"nonconstant_unit_evidence={unit_nonconstant}")

    nonfinite = {
        "enhanced": _nonfinite_counts(enhanced, UNIT_TEMPORAL_EVIDENCE_COLUMNS),
        "intervals": _nonfinite_counts(intervals, UNIT_TEMPORAL_EVIDENCE_COLUMNS),
        "windows": _nonfinite_counts(windows, WINDOW_TEMPORAL_EVIDENCE_COLUMNS),
    }
    nonfinite_errors = {
        table: values for table, values in nonfinite.items() if values
    }
    if nonfinite_errors:
        errors.append(f"nonfinite_evidence={nonfinite_errors}")

    out_of_bounds = {
        "enhanced": _bounded_value_errors(enhanced, UNIT_TEMPORAL_EVIDENCE_COLUMNS),
        "intervals": _bounded_value_errors(
            intervals,
            UNIT_TEMPORAL_EVIDENCE_COLUMNS,
        ),
        "windows": _bounded_value_errors(windows, WINDOW_TEMPORAL_EVIDENCE_COLUMNS),
    }
    bound_errors = {
        table: values for table, values in out_of_bounds.items() if values
    }
    if bound_errors:
        errors.append(f"out_of_bounds_evidence={bound_errors}")

    native_lengths = _native_length_audit(intervals)
    errors.extend(native_lengths["errors"])
    hidden_trust = _cvat_hidden_trust_audit(enhanced)
    if hidden_trust["trusted_without_review_rows"]:
        errors.append(
            "cvat_hidden_trusted_without_review="
            f"{hidden_trust['trusted_without_review_rows']}"
        )

    whitelist = [
        str(column)
        for column in trainer_contract.get("tabular_feature_whitelist", [])
    ]
    missing_whitelist = sorted(
        set(WINDOW_TEMPORAL_EVIDENCE_COLUMNS).difference(whitelist)
    )
    leaked_review_columns = sorted(
        set(REVIEW_EVIDENCE_COLUMNS).intersection(whitelist)
    )
    forbidden_raw_names = _forbidden_raw_evidence_names()
    if missing_whitelist:
        errors.append(f"temporal_evidence_missing_from_whitelist={missing_whitelist}")
    if leaked_review_columns:
        errors.append(f"review_evidence_in_model_whitelist={leaked_review_columns}")
    if forbidden_raw_names:
        errors.append(f"forbidden_raw_evidence_names={forbidden_raw_names}")

    review_audit = _review_unit_audit(review_units, interval_units)
    errors.extend(review_audit["errors"])
    warnings.extend(review_audit["warnings"])
    return {
        "valid": not errors,
        "rows": {
            "enhanced": int(len(enhanced)),
            "intervals": int(len(intervals)),
            "windows": int(len(windows)),
            "review_units": int(len(review_units)) if review_units is not None else 0,
        },
        "keys": {
            "enhanced_unique_temporal_units": enhanced_units,
            "interval_unique_temporal_units": interval_units,
            "duplicate_temporal_unit_key": duplicate_intervals,
            "duplicate_window_id": duplicate_windows,
        },
        "evidence_column_counts": {
            "base": int(len(TEMPORAL_EVIDENCE_BASE_COLUMNS)),
            "unit": int(len(UNIT_TEMPORAL_EVIDENCE_COLUMNS)),
            "window": int(len(WINDOW_TEMPORAL_EVIDENCE_COLUMNS)),
            "trainer_whitelist": int(len(whitelist)),
        },
        "unit_nonconstant_counts": unit_nonconstant,
        "nonfinite_counts": nonfinite,
        "out_of_bounds_counts": out_of_bounds,
        "native_lengths": native_lengths,
        "cvat_hidden_trust": hidden_trust,
        "review_units": review_audit,
        "source_support": _source_support(intervals),
        "missing_temporal_evidence_from_whitelist": missing_whitelist,
        "review_evidence_in_model_whitelist": leaked_review_columns,
        "forbidden_raw_evidence_names": forbidden_raw_names,
        "errors": errors,
        "warnings": warnings,
    }


def _require_columns(
    table: pd.DataFrame,
    required: list[str],
    name: str,
    errors: list[str],
) -> None:
    missing = [column for column in required if column not in table.columns]
    if missing:
        errors.append(f"{name}_missing_columns={missing}")


def _unit_nonconstant_counts(enhanced: pd.DataFrame) -> dict[str, int]:
    available = [
        column
        for column in UNIT_TEMPORAL_EVIDENCE_COLUMNS
        if column in enhanced.columns
    ]
    if "temporal_unit_key" not in enhanced.columns or not available:
        return {}
    variation = enhanced.groupby(
        "temporal_unit_key",
        dropna=False,
        sort=False,
    )[available].nunique(dropna=False)
    return {
        column: int(variation[column].gt(1).sum())
        for column in available
        if variation[column].gt(1).any()
    }


def _nonfinite_counts(
    table: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for column in columns:
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(
            dtype="float64"
        )
        count = int((~np.isfinite(values)).sum())
        if count:
            out[column] = count
    return out


def _bounded_value_errors(
    table: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for column in columns:
        if column not in table.columns or not _is_bounded_evidence(column):
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        tolerance = 1e-9
        count = int(
            (values.lt(-tolerance) | values.gt(1.0 + tolerance)).sum()
        )
        if count:
            out[column] = count
    return out


def _is_bounded_evidence(column: str) -> bool:
    return any(
        token in column
        for token in (
            "_ratio_",
            "_concentration_",
            "_straightness_",
            "_availability_",
        )
    )


def _native_length_audit(intervals: pd.DataFrame) -> dict[str, Any]:
    if not {"source_type", "label_frame_count"}.issubset(intervals.columns):
        return {"cvat_invalid": 0, "legacy_invalid": 0, "errors": []}
    source = intervals["source_type"].fillna("").astype(str)
    lengths = pd.to_numeric(intervals["label_frame_count"], errors="coerce")
    cvat_invalid = int((source.eq("cvat_tracking_xml") & lengths.ne(6)).sum())
    legacy_invalid = int((source.eq("legacy_recovered") & lengths.ne(16)).sum())
    errors = []
    if cvat_invalid:
        errors.append(f"cvat_native_length_not_6={cvat_invalid}")
    if legacy_invalid:
        errors.append(f"legacy_native_length_not_16={legacy_invalid}")
    return {
        "cvat_invalid": cvat_invalid,
        "legacy_invalid": legacy_invalid,
        "errors": errors,
    }


def _cvat_hidden_trust_audit(enhanced: pd.DataFrame) -> dict[str, int]:
    required = {"source_type", "hidden_is_trusted", "hidden_review_status"}
    if not required.issubset(enhanced.columns):
        return {"cvat_rows": 0, "trusted_rows": 0, "trusted_without_review_rows": 0}
    cvat = enhanced["source_type"].fillna("").astype(str).eq("cvat_tracking_xml")
    trusted = _bool_series(enhanced["hidden_is_trusted"])
    reviewed = (
        enhanced["hidden_review_status"]
        .fillna("")
        .astype(str)
        .str.lower()
        .isin({"reviewed", "resolved", "complete"})
    )
    return {
        "cvat_rows": int(cvat.sum()),
        "trusted_rows": int((cvat & trusted).sum()),
        "trusted_without_review_rows": int((cvat & trusted & ~reviewed).sum()),
    }


def _review_unit_audit(
    review_units: pd.DataFrame | None,
    expected_units: int,
) -> dict[str, Any]:
    if review_units is None:
        return {
            "available": False,
            "rows": 0,
            "duplicate_review_unit_id": 0,
            "evidence_available_rows": 0,
            "errors": [],
            "warnings": ["review_unit_manifest_not_provided"],
        }
    errors: list[str] = []
    duplicate = _duplicate_count(review_units, "review_unit_id")
    if duplicate:
        errors.append(f"duplicate_review_unit_id={duplicate}")
    if len(review_units) != expected_units:
        errors.append(
            f"review_interval_row_mismatch={len(review_units)}!={expected_units}"
        )
    missing = [
        column for column in REVIEW_EVIDENCE_COLUMNS if column not in review_units
    ]
    if missing:
        errors.append(f"review_units_missing_evidence={missing}")
    available = _bool_series(
        review_units.get(
            "review_evidence_available",
            pd.Series(False, index=review_units.index),
        )
    )
    return {
        "available": True,
        "rows": int(len(review_units)),
        "duplicate_review_unit_id": duplicate,
        "evidence_available_rows": int(available.sum()),
        "errors": errors,
        "warnings": [],
    }


def _source_support(intervals: pd.DataFrame) -> dict[str, Any]:
    if "source_type" not in intervals.columns:
        return {}
    return {
        str(source): {
            "rows": int(len(group)),
            "motion_active_ratio_mean": _mean(
                group,
                "motion_active_ratio_unit",
            ),
            "roi_feeder_availability_mean": _mean(
                group,
                "roi_feeder_availability_ratio_unit",
            ),
            "social_neighbor_availability_mean": _mean(
                group,
                "social_neighbor_availability_ratio_unit",
            ),
        }
        for source, group in intervals.groupby("source_type", dropna=False)
    }


def _forbidden_raw_evidence_names() -> list[str]:
    forbidden = ("behavior", "label", "manual", "review", "hidden", "target_roi")
    return sorted(
        column
        for column in TEMPORAL_EVIDENCE_BASE_COLUMNS
        if any(token in column for token in forbidden)
    )


def _mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def _unique_count(df: pd.DataFrame, column: str) -> int:
    return int(df[column].nunique(dropna=False)) if column in df.columns else 0


def _duplicate_count(df: pd.DataFrame, column: str) -> int:
    return int(df[column].duplicated().sum()) if column in df.columns else 0
