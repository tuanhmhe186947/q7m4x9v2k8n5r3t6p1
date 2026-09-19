"""Event-overlap weighting for classification_v2 sequence windows."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EventWeightTables:
    weights: pd.DataFrame
    audit: dict[str, Any]


def audit_event_weight_manifest(
    weights: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Rebuild expected weights and compare every window without modifying data."""

    required = {
        "window_id",
        "event_overlap_cluster_id",
        "event_count_window",
        "windows_per_event",
        "valid_windows_per_event",
        "window_sample_weight",
        "inverse_windows_per_event",
        "event_balanced_sample_weight",
        "window_valid_for_event_weight",
    }
    missing = sorted(required.difference(weights.columns))
    errors = [f"missing_columns={missing}"] if missing else []
    blank_window_id = 0
    duplicate_window_id = 0
    if "window_id" in weights.columns:
        window_ids = weights["window_id"].fillna("").astype(str).str.strip()
        blank_window_id = int(window_ids.eq("").sum())
        duplicate_window_id = int(window_ids.duplicated(keep=False).sum())
    if blank_window_id:
        errors.append(f"blank_window_id={blank_window_id}")
    if duplicate_window_id:
        errors.append(f"duplicate_window_id_rows={duplicate_window_id}")
    invalid_validity_values = 0
    if "window_valid_for_event_weight" in weights.columns:
        invalid_validity_values = _invalid_bool_count(
            weights["window_valid_for_event_weight"]
        )
        if invalid_validity_values:
            errors.append(
                f"invalid_validity_values={invalid_validity_values}"
            )

    expected_tables: EventWeightTables | None = None
    try:
        expected_tables = build_event_weight_manifest(windows)
    except ValueError as exc:
        errors.append(f"window_manifest_contract={exc}")

    row_count_mismatch = 0
    missing_weight_window_ids: list[str] = []
    extra_weight_window_ids: list[str] = []
    numeric_mismatch_counts: dict[str, int] = {}
    text_mismatch_counts: dict[str, int] = {}
    nonfinite_counts: dict[str, int] = {}
    expected_event_key_column: str | None = None
    if expected_tables is not None:
        expected = expected_tables.weights
        expected_event_key_column = expected_tables.audit["event_key_column"]
        if expected_event_key_column not in weights.columns:
            errors.append(
                f"missing_event_key_column={expected_event_key_column}"
            )
        row_count_mismatch = abs(len(weights) - len(expected))
        if row_count_mismatch:
            errors.append(
                f"row_count_mismatch weights={len(weights)} windows={len(expected)}"
            )
        if "window_id" in weights.columns and not duplicate_window_id:
            actual_ids = set(weights["window_id"].fillna("").astype(str))
            expected_ids = set(expected["window_id"].fillna("").astype(str))
            missing_weight_window_ids = sorted(expected_ids - actual_ids)
            extra_weight_window_ids = sorted(actual_ids - expected_ids)
            if missing_weight_window_ids:
                errors.append(
                    "missing_weight_window_ids="
                    f"{len(missing_weight_window_ids)}"
                )
            if extra_weight_window_ids:
                errors.append(
                    f"extra_weight_window_ids={len(extra_weight_window_ids)}"
                )
        can_compare = (
            not missing
            and not duplicate_window_id
            and not blank_window_id
            and expected_event_key_column in weights.columns
        )
        if can_compare:
            comparison = weights.merge(
                expected,
                on="window_id",
                how="inner",
                suffixes=("_actual", "_expected"),
                validate="one_to_one",
            )
            numeric_mismatch_counts, nonfinite_counts = _numeric_mismatches(
                comparison,
                tolerance=tolerance,
            )
            text_mismatch_counts = _text_mismatches(
                comparison,
                event_key_column=expected_event_key_column,
            )
            for column, count in numeric_mismatch_counts.items():
                if count:
                    errors.append(f"numeric_mismatch_{column}={count}")
            for column, count in text_mismatch_counts.items():
                if count:
                    errors.append(f"text_mismatch_{column}={count}")
            for column, count in nonfinite_counts.items():
                if count:
                    errors.append(f"nonfinite_{column}={count}")

    expected_audit = expected_tables.audit if expected_tables is not None else {}
    return {
        "schema_version": "classification_v2_native_event_mass_check_v2",
        "rows": int(len(weights)),
        "expected_rows": (
            int(len(expected_tables.weights))
            if expected_tables is not None
            else None
        ),
        "row_count_mismatch": row_count_mismatch,
        "blank_window_id": blank_window_id,
        "duplicate_window_id_rows": duplicate_window_id,
        "invalid_validity_values": invalid_validity_values,
        "event_key_column": expected_event_key_column,
        "missing_weight_window_ids": missing_weight_window_ids[:20],
        "extra_weight_window_ids": extra_weight_window_ids[:20],
        "numeric_mismatch_counts": numeric_mismatch_counts,
        "text_mismatch_counts": text_mismatch_counts,
        "nonfinite_counts": nonfinite_counts,
        "unique_native_event_count": expected_audit.get(
            "unique_native_event_count"
        ),
        "event_mass_conservation_error": expected_audit.get(
            "event_mass_conservation_error"
        ),
        "warnings": [
            "do not use overlapping windows as independent statistical units"
        ],
        "errors": errors,
    }


def build_event_weight_manifest(
    windows: pd.DataFrame,
    *,
    window_id_col: str = "window_id",
    event_key_col: str | None = None,
    valid_col: str = "window_valid_for_main_train",
    base_weight_col: str = "window_sample_weight",
) -> EventWeightTables:
    """Allocate one unit of mass per native event across overlapping windows."""

    if event_key_col is None:
        event_key_col = (
            "temporal_unit_keys_json"
            if "temporal_unit_keys_json" in windows.columns
            else "temporal_unit_keys_window"
        )
    required = [window_id_col, event_key_col, valid_col]
    missing = [c for c in required if c not in windows.columns]
    if missing:
        raise ValueError(f"Missing event-weight input columns: {missing}")

    work = windows[[window_id_col, event_key_col, valid_col]].copy()
    window_ids = work[window_id_col].fillna("").astype(str).str.strip()
    valid = _as_bool(work[valid_col])
    invalid_bool = _invalid_bool_count(work[valid_col])
    if base_weight_col in windows.columns:
        base_text = windows[base_weight_col].fillna("").astype(str).str.strip()
        base_missing = base_text.eq("")
        base_weight = pd.to_numeric(windows[base_weight_col], errors="coerce")
        invalid_base = int(
            (
                ~base_missing
                & (
                    base_weight.isna()
                    | ~np.isfinite(base_weight)
                    | base_weight.lt(0)
                )
            ).sum()
        )
        base_weight = base_weight.fillna(1.0)
    else:
        base_missing = pd.Series(False, index=work.index)
        base_weight = pd.Series(1.0, index=work.index, dtype="float64")
        invalid_base = 0
    work[base_weight_col] = base_weight
    work["window_valid_for_event_weight"] = valid

    event_lists, parse_errors, legacy_fallback = _parse_event_lists(
        work[event_key_col],
        event_key_col,
    )
    valid_missing_event = int(
        sum(
            is_valid and not events
            for is_valid, events in zip(valid, event_lists, strict=True)
        )
    )
    counts = {
        "blank_window_id": int(window_ids.eq("").sum()),
        "duplicate_window_id_rows": int(
            window_ids.duplicated(keep=False).sum()
        ),
        "invalid_validity_values": invalid_bool,
        "invalid_base_weight_rows": invalid_base,
        "event_key_parse_error_rows": parse_errors,
        "valid_windows_without_native_event": valid_missing_event,
    }
    errors = [f"{name}={count}" for name, count in counts.items() if count]
    if errors:
        raise ValueError("event weight input contract failed: " + "; ".join(errors))

    all_event_counts: Counter[str] = Counter()
    valid_event_counts: Counter[str] = Counter()
    for is_valid, events in zip(valid, event_lists, strict=True):
        all_event_counts.update(events)
        if is_valid:
            valid_event_counts.update(events)

    event_mass: list[float] = []
    max_all_counts: list[int] = []
    max_valid_counts: list[int] = []
    for is_valid, events in zip(valid, event_lists, strict=True):
        if not is_valid or not events:
            event_mass.append(0.0)
        else:
            event_mass.append(
                float(sum(1.0 / valid_event_counts[event] for event in events))
            )
        max_all_counts.append(
            max((all_event_counts[event] for event in events), default=0)
        )
        max_valid_counts.append(
            max((valid_event_counts[event] for event in events), default=0)
        )

    work["event_overlap_cluster_id"] = [
        json.dumps(events, ensure_ascii=True, separators=(",", ":"))
        for events in event_lists
    ]
    work["event_count_window"] = [len(events) for events in event_lists]
    work["windows_per_event"] = max_all_counts
    work["valid_windows_per_event"] = max_valid_counts
    work["inverse_windows_per_event"] = event_mass
    work["event_balanced_sample_weight"] = (
        work[base_weight_col] * work["inverse_windows_per_event"]
    )

    out_cols = [
        window_id_col,
        event_key_col,
        "event_overlap_cluster_id",
        "event_count_window",
        "windows_per_event",
        "valid_windows_per_event",
        base_weight_col,
        "inverse_windows_per_event",
        "event_balanced_sample_weight",
        "window_valid_for_event_weight",
    ]
    weights = work[out_cols].copy()
    mass_sum = float(work["inverse_windows_per_event"].sum())
    expected_mass = float(len(valid_event_counts))
    mass_error = abs(mass_sum - expected_mass)
    audit = {
        "schema_version": "classification_v2_native_event_mass_v2",
        "rows": int(len(weights)),
        "unique_window_ids": int(weights[window_id_col].nunique(dropna=False)),
        "duplicate_window_id": 0,
        "event_key_column": event_key_col,
        "event_key_encoding": (
            "legacy_exact_cluster_fallback"
            if legacy_fallback
            else "json_native_unit_list"
        ),
        "event_overlap_cluster_count": int(
            weights["event_overlap_cluster_id"].nunique(dropna=False)
        ),
        "unique_native_event_count": int(len(valid_event_counts)),
        "missing_event_key_rows": int(sum(not events for events in event_lists)),
        "multi_event_window_rows": int(sum(len(events) > 1 for events in event_lists)),
        "max_windows_per_event": int(weights["windows_per_event"].max()) if len(weights) else 0,
        "mean_windows_per_event": (
            float(weights["windows_per_event"].mean()) if len(weights) else 0.0
        ),
        "unweighted_event_mass_sum": mass_sum,
        "expected_unweighted_event_mass_sum": expected_mass,
        "event_mass_conservation_error": mass_error,
        "event_balanced_weight_sum": float(weights["event_balanced_sample_weight"].sum()),
        "base_weight_sum": float(weights[base_weight_col].sum()),
        "defaulted_base_weight_rows": int(base_missing.sum()),
        "invalid_weight_zero_count": int(
            (
                (~weights["window_valid_for_event_weight"])
                & weights["event_balanced_sample_weight"].eq(0.0)
            ).sum()
        ),
        "warnings": [
            "event_balanced_sample_weight is for training augmentation, not "
            "independent test sample size",
            *(
                ["legacy event-key fallback cannot decompose multi-unit windows"]
                if legacy_fallback
                else []
            ),
            *(
                ["blank base weights default to one for backward compatibility"]
                if base_missing.any()
                else []
            ),
        ],
        "errors": (
            []
            if mass_error <= 1e-8
            else [f"event_mass_not_conserved={mass_error}"]
        ),
    }
    return EventWeightTables(weights=weights, audit=audit)


def _numeric_mismatches(
    comparison: pd.DataFrame,
    *,
    tolerance: float,
) -> tuple[dict[str, int], dict[str, int]]:
    """Compare persisted numeric fields with values rebuilt from windows."""

    columns = (
        "event_count_window",
        "windows_per_event",
        "valid_windows_per_event",
        "window_sample_weight",
        "inverse_windows_per_event",
        "event_balanced_sample_weight",
    )
    mismatches: dict[str, int] = {}
    nonfinite: dict[str, int] = {}
    for column in columns:
        actual = pd.to_numeric(
            comparison[f"{column}_actual"],
            errors="coerce",
        ).to_numpy(dtype=float)
        expected = pd.to_numeric(
            comparison[f"{column}_expected"],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite = np.isfinite(actual)
        nonfinite[column] = int((~finite).sum())
        mismatches[column] = int(
            (finite & ~np.isclose(actual, expected, atol=tolerance, rtol=0.0)).sum()
        )
    return mismatches, nonfinite


def _text_mismatches(
    comparison: pd.DataFrame,
    *,
    event_key_column: str,
) -> dict[str, int]:
    """Compare event lineage and explicit validity fields by window ID."""

    mismatch: dict[str, int] = {}
    for column in (event_key_column, "event_overlap_cluster_id"):
        actual = comparison[f"{column}_actual"].fillna("").astype(str)
        expected = comparison[f"{column}_expected"].fillna("").astype(str)
        mismatch[column] = int(actual.ne(expected).sum())
    actual_valid = _as_bool(
        comparison["window_valid_for_event_weight_actual"]
    )
    expected_valid = _as_bool(
        comparison["window_valid_for_event_weight_expected"]
    )
    mismatch["window_valid_for_event_weight"] = int(
        actual_valid.ne(expected_valid).sum()
    )
    return mismatch


def _parse_event_lists(
    values: pd.Series,
    event_key_col: str,
) -> tuple[list[list[str]], int, bool]:
    """Parse unambiguous JSON lists or retain one legacy exact cluster key."""

    use_json = event_key_col == "temporal_unit_keys_json"
    parsed_rows: list[list[str]] = []
    errors = 0
    for value in values:
        text = "" if pd.isna(value) else str(value).strip()
        if use_json:
            if not text:
                parsed_rows.append([])
                continue
            try:
                parsed = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                parsed_rows.append([])
                errors += 1
                continue
            if not isinstance(parsed, list):
                parsed_rows.append([])
                errors += 1
                continue
            cleaned = [str(item).strip() for item in parsed]
            if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
                parsed_rows.append([])
                errors += 1
                continue
            parsed_rows.append(sorted(cleaned))
        else:
            parsed_rows.append([text] if text else [])
    return parsed_rows, errors, not use_json


def _invalid_bool_count(series: pd.Series) -> int:
    """Count values outside the explicit bool-like CSV contract."""

    if pd.api.types.is_bool_dtype(series):
        return int(series.isna().sum())
    allowed = {"true", "1", "yes", "y", "t", "false", "0", "no", "n", "f"}
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return int((~normalized.isin(allowed)).sum())


def json_default(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
