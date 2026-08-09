"""Primary S1 validation evaluation at canonical native-unit grain only."""

from __future__ import annotations

import pandas as pd

from pig_behavior.classification_v2.evaluation.native_temporal_collapse import (
    NativeTemporalCollapseResult,
    collapse_window_predictions_to_native_units,
)


class S1PrimaryEvaluationError(ValueError):
    """Raised when a primary S1 validation evaluation contract is violated."""


def evaluate_primary_s1_validation(
    window_predictions: pd.DataFrame,
    eligibility_windows: pd.DataFrame,
    expected_native_units: pd.DataFrame,
) -> NativeTemporalCollapseResult:
    """Validate window coverage then collapse explicitly to native predictions."""

    _require(
        window_predictions,
        ["window_id", "y_pred", "confidence"],
        "window_predictions",
    )
    _require(
        eligibility_windows,
        [
            "window_id",
            "temporal_unit_keys_json",
            "primary_s1_role",
            "primary_s1_eligible",
        ],
        "eligibility_windows",
    )
    _require(
        expected_native_units,
        ["temporal_unit_key", "behavior_label"],
        "expected_native_units",
    )
    if "temporal_unit_key" in window_predictions.columns:
        raise S1PrimaryEvaluationError(
            "primary S1 metric rejects direct temporal_unit_key prediction input"
        )
    selected = eligibility_windows.loc[
        _strict_bool(eligibility_windows["primary_s1_eligible"])
        & eligibility_windows["primary_s1_role"].astype(str).eq("validation")
    ].copy()
    expected_window_ids = selected["window_id"].astype(str).tolist()
    observed_window_ids = window_predictions["window_id"].astype(str).tolist()
    duplicate_predictions = int(window_predictions["window_id"].duplicated().sum())
    missing = sorted(set(expected_window_ids).difference(observed_window_ids))
    unexpected = sorted(set(observed_window_ids).difference(expected_window_ids))
    if (
        duplicate_predictions
        or missing
        or unexpected
        or len(expected_window_ids) != len(observed_window_ids)
    ):
        raise S1PrimaryEvaluationError(
            "primary S1 window prediction coverage mismatch="
            f"duplicates:{duplicate_predictions},missing:{len(missing)},"
            f"unexpected:{len(unexpected)}"
        )
    return collapse_window_predictions_to_native_units(
        window_predictions,
        selected[["window_id", "temporal_unit_keys_json"]],
        expected_native_units,
    )


def _strict_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise S1PrimaryEvaluationError("primary_s1_eligible contains null values")
        return series.astype(bool)
    values = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    if (~values.isin(true_values | false_values)).any():
        raise S1PrimaryEvaluationError("primary_s1_eligible contains invalid values")
    return values.isin(true_values)


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise S1PrimaryEvaluationError(f"{name} missing columns={missing}")
