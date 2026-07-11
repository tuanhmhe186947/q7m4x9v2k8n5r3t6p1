"""Paper-facing fixed-label metrics at the native temporal-unit level."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.calibration import probability_calibration_metrics
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_predictions,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def evaluate_native_oof(
    window_predictions: pd.DataFrame,
    fold_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse overlapping windows once, validate OOF lineage, and compute Q2 metrics."""

    config = NativeTemporalMetricsConfig(
        true_col="true_label",
        pred_col="predicted_label",
        weight_col=None,
        valid_col=None,
        bootstrap_iterations=0,
    )
    units, collapse_audit = build_native_temporal_predictions(window_predictions, config)
    assignment_columns = [
        "temporal_unit_key",
        "recording_group_id",
        "outer_fold_id",
        "behavior_label",
        "source_type",
        "video_key",
        "native_unit_valid_for_main_eval",
    ]
    missing = [column for column in assignment_columns if column not in fold_assignments.columns]
    if missing:
        raise ValueError(f"Q2 fold assignments missing columns: {missing}")
    assignments = fold_assignments[assignment_columns].copy()
    if assignments["temporal_unit_key"].duplicated().any():
        raise ValueError("duplicate temporal_unit_key in Q2 fold assignments")
    merged = assignments.merge(units, on="temporal_unit_key", how="left", validate="one_to_one")
    merged["native_metric_include"] = _to_bool(merged["native_metric_include"]) & _to_bool(
        merged["native_unit_valid_for_main_eval"]
    )
    evaluable = merged.loc[merged["native_metric_include"]].copy()
    fold_conflict = evaluable.get("oof_fold_id", pd.Series("", index=evaluable.index)).astype(str).ne(
        evaluable["outer_fold_id"].astype(str)
    )
    errors: list[str] = []
    if fold_conflict.any():
        errors.append(f"prediction_outer_fold_mismatch={int(fold_conflict.sum())}")
    expected = assignments.loc[_to_bool(assignments["native_unit_valid_for_main_eval"]), "temporal_unit_key"]
    predicted = set(evaluable["temporal_unit_key"].astype(str))
    missing_predictions = sorted(set(expected.astype(str)) - predicted)
    duplicate_predictions = int(evaluable["temporal_unit_key"].duplicated().sum())
    if missing_predictions:
        errors.append(f"missing_valid_native_predictions={len(missing_predictions)}")
    if duplicate_predictions:
        errors.append(f"duplicate_native_predictions={duplicate_predictions}")
    metrics = _metric_bundle(evaluable)
    fold_metrics = {
        str(fold): _metric_bundle(group)
        for fold, group in evaluable.groupby("outer_fold_id", sort=True)
    }
    slices = {
        column: {
            str(value): _metric_bundle(group)
            for value, group in evaluable.groupby(column, sort=True, dropna=False)
        }
        for column in ["source_type", "recording_group_id", "behavior_label"]
    }
    audit = {
        "schema_version": "classification_v2_q2_native_unit_metrics_v1",
        "statistical_unit": "native_temporal_unit",
        "fixed_label_order": list(VALID_BEHAVIORS),
        "fold_count": int(evaluable["outer_fold_id"].nunique()),
        "expected_valid_native_units": int(len(expected)),
        "predicted_valid_native_units": int(len(evaluable)),
        "missing_valid_native_unit_count": len(missing_predictions),
        "duplicate_native_prediction_count": duplicate_predictions,
        "collapse_audit": collapse_audit,
        "pooled_metrics": metrics,
        "fold_metrics": fold_metrics,
        "slice_metrics": slices,
        "errors": errors,
        "valid": not errors,
    }
    return merged, audit


def _metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    base = evaluate_predictions(
        frame,
        y_true_col="true_label",
        y_pred_col="native_predicted_behavior",
        label_order=list(VALID_BEHAVIORS),
    )
    confusion = np.asarray(base["confusion_matrix"]["values"], dtype=float)
    base["balanced_accuracy"] = float(base["macro_recall"])
    base["multiclass_mcc"] = _multiclass_mcc(confusion)
    probability_columns = [f"prob_{label}" for label in VALID_BEHAVIORS]
    if set(probability_columns).issubset(frame.columns) and len(frame):
        probabilities = frame[probability_columns].to_numpy(dtype=float)
        targets = np.asarray(
            [{label: index for index, label in enumerate(VALID_BEHAVIORS)}[label] for label in frame["true_label"]],
            dtype=int,
        )
        base.update(probability_calibration_metrics(probabilities, targets, ece_bins=15))
    return base


def _multiclass_mcc(confusion: np.ndarray) -> float:
    total = float(confusion.sum())
    correct = float(np.trace(confusion))
    true_marginal = confusion.sum(axis=1)
    predicted_marginal = confusion.sum(axis=0)
    numerator = correct * total - float(np.dot(true_marginal, predicted_marginal))
    denominator = np.sqrt(
        (total**2 - float(np.dot(predicted_marginal, predicted_marginal)))
        * (total**2 - float(np.dot(true_marginal, true_marginal)))
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
