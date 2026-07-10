"""Aggregate window predictions into native temporal-unit metrics.

Training can use overlapping sequence windows, but publication-facing metrics
should score each temporal/review unit once. This module converts window-level
predictions to one deterministic native-unit prediction, preserving audit counts
for rows that cannot be used instead of silently dropping them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions


@dataclass(frozen=True, slots=True)
class NativeTemporalMetricsConfig:
    """Column contract for aggregating window predictions to native units."""

    unit_id_col: str = "temporal_unit_key"
    true_col: str = "behavior_true"
    pred_col: str = "behavior_pred"
    weight_col: str | None = "window_sample_weight"
    valid_col: str | None = "window_valid_for_main_train"
    window_id_col: str = "window_id"
    prob_prefix: str = "prob_"
    include_invalid_windows: bool = False


def build_native_temporal_predictions(
    predictions: pd.DataFrame,
    config: NativeTemporalMetricsConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one deterministic prediction row per native temporal unit."""

    cfg = config or NativeTemporalMetricsConfig()
    errors: list[str] = []
    warnings: list[str] = []
    _validate_required_columns(predictions, cfg, errors)
    if errors:
        return pd.DataFrame(), _audit(predictions, pd.DataFrame(), cfg, errors, warnings)

    frame = predictions.copy()
    frame[cfg.unit_id_col] = frame[cfg.unit_id_col].fillna("").astype(str).str.strip()
    missing_unit_mask = frame[cfg.unit_id_col].eq("")
    usable = frame.loc[~missing_unit_mask].copy()

    invalid_window_count = 0
    if cfg.valid_col and cfg.valid_col in usable.columns and not cfg.include_invalid_windows:
        valid = _to_bool_series(usable[cfg.valid_col])
        invalid_window_count = int((~valid).sum())
        usable = usable.loc[valid].copy()

    rows: list[dict[str, Any]] = []
    prob_cols = [col for col in usable.columns if col.startswith(cfg.prob_prefix)]
    for unit_id, group in usable.groupby(cfg.unit_id_col, sort=True, dropna=False):
        rows.append(_aggregate_one_unit(str(unit_id), group, cfg, prob_cols))

    units = pd.DataFrame(rows)
    if not units.empty:
        units = units.sort_values(cfg.unit_id_col, kind="mergesort").reset_index(drop=True)
        units.insert(0, "native_temporal_row_index", np.arange(len(units), dtype="int64"))

    audit = _audit(predictions, units, cfg, errors, warnings)
    audit["missing_unit_id_window_rows"] = int(missing_unit_mask.sum())
    audit["invalid_window_rows_excluded"] = invalid_window_count
    audit["prediction_aggregation_method"] = "mean_probability" if prob_cols else "weighted_vote"
    return units, audit


def build_native_temporal_metrics(
    predictions: pd.DataFrame,
    config: NativeTemporalMetricsConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate predictions and compute native-unit multiclass metrics."""

    cfg = config or NativeTemporalMetricsConfig()
    units, audit = build_native_temporal_predictions(predictions, cfg)
    if units.empty:
        metrics: dict[str, Any] = {}
    else:
        metric_rows = units.loc[units["native_metric_include"]].copy()
        metrics = (
            evaluate_predictions(metric_rows, y_true_col=cfg.true_col, y_pred_col="native_predicted_behavior")
            if not metric_rows.empty
            else {}
        )
    payload = {
        "primary_metric_unit": "native_temporal_unit",
        "native_temporal_metrics": metrics,
        "native_temporal_prediction_audit": audit,
    }
    return units, payload


def _aggregate_one_unit(
    unit_id: str,
    group: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
    prob_cols: list[str],
) -> dict[str, Any]:
    """Reduce all window predictions for one native unit to one row."""

    true_values = [str(v) for v in group[cfg.true_col].fillna("").astype(str).tolist() if str(v)]
    unique_true = sorted(set(true_values))
    true_label = _deterministic_vote(true_values, weights=_weights(group, cfg)) if true_values else ""
    if prob_cols:
        pred_label, confidence = _mean_probability_prediction(group, cfg, prob_cols)
        aggregation_method = "mean_probability"
    else:
        pred_values = [str(v) for v in group[cfg.pred_col].fillna("").astype(str).tolist() if str(v)]
        pred_label = _deterministic_vote(pred_values, weights=_weights(group, cfg))
        confidence = _vote_confidence(pred_values, pred_label, _weights(group, cfg))
        aggregation_method = "weighted_vote"

    window_ids = (
        sorted(set(group[cfg.window_id_col].fillna("").astype(str).tolist()))
        if cfg.window_id_col in group.columns
        else []
    )
    true_conflict = len(unique_true) > 1
    return {
        cfg.unit_id_col: unit_id,
        cfg.true_col: true_label,
        "native_predicted_behavior": pred_label,
        "native_prediction_confidence": float(confidence),
        "native_metric_include": bool(true_label and pred_label and not true_conflict),
        "native_metric_exclusion_reason": "true_label_conflict" if true_conflict else "",
        "true_label_conflict": bool(true_conflict),
        "unique_true_labels_in_unit": "|".join(unique_true),
        "contributing_window_count": int(len(group)),
        "contributing_window_ids": "|".join(window_ids[:50]),
        "prediction_aggregation_method": aggregation_method,
    }


def _validate_required_columns(
    predictions: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
    errors: list[str],
) -> None:
    """Check hard requirements before any aggregation work starts."""

    required = [cfg.unit_id_col, cfg.true_col, cfg.pred_col]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        errors.append(f"missing_prediction_columns={missing}")


def _weights(group: pd.DataFrame, cfg: NativeTemporalMetricsConfig) -> pd.Series:
    """Return non-negative aggregation weights, defaulting to equal windows."""

    if cfg.weight_col and cfg.weight_col in group.columns:
        weights = pd.to_numeric(group[cfg.weight_col], errors="coerce").fillna(1.0).clip(lower=0.0)
    else:
        weights = pd.Series(1.0, index=group.index, dtype="float64")
    if float(weights.sum()) <= 0:
        weights = pd.Series(1.0, index=group.index, dtype="float64")
    return weights


def _mean_probability_prediction(
    group: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
    prob_cols: list[str],
) -> tuple[str, float]:
    """Average class probabilities with window weights and choose max class."""

    weights = _weights(group, cfg).to_numpy(dtype="float64")
    probs = group[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    weighted = np.average(probs, axis=0, weights=weights)
    best_idx = int(np.argmax(weighted))
    label = prob_cols[best_idx][len(cfg.prob_prefix) :]
    return label, float(weighted[best_idx])


def _deterministic_vote(values: list[str], weights: pd.Series) -> str:
    """Weighted vote with lexical tie-break for reproducible aggregation."""

    if not values:
        return ""
    scores: Counter[str] = Counter()
    for value, weight in zip(values, weights.tolist(), strict=False):
        scores[str(value)] += float(weight)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _vote_confidence(values: list[str], pred_label: str, weights: pd.Series) -> float:
    """Return weighted vote share for the selected class."""

    if not values or not pred_label:
        return 0.0
    total = float(weights.iloc[: len(values)].sum())
    if total <= 0:
        return 0.0
    score = 0.0
    for value, weight in zip(values, weights.tolist(), strict=False):
        if value == pred_label:
            score += float(weight)
    return float(score / total)


def _audit(
    predictions: pd.DataFrame,
    units: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the audit payload consumed by experiment-record gates."""

    conflict_count = int(units.get("true_label_conflict", pd.Series(dtype=bool)).sum()) if not units.empty else 0
    metric_include_count = (
        int(units.get("native_metric_include", pd.Series(dtype=bool)).sum()) if not units.empty else 0
    )
    return {
        "schema_version": "classification_v2_native_temporal_prediction_audit_v1",
        "input_window_rows": int(len(predictions)),
        "native_temporal_unit_rows": int(len(units)),
        "metric_included_unit_rows": metric_include_count,
        "true_label_conflict_units": conflict_count,
        "unit_id_col": cfg.unit_id_col,
        "true_col": cfg.true_col,
        "pred_col": cfg.pred_col,
        "weight_col": cfg.weight_col,
        "valid_col": cfg.valid_col,
        "include_invalid_windows": bool(cfg.include_invalid_windows),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors and conflict_count == 0 and metric_include_count > 0,
    }


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Normalize bool-like CSV values from pandas into a boolean mask."""

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
