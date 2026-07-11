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

from pig_behavior.classification_v2.evaluation.metrics import DEFAULT_LABEL_ORDER, evaluate_predictions


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
    bootstrap_iterations: int = 200
    bootstrap_seed: int = 20260710
    sesoi_primary_metric: str = "macro_f1_supported"
    sesoi_minimum_effect_size: float = 0.02


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
        "statistical_unit": "native_temporal_unit",
        "native_temporal_metrics": metrics,
        "confidence_intervals": _bootstrap_confidence_intervals(metric_rows, cfg)
        if not units.empty and not metric_rows.empty
        else {},
        "sesoi": {
            "primary_metric": cfg.sesoi_primary_metric,
            "minimum_effect_size": float(cfg.sesoi_minimum_effect_size),
            "comparison_required": True,
            "status": "comparison_required_for_claim",
        },
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
        pred_label, confidence, mean_probabilities = _mean_probability_prediction(group, cfg, prob_cols)
        aggregation_method = "mean_probability"
    else:
        pred_values = [str(v) for v in group[cfg.pred_col].fillna("").astype(str).tolist() if str(v)]
        pred_label = _deterministic_vote(pred_values, weights=_weights(group, cfg))
        confidence = _vote_confidence(pred_values, pred_label, _weights(group, cfg))
        mean_probabilities = {}
        aggregation_method = "weighted_vote"

    window_ids = (
        sorted(set(group[cfg.window_id_col].fillna("").astype(str).tolist()))
        if cfg.window_id_col in group.columns
        else []
    )
    true_conflict = len(unique_true) > 1
    row = {
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
    row.update(mean_probabilities)
    if "oof_fold_id" in group.columns:
        fold_ids = sorted(set(group["oof_fold_id"].fillna("").astype(str)) - {""})
        row["oof_fold_id"] = fold_ids[0] if len(fold_ids) == 1 else ""
        row["oof_fold_conflict"] = len(fold_ids) > 1
    return row


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
) -> tuple[str, float, dict[str, float]]:
    """Average class probabilities with window weights and choose max class."""

    weights = _weights(group, cfg).to_numpy(dtype="float64")
    probs = group[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    weighted = np.average(probs, axis=0, weights=weights)
    best_idx = int(np.argmax(weighted))
    label = prob_cols[best_idx][len(cfg.prob_prefix) :]
    return label, float(weighted[best_idx]), {column: float(weighted[idx]) for idx, column in enumerate(prob_cols)}


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


def _bootstrap_confidence_intervals(metric_rows: pd.DataFrame, cfg: NativeTemporalMetricsConfig) -> dict[str, Any]:
    """Compute deterministic bootstrap CIs over native temporal units.

    The resampling unit is the native temporal unit row. This keeps uncertainty
    aligned with the paper-facing statistical unit rather than overlapping
    training windows.
    """

    if metric_rows.empty or cfg.bootstrap_iterations <= 0:
        return {}
    metrics = ["accuracy", "macro_f1_supported", "macro_recall_supported"]
    estimates = evaluate_predictions(
        metric_rows, y_true_col=cfg.true_col, y_pred_col="native_predicted_behavior"
    )
    if "oof_fold_id" in metric_rows.columns and metric_rows["oof_fold_id"].fillna("").nunique() >= 2:
        samples = _fold_cluster_bootstrap_samples(metric_rows, cfg)
        method = "oof_fold_cluster_bootstrap_percentile"
        resample_unit = "oof_fold_id"
    else:
        samples = _unit_bootstrap_samples(metric_rows, cfg, metrics)
        method = "unit_bootstrap_percentile"
        resample_unit = "native_temporal_unit"
    return {
        name: {
            "estimate": float(estimates.get(name, 0.0)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "method": method,
            "n_bootstrap": int(cfg.bootstrap_iterations),
            "resample_unit": resample_unit,
        }
        for name, values in samples.items()
    }


def _unit_bootstrap_samples(
    metric_rows: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
    metrics: list[str],
) -> dict[str, list[float]]:
    """Fallback uncertainty for one-fold engineering pilots only."""

    rng = np.random.default_rng(int(cfg.bootstrap_seed))
    samples: dict[str, list[float]] = {name: [] for name in metrics}
    n_rows = len(metric_rows)
    for _ in range(int(cfg.bootstrap_iterations)):
        indices = rng.integers(0, n_rows, size=n_rows)
        boot = metric_rows.iloc[indices].reset_index(drop=True)
        boot_metrics = evaluate_predictions(boot, y_true_col=cfg.true_col, y_pred_col="native_predicted_behavior")
        for name in metrics:
            samples[name].append(float(boot_metrics.get(name, 0.0)))
    return samples


def _fold_cluster_bootstrap_samples(
    metric_rows: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
) -> dict[str, list[float]]:
    """Resample complete held-out recording folds using precomputed confusion matrices."""

    labels = list(DEFAULT_LABEL_ORDER)
    observed = set(metric_rows[cfg.true_col]).union(metric_rows["native_predicted_behavior"])
    labels.extend(sorted(observed - set(labels)))
    fold_ids = sorted(metric_rows["oof_fold_id"].fillna("").astype(str).unique())
    confusion_by_fold = np.stack(
        [
            pd.crosstab(
                metric_rows.loc[metric_rows["oof_fold_id"].astype(str).eq(fold_id), cfg.true_col],
                metric_rows.loc[
                    metric_rows["oof_fold_id"].astype(str).eq(fold_id), "native_predicted_behavior"
                ],
                dropna=False,
            )
            .reindex(index=labels, columns=labels, fill_value=0)
            .to_numpy(dtype=np.int64)
            for fold_id in fold_ids
        ]
    )
    rng = np.random.default_rng(int(cfg.bootstrap_seed))
    samples = {"accuracy": [], "macro_f1_supported": [], "macro_recall_supported": []}
    for _ in range(int(cfg.bootstrap_iterations)):
        multiplicity = np.bincount(
            rng.integers(0, len(fold_ids), size=len(fold_ids)), minlength=len(fold_ids)
        )
        confusion = np.tensordot(multiplicity, confusion_by_fold, axes=(0, 0))
        values = _metrics_from_confusion(confusion)
        for name in samples:
            samples[name].append(values[name])
    return samples


def _metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    """Compute primary supported-class metrics from a fixed-order confusion matrix."""

    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2.0 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    supported = support > 0
    total = float(confusion.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total > 0.0 else 0.0,
        "macro_f1_supported": float(f1[supported].mean()) if supported.any() else 0.0,
        "macro_recall_supported": float(recall[supported].mean()) if supported.any() else 0.0,
    }


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Normalize bool-like CSV values from pandas into a boolean mask."""

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
