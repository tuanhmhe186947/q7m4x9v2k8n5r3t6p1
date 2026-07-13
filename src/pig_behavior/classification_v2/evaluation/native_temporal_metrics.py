"""Aggregate window predictions into native temporal-unit metrics.

Training can use overlapping sequence windows, but publication-facing metrics
should score each temporal/review unit once. This module converts window-level
predictions to one deterministic native-unit prediction, preserving audit counts
for rows that cannot be used instead of silently dropping them.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import (
    DEFAULT_LABEL_ORDER,
    evaluate_predictions,
)


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
    label_order: tuple[str, ...] = tuple(DEFAULT_LABEL_ORDER)
    require_complete_probability_vector: bool = False
    require_oof_fold: bool = False
    bootstrap_iterations: int = 200
    bootstrap_seed: int = 20260710
    sesoi_primary_metric: str = "macro_f1"
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
    if not errors:
        _validate_prediction_rows(predictions, cfg, errors)
    if errors:
        return pd.DataFrame(), _audit(predictions, pd.DataFrame(), cfg, errors, warnings)

    frame = predictions.copy()
    frame[cfg.unit_id_col] = frame[cfg.unit_id_col].fillna("").astype(str).str.strip()
    missing_unit_mask = frame[cfg.unit_id_col].eq("")
    usable = frame.loc[~missing_unit_mask].copy()

    invalid_window_count = 0
    if cfg.valid_col and cfg.valid_col in usable.columns and not cfg.include_invalid_windows:
        valid = _strict_bool_series(usable[cfg.valid_col], cfg.valid_col)
        invalid_window_count = int((~valid).sum())
        usable = usable.loc[valid].copy()
        if invalid_window_count:
            warnings.append(
                f"invalid_window_rows_excluded={invalid_window_count}"
            )

    if usable.empty:
        errors.append("no_usable_window_predictions")
        return pd.DataFrame(), _audit(
            predictions,
            pd.DataFrame(),
            cfg,
            errors,
            warnings,
        )

    rows: list[dict[str, Any]] = []
    prob_cols = _ordered_probability_columns(usable, cfg)
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
    metric_rows = pd.DataFrame()
    if units.empty:
        metrics: dict[str, Any] = {}
    else:
        metric_rows = units.loc[units["native_metric_include"]].copy()
        metrics = (
            evaluate_predictions(
                metric_rows,
                y_true_col=cfg.true_col,
                y_pred_col="native_predicted_behavior",
                label_order=list(cfg.label_order),
            )
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

    weights = _weights(group, cfg)
    true_values = group[cfg.true_col].astype(str).tolist()
    unique_true = sorted(set(true_values))
    true_label = _deterministic_vote(true_values, weights=weights)
    if prob_cols:
        pred_label, confidence, mean_probabilities = _mean_probability_prediction(
            group,
            cfg,
            prob_cols,
            weights,
        )
        aggregation_method = "mean_probability"
    else:
        pred_values = group[cfg.pred_col].astype(str).tolist()
        pred_label = _deterministic_vote(pred_values, weights=weights)
        confidence = _vote_confidence(pred_values, pred_label, weights)
        mean_probabilities = {}
        aggregation_method = "weighted_vote"

    window_ids = (
        sorted(set(group[cfg.window_id_col].fillna("").astype(str).tolist()))
        if cfg.window_id_col in group.columns
        else []
    )
    true_conflict = len(unique_true) > 1
    fold_ids: list[str] = []
    if "oof_fold_id" in group.columns:
        fold_ids = sorted(
            set(group["oof_fold_id"].fillna("").astype(str)) - {""}
        )
    fold_conflict = len(fold_ids) > 1
    exclusion_reasons = []
    if true_conflict:
        exclusion_reasons.append("true_label_conflict")
    if fold_conflict:
        exclusion_reasons.append("oof_fold_conflict")
    row = {
        cfg.unit_id_col: unit_id,
        cfg.true_col: true_label,
        "native_predicted_behavior": pred_label,
        "native_prediction_confidence": float(confidence),
        "native_metric_include": bool(
            true_label
            and pred_label
            and not true_conflict
            and not fold_conflict
        ),
        "native_metric_exclusion_reason": "|".join(exclusion_reasons),
        "true_label_conflict": bool(true_conflict),
        "unique_true_labels_in_unit": "|".join(unique_true),
        "contributing_window_count": int(len(group)),
        "contributing_window_ids": "|".join(window_ids[:50]),
        "contributing_window_id_sha256": _ordered_ids_hash(window_ids),
        "prediction_aggregation_method": aggregation_method,
    }
    row.update(mean_probabilities)
    if "oof_fold_id" in group.columns:
        row["oof_fold_id"] = fold_ids[0] if len(fold_ids) == 1 else ""
        row["oof_fold_conflict"] = fold_conflict
    return row


def _validate_required_columns(
    predictions: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
    errors: list[str],
) -> None:
    """Check hard requirements before any aggregation work starts."""

    required = [
        cfg.unit_id_col,
        cfg.true_col,
        cfg.pred_col,
        cfg.window_id_col,
    ]
    if cfg.require_oof_fold:
        required.append("oof_fold_id")
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        errors.append(f"missing_prediction_columns={missing}")


def _validate_prediction_rows(
    predictions: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
    errors: list[str],
) -> None:
    """Reject malformed keys, labels, masks, weights, and probabilities."""

    if predictions.empty:
        errors.append("window_predictions_empty")
        return
    if not cfg.label_order or len(set(cfg.label_order)) != len(cfg.label_order):
        errors.append("label_order_empty_or_duplicated")
        return

    for column in [cfg.unit_id_col, cfg.window_id_col, cfg.true_col, cfg.pred_col]:
        blank = predictions[column].fillna("").astype(str).str.strip().eq("")
        if blank.any():
            errors.append(f"blank_{column}_rows={int(blank.sum())}")
    duplicate_windows = predictions[cfg.window_id_col].duplicated(keep=False)
    if duplicate_windows.any():
        errors.append(f"duplicate_window_id_rows={int(duplicate_windows.sum())}")

    labels = set(cfg.label_order)
    for column in [cfg.true_col, cfg.pred_col]:
        values = predictions[column].fillna("").astype(str).str.strip()
        invalid = sorted(set(values) - labels)
        if invalid:
            errors.append(f"invalid_{column}_labels={invalid}")

    if cfg.valid_col and cfg.valid_col in predictions.columns:
        try:
            _strict_bool_series(predictions[cfg.valid_col], cfg.valid_col)
        except ValueError as exc:
            errors.append(str(exc))

    if cfg.weight_col and cfg.weight_col in predictions.columns:
        weights = pd.to_numeric(predictions[cfg.weight_col], errors="coerce")
        invalid_weights = weights.isna() | ~np.isfinite(weights) | weights.lt(0.0)
        if invalid_weights.any():
            errors.append(
                f"invalid_aggregation_weight_rows={int(invalid_weights.sum())}"
            )

    prob_cols = _ordered_probability_columns(predictions, cfg)
    observed_prob_cols = sorted(
        column for column in predictions if column.startswith(cfg.prob_prefix)
    )
    invalid_prob_cols = sorted(set(observed_prob_cols) - set(prob_cols))
    if invalid_prob_cols:
        errors.append(f"invalid_probability_columns={invalid_prob_cols}")
    expected_prob_cols = [f"{cfg.prob_prefix}{label}" for label in cfg.label_order]
    if cfg.require_complete_probability_vector and set(prob_cols) != set(
        expected_prob_cols
    ):
        missing = sorted(set(expected_prob_cols) - set(prob_cols))
        extra = sorted(set(prob_cols) - set(expected_prob_cols))
        errors.append(
            "probability_vector_schema_mismatch="
            f"missing:{missing},extra:{extra}"
        )
    if prob_cols:
        probabilities = predictions[prob_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )
        values = probabilities.to_numpy(dtype=np.float64)
        invalid_values = ~np.isfinite(values) | (values < 0.0) | (values > 1.0)
        if invalid_values.any():
            errors.append(
                f"invalid_probability_values={int(invalid_values.sum())}"
            )
        elif not np.allclose(values.sum(axis=1), 1.0, atol=1e-4, rtol=0.0):
            errors.append("probability_row_sum_mismatch")
        else:
            probability_labels = [
                column[len(cfg.prob_prefix) :] for column in prob_cols
            ]
            argmax_labels = np.asarray(probability_labels, dtype=object)[
                values.argmax(axis=1)
            ]
            declared = predictions[cfg.pred_col].astype(str).to_numpy()
            mismatch = argmax_labels != declared
            if mismatch.any():
                errors.append(
                    f"predicted_label_probability_argmax_mismatch={int(mismatch.sum())}"
                )
    elif cfg.require_complete_probability_vector:
        errors.append("complete_probability_vector_required")

    if cfg.require_oof_fold:
        blank_fold = predictions["oof_fold_id"].fillna("").astype(str).str.strip()
        if blank_fold.eq("").any():
            errors.append(f"blank_oof_fold_rows={int(blank_fold.eq('').sum())}")


def _ordered_probability_columns(
    predictions: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
) -> list[str]:
    """Return probability columns in the frozen behavior-label order."""

    return [
        f"{cfg.prob_prefix}{label}"
        for label in cfg.label_order
        if f"{cfg.prob_prefix}{label}" in predictions.columns
    ]


def _weights(group: pd.DataFrame, cfg: NativeTemporalMetricsConfig) -> pd.Series:
    """Return non-negative aggregation weights, defaulting to equal windows."""

    if cfg.weight_col and cfg.weight_col in group.columns:
        weights = pd.to_numeric(group[cfg.weight_col], errors="raise").astype(
            "float64"
        )
    else:
        weights = pd.Series(1.0, index=group.index, dtype="float64")
    if float(weights.sum()) <= 0:
        raise ValueError("native-unit aggregation weights have non-positive total")
    return weights


def _mean_probability_prediction(
    group: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
    prob_cols: list[str],
    weights: pd.Series,
) -> tuple[str, float, dict[str, float]]:
    """Average class probabilities with window weights and choose max class."""

    weight_values = weights.to_numpy(dtype="float64")
    probs = group[prob_cols].to_numpy(dtype="float64")
    weighted = np.average(probs, axis=0, weights=weight_values)
    best_idx = int(np.argmax(weighted))
    label = prob_cols[best_idx][len(cfg.prob_prefix) :]
    mean_probabilities = {
        column: float(weighted[index])
        for index, column in enumerate(prob_cols)
    }
    return label, float(weighted[best_idx]), mean_probabilities


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

    conflict_count = (
        int(units.get("true_label_conflict", pd.Series(dtype=bool)).sum())
        if not units.empty
        else 0
    )
    fold_conflict_count = (
        int(units.get("oof_fold_conflict", pd.Series(dtype=bool)).sum())
        if not units.empty
        else 0
    )
    metric_include_count = (
        int(units.get("native_metric_include", pd.Series(dtype=bool)).sum())
        if not units.empty
        else 0
    )
    input_window_ids = (
        predictions[cfg.window_id_col].fillna("").astype(str).tolist()
        if cfg.window_id_col in predictions.columns
        else []
    )
    output_unit_ids = (
        units[cfg.unit_id_col].fillna("").astype(str).tolist()
        if cfg.unit_id_col in units.columns
        else []
    )
    return {
        "schema_version": "classification_v2_native_temporal_prediction_audit_v2",
        "input_window_rows": int(len(predictions)),
        "native_temporal_unit_rows": int(len(units)),
        "metric_included_unit_rows": metric_include_count,
        "true_label_conflict_units": conflict_count,
        "oof_fold_conflict_units": fold_conflict_count,
        "duplicate_native_unit_rows": int(
            units.get(cfg.unit_id_col, pd.Series(dtype=str)).duplicated().sum()
        ),
        "ordered_window_id_sha256": _ordered_ids_hash(input_window_ids),
        "ordered_native_unit_id_sha256": _ordered_ids_hash(output_unit_ids),
        "unit_id_col": cfg.unit_id_col,
        "true_col": cfg.true_col,
        "pred_col": cfg.pred_col,
        "weight_col": cfg.weight_col,
        "valid_col": cfg.valid_col,
        "include_invalid_windows": bool(cfg.include_invalid_windows),
        "require_complete_probability_vector": bool(
            cfg.require_complete_probability_vector
        ),
        "require_oof_fold": bool(cfg.require_oof_fold),
        "errors": errors,
        "warnings": warnings,
        "valid": bool(
            not errors
            and conflict_count == 0
            and fold_conflict_count == 0
            and metric_include_count > 0
        ),
    }


def _bootstrap_confidence_intervals(
    metric_rows: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
) -> dict[str, Any]:
    """Compute deterministic bootstrap CIs over native temporal units.

    The resampling unit is the native temporal unit row. This keeps uncertainty
    aligned with the paper-facing statistical unit rather than overlapping
    training windows.
    """

    if metric_rows.empty or cfg.bootstrap_iterations <= 0:
        return {}
    metrics = [
        "accuracy",
        "macro_f1",
        "macro_f1_supported",
        "macro_recall_supported",
    ]
    estimates = evaluate_predictions(
        metric_rows,
        y_true_col=cfg.true_col,
        y_pred_col="native_predicted_behavior",
        label_order=list(cfg.label_order),
    )
    has_multiple_folds = (
        "oof_fold_id" in metric_rows.columns
        and metric_rows["oof_fold_id"].fillna("").nunique() >= 2
    )
    if has_multiple_folds:
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
        boot_metrics = evaluate_predictions(
            boot,
            y_true_col=cfg.true_col,
            y_pred_col="native_predicted_behavior",
            label_order=list(cfg.label_order),
        )
        for name in metrics:
            samples[name].append(float(boot_metrics.get(name, 0.0)))
    return samples


def _fold_cluster_bootstrap_samples(
    metric_rows: pd.DataFrame,
    cfg: NativeTemporalMetricsConfig,
) -> dict[str, list[float]]:
    """Resample complete held-out recording folds using precomputed confusion matrices."""

    labels = list(cfg.label_order)
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
    samples = {
        "accuracy": [],
        "macro_f1": [],
        "macro_f1_supported": [],
        "macro_recall_supported": [],
    }
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
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    supported = support > 0
    total = float(confusion.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total > 0.0 else 0.0,
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "macro_f1_supported": float(f1[supported].mean()) if supported.any() else 0.0,
        "macro_recall_supported": float(recall[supported].mean()) if supported.any() else 0.0,
    }


def _strict_bool_series(series: pd.Series, name: str) -> pd.Series:
    """Parse bool-like CSV values while rejecting blanks and unknown tokens."""

    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"invalid_{name}_bool_rows={int(series.isna().sum())}")
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    invalid = ~normalized.isin(true_values | false_values)
    if invalid.any():
        raise ValueError(f"invalid_{name}_bool_rows={int(invalid.sum())}")
    return normalized.isin(true_values)


def _ordered_ids_hash(values: list[str]) -> str:
    """Hash ordered identifiers for external row-preservation checks."""

    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()
