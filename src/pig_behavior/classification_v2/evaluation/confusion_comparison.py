"""Paired confusion-focused comparison at native units with fold-cluster uncertainty."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import (
    DEFAULT_LABEL_ORDER,
    FOCUS_PAIRS,
    evaluate_predictions,
)


def compare_confusion_focus(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    proposed_pred_col: str = "native_predicted_behavior",
    baseline_pred_col: str = "native_predicted_behavior",
    true_col: str = "behavior_true",
    unit_col: str = "temporal_unit_key",
    fold_col: str = "oof_fold_id",
    proposed_confidence_col: str = "calibrated_confidence",
    expected_fold_count: int | None = None,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260710,
    high_confidence_threshold: float = 0.7,
    sesoi_macro_f1: float = 0.02,
    paper_facing_inputs_verified: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare aligned model outputs and quantify predefined confusion-pair deltas."""

    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if not 0.0 <= high_confidence_threshold <= 1.0:
        raise ValueError("high_confidence_threshold must be in [0, 1]")
    if sesoi_macro_f1 <= 0.0:
        raise ValueError("sesoi_macro_f1 must be positive")
    paired = _align_predictions(
        proposed,
        baseline,
        proposed_pred_col=proposed_pred_col,
        baseline_pred_col=baseline_pred_col,
        proposed_confidence_col=proposed_confidence_col,
        true_col=true_col,
        unit_col=unit_col,
        fold_col=fold_col,
    )
    fold_count = int(paired[fold_col].nunique())
    errors: list[str] = []
    warnings: list[str] = []
    if expected_fold_count is None:
        warnings.append("expected_fold_count_not_declared_complete_oof_coverage_unproven")
    elif fold_count != int(expected_fold_count):
        errors.append(f"oof_fold_count_mismatch=expected:{expected_fold_count},observed:{fold_count}")
    if bootstrap_iterations < 1000:
        warnings.append("bootstrap_iterations_below_paper_gate_1000")
    if not paper_facing_inputs_verified:
        warnings.append("full_run_audits_not_verified_paper_facing_comparison_blocked")

    baseline_metrics = evaluate_predictions(
        paired,
        y_true_col=true_col,
        y_pred_col="baseline_pred",
    )
    proposed_metrics = evaluate_predictions(
        paired,
        y_true_col=true_col,
        y_pred_col="proposed_pred",
    )
    pair_table = _pair_metrics(paired, true_col=true_col)
    bootstrap = _cluster_bootstrap_deltas(
        paired,
        true_col=true_col,
        fold_col=fold_col,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    macro_delta = float(
        proposed_metrics["macro_f1_supported"] - baseline_metrics["macro_f1_supported"]
    )
    hard_errors = _hard_errors(
        paired,
        true_col=true_col,
        unit_col=unit_col,
        fold_col=fold_col,
        threshold=high_confidence_threshold,
    )
    complete_folds = expected_fold_count is not None and fold_count == int(expected_fold_count)
    report = {
        "schema_version": "classification_v2_confusion_comparison_v1",
        "statistical_unit": "native_temporal_unit",
        "uncertainty_resample_unit": "oof_fold_id",
        "paired_native_unit_rows": int(len(paired)),
        "paired_unit_ids_sha256": _ids_hash(paired[unit_col]),
        "oof_fold_count": fold_count,
        "expected_fold_count": expected_fold_count,
        "complete_oof_fold_coverage": bool(complete_folds),
        "baseline_metrics": baseline_metrics,
        "proposed_metrics": proposed_metrics,
        "macro_f1_supported_delta": macro_delta,
        "macro_f1_supported_delta_ci": bootstrap["macro_f1_supported_delta"],
        "sesoi": {
            "metric": "macro_f1_supported_delta",
            "minimum_effect_size": float(sesoi_macro_f1),
            "point_estimate_exceeds_sesoi": bool(macro_delta >= sesoi_macro_f1),
            "ci_low_exceeds_zero": bool(bootstrap["macro_f1_supported_delta"]["ci_low"] > 0.0),
        },
        "focus_pairs": pair_table,
        "focus_pair_delta_confidence_intervals": bootstrap["focus_pair_error_rate_delta"],
        "high_confidence_threshold": float(high_confidence_threshold),
        "high_confidence_hard_error_rows": int(len(hard_errors)),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "paper_facing_inputs_verified": bool(paper_facing_inputs_verified),
        "paper_facing_ready": bool(
            complete_folds and bootstrap_iterations >= 1000 and paper_facing_inputs_verified and not errors
        ),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    return report, hard_errors


def _align_predictions(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    proposed_pred_col: str,
    baseline_pred_col: str,
    proposed_confidence_col: str,
    true_col: str,
    unit_col: str,
    fold_col: str,
) -> pd.DataFrame:
    """Create a strict one-to-one paired table; mismatched lineage is a hard failure."""

    proposed_required = [unit_col, fold_col, true_col, proposed_pred_col]
    baseline_required = [unit_col, fold_col, true_col, baseline_pred_col]
    missing_proposed = [column for column in proposed_required if column not in proposed.columns]
    missing_baseline = [column for column in baseline_required if column not in baseline.columns]
    if missing_proposed or missing_baseline:
        raise ValueError(
            f"missing comparison columns: proposed={missing_proposed}, baseline={missing_baseline}"
        )
    if proposed[unit_col].duplicated().any() or baseline[unit_col].duplicated().any():
        raise ValueError("comparison inputs require one row per native temporal unit")
    proposed_columns = proposed_required + (
        [proposed_confidence_col] if proposed_confidence_col in proposed.columns else []
    )
    left = proposed[proposed_columns].rename(
        columns={
            fold_col: f"{fold_col}_proposed",
            true_col: f"{true_col}_proposed",
            proposed_pred_col: "proposed_pred",
            proposed_confidence_col: "proposed_confidence",
        }
    )
    right = baseline[baseline_required].rename(
        columns={
            fold_col: f"{fold_col}_baseline",
            true_col: f"{true_col}_baseline",
            baseline_pred_col: "baseline_pred",
        }
    )
    paired = left.merge(right, on=unit_col, how="outer", indicator=True, validate="one_to_one")
    if not paired["_merge"].eq("both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"native unit alignment mismatch: {counts}")
    if not paired[f"{true_col}_proposed"].astype(str).eq(paired[f"{true_col}_baseline"].astype(str)).all():
        raise ValueError("true label mismatch between proposed and baseline")
    if not paired[f"{fold_col}_proposed"].astype(str).eq(paired[f"{fold_col}_baseline"].astype(str)).all():
        raise ValueError("OOF fold mismatch between proposed and baseline")
    paired[true_col] = paired[f"{true_col}_proposed"].astype(str)
    paired[fold_col] = paired[f"{fold_col}_proposed"].astype(str)
    paired["proposed_confidence"] = pd.to_numeric(
        paired.get("proposed_confidence", pd.Series(np.nan, index=paired.index)), errors="coerce"
    )
    return paired.sort_values([fold_col, unit_col], kind="mergesort").reset_index(drop=True)


def _pair_metrics(paired: pd.DataFrame, *, true_col: str) -> dict[str, dict[str, float | int]]:
    """Report directional and total error rates for every predeclared confusion pair."""

    result: dict[str, dict[str, float | int]] = {}
    for first, second in FOCUS_PAIRS:
        key = f"{first}__vs__{second}"
        first_support = int(paired[true_col].eq(first).sum())
        second_support = int(paired[true_col].eq(second).sum())
        pair_support = first_support + second_support
        baseline_errors = int(
            ((paired[true_col].eq(first) & paired["baseline_pred"].eq(second))
             | (paired[true_col].eq(second) & paired["baseline_pred"].eq(first))).sum()
        )
        proposed_errors = int(
            ((paired[true_col].eq(first) & paired["proposed_pred"].eq(second))
             | (paired[true_col].eq(second) & paired["proposed_pred"].eq(first))).sum()
        )
        baseline_rate = float(baseline_errors / pair_support) if pair_support else 0.0
        proposed_rate = float(proposed_errors / pair_support) if pair_support else 0.0
        result[key] = {
            "first_label_support": first_support,
            "second_label_support": second_support,
            "pair_support": pair_support,
            "baseline_pair_errors": baseline_errors,
            "proposed_pair_errors": proposed_errors,
            "baseline_pair_error_rate": baseline_rate,
            "proposed_pair_error_rate": proposed_rate,
            "proposed_minus_baseline_error_rate": proposed_rate - baseline_rate,
        }
    return result


def _cluster_bootstrap_deltas(
    paired: pd.DataFrame,
    *,
    true_col: str,
    fold_col: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap complete held-out folds to preserve within-session dependence."""

    fold_ids = sorted(paired[fold_col].unique())
    if len(fold_ids) < 2:
        raise ValueError("cluster bootstrap requires at least two OOF folds")
    labels = list(DEFAULT_LABEL_ORDER)
    label_to_index = {label: index for index, label in enumerate(labels)}
    observed = set(paired[true_col]).union(paired["baseline_pred"]).union(paired["proposed_pred"])
    invalid = sorted(observed - set(labels))
    if invalid:
        raise ValueError(f"unsupported labels in confusion bootstrap: {invalid}")
    baseline_by_fold = np.stack(
        [
            _confusion_array(
                paired.loc[paired[fold_col].eq(fold_id)], true_col, "baseline_pred", labels
            )
            for fold_id in fold_ids
        ]
    )
    proposed_by_fold = np.stack(
        [
            _confusion_array(
                paired.loc[paired[fold_col].eq(fold_id)], true_col, "proposed_pred", labels
            )
            for fold_id in fold_ids
        ]
    )
    rng = np.random.default_rng(int(seed))
    macro_deltas: list[float] = []
    pair_deltas: dict[str, list[float]] = {f"{a}__vs__{b}": [] for a, b in FOCUS_PAIRS}
    for _ in range(int(iterations)):
        multiplicity = np.bincount(
            rng.integers(0, len(fold_ids), size=len(fold_ids)), minlength=len(fold_ids)
        )
        baseline_confusion = np.tensordot(multiplicity, baseline_by_fold, axes=(0, 0))
        proposed_confusion = np.tensordot(multiplicity, proposed_by_fold, axes=(0, 0))
        macro_deltas.append(
            _macro_f1_supported(proposed_confusion) - _macro_f1_supported(baseline_confusion)
        )
        for first, second in FOCUS_PAIRS:
            key = f"{first}__vs__{second}"
            first_idx, second_idx = label_to_index[first], label_to_index[second]
            support = int(
                baseline_confusion[first_idx].sum() + baseline_confusion[second_idx].sum()
            )
            baseline_errors = int(
                baseline_confusion[first_idx, second_idx] + baseline_confusion[second_idx, first_idx]
            )
            proposed_errors = int(
                proposed_confusion[first_idx, second_idx] + proposed_confusion[second_idx, first_idx]
            )
            delta = float((proposed_errors - baseline_errors) / support) if support else 0.0
            pair_deltas[key].append(delta)
    return {
        "macro_f1_supported_delta": _interval(macro_deltas, iterations),
        "focus_pair_error_rate_delta": {
            key: _interval(values, iterations) for key, values in pair_deltas.items()
        },
    }


def _confusion_array(frame: pd.DataFrame, true_col: str, pred_col: str, labels: list[str]) -> np.ndarray:
    """Build one fixed-order fold confusion matrix for fast cluster resampling."""

    return (
        pd.crosstab(frame[true_col], frame[pred_col], dropna=False)
        .reindex(index=labels, columns=labels, fill_value=0)
        .to_numpy(dtype=np.int64)
    )


def _macro_f1_supported(confusion: np.ndarray) -> float:
    """Compute macro F1 over classes supported in a resampled fold set."""

    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2.0 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return float(f1[support > 0].mean()) if (support > 0).any() else 0.0


def _hard_errors(
    paired: pd.DataFrame,
    *,
    true_col: str,
    unit_col: str,
    fold_col: str,
    threshold: float,
) -> pd.DataFrame:
    """Return high-confidence proposed errors restricted to declared confusion pairs."""

    pair_lookup = {frozenset(pair): f"{pair[0]}__vs__{pair[1]}" for pair in FOCUS_PAIRS}
    rows: list[dict[str, Any]] = []
    for _, row in paired.iterrows():
        pair_key = pair_lookup.get(frozenset((str(row[true_col]), str(row["proposed_pred"]))))
        confidence = float(row["proposed_confidence"]) if pd.notna(row["proposed_confidence"]) else np.nan
        if pair_key and np.isfinite(confidence) and confidence >= threshold:
            rows.append(
                {
                    unit_col: str(row[unit_col]),
                    fold_col: str(row[fold_col]),
                    true_col: str(row[true_col]),
                    "baseline_pred": str(row["baseline_pred"]),
                    "proposed_pred": str(row["proposed_pred"]),
                    "proposed_confidence": confidence,
                    "focus_pair": pair_key,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["proposed_confidence", fold_col, unit_col], ascending=[False, True, True], kind="mergesort"
    ) if rows else pd.DataFrame(
        columns=[unit_col, fold_col, true_col, "baseline_pred", "proposed_pred", "proposed_confidence", "focus_pair"]
    )


def _interval(values: list[float], iterations: int) -> dict[str, float | int | str]:
    """Build a deterministic percentile interval for paired deltas."""

    return {
        "estimate_bootstrap_mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "method": "oof_fold_cluster_bootstrap_percentile",
        "n_bootstrap": int(iterations),
    }


def _ids_hash(values: pd.Series) -> str:
    """Hash paired unit identities so comparison lineage is externally checkable."""

    return hashlib.sha256("\n".join(sorted(values.astype(str))).encode("utf-8")).hexdigest()
