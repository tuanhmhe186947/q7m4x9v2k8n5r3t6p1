"""Cross-fitted probability calibration for leakage-safe native-unit evaluation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def cross_fit_temperature_scaling(
    native_predictions: pd.DataFrame,
    *,
    true_col: str = "behavior_true",
    fold_col: str = "oof_fold_id",
    unit_col: str = "temporal_unit_key",
    prob_prefix: str = "prob_",
    calibrated_prefix: str = "cal_prob_",
    ece_bins: int = 15,
    expected_fold_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit temperature on non-evaluation folds and calibrate each held-out fold once."""

    labels = [column[len(prob_prefix) :] for column in native_predictions.columns if column.startswith(prob_prefix)]
    required = [true_col, fold_col, unit_col]
    missing = [column for column in required if column not in native_predictions.columns]
    if missing or len(labels) < 2:
        raise ValueError(f"calibration input contract failed: missing={missing}, probability_labels={labels}")
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least 2")
    if expected_fold_count is not None and expected_fold_count < 2:
        raise ValueError("expected_fold_count must be at least 2")

    frame = native_predictions.copy()
    unit_ids = frame[unit_col].fillna("").astype(str)
    folds = frame[fold_col].fillna("").astype(str)
    if unit_ids.eq("").any() or unit_ids.duplicated().any():
        raise ValueError("native calibration rows require unique non-empty temporal unit keys")
    if folds.eq("").any() or folds.nunique() < 2:
        raise ValueError("cross-fitted calibration requires at least two non-empty OOF folds")
    if "oof_fold_conflict" in frame.columns and _as_bool(frame["oof_fold_conflict"]).any():
        raise ValueError("native calibration rows contain OOF fold conflicts")

    prob_cols = [f"{prob_prefix}{label}" for label in labels]
    probabilities = frame[prob_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    probabilities = _validated_probabilities(probabilities)
    true_labels = frame[true_col].fillna("").astype(str).to_numpy()
    label_to_index = {label: index for index, label in enumerate(labels)}
    invalid_labels = sorted(set(true_labels) - set(labels))
    if invalid_labels:
        raise ValueError(f"true labels missing probability columns: {invalid_labels}")
    targets = np.asarray([label_to_index[label] for label in true_labels], dtype=np.int64)

    calibrated = np.zeros_like(probabilities)
    fold_audits: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for fold_id in sorted(folds.unique()):
        eval_mask = folds.eq(fold_id).to_numpy()
        calibration_mask = ~eval_mask
        temperature = _fit_temperature(probabilities[calibration_mask], targets[calibration_mask])
        calibrated[eval_mask] = _apply_temperature(probabilities[eval_mask], temperature)
        calibration_label_counts = pd.Series(true_labels[calibration_mask]).value_counts().reindex(
            labels, fill_value=0
        )
        missing_calibration_labels = calibration_label_counts[calibration_label_counts.eq(0)].index.tolist()
        if missing_calibration_labels:
            warnings.append(f"calibration_fold_missing_labels={fold_id}:{missing_calibration_labels}")
        if temperature <= 0.051 or temperature >= 19.9:
            warnings.append(f"temperature_at_search_boundary={fold_id}:{temperature}")
        fold_audits.append(
            {
                "oof_fold_id": str(fold_id),
                "calibration_rows": int(calibration_mask.sum()),
                "evaluation_rows": int(eval_mask.sum()),
                "temperature": float(temperature),
                "calibration_label_counts": {
                    str(label): int(count) for label, count in calibration_label_counts.items()
                },
                "calibration_unit_ids_sha256": _ids_hash(unit_ids[calibration_mask]),
                "evaluation_unit_ids_sha256": _ids_hash(unit_ids[eval_mask]),
                "fold_excluded_from_temperature_fit": True,
            }
        )

    for index, label in enumerate(labels):
        frame[f"{calibrated_prefix}{label}"] = calibrated[:, index]
    frame["behavior_pred_calibrated"] = np.asarray(labels)[np.argmax(calibrated, axis=1)]
    frame["calibrated_confidence"] = calibrated.max(axis=1)
    before = probability_calibration_metrics(probabilities, targets, ece_bins=ece_bins)
    after = probability_calibration_metrics(calibrated, targets, ece_bins=ece_bins)
    observed_fold_count = int(folds.nunique())
    if expected_fold_count is None:
        warnings.append("expected_fold_count_not_declared_complete_oof_coverage_unproven")
    elif observed_fold_count != int(expected_fold_count):
        errors.append(f"oof_fold_count_mismatch=expected:{expected_fold_count},observed:{observed_fold_count}")
    audit = {
        "schema_version": "classification_v2_cross_fitted_calibration_v1",
        "statistical_unit": "native_temporal_unit",
        "method": "cross_fitted_scalar_temperature_scaling",
        "labels": labels,
        "native_unit_rows": int(len(frame)),
        "oof_fold_count": observed_fold_count,
        "expected_fold_count": expected_fold_count,
        "complete_oof_fold_coverage": bool(
            expected_fold_count is not None and observed_fold_count == expected_fold_count
        ),
        "ece_bins": int(ece_bins),
        "metrics_before": before,
        "metrics_after": after,
        "fold_audits": fold_audits,
        "errors": errors,
        "warnings": warnings
        + ["Calibration quality is claimable only on complete OOF predictions, not bounded pilots."],
        "valid": not errors,
    }
    return frame, audit


def probability_calibration_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    ece_bins: int,
) -> dict[str, float]:
    """Compute proper scoring rules and fixed-bin top-label calibration error."""

    probabilities = _validated_probabilities(np.asarray(probabilities, dtype=np.float64))
    targets = np.asarray(targets, dtype=np.int64)
    if len(probabilities) != len(targets) or not len(targets):
        raise ValueError("probabilities and targets must have matching non-zero rows")
    true_probability = probabilities[np.arange(len(targets)), targets]
    nll = float(-np.log(np.clip(true_probability, 1e-12, 1.0)).mean())
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    confidence = probabilities.max(axis=1)
    correct = np.argmax(probabilities, axis=1) == targets
    edges = np.linspace(0.0, 1.0, int(ece_bins) + 1)
    bin_index = np.minimum(np.digitize(confidence, edges[1:-1], right=True), int(ece_bins) - 1)
    ece = 0.0
    for index in range(int(ece_bins)):
        mask = bin_index == index
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {"negative_log_likelihood": nll, "multiclass_brier": brier, "top_label_ece": float(ece)}


def _fit_temperature(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Deterministically minimize NLL with bounded golden search in log-temperature."""

    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    low, high = float(np.log(0.05)), float(np.log(20.0))
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    left_loss = _temperature_nll(log_probabilities, targets, left)
    right_loss = _temperature_nll(log_probabilities, targets, right)
    for _ in range(30):
        if left_loss <= right_loss:
            high, right, right_loss = right, left, left_loss
            left = high - ratio * (high - low)
            left_loss = _temperature_nll(log_probabilities, targets, left)
        else:
            low, left, left_loss = left, right, right_loss
            right = low + ratio * (high - low)
            right_loss = _temperature_nll(log_probabilities, targets, right)
    return float(np.exp((low + high) / 2.0))


def _temperature_nll(log_probabilities: np.ndarray, targets: np.ndarray, log_temperature: float) -> float:
    """Evaluate mean NLL without computing calibration metrics inside the optimizer."""

    scaled = log_probabilities / float(np.exp(log_temperature))
    row_max = scaled.max(axis=1, keepdims=True)
    log_normalizer = row_max[:, 0] + np.log(np.exp(scaled - row_max).sum(axis=1))
    true_logits = scaled[np.arange(len(targets)), targets]
    return float((log_normalizer - true_logits).mean())


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature to probabilities through stable log-softmax."""

    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _validated_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Validate and normalize finite non-negative class probabilities."""

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional multiclass matrix")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError("probability rows must have positive mass")
    return probabilities / row_sums


def _ids_hash(values: pd.Series) -> str:
    """Hash sorted native-unit identities for fold lineage audits."""

    payload = "\n".join(sorted(values.astype(str).tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_bool(values: pd.Series) -> pd.Series:
    """Normalize CSV bool-like values for fold-conflict validation."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
