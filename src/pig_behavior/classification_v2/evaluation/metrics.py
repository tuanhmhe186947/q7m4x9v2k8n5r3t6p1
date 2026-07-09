"""Shared metrics for classification_v2 predictions."""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_LABEL_ORDER = [
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
]

FOCUS_PAIRS = [
    ("fight", "social-nose"),
    ("fight", "stand"),
    ("fight", "move"),
    ("eat", "stand"),
    ("eat", "explore"),
    ("drink", "stand"),
    ("drink", "explore"),
    ("playwithtoy", "explore"),
    ("playwithtoy", "stand"),
    ("playwithtoy", "move"),
    ("lying", "sitting"),
    ("move", "explore"),
    ("move", "stand"),
]


def evaluate_predictions(
    frame: pd.DataFrame,
    *,
    y_true_col: str,
    y_pred_col: str,
    label_order: list[str] | None = None,
) -> dict[str, Any]:
    """Compute deterministic multiclass classification metrics."""
    missing = [c for c in [y_true_col, y_pred_col] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    y_true = frame[y_true_col].fillna("").astype(str)
    y_pred = frame[y_pred_col].fillna("").astype(str)
    labels = _ordered_labels(y_true, y_pred, label_order)
    confusion = pd.crosstab(y_true, y_pred, dropna=False).reindex(index=labels, columns=labels, fill_value=0)
    per_class = _per_class_metrics(confusion, labels)
    macro_f1 = _mean([m["f1"] for m in per_class.values()])
    macro_precision = _mean([m["precision"] for m in per_class.values()])
    macro_recall = _mean([m["recall"] for m in per_class.values()])
    supported = [m for m in per_class.values() if int(m["support"]) > 0]
    accuracy = float((y_true.to_numpy() == y_pred.to_numpy()).mean()) if len(y_true) else 0.0
    return {
        "rows": int(len(frame)),
        "labels": labels,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_precision_supported": _mean([float(m["precision"]) for m in supported]),
        "macro_recall_supported": _mean([float(m["recall"]) for m in supported]),
        "macro_f1_supported": _mean([float(m["f1"]) for m in supported]),
        "supported_label_count": int(len(supported)),
        "per_class": per_class,
        "confusion_matrix": {
            "index": labels,
            "columns": labels,
            "values": confusion.astype(int).values.tolist(),
        },
        "focus_pair_confusions": _focus_pair_confusions(confusion),
    }


def evaluate_predictions_by_slice(
    frame: pd.DataFrame,
    *,
    y_true_col: str,
    y_pred_col: str,
    slice_col: str,
    label_order: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate predictions for each value of one slice column."""
    if slice_col not in frame.columns:
        raise ValueError(f"Missing slice column: {slice_col}")
    out: dict[str, Any] = {}
    for value, group in frame.groupby(slice_col, dropna=False, sort=True):
        out[str(value)] = evaluate_predictions(
            group,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            label_order=label_order,
        )
    return out


def _ordered_labels(y_true: pd.Series, y_pred: pd.Series, label_order: list[str] | None) -> list[str]:
    observed = set(y_true.tolist()).union(y_pred.tolist())
    ordered = list(label_order or DEFAULT_LABEL_ORDER)
    for label in sorted(observed):
        if label not in ordered:
            ordered.append(label)
    return ordered


def _per_class_metrics(confusion: pd.DataFrame, labels: list[str]) -> dict[str, dict[str, float | int]]:
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = int(confusion.loc[label, label]) if label in confusion.index and label in confusion.columns else 0
        support = int(confusion.loc[label].sum()) if label in confusion.index else 0
        predicted = int(confusion[label].sum()) if label in confusion.columns else 0
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return per_class


def _focus_pair_confusions(confusion: pd.DataFrame) -> dict[str, dict[str, int]]:
    focus: dict[str, dict[str, int]] = {}
    for a, b in FOCUS_PAIRS:
        a_to_b = int(confusion.loc[a, b]) if a in confusion.index and b in confusion.columns else 0
        b_to_a = int(confusion.loc[b, a]) if b in confusion.index and a in confusion.columns else 0
        focus[f"{a}__vs__{b}"] = {
            f"{a}_predicted_as_{b}": a_to_b,
            f"{b}_predicted_as_{a}": b_to_a,
            "total_pair_confusions": a_to_b + b_to_a,
        }
    return focus


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
