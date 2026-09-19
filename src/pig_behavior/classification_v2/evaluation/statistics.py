"""Cluster-aware paired uncertainty and multiplicity control for Q2 comparisons."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def paired_cluster_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    cluster_col: str = "recording_group_id",
    unit_col: str = "temporal_unit_key",
    fold_col: str = "outer_fold_id",
    true_col: str = "true_label",
    pred_col: str = "native_predicted_behavior",
    iterations: int = 2000,
    seed: int = 20260710,
    outer_predictions_used_for_model_selection: bool = False,
) -> dict[str, Any]:
    """Bootstrap paired model deltas by recording date, never overlapping windows."""

    if iterations <= 0:
        raise ValueError("paired bootstrap iterations must be positive")
    if outer_predictions_used_for_model_selection:
        raise ValueError("outer OOF predictions cannot select or tune a model")
    required = [cluster_col, unit_col, fold_col, true_col, pred_col]
    for name, frame in [("candidate", candidate), ("baseline", baseline)]:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} comparison input missing columns: {missing}")
        if frame[unit_col].duplicated().any():
            raise ValueError(f"{name} contains duplicate native units")
        _validate_comparison_rows(
            frame,
            name=name,
            cluster_col=cluster_col,
            unit_col=unit_col,
            fold_col=fold_col,
            true_col=true_col,
            pred_col=pred_col,
        )
    left = candidate[required].rename(columns={pred_col: "candidate_pred"})
    right = baseline[required].rename(
        columns={
            cluster_col: "baseline_cluster",
            fold_col: "baseline_fold",
            true_col: "baseline_true",
            pred_col: "baseline_pred",
        }
    )
    paired = left.merge(
        right,
        on=unit_col,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        counts = paired["_merge"].value_counts().to_dict()
        raise ValueError(f"candidate and baseline native-unit sets differ: {counts}")
    if paired[true_col].astype(str).ne(paired["baseline_true"].astype(str)).any():
        raise ValueError("candidate and baseline true labels disagree")
    if paired[cluster_col].astype(str).ne(paired["baseline_cluster"].astype(str)).any():
        raise ValueError("candidate and baseline recording clusters disagree")
    if paired[fold_col].astype(str).ne(paired["baseline_fold"].astype(str)).any():
        raise ValueError("candidate and baseline outer folds disagree")
    paired = paired.drop(columns=["_merge"])
    clusters = sorted(paired[cluster_col].astype(str).unique())
    if len(clusters) < 2:
        raise ValueError("cluster bootstrap requires at least two recording groups")
    observed = _macro_f1(paired, true_col, "candidate_pred") - _macro_f1(
        paired, true_col, "baseline_pred"
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    grouped = {
        cluster: paired.loc[paired[cluster_col].astype(str).eq(cluster)]
        for cluster in clusters
    }
    for index in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[cluster] for cluster in sampled], ignore_index=True)
        deltas[index] = _macro_f1(sample, true_col, "candidate_pred") - _macro_f1(
            sample, true_col, "baseline_pred"
        )
    return {
        "schema_version": "classification_v2_paired_cluster_bootstrap_v2",
        "cluster_unit": cluster_col,
        "fold_unit": fold_col,
        "paired_native_units": int(len(paired)),
        "paired_unit_ids_sha256": _mapping_hash(paired, [unit_col]),
        "paired_fold_mapping_sha256": _mapping_hash(
            paired,
            [unit_col, cluster_col, fold_col, true_col],
        ),
        "cluster_count": len(clusters),
        "iterations": iterations,
        "seed": seed,
        "macro_f1_delta": float(observed),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "uncertainty_method": "paired_recording_cluster_bootstrap_percentile",
        "two_sided_bootstrap_p": None,
        "p_value_status": (
            "not_reported_percentile_bootstrap_is_not_a_null_test"
        ),
        "bootstrap_fraction_delta_le_zero": float(np.mean(deltas <= 0.0)),
        "bootstrap_fraction_delta_ge_zero": float(np.mean(deltas >= 0.0)),
        "effect_size_definition": "candidate_minus_baseline_fixed_10_class_macro_f1",
        "outer_predictions_used_for_model_selection": False,
    }


def _validate_comparison_rows(
    frame: pd.DataFrame,
    *,
    name: str,
    cluster_col: str,
    unit_col: str,
    fold_col: str,
    true_col: str,
    pred_col: str,
) -> None:
    """Reject blank lineage, unsupported labels, and split-group conflicts."""

    for column in [cluster_col, unit_col, fold_col, true_col, pred_col]:
        blank = frame[column].fillna("").astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(
                f"{name} contains blank {column} rows={int(blank.sum())}"
            )
    labels = set(VALID_BEHAVIORS)
    invalid_true = sorted(set(frame[true_col].astype(str)) - labels)
    invalid_pred = sorted(set(frame[pred_col].astype(str)) - labels)
    if invalid_true or invalid_pred:
        raise ValueError(
            f"{name} contains unsupported labels: "
            f"true={invalid_true},pred={invalid_pred}"
        )
    cluster_folds = frame.groupby(cluster_col)[fold_col].nunique()
    if cluster_folds.gt(1).any():
        raise ValueError(
            f"{name} recording clusters cross outer folds="
            f"{int(cluster_folds.gt(1).sum())}"
        )


def _mapping_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash a sorted paired lineage mapping for manifest equality evidence."""

    rows = frame[columns].astype(str).sort_values(columns, kind="mergesort")
    payload = "\n".join("\x1f".join(row) for row in rows.itertuples(index=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Apply deterministic Holm family-wise error correction to ablation p-values."""

    ordered = sorted(p_values.items(), key=lambda item: (float(item[1]), item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in sorted(adjusted)}


def _macro_f1(frame: pd.DataFrame, true_col: str, pred_col: str) -> float:
    return float(
        evaluate_predictions(
            frame,
            y_true_col=true_col,
            y_pred_col=pred_col,
            label_order=list(VALID_BEHAVIORS),
        )["macro_f1"]
    )
