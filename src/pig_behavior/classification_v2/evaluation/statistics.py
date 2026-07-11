"""Cluster-aware paired uncertainty and multiplicity control for Q2 comparisons."""

from __future__ import annotations

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
    true_col: str = "true_label",
    pred_col: str = "native_predicted_behavior",
    iterations: int = 2000,
    seed: int = 20260710,
) -> dict[str, Any]:
    """Bootstrap paired model deltas by recording date, never overlapping windows."""

    required = [cluster_col, unit_col, true_col, pred_col]
    for name, frame in [("candidate", candidate), ("baseline", baseline)]:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} comparison input missing columns: {missing}")
        if frame[unit_col].duplicated().any():
            raise ValueError(f"{name} contains duplicate native units")
    left = candidate[required].rename(columns={pred_col: "candidate_pred"})
    right = baseline[[unit_col, true_col, pred_col]].rename(
        columns={true_col: "baseline_true", pred_col: "baseline_pred"}
    )
    paired = left.merge(right, on=unit_col, how="inner", validate="one_to_one")
    if len(paired) != len(candidate) or len(paired) != len(baseline):
        raise ValueError("candidate and baseline native-unit sets are not identical")
    if paired[true_col].astype(str).ne(paired["baseline_true"].astype(str)).any():
        raise ValueError("candidate and baseline true labels disagree")
    clusters = sorted(paired[cluster_col].astype(str).unique())
    if len(clusters) < 2:
        raise ValueError("cluster bootstrap requires at least two recording groups")
    observed = _macro_f1(paired, true_col, "candidate_pred") - _macro_f1(
        paired, true_col, "baseline_pred"
    )
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    grouped = {cluster: paired.loc[paired[cluster_col].astype(str).eq(cluster)] for cluster in clusters}
    for index in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[cluster] for cluster in sampled], ignore_index=True)
        deltas[index] = _macro_f1(sample, true_col, "candidate_pred") - _macro_f1(
            sample, true_col, "baseline_pred"
        )
    return {
        "schema_version": "classification_v2_paired_cluster_bootstrap_v1",
        "cluster_unit": cluster_col,
        "paired_native_units": int(len(paired)),
        "cluster_count": len(clusters),
        "iterations": iterations,
        "seed": seed,
        "macro_f1_delta": float(observed),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "two_sided_bootstrap_p": float(
            min(
                1.0,
                2.0
                * min(
                    (float(np.count_nonzero(deltas <= 0)) + 1.0) / (iterations + 1.0),
                    (float(np.count_nonzero(deltas >= 0)) + 1.0) / (iterations + 1.0),
                ),
            )
        ),
        "effect_size_definition": "candidate_minus_baseline_fixed_10_class_macro_f1",
    }


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
