"""Training-fold-only weighting policies without duplicating sequence windows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

WEIGHT_POLICIES = ("uniform", "class_balanced", "event_balanced", "source_class_balanced")


def build_training_weights(
    training_rows: pd.DataFrame,
    *,
    policy: str,
    label_col: str = "behavior_true",
    source_col: str = "source_type",
    event_col: str = "temporal_unit_key",
    max_weight: float = 10.0,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fit a declared weighting policy using training rows only and preserve row order."""

    if policy not in WEIGHT_POLICIES:
        raise ValueError(f"unsupported training weight policy={policy}")
    required = [label_col]
    if policy == "event_balanced":
        required.append(event_col)
    if policy == "source_class_balanced":
        required.append(source_col)
    missing = [column for column in required if column not in training_rows.columns]
    if missing:
        raise ValueError(f"training weight input missing columns: {missing}")
    if len(training_rows) == 0:
        raise ValueError("cannot fit training weights on zero rows")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    weights = pd.Series(1.0, index=training_rows.index, dtype=float)
    if policy == "class_balanced":
        counts = training_rows[label_col].astype(str).value_counts()
        weights = training_rows[label_col].astype(str).map(1.0 / counts)
    elif policy == "event_balanced":
        counts = training_rows[event_col].astype(str).value_counts()
        weights = training_rows[event_col].astype(str).map(1.0 / counts)
    elif policy == "source_class_balanced":
        keys = pd.MultiIndex.from_frame(training_rows[[source_col, label_col]].astype(str))
        counts = pd.Series(list(keys), dtype=object).value_counts()
        weights = pd.Series([1.0 / counts[key] for key in keys], index=training_rows.index)
    weights = weights.astype(float)
    weights *= 1.0 / float(weights.mean())
    weights = weights.clip(upper=float(max_weight))
    weights *= 1.0 / float(weights.mean())
    effective_sample_size = float(weights.sum() ** 2 / np.square(weights).sum())
    audit = {
        "schema_version": "classification_v2_training_weight_policy_v1",
        "policy": policy,
        "fit_scope": "training_fold_only",
        "rows": int(len(training_rows)),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "effective_sample_size": effective_sample_size,
        "row_duplication_used": False,
        "source_is_model_input": False,
        "errors": [],
        "valid": True,
    }
    return weights, audit
