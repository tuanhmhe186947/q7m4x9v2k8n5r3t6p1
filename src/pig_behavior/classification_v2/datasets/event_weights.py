"""Event-overlap weighting for classification_v2 sequence windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class EventWeightTables:
    weights: pd.DataFrame
    audit: dict[str, Any]


def build_event_weight_manifest(
    windows: pd.DataFrame,
    *,
    window_id_col: str = "window_id",
    event_key_col: str = "temporal_unit_keys_window",
    valid_col: str = "window_valid_for_main_train",
    base_weight_col: str = "window_sample_weight",
) -> EventWeightTables:
    """Build inverse-window-count weights for overlapping temporal events."""
    required = [window_id_col, event_key_col, valid_col]
    missing = [c for c in required if c not in windows.columns]
    if missing:
        raise ValueError(f"Missing event-weight input columns: {missing}")

    work = windows[[window_id_col, event_key_col, valid_col]].copy()
    if base_weight_col in windows.columns:
        work[base_weight_col] = pd.to_numeric(windows[base_weight_col], errors="coerce").fillna(1.0)
    else:
        work[base_weight_col] = 1.0

    work["event_overlap_cluster_id"] = work[event_key_col].fillna("").astype(str)
    missing_event_key = work["event_overlap_cluster_id"].str.strip().eq("")
    work.loc[missing_event_key, "event_overlap_cluster_id"] = (
        "missing_event_key|" + work.loc[missing_event_key, window_id_col].astype(str)
    )
    work["window_valid_for_event_weight"] = _as_bool(work[valid_col])

    total_counts = work.groupby("event_overlap_cluster_id", sort=False)[window_id_col].transform("size")
    valid_counts = (
        work.groupby("event_overlap_cluster_id", sort=False)["window_valid_for_event_weight"]
        .transform("sum")
        .astype(int)
    )
    denominator = valid_counts.where(valid_counts > 0, total_counts).clip(lower=1)
    work["windows_per_event"] = total_counts.astype(int)
    work["valid_windows_per_event"] = valid_counts.astype(int)
    work["inverse_windows_per_event"] = 1.0 / denominator.astype(float)
    work["event_balanced_sample_weight"] = (
        work[base_weight_col].clip(lower=0.0) * work["inverse_windows_per_event"]
    )
    work.loc[~work["window_valid_for_event_weight"], "event_balanced_sample_weight"] = 0.0

    out_cols = [
        window_id_col,
        event_key_col,
        "event_overlap_cluster_id",
        "windows_per_event",
        "valid_windows_per_event",
        base_weight_col,
        "inverse_windows_per_event",
        "event_balanced_sample_weight",
        "window_valid_for_event_weight",
    ]
    weights = work[out_cols].copy()
    duplicate_window_ids = int(weights[window_id_col].duplicated().sum())
    audit = {
        "rows": int(len(weights)),
        "unique_window_ids": int(weights[window_id_col].nunique(dropna=False)),
        "duplicate_window_id": duplicate_window_ids,
        "event_overlap_cluster_count": int(weights["event_overlap_cluster_id"].nunique(dropna=False)),
        "missing_event_key_rows": int(missing_event_key.sum()),
        "max_windows_per_event": int(weights["windows_per_event"].max()) if len(weights) else 0,
        "mean_windows_per_event": float(weights["windows_per_event"].mean()) if len(weights) else 0.0,
        "event_balanced_weight_sum": float(weights["event_balanced_sample_weight"].sum()),
        "base_weight_sum": float(weights[base_weight_col].sum()),
        "invalid_weight_zero_count": int(
            ((~weights["window_valid_for_event_weight"]) & weights["event_balanced_sample_weight"].eq(0.0)).sum()
        ),
        "warnings": ["event_balanced_sample_weight is for training augmentation, not independent test sample size"],
        "errors": [] if duplicate_window_ids == 0 else [f"duplicate_window_id={duplicate_window_ids}"],
    }
    return EventWeightTables(weights=weights, audit=audit)


def json_default(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
