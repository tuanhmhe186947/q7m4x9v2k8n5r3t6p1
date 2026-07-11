"""Source/domain control views that preserve every classification_v2 window."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_source_matched_views(windows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark source and matched-length controls without dropping or relabeling rows."""

    required = {
        "window_id",
        "source_type",
        "behavior_window_label",
        "window_length_frames",
        "window_valid_for_main_train",
    }
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError(f"source-matched view input missing columns: {missing}")
    if windows["window_id"].duplicated().any():
        raise ValueError("duplicate window_id in source-matched view input")
    result = windows.copy()
    valid = _to_bool(result["window_valid_for_main_train"])
    source = result["source_type"].astype(str)
    length = pd.to_numeric(result["window_length_frames"], errors="coerce")
    result["view_combined"] = valid
    result["view_cvat_only"] = valid & source.eq("cvat_tracking_xml")
    result["view_legacy_only"] = valid & source.eq("legacy_recovered")
    result["view_matched_6frame"] = valid & length.eq(6)
    result["source_class_balance_keep"] = False
    result["source_class_balance_rank"] = 0
    result["source_class_balance_quota"] = 0
    candidates = result.loc[result["view_matched_6frame"]].copy()
    sources = sorted(candidates["source_type"].astype(str).unique())
    quotas: dict[str, int] = {}
    counts = candidates.groupby(["behavior_window_label", "source_type"])["window_id"].count()
    for label in sorted(candidates["behavior_window_label"].astype(str).unique()):
        values = [int(counts.get((label, source_name), 0)) for source_name in sources]
        quotas[label] = min(values) if len(sources) >= 2 and all(value > 0 for value in values) else 0
    ordered = candidates.sort_values(
        ["behavior_window_label", "source_type", "window_id"], kind="mergesort"
    )
    ranks = ordered.groupby(["behavior_window_label", "source_type"]).cumcount() + 1
    result.loc[ordered.index, "source_class_balance_rank"] = ranks.astype(int)
    result["source_class_balance_quota"] = (
        result["behavior_window_label"].astype(str).map(quotas).fillna(0).astype(int)
    )
    result["source_class_balance_keep"] = (
        result["view_matched_6frame"]
        & result["source_class_balance_quota"].gt(0)
        & result["source_class_balance_rank"].le(result["source_class_balance_quota"])
    )
    result["source_control_exclusion_reason"] = "not_valid_for_main_train"
    result.loc[valid, "source_control_exclusion_reason"] = "valid_not_in_matched_6frame"
    result.loc[result["view_matched_6frame"], "source_control_exclusion_reason"] = (
        "matched_6frame_above_source_class_quota"
    )
    result.loc[result["source_class_balance_keep"], "source_control_exclusion_reason"] = (
        "source_class_matched_keep"
    )
    audit = {
        "schema_version": "classification_v2_source_matched_views_v1",
        "rows_input": int(len(windows)),
        "rows_output": int(len(result)),
        "duplicate_window_id": int(result["window_id"].duplicated().sum()),
        "source_counts": source.value_counts().sort_index().to_dict(),
        "view_counts": {
            column: int(_to_bool(result[column]).sum())
            for column in [
                "view_combined",
                "view_cvat_only",
                "view_legacy_only",
                "view_matched_6frame",
                "source_class_balance_keep",
            ]
        },
        "matched_6frame_source_behavior_counts": _contingency(
            result.loc[result["view_matched_6frame"]]
        ),
        "source_class_balanced_counts": _contingency(
            result.loc[result["source_class_balance_keep"]]
        ),
        "rows_dropped": 0,
        "labels_changed": 0,
        "errors": [],
        "valid": len(result) == len(windows) and not result["window_id"].duplicated().any(),
    }
    return result, audit


def _contingency(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    table = pd.crosstab(frame["source_type"], frame["behavior_window_label"])
    return {
        label: {source: int(count) for source, count in values.items()}
        for label, values in table.to_dict().items()
    }


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
