"""Collapse window-level predictions to native temporal-unit predictions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import DEFAULT_LABEL_ORDER, evaluate_predictions

SOURCE_MARKER_RE = re.compile(r"(?=(?:^|\|)(?:cvat_tracking_xml|legacy_recovered)\|)")


@dataclass(slots=True)
class NativeTemporalCollapseResult:
    predictions: pd.DataFrame
    audit: dict[str, Any]


def collapse_window_predictions_to_native_units(
    window_predictions: pd.DataFrame,
    window_manifest: pd.DataFrame,
    native_units: pd.DataFrame,
) -> NativeTemporalCollapseResult:
    """Aggregate window predictions by temporal unit using confidence-weighted vote."""
    _require(window_predictions, ["window_id", "y_pred", "confidence"], "window_predictions")
    _require(window_manifest, ["window_id", "temporal_unit_keys_window"], "window_manifest")
    _require(native_units, ["temporal_unit_key", "behavior_label"], "native_units")

    pred = window_predictions.copy()
    pred["confidence"] = pd.to_numeric(pred["confidence"], errors="coerce").fillna(0.0).clip(lower=0.0)
    joined = pred.merge(
        window_manifest[["window_id", "temporal_unit_keys_window"]],
        on="window_id",
        how="left",
        validate="many_to_one",
    )
    exploded_rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        temporal_keys = parse_temporal_unit_keys(str(row.temporal_unit_keys_window))
        for temporal_key in temporal_keys:
            exploded_rows.append(
                {
                    "temporal_unit_key": temporal_key,
                    "window_id": str(row.window_id),
                    "window_y_pred": str(row.y_pred),
                    "window_confidence": float(row.confidence),
                    "prediction_split": getattr(row, "prediction_split", ""),
                }
            )
    exploded = pd.DataFrame(exploded_rows)
    if exploded.empty:
        result = native_units[["temporal_unit_key", "behavior_label"]].copy()
        result["y_pred"] = ""
        result["confidence"] = 0.0
        result["supporting_window_count"] = 0
        result["native_prediction_status"] = "no_window_predictions"
    else:
        collapsed = _collapse_exploded(exploded)
        result = native_units.merge(collapsed, on="temporal_unit_key", how="left", validate="one_to_one")
        result["y_pred"] = result["y_pred"].fillna("")
        result["confidence"] = pd.to_numeric(result["confidence"], errors="coerce").fillna(0.0)
        result["supporting_window_count"] = (
            pd.to_numeric(result["supporting_window_count"], errors="coerce").fillna(0).astype(int)
        )
        result["native_prediction_status"] = result["native_prediction_status"].fillna("no_window_predictions")

    result["y_true"] = result["behavior_label"].fillna("").astype(str)
    evaluated = result[result["native_prediction_status"].eq("predicted")].copy()
    metrics = (
        evaluate_predictions(evaluated, y_true_col="y_true", y_pred_col="y_pred", label_order=DEFAULT_LABEL_ORDER)
        if not evaluated.empty
        else {}
    )
    audit = {
        "window_prediction_rows": int(len(window_predictions)),
        "window_predictions_joined": int(len(joined)),
        "exploded_temporal_votes": int(len(exploded)),
        "native_unit_rows": int(len(native_units)),
        "native_units_predicted": int(result["native_prediction_status"].eq("predicted").sum()),
        "native_units_unpredicted": int(result["native_prediction_status"].ne("predicted").sum()),
        "metrics_on_predicted_units": metrics,
        "errors": [],
        "warnings": [],
    }
    if audit["native_units_unpredicted"]:
        audit["warnings"].append(f"native_units_unpredicted={audit['native_units_unpredicted']}")
    return NativeTemporalCollapseResult(predictions=result, audit=audit)


def parse_temporal_unit_keys(value: str) -> list[str]:
    """Parse concatenated temporal-unit keys without splitting inside key fields."""
    if not value or value.lower() in {"nan", "none", "<na>"}:
        return []
    starts = [match.start() for match in SOURCE_MARKER_RE.finditer(value)]
    if not starts:
        return [value.strip("|")] if value.strip("|") else []
    keys: list[str] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(value)
        key = value[start:end].strip("|")
        if key:
            keys.append(key)
    return keys


def _collapse_exploded(exploded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for temporal_key, group in exploded.groupby("temporal_unit_key", sort=True):
        scores = group.groupby("window_y_pred")["window_confidence"].sum().sort_values(ascending=False)
        y_pred = str(scores.index[0])
        confidence = float(scores.iloc[0] / max(1e-12, scores.sum()))
        rows.append(
            {
                "temporal_unit_key": temporal_key,
                "y_pred": y_pred,
                "confidence": confidence,
                "supporting_window_count": int(len(group)),
                "supporting_window_ids": "|".join(group["window_id"].astype(str).tolist()),
                "native_prediction_status": "predicted",
            }
        )
    return pd.DataFrame(rows)


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
