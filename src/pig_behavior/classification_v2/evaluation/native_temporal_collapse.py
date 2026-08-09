"""Collapse window-level predictions to native temporal-unit predictions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import (
    DEFAULT_LABEL_ORDER,
    evaluate_predictions,
)

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
    _require(window_manifest, ["window_id"], "window_manifest")
    _require(native_units, ["temporal_unit_key", "behavior_label"], "native_units")
    if not {
        "temporal_unit_keys_json",
        "temporal_unit_keys_window",
    }.intersection(window_manifest.columns):
        raise ValueError("window_manifest missing native temporal-unit keys")
    if window_predictions["window_id"].astype(str).duplicated().any():
        raise ValueError("window_predictions has duplicate window_id")
    if window_manifest["window_id"].astype(str).duplicated().any():
        raise ValueError("window_manifest has duplicate window_id")
    if native_units["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("native_units has duplicate temporal_unit_key")

    pred = window_predictions.copy()
    pred["confidence"] = (
        pd.to_numeric(pred["confidence"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    key_column = (
        "temporal_unit_keys_json"
        if "temporal_unit_keys_json" in window_manifest.columns
        else "temporal_unit_keys_window"
    )
    joined = pred.merge(
        window_manifest[["window_id", key_column]],
        on="window_id",
        how="left",
        validate="many_to_one",
    )
    exploded_rows: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        temporal_keys = parse_temporal_unit_keys(getattr(row, key_column))
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
    expected_keys = set(native_units["temporal_unit_key"].astype(str))
    observed_keys = set(exploded.get("temporal_unit_key", pd.Series(dtype=str)).astype(str))
    extra_keys = sorted(observed_keys.difference(expected_keys))
    if exploded.empty:
        result = native_units[["temporal_unit_key", "behavior_label"]].copy()
        result["y_pred"] = ""
        result["confidence"] = 0.0
        result["supporting_window_count"] = 0
        result["native_prediction_status"] = "no_window_predictions"
    else:
        collapsed = _collapse_exploded(exploded)
        result = native_units.merge(
            collapsed,
            on="temporal_unit_key",
            how="left",
            validate="one_to_one",
        )
        result["y_pred"] = result["y_pred"].fillna("")
        result["confidence"] = pd.to_numeric(result["confidence"], errors="coerce").fillna(0.0)
        result["supporting_window_count"] = (
            pd.to_numeric(result["supporting_window_count"], errors="coerce").fillna(0).astype(int)
        )
        result["native_prediction_status"] = result[
            "native_prediction_status"
        ].fillna("no_window_predictions")

    result["y_true"] = result["behavior_label"].fillna("").astype(str)
    missing_keys = sorted(
        result.loc[
            result["native_prediction_status"].ne("predicted"), "temporal_unit_key"
        ].astype(str)
    )
    errors: list[str] = []
    if missing_keys:
        errors.append(f"native_units_unpredicted={len(missing_keys)}")
    if extra_keys:
        errors.append(f"unexpected_native_predictions={len(extra_keys)}")
    evaluated = result[result["native_prediction_status"].eq("predicted")].copy()
    metrics = (
        evaluate_predictions(
            evaluated,
            y_true_col="y_true",
            y_pred_col="y_pred",
            label_order=DEFAULT_LABEL_ORDER,
        )
        if not evaluated.empty and not errors
        else {}
    )
    audit = {
        "window_prediction_rows": int(len(window_predictions)),
        "window_predictions_joined": int(len(joined)),
        "exploded_temporal_votes": int(len(exploded)),
        "native_unit_rows": int(len(native_units)),
        "native_units_predicted": int(result["native_prediction_status"].eq("predicted").sum()),
        "native_units_unpredicted": int(result["native_prediction_status"].ne("predicted").sum()),
        "expected_native_units": int(len(native_units)),
        "missing_native_unit_examples": missing_keys[:20],
        "unexpected_native_prediction_examples": extra_keys[:20],
        "duplicate_collapsed_native_predictions": int(
            result["temporal_unit_key"].duplicated().sum()
        ),
        "metrics_on_predicted_units": metrics,
        "errors": errors,
        "warnings": [],
        "valid": not errors,
    }
    return NativeTemporalCollapseResult(predictions=result, audit=audit)


def parse_temporal_unit_keys(value: object) -> list[str]:
    """Parse concatenated temporal-unit keys without splitting inside key fields."""
    if isinstance(value, list):
        return _validated_temporal_unit_keys(value)
    text = str(value)
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid temporal_unit_keys_json") from exc
        if not isinstance(parsed, list):
            raise ValueError("temporal_unit_keys_json is not a list")
        return _validated_temporal_unit_keys(parsed)
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return []
    starts = [match.start() for match in SOURCE_MARKER_RE.finditer(text)]
    if not starts:
        return [text.strip("|")] if text.strip("|") else []
    keys: list[str] = []
    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(text)
        key = text[start:end].strip("|")
        if key:
            keys.append(key)
    return _validated_temporal_unit_keys(keys)


def _collapse_exploded(exploded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for temporal_key, group in exploded.groupby("temporal_unit_key", sort=True):
        scores = group.groupby("window_y_pred", sort=True)["window_confidence"].sum()
        y_pred = _canonical_vote_winner(scores)
        confidence = float(scores.loc[y_pred] / max(1e-12, scores.sum()))
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


def _validated_temporal_unit_keys(values: list[object]) -> list[str]:
    keys = [str(value).strip() for value in values]
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("invalid temporal-unit key list")
    return keys


def _canonical_vote_winner(scores: pd.Series) -> str:
    """Resolve equal weighted votes by the frozen canonical ten-class order."""

    rank = {label: index for index, label in enumerate(DEFAULT_LABEL_ORDER)}
    ordered = sorted(
        ((str(label), float(score)) for label, score in scores.items()),
        key=lambda item: (-item[1], rank.get(item[0], len(rank)), item[0]),
    )
    return ordered[0][0]
