"""Native-unit-safe inner-validation model-selection contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import (
    evaluate_predictions,
)
from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_predictions,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

VALIDATION_SELECTION_CONTRACT_VERSION = (
    "classification_v2.validation_selection.v1"
)
NATIVE_PREDICTION_SCHEMA_VERSION = (
    "classification_v2.native_unit_predictions.v1"
)
VALIDATION_PRIMARY_METRIC = (
    "validation_native_unit_macro_f1_supported"
)
VALIDATION_TIEBREAKER = "validation_native_unit_nll"
SUPPORTED_EVALUATION_SPLITS = frozenset({"validation", "test"})
NATIVE_UNIT_METADATA_COLUMNS = ("source_type", "split_group_key")


@dataclass(frozen=True, slots=True)
class ValidationSelectionScore:
    """Primary maximize score and secondary minimize score for one epoch."""

    primary: float
    tiebreaker: float


def resolve_source_aware_native_unit_key(
    row: pd.Series | dict[str, Any],
) -> str:
    """Derive canonical native evaluation unit key for CVAT and Legacy."""
    raw_unit = (
        row.get("temporal_unit_key", "")
        if "temporal_unit_key" in row and pd.notna(row["temporal_unit_key"])
        else ""
    )
    unit_key = str(raw_unit).strip()
    if "temporal_unit_key" in row and str(row["temporal_unit_key"]).strip() == "":
        return ""

    source_type = str(
        row.get("source_type", "")
        if "source_type" in row and pd.notna(row["source_type"])
        else ""
    ).strip()

    if source_type == "cvat_tracking_xml":
        # Each CVAT 6-frame anchor interval is uniquely identified by its window_id / target_id
        # ensuring multiple 6-frame windows in a continuous behavior run are not collapsed.
        anchor_key = (
            row.get("native_anchor_interval_id")
            or row.get("window_id")
            or row.get("target_id")
        )
        if anchor_key and (
            "run=" in unit_key
            or "run_" in unit_key
            or "cvat" in str(anchor_key).lower()
            or "ordinal=" in str(anchor_key)
            or not unit_key
        ):
            return str(anchor_key).strip()
        return unit_key if unit_key else str(anchor_key).strip()

    if source_type == "legacy_recovered":
        # Each Legacy 16-frame burst is uniquely identified by its native_unit_id (burst sequence)
        burst_key = (
            row.get("native_unit_id")
            or unit_key
            or row.get("window_id")
            or row.get("target_id")
        )
        return str(burst_key).strip()

    fallback = unit_key or row.get("window_id") or row.get("target_id")
    return str(fallback).strip()


def _resolve_source_aware_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure predictions DataFrame uses source-aware native evaluation unit keys."""
    if predictions.empty:
        return predictions.copy()
    frame = predictions.copy()
    if "source_type" in frame.columns and (
        "window_id" in frame.columns or "target_id" in frame.columns
    ):
        frame["temporal_unit_key"] = [
            resolve_source_aware_native_unit_key(row)
            for _, row in frame.iterrows()
        ]
    return frame


def build_native_split_evaluation(
    predictions: pd.DataFrame,
    *,
    split: str,
    min_supported_classes: int = 1,
    label_order: tuple[str, ...] = tuple(VALID_BEHAVIORS),
) -> tuple[pd.DataFrame, dict[str, float | int], dict[str, Any]]:
    """Collapse windows and compute strict native-unit metrics for one split."""

    if split not in SUPPORTED_EVALUATION_SPLITS:
        raise ValueError(f"unsupported evaluation split={split}")
    if min_supported_classes <= 0:
        raise ValueError("min_supported_classes must be positive")
    _validate_prediction_scope(predictions, split)
    predictions = _resolve_source_aware_predictions(predictions)

    config = NativeTemporalMetricsConfig(
        true_col="true_label",
        pred_col="predicted_label",
        weight_col=None,
        valid_col=None,
        label_order=label_order,
        require_complete_probability_vector=True,
        require_oof_fold=True,
        bootstrap_iterations=0,
    )
    native, collapse_audit = build_native_temporal_predictions(
        predictions,
        config,
    )
    if collapse_audit.get("valid") is not True:
        reasons = list(collapse_audit.get("errors", []))
        if int(collapse_audit.get("true_label_conflict_units", 0)):
            reasons.append(
                "true_label_conflict_units="
                f"{collapse_audit['true_label_conflict_units']}"
            )
        if int(collapse_audit.get("oof_fold_conflict_units", 0)):
            reasons.append(
                "oof_fold_conflict_units="
                f"{collapse_audit['oof_fold_conflict_units']}"
            )
        raise ValueError(
            "native-unit validation aggregation failed: "
            f"{reasons}"
        )
    expected_units = int(predictions["temporal_unit_key"].nunique())
    if len(native) != expected_units:
        raise ValueError(
            "native-unit validation row loss: "
            f"expected={expected_units}, observed={len(native)}"
        )
    if not native["native_metric_include"].astype(bool).all():
        raise ValueError("native-unit validation contains excluded units")

    native = native.copy()
    _attach_native_unit_metadata(native, predictions)
    native["schema_version"] = NATIVE_PREDICTION_SCHEMA_VERSION
    native["native_unit_nll"] = _native_unit_losses(native, label_order)
    native["split"] = split
    native["prediction_split"] = split
    native["y_true"] = native["true_label"]
    native["y_pred"] = native["native_predicted_behavior"]
    native_metrics = evaluate_predictions(
        native,
        y_true_col="true_label",
        y_pred_col="native_predicted_behavior",
        label_order=list(label_order),
    )
    window_metrics = evaluate_predictions(
        predictions,
        y_true_col="true_label",
        y_pred_col="predicted_label",
        label_order=list(label_order),
    )
    supported_count = int(native_metrics["supported_label_count"])
    if supported_count < min_supported_classes:
        raise ValueError(
            "inner validation has inadequate class support: "
            f"observed={supported_count}, required={min_supported_classes}"
        )
    metrics: dict[str, float | int] = {
        f"{split}_window_macro_f1": float(window_metrics["macro_f1"]),
        f"{split}_native_unit_macro_f1_global": float(
            native_metrics["macro_f1"]
        ),
        f"{split}_native_unit_macro_f1_supported": float(
            native_metrics["macro_f1_supported"]
        ),
        f"{split}_native_unit_nll": float(native["native_unit_nll"].mean()),
        f"{split}_native_unit_count": int(len(native)),
        f"{split}_supported_class_count": supported_count,
    }
    _validate_metric_values(metrics, split)
    audit = {
        "schema_version": VALIDATION_SELECTION_CONTRACT_VERSION,
        "split": split,
        "evaluation_scope": (
            "grouped_inner_validation_model_selection"
            if split == "validation"
            else "held_out_outer_test_evaluation_only"
        ),
        "eligible_for_model_selection": split == "validation",
        "outer_predictions_used_for_model_selection": False,
        "aggregation_method": "mean_probability_by_temporal_unit_key",
        "primary_metric": (
            VALIDATION_PRIMARY_METRIC if split == "validation" else None
        ),
        "tiebreaker": VALIDATION_TIEBREAKER if split == "validation" else None,
        "preserved_unit_metadata": list(NATIVE_UNIT_METADATA_COLUMNS),
        "min_supported_classes": int(min_supported_classes),
        "input_window_rows": int(len(predictions)),
        "expected_native_unit_rows": expected_units,
        "output_native_unit_rows": int(len(native)),
        "row_loss": int(expected_units - len(native)),
        "supported_classes": [
            label
            for label, values in native_metrics["per_class"].items()
            if int(values["support"]) > 0
        ],
        "metrics": metrics,
        "collapse_audit": collapse_audit,
        "errors": [],
        "valid": True,
    }
    return native, metrics, audit


def _validate_prediction_scope(
    predictions: pd.DataFrame,
    split: str,
) -> None:
    """Require one declared split and complete grouped audit metadata."""

    required = {
        "split",
        "prediction_split",
        "oof_fold_id",
        *NATIVE_UNIT_METADATA_COLUMNS,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"window predictions are missing scope columns={missing}")
    for column in ["split", "prediction_split"]:
        observed = sorted(
            predictions[column].fillna("").astype(str).str.strip().unique()
        )
        if observed != [split]:
            raise ValueError(
                "window prediction split mismatch: "
                f"column={column}, observed={observed}, expected={split}"
            )


def _attach_native_unit_metadata(
    native: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Copy invariant source/group metadata without exposing it to model X."""

    unit_col = "temporal_unit_key"
    for column in NATIVE_UNIT_METADATA_COLUMNS:
        normalized = predictions[column].fillna("").astype(str).str.strip()
        if normalized.eq("").any():
            raise ValueError(f"blank_{column}_rows={int(normalized.eq('').sum())}")
        metadata = pd.DataFrame(
            {
                unit_col: predictions[unit_col].astype(str).str.strip(),
                column: normalized,
            }
        ).drop_duplicates()
        conflicts = metadata[unit_col].duplicated(keep=False)
        if conflicts.any():
            conflict_units = int(metadata.loc[conflicts, unit_col].nunique())
            raise ValueError(
                f"native_unit_{column}_conflicts={conflict_units}"
            )
        lookup = metadata.set_index(unit_col)[column]
        native[column] = native[unit_col].map(lookup)
        if native[column].isna().any():
            raise ValueError(f"native_unit_{column}_mapping_incomplete")


def selection_score_from_metrics(
    metrics: dict[str, Any],
) -> ValidationSelectionScore:
    """Read the predeclared inner-validation score from an epoch record."""

    try:
        primary = float(metrics[VALIDATION_PRIMARY_METRIC])
        tiebreaker = float(metrics[VALIDATION_TIEBREAKER])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("validation selection metrics are missing or invalid") from exc
    score = ValidationSelectionScore(primary, tiebreaker)
    if not math.isfinite(score.primary) or not 0.0 <= score.primary <= 1.0:
        raise ValueError("validation primary metric must be finite in [0, 1]")
    if not math.isfinite(score.tiebreaker) or score.tiebreaker < 0.0:
        raise ValueError("validation NLL tiebreaker must be finite and non-negative")
    return score


def validation_score_is_better(
    candidate: ValidationSelectionScore,
    best: ValidationSelectionScore | None,
    *,
    tolerance: float,
) -> bool:
    """Maximize supported macro-F1, then minimize NLL inside a fixed tolerance."""

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("validation selection tolerance must be finite and non-negative")
    if best is None:
        return True
    if candidate.primary > best.primary + tolerance:
        return True
    tied = abs(candidate.primary - best.primary) <= tolerance
    return tied and candidate.tiebreaker < best.tiebreaker - tolerance


def validation_selection_policy(
    *,
    tolerance: float,
    min_supported_classes: int,
) -> dict[str, Any]:
    """Serialize model-selection semantics for config, checkpoint, and audit."""

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("validation selection tolerance must be finite and non-negative")
    if min_supported_classes <= 0:
        raise ValueError("min_supported_classes must be positive")
    return {
        "contract_version": VALIDATION_SELECTION_CONTRACT_VERSION,
        "selection_scope": "grouped_inner_validation_native_unit_only",
        "primary_metric": VALIDATION_PRIMARY_METRIC,
        "primary_direction": "maximize",
        "tiebreaker": VALIDATION_TIEBREAKER,
        "tiebreaker_direction": "minimize",
        "tie_tolerance": float(tolerance),
        "min_supported_classes": int(min_supported_classes),
        "outer_predictions_used_for_model_selection": False,
    }


def _native_unit_losses(
    native: pd.DataFrame,
    label_order: tuple[str, ...],
) -> np.ndarray:
    probability_columns = [f"prob_{label}" for label in label_order]
    probabilities = native[probability_columns].to_numpy(dtype=np.float64)
    target_index = {label: index for index, label in enumerate(label_order)}
    target_probabilities = np.asarray(
        [
            probabilities[row_index, target_index[str(label)]]
            for row_index, label in enumerate(native["true_label"])
        ],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        losses = -np.log(target_probabilities)
    if not np.isfinite(losses).all():
        invalid = int((~np.isfinite(losses)).sum())
        raise ValueError(f"nonfinite_native_unit_loss_rows={invalid}")
    return losses


def _validate_metric_values(
    metrics: dict[str, float | int],
    split: str,
) -> None:
    for name, value in metrics.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"nonfinite_{name}")
    if int(metrics[f"{split}_native_unit_count"]) <= 0:
        raise ValueError("native-unit evaluation produced no rows")


__all__ = [
    "NATIVE_PREDICTION_SCHEMA_VERSION",
    "VALIDATION_PRIMARY_METRIC",
    "VALIDATION_SELECTION_CONTRACT_VERSION",
    "VALIDATION_TIEBREAKER",
    "ValidationSelectionScore",
    "build_native_split_evaluation",
    "resolve_source_aware_native_unit_key",
    "selection_score_from_metrics",
    "validation_score_is_better",
    "validation_selection_policy",
]
