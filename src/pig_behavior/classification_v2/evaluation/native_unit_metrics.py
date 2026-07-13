"""Paper-facing fixed-label metrics at the native temporal-unit level."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation.calibration import probability_calibration_metrics
from pig_behavior.classification_v2.evaluation.metrics import evaluate_predictions
from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_predictions,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

CLASS_GROUPS = {
    "interaction": ("fight", "social-nose"),
    "roi_behavior": ("eat", "drink", "playwithtoy"),
    "posture": ("lying", "sitting"),
    "locomotion_context": ("move", "explore", "stand"),
    "rare": ("fight", "social-nose", "playwithtoy", "move"),
}


def evaluate_native_oof(
    window_predictions: pd.DataFrame,
    fold_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse overlapping windows once, validate OOF lineage, and compute Q2 metrics."""

    config = NativeTemporalMetricsConfig(
        true_col="true_label",
        pred_col="predicted_label",
        weight_col=None,
        valid_col=None,
        label_order=tuple(VALID_BEHAVIORS),
        require_complete_probability_vector=True,
        require_oof_fold=True,
        bootstrap_iterations=0,
    )
    units, collapse_audit = build_native_temporal_predictions(window_predictions, config)
    assignment_columns = [
        "temporal_unit_key",
        "recording_group_id",
        "outer_fold_id",
        "behavior_label",
        "source_type",
        "video_key",
        "native_unit_valid_for_main_eval",
    ]
    missing = [column for column in assignment_columns if column not in fold_assignments.columns]
    if missing:
        raise ValueError(f"Q2 fold assignments missing columns: {missing}")
    assignments = fold_assignments[assignment_columns].copy()
    if assignments["temporal_unit_key"].duplicated().any():
        raise ValueError("duplicate temporal_unit_key in Q2 fold assignments")
    errors = list(collapse_audit.get("errors", []))
    if not collapse_audit.get("valid", False):
        errors.append("native_temporal_collapse_invalid")
    _validate_assignment_rows(assignments, errors)
    try:
        valid_assignment = _strict_bool(
            assignments["native_unit_valid_for_main_eval"],
            "native_unit_valid_for_main_eval",
        )
    except ValueError:
        valid_assignment = pd.Series(False, index=assignments.index)
    expected = assignments.loc[valid_assignment, "temporal_unit_key"].astype(str)
    all_assignment_ids = set(assignments["temporal_unit_key"].astype(str))
    if "temporal_unit_key" not in units.columns:
        units = pd.DataFrame(
            {"temporal_unit_key": pd.Series(dtype="object")}
        )
    unit_ids = set(units["temporal_unit_key"].astype(str))
    missing_predictions = sorted(set(expected) - unit_ids)
    extra_predictions = sorted(unit_ids - all_assignment_ids)
    invalid_unit_predictions = sorted(
        (unit_ids & all_assignment_ids) - set(expected)
    )
    duplicate_predictions = int(units["temporal_unit_key"].duplicated().sum())
    if missing_predictions:
        errors.append(f"missing_valid_native_predictions={len(missing_predictions)}")
    if extra_predictions:
        errors.append(f"extra_native_predictions={len(extra_predictions)}")
    if invalid_unit_predictions:
        errors.append(
            f"predictions_for_non_evaluable_native_units={len(invalid_unit_predictions)}"
        )
    if duplicate_predictions:
        errors.append(f"duplicate_native_predictions={duplicate_predictions}")

    merged = assignments.merge(
        units,
        on="temporal_unit_key",
        how="left",
        validate="one_to_one",
    )
    _ensure_prediction_columns(merged)
    has_prediction = merged["native_predicted_behavior"].fillna("").ne("")
    target_mismatch = has_prediction & merged["true_label"].fillna("").astype(
        str
    ).ne(merged["behavior_label"].astype(str))
    fold_mismatch = has_prediction & merged["oof_fold_id"].fillna("").astype(
        str
    ).ne(merged["outer_fold_id"].astype(str))
    if target_mismatch.any():
        errors.append(f"prediction_true_label_mismatch={int(target_mismatch.sum())}")
    if fold_mismatch.any():
        errors.append(f"prediction_outer_fold_mismatch={int(fold_mismatch.sum())}")
    merged["prediction_true_label"] = merged["true_label"].fillna("").astype(str)
    merged["true_label"] = merged["behavior_label"].astype(str)
    merged["native_metric_include"] = (
        _to_bool(merged["native_metric_include"])
        & _to_bool(merged["native_unit_valid_for_main_eval"])
        & ~target_mismatch
        & ~fold_mismatch
    )
    evaluable = merged.loc[merged["native_metric_include"]].copy()
    metrics = _metric_bundle(evaluable)
    fold_metrics = {
        str(fold): _metric_bundle(group)
        for fold, group in evaluable.groupby("outer_fold_id", sort=True)
    }
    slices = {
        column: {
            str(value): _metric_bundle(group)
            for value, group in evaluable.groupby(column, sort=True, dropna=False)
        }
        for column in [
            "source_type",
            "video_key",
            "recording_group_id",
            "behavior_label",
        ]
    }
    class_fold_support = _class_fold_support(
        assignments,
        valid_assignment,
    )
    audit = {
        "schema_version": "classification_v2_q2_native_unit_metrics_v2",
        "statistical_unit": "native_temporal_unit",
        "fixed_label_order": list(VALID_BEHAVIORS),
        "fold_count": int(evaluable["outer_fold_id"].nunique()),
        "assignment_native_unit_rows": int(len(assignments)),
        "output_native_unit_rows": int(len(merged)),
        "assignment_to_output_row_loss": int(len(assignments) - len(merged)),
        "expected_valid_native_units": int(len(expected)),
        "predicted_valid_native_units": int(len(evaluable)),
        "missing_valid_native_unit_count": len(missing_predictions),
        "extra_native_prediction_count": len(extra_predictions),
        "non_evaluable_native_prediction_count": len(invalid_unit_predictions),
        "duplicate_native_prediction_count": duplicate_predictions,
        "prediction_true_label_mismatch_count": int(target_mismatch.sum()),
        "prediction_outer_fold_mismatch_count": int(fold_mismatch.sum()),
        "missing_valid_native_unit_examples": missing_predictions[:10],
        "extra_native_prediction_examples": extra_predictions[:10],
        "collapse_audit": collapse_audit,
        "pooled_metrics": metrics,
        "fold_metrics": fold_metrics,
        "fold_macro_f1_definition": "supported_true_classes_only",
        "class_fold_support": class_fold_support,
        "slice_metrics": slices,
        "errors": errors,
        "valid": not errors,
    }
    return merged, audit


def _metric_bundle(frame: pd.DataFrame) -> dict[str, Any]:
    base = evaluate_predictions(
        frame,
        y_true_col="true_label",
        y_pred_col="native_predicted_behavior",
        label_order=list(VALID_BEHAVIORS),
    )
    confusion = np.asarray(base["confusion_matrix"]["values"], dtype=float)
    base["balanced_accuracy"] = float(base["macro_recall"])
    base["multiclass_mcc"] = _multiclass_mcc(confusion)
    total_support = sum(
        int(values["support"]) for values in base["per_class"].values()
    )
    base["weighted_f1"] = (
        float(
            sum(
                float(values["f1"]) * int(values["support"])
                for values in base["per_class"].values()
            )
            / total_support
        )
        if total_support
        else 0.0
    )
    base["class_group_metrics"] = _class_group_metrics(base["per_class"])
    probability_columns = [f"prob_{label}" for label in VALID_BEHAVIORS]
    if set(probability_columns).issubset(frame.columns) and len(frame):
        probabilities = frame[probability_columns].to_numpy(dtype=float)
        label_to_index = {
            label: index for index, label in enumerate(VALID_BEHAVIORS)
        }
        targets = np.asarray(
            [label_to_index[label] for label in frame["true_label"]],
            dtype=int,
        )
        base.update(probability_calibration_metrics(probabilities, targets, ece_bins=15))
    return base


def _class_group_metrics(
    per_class: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, Any]]:
    """Summarize fixed scientific behavior groups without remapping labels."""

    result: dict[str, dict[str, Any]] = {}
    for group_name, labels in CLASS_GROUPS.items():
        values = [per_class[label] for label in labels]
        supported = [value for value in values if int(value["support"]) > 0]
        result[group_name] = {
            "labels": "|".join(labels),
            "support": sum(int(value["support"]) for value in values),
            "macro_f1": float(
                sum(float(value["f1"]) for value in values) / len(values)
            ),
            "macro_recall": float(
                sum(float(value["recall"]) for value in values) / len(values)
            ),
            "macro_f1_supported": (
                float(
                    sum(float(value["f1"]) for value in supported)
                    / len(supported)
                )
                if supported
                else 0.0
            ),
        }
    return result


def _class_fold_support(
    assignments: pd.DataFrame,
    valid_mask: pd.Series,
) -> list[dict[str, Any]]:
    """Return the complete global-class by outer-fold support matrix."""

    valid = assignments.loc[valid_mask].copy()
    folds = sorted(valid["outer_fold_id"].astype(str).unique())
    counts = valid.groupby(["outer_fold_id", "behavior_label"]).size()
    records: list[dict[str, Any]] = []
    for fold_id in folds:
        supported_count = sum(
            int(counts.get((fold_id, label), 0)) > 0
            for label in VALID_BEHAVIORS
        )
        for label in VALID_BEHAVIORS:
            support = int(counts.get((fold_id, label), 0))
            records.append(
                {
                    "outer_fold_id": fold_id,
                    "behavior_label": label,
                    "support": support,
                    "class_supported": support > 0,
                    "supported_class_count_in_fold": supported_count,
                }
            )
    return records


def _validate_assignment_rows(
    assignments: pd.DataFrame,
    errors: list[str],
) -> None:
    """Validate authority labels and recording/video-safe fold membership."""

    text_columns = [
        "temporal_unit_key",
        "recording_group_id",
        "outer_fold_id",
        "behavior_label",
        "source_type",
        "video_key",
    ]
    for column in text_columns:
        blank = assignments[column].fillna("").astype(str).str.strip().eq("")
        if blank.any():
            errors.append(f"blank_assignment_{column}_rows={int(blank.sum())}")
    invalid_labels = sorted(
        set(assignments["behavior_label"].astype(str)) - set(VALID_BEHAVIORS)
    )
    if invalid_labels:
        errors.append(f"invalid_assignment_behavior_labels={invalid_labels}")
    valid_sources = {"legacy_recovered", "cvat_tracking_xml"}
    invalid_sources = sorted(
        set(assignments["source_type"].astype(str)) - valid_sources
    )
    if invalid_sources:
        errors.append(f"invalid_assignment_source_types={invalid_sources}")
    try:
        _strict_bool(
            assignments["native_unit_valid_for_main_eval"],
            "native_unit_valid_for_main_eval",
        )
    except ValueError as exc:
        errors.append(str(exc))
    for group_column in ["recording_group_id", "video_key"]:
        conflicts = assignments.groupby(group_column)["outer_fold_id"].nunique()
        conflict_count = int(conflicts.gt(1).sum())
        if conflict_count:
            errors.append(
                f"{group_column}_crosses_outer_folds={conflict_count}"
            )


def _ensure_prediction_columns(frame: pd.DataFrame) -> None:
    """Add explicit missing-prediction values after row-preserving left join."""

    defaults: dict[str, object] = {
        "native_metric_include": False,
        "native_predicted_behavior": "",
        "true_label": "",
        "oof_fold_id": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default


def _multiclass_mcc(confusion: np.ndarray) -> float:
    total = float(confusion.sum())
    correct = float(np.trace(confusion))
    true_marginal = confusion.sum(axis=1)
    predicted_marginal = confusion.sum(axis=0)
    numerator = correct * total - float(np.dot(true_marginal, predicted_marginal))
    denominator = np.sqrt(
        (total**2 - float(np.dot(predicted_marginal, predicted_marginal)))
        * (total**2 - float(np.dot(true_marginal, true_marginal)))
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    """Parse authority booleans without treating unknown values as false."""

    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"invalid_{name}_bool_rows={int(series.isna().sum())}")
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    invalid = ~normalized.isin(true_values | false_values)
    if invalid.any():
        raise ValueError(f"invalid_{name}_bool_rows={int(invalid.sum())}")
    return normalized.isin(true_values)
