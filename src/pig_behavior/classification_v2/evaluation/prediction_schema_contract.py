"""Prediction-file schema contract for classification_v2 experiments.

Window-level prediction CSVs are an exchange format between model/baseline
code and native-temporal metrics. This checker keeps that boundary explicit:
required prediction fields must exist, labels must be valid, optional
probability columns must be numeric, and high-risk leakage fields are rejected
before any paper-facing metric is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import VALID_BEHAVIOR_SET, VALID_BEHAVIORS

REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "temporal_unit_key",
    "window_id",
    "behavior_true",
    "behavior_pred",
    "window_sample_weight",
    "window_valid_for_main_train",
    "oof_fold_id",
    "experiment_role",
)

FORBIDDEN_PREDICTION_COLUMNS: tuple[str, ...] = (
    "review_item_id",
    "review_unit_id",
    "review_unit_type",
    "review_template",
    "review_reason",
    "review_include_in_training",
    "review_training_action",
    "review_sample_weight",
    "behavior_before_review",
    "original_behavior",
    "manual_review_decision",
    "manual_corrected_behavior",
    "manual_label_strength",
    "manual_training_action",
    "manual_sample_weight",
    "manual_note",
    "source_type",
    "dataset_id",
    "video_key",
    "source_video_key",
    "pig_id",
    "track_id",
    "track_label",
    "object_track_key",
    "frame_uid",
    "image_key",
    "image_name",
    "crop_path",
    "video_path",
)

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "manual_",
    "review_",
    "path_",
)


@dataclass(frozen=True, slots=True)
class PredictionSchemaContract:
    """Configurable column names for validating experiment prediction CSVs."""

    required_columns: tuple[str, ...] = REQUIRED_PREDICTION_COLUMNS
    forbidden_columns: tuple[str, ...] = FORBIDDEN_PREDICTION_COLUMNS
    forbidden_prefixes: tuple[str, ...] = FORBIDDEN_PREFIXES
    prob_prefix: str = "prob_"
    true_col: str = "behavior_true"
    pred_col: str = "behavior_pred"
    weight_col: str = "window_sample_weight"
    valid_col: str = "window_valid_for_main_train"
    window_id_col: str = "window_id"
    unit_id_col: str = "temporal_unit_key"


def check_prediction_schema(
    predictions: pd.DataFrame,
    contract: PredictionSchemaContract | None = None,
) -> dict[str, Any]:
    """Validate a prediction table before native temporal metric aggregation.

    The contract deliberately checks the prediction exchange file, not model
    training quality. A failure means the artifact is unsafe or ambiguous for
    downstream metrics because it is missing required keys, contains invalid
    labels, contains duplicate window rows, or carries leakage-prone columns.
    """

    cfg = contract or PredictionSchemaContract()
    errors: list[str] = []
    warnings: list[str] = []
    columns = list(predictions.columns)

    _check_required_columns(columns, cfg, errors)
    _check_forbidden_columns(columns, cfg, errors)
    if errors:
        return _audit(predictions, cfg, errors, warnings)

    _check_identity_columns(predictions, cfg, errors)
    _check_behavior_labels(predictions, cfg, errors)
    _check_weights(predictions, cfg, errors)
    _check_valid_flags(predictions, cfg, errors)
    _check_probability_columns(predictions, cfg, errors, warnings)
    _check_fold_and_role(predictions, cfg, errors, warnings)

    return _audit(predictions, cfg, errors, warnings)


def check_prediction_schema_csv(
    predictions_csv: Path,
    contract: PredictionSchemaContract | None = None,
) -> dict[str, Any]:
    """Read and validate a prediction CSV path with a file-level audit."""

    errors: list[str] = []
    warnings: list[str] = []
    if not predictions_csv.exists():
        cfg = contract or PredictionSchemaContract()
        errors.append(f"missing_predictions_csv={predictions_csv}")
        return _audit(pd.DataFrame(), cfg, errors, warnings, path=predictions_csv)
    try:
        predictions = pd.read_csv(predictions_csv, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive file IO boundary.
        cfg = contract or PredictionSchemaContract()
        errors.append(f"invalid_predictions_csv={predictions_csv}:{exc}")
        return _audit(pd.DataFrame(), cfg, errors, warnings, path=predictions_csv)
    result = check_prediction_schema(predictions, contract)
    result["predictions_csv"] = str(predictions_csv)
    return result


def _check_required_columns(columns: list[str], cfg: PredictionSchemaContract, errors: list[str]) -> None:
    """Require the metric exchange keys used by every model/baseline output."""

    missing = sorted(set(cfg.required_columns).difference(columns))
    if missing:
        errors.append(f"missing_prediction_columns={missing}")


def _check_forbidden_columns(columns: list[str], cfg: PredictionSchemaContract, errors: list[str]) -> None:
    """Reject leakage-prone audit/source/identity columns in prediction files."""

    exact = sorted(set(columns).intersection(cfg.forbidden_columns))
    prefixed = sorted(
        column
        for column in columns
        if any(column.startswith(prefix) for prefix in cfg.forbidden_prefixes)
        and column not in set(cfg.required_columns)
    )
    forbidden = sorted(set(exact + prefixed))
    if forbidden:
        errors.append(f"forbidden_prediction_columns={forbidden}")


def _check_identity_columns(predictions: pd.DataFrame, cfg: PredictionSchemaContract, errors: list[str]) -> None:
    """Ensure native unit and window identifiers are present and deterministic."""

    unit_values = predictions[cfg.unit_id_col].fillna("").astype(str).str.strip()
    window_values = predictions[cfg.window_id_col].fillna("").astype(str).str.strip()
    missing_unit_rows = int(unit_values.eq("").sum())
    missing_window_rows = int(window_values.eq("").sum())
    duplicate_window_rows = int(window_values.loc[window_values.ne("")].duplicated().sum())
    if missing_unit_rows:
        errors.append(f"missing_temporal_unit_key_rows={missing_unit_rows}")
    if missing_window_rows:
        errors.append(f"missing_window_id_rows={missing_window_rows}")
    if duplicate_window_rows:
        errors.append(f"duplicate_window_id_rows={duplicate_window_rows}")


def _check_behavior_labels(predictions: pd.DataFrame, cfg: PredictionSchemaContract, errors: list[str]) -> None:
    """Validate true and predicted behavior labels against project schema."""

    for column in (cfg.true_col, cfg.pred_col):
        values = predictions[column].fillna("").astype(str).str.strip()
        blank_count = int(values.eq("").sum())
        invalid = sorted(set(values.loc[values.ne("")]).difference(VALID_BEHAVIOR_SET))
        if blank_count:
            errors.append(f"blank_behavior_label_rows={column}:{blank_count}")
        if invalid:
            errors.append(f"invalid_behavior_labels={column}:{invalid}")


def _check_weights(predictions: pd.DataFrame, cfg: PredictionSchemaContract, errors: list[str]) -> None:
    """Require numeric non-negative sample weights for reproducible metrics."""

    weights = pd.to_numeric(predictions[cfg.weight_col], errors="coerce")
    non_numeric = int(weights.isna().sum())
    negative = int(weights.lt(0).sum())
    if non_numeric:
        errors.append(f"non_numeric_sample_weight_rows={non_numeric}")
    if negative:
        errors.append(f"negative_sample_weight_rows={negative}")


def _check_valid_flags(predictions: pd.DataFrame, cfg: PredictionSchemaContract, errors: list[str]) -> None:
    """Ensure main-train validity flags are explicit bool-like values."""

    raw = predictions[cfg.valid_col]
    if pd.api.types.is_bool_dtype(raw):
        return
    normalized = raw.fillna("").astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}
    invalid = int((~normalized.isin(allowed)).sum())
    if invalid:
        errors.append(f"invalid_window_valid_for_main_train_rows={invalid}")


def _check_probability_columns(
    predictions: pd.DataFrame,
    cfg: PredictionSchemaContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate optional class-probability columns if a learned model emits them."""

    prob_cols = [col for col in predictions.columns if col.startswith(cfg.prob_prefix)]
    if not prob_cols:
        warnings.append("no_probability_columns_present_weighted_vote_metrics_only")
        return
    labels = sorted(col[len(cfg.prob_prefix) :] for col in prob_cols)
    invalid_labels = sorted(set(labels).difference(VALID_BEHAVIOR_SET))
    missing_labels = sorted(set(VALID_BEHAVIORS).difference(labels))
    if invalid_labels:
        errors.append(f"invalid_probability_label_columns={invalid_labels}")
    if missing_labels:
        errors.append(f"missing_probability_label_columns={missing_labels}")
    for column in prob_cols:
        values = pd.to_numeric(predictions[column], errors="coerce")
        if int(values.isna().sum()):
            errors.append(f"non_numeric_probability_column={column}")
        if int(values.lt(0).sum()):
            errors.append(f"negative_probability_column={column}")


def _check_fold_and_role(
    predictions: pd.DataFrame,
    cfg: PredictionSchemaContract,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check OOF bookkeeping fields needed for leakage-safe evaluation audits."""

    fold_values = predictions["oof_fold_id"].fillna("").astype(str).str.strip()
    role_values = predictions["experiment_role"].fillna("").astype(str).str.strip()
    if int(fold_values.eq("").sum()):
        errors.append(f"blank_oof_fold_id_rows={int(fold_values.eq('').sum())}")
    if int(role_values.eq("").sum()):
        errors.append(f"blank_experiment_role_rows={int(role_values.eq('').sum())}")
    if fold_values.nunique(dropna=True) < 2:
        warnings.append("single_oof_fold_present")


def _audit(
    predictions: pd.DataFrame,
    cfg: PredictionSchemaContract,
    errors: list[str],
    warnings: list[str],
    path: Path | None = None,
) -> dict[str, Any]:
    """Build a compact audit payload suitable for committing as evidence."""

    prob_cols = [col for col in predictions.columns if col.startswith(cfg.prob_prefix)]
    result: dict[str, Any] = {
        "schema_version": "classification_v2_prediction_schema_contract_v1",
        "prediction_rows": int(len(predictions)),
        "prediction_columns": list(predictions.columns),
        "required_columns": list(cfg.required_columns),
        "forbidden_columns": list(cfg.forbidden_columns),
        "probability_columns": prob_cols,
        "valid_behavior_labels": list(VALID_BEHAVIORS),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    if path is not None:
        result["predictions_csv"] = str(path)
    if not predictions.empty and cfg.true_col in predictions.columns:
        true_counts = predictions[cfg.true_col].fillna("").astype(str).value_counts().sort_index()
        result["true_label_counts"] = {str(k): int(v) for k, v in true_counts.items()}
    if not predictions.empty and cfg.pred_col in predictions.columns:
        pred_counts = predictions[cfg.pred_col].fillna("").astype(str).value_counts().sort_index()
        result["pred_label_counts"] = {str(k): int(v) for k, v in pred_counts.items()}
    return result
