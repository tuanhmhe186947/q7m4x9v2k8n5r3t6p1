"""Native temporal majority baseline for classification_v2 smoke experiments.

This is a no-training baseline: for each out-of-fold test split, it predicts the
most frequent behavior observed in the remaining folds. The purpose is to prove
the experiment plumbing, native temporal metric schema, and registry gates before
running any learned multimodal model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_metrics,
)
from pig_behavior.classification_v2.evaluation.prediction_schema_contract import check_prediction_schema


@dataclass(frozen=True, slots=True)
class NativeMajorityBaselineConfig:
    """Input/output paths and deterministic options for the smoke baseline."""

    native_manifest_csv: Path = Path(
        "outputs/classification_v2/native_temporal_units/native_temporal_unit_manifest.csv"
    )
    native_oof_fold_manifest_csv: Path = Path(
        "outputs/classification_v2/native_temporal_units_oof_folds/native_oof_fold_manifest.csv"
    )
    output_dir: Path = Path("outputs/classification_v2/model_smoke/native_majority_baseline")
    bootstrap_iterations: int = 200


def run_native_majority_baseline(config: NativeMajorityBaselineConfig) -> dict[str, Any]:
    """Build OOF majority predictions and write prediction/metric artifacts."""

    native_units = pd.read_csv(config.native_manifest_csv, low_memory=False)
    folds = pd.read_csv(config.native_oof_fold_manifest_csv, low_memory=False)
    predictions, audit = build_native_majority_predictions(native_units, folds)
    metrics_units, metrics_payload = build_native_temporal_metrics(
        predictions,
        NativeTemporalMetricsConfig(
            unit_id_col="temporal_unit_key",
            true_col="behavior_true",
            pred_col="behavior_pred",
            weight_col="window_sample_weight",
            valid_col="window_valid_for_main_train",
            window_id_col="window_id",
            bootstrap_iterations=int(config.bootstrap_iterations),
        ),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.output_dir / "native_majority_predictions.csv"
    native_units_path = config.output_dir / "native_majority_unit_predictions.csv"
    metrics_path = config.output_dir / "native_majority_metrics.json"
    audit_path = config.output_dir / "native_majority_audit.json"
    prediction_schema_path = config.output_dir / "native_majority_prediction_schema_audit.json"
    predictions.to_csv(predictions_path, index=False)
    metrics_units.to_csv(native_units_path, index=False)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    prediction_schema_audit = check_prediction_schema(predictions)
    prediction_schema_path.write_text(json.dumps(prediction_schema_audit, indent=2), encoding="utf-8")

    audit.update(
        {
            "predictions_csv": str(predictions_path),
            "native_unit_predictions_csv": str(native_units_path),
            "metrics_json": str(metrics_path),
            "prediction_schema_audit_json": str(prediction_schema_path),
            "prediction_schema_valid": bool(prediction_schema_audit.get("valid")),
        }
    )
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {
        "predictions_csv": str(predictions_path),
        "native_unit_predictions_csv": str(native_units_path),
        "metrics_json": str(metrics_path),
        "audit_json": str(audit_path),
        "prediction_schema_audit_json": str(prediction_schema_path),
        "audit": audit,
    }


def build_native_majority_predictions(
    native_units: pd.DataFrame,
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one pseudo-window prediction per native temporal unit.

    The output intentionally uses the window-prediction schema consumed by the
    native temporal metric aggregator, with `window_id` set to a synthetic native
    smoke identifier. No source, path, review, or identity fields are model
    inputs here; they are retained only as audit metadata.
    """

    required_native = [
        "temporal_unit_key",
        "behavior_label",
        "native_unit_valid_for_main_eval",
        "native_unit_sample_weight",
    ]
    required_folds = ["temporal_unit_key", "oof_fold_id"]
    missing_native = [col for col in required_native if col not in native_units.columns]
    missing_folds = [col for col in required_folds if col not in folds.columns]
    if missing_native or missing_folds:
        raise ValueError(f"missing_columns native={missing_native} folds={missing_folds}")

    frame = native_units.merge(
        folds[required_folds].drop_duplicates("temporal_unit_key"),
        on="temporal_unit_key",
        how="left",
    )
    frame["native_unit_valid_for_main_eval"] = _to_bool_series(frame["native_unit_valid_for_main_eval"])
    frame = frame.loc[frame["native_unit_valid_for_main_eval"]].copy()
    frame["behavior_label"] = frame["behavior_label"].fillna("").astype(str)
    frame = frame.loc[frame["behavior_label"].ne("")].copy()
    frame["oof_fold_id"] = frame["oof_fold_id"].fillna("").astype(str)
    frame = frame.loc[frame["oof_fold_id"].ne("")].copy()

    global_majority = _majority_label(frame["behavior_label"])
    rows: list[dict[str, Any]] = []
    for fold_id, test in frame.groupby("oof_fold_id", sort=True):
        train = frame.loc[frame["oof_fold_id"].ne(fold_id)]
        pred_label = _majority_label(train["behavior_label"]) if not train.empty else global_majority
        for _, row in test.sort_values("temporal_unit_key", kind="mergesort").iterrows():
            rows.append(
                {
                    "temporal_unit_key": str(row["temporal_unit_key"]),
                    "window_id": f"native_majority|{row['temporal_unit_key']}",
                    "behavior_true": str(row["behavior_label"]),
                    "behavior_pred": str(pred_label),
                    "window_sample_weight": float(row.get("native_unit_sample_weight", 1.0)),
                    "window_valid_for_main_train": True,
                    "oof_fold_id": str(fold_id),
                    "experiment_role": "native_temporal_oof_majority_baseline",
                }
            )

    predictions = pd.DataFrame(rows).sort_values(["oof_fold_id", "temporal_unit_key"], kind="mergesort").reset_index(
        drop=True
    )
    audit = {
        "schema_version": "classification_v2_native_majority_baseline_audit_v1",
        "native_unit_rows_input": int(len(native_units)),
        "fold_rows_input": int(len(folds)),
        "eligible_native_unit_rows": int(len(frame)),
        "prediction_rows": int(len(predictions)),
        "fold_count": int(frame["oof_fold_id"].nunique()) if not frame.empty else 0,
        "global_majority_label": global_majority,
        "label_counts": {str(k): int(v) for k, v in frame["behavior_label"].value_counts().sort_index().items()},
        "errors": [],
        "warnings": ["no_model_training_majority_baseline_only"],
        "valid": not predictions.empty,
    }
    return predictions, audit


def _majority_label(labels: pd.Series) -> str:
    """Choose majority label with lexical tie-break for reproducibility."""

    counts = labels.fillna("").astype(str)
    counts = counts.loc[counts.ne("")].value_counts()
    if counts.empty:
        return ""
    return sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Normalize bool-like CSV strings from audit artifacts."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
