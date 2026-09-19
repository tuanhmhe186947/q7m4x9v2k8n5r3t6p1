"""Nonlinear tabular whitelist baseline for classification_v2.

This is the B2 ablation control. It uses the same leakage-safe input contract
as B1, but replaces the linear classifier with a tree-boosting model so the
paper ladder can separate tabular nonlinear signal from image/temporal fusion
signal. It never expands model inputs beyond the audited feature whitelist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_metrics,
)
from pig_behavior.classification_v2.evaluation.prediction_schema_contract import check_prediction_schema
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


@dataclass(frozen=True, slots=True)
class TabularNonlinearBaselineConfig:
    """Paths and deterministic model options for the B2 nonlinear baseline."""

    trainer_contract_json: Path = Path("configs/classification_v2/trainer_contract_v1.json")
    root: Path = Path("outputs/classification_v2/train_ready_windows")
    sequence_manifest_csv: Path = Path(
        "outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv"
    )
    native_oof_fold_manifest_csv: Path = Path(
        "outputs/classification_v2/native_temporal_units_oof_folds/native_oof_fold_manifest.csv"
    )
    output_dir: Path = Path("outputs/classification_v2/model_smoke/tabular_nonlinear_baseline")
    max_iter: int = 120
    learning_rate: float = 0.06
    l2_regularization: float = 0.02
    max_leaf_nodes: int = 31
    random_state: int = 20260710
    bootstrap_iterations: int = 100


def run_tabular_nonlinear_baseline(config: TabularNonlinearBaselineConfig) -> dict[str, Any]:
    """Train native-OOF nonlinear whitelist baseline and write audit artifacts."""

    data, load_audit = _load_training_frame(config)
    predictions, train_audit = _fit_oof_predictions(data, config)
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
    prediction_schema_audit = check_prediction_schema(predictions)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = config.output_dir / "tabular_nonlinear_predictions.csv"
    native_units_path = config.output_dir / "tabular_nonlinear_unit_predictions.csv"
    metrics_path = config.output_dir / "tabular_nonlinear_metrics.json"
    audit_path = config.output_dir / "tabular_nonlinear_audit.json"
    schema_path = config.output_dir / "tabular_nonlinear_prediction_schema_audit.json"
    predictions.to_csv(predictions_path, index=False)
    metrics_units.to_csv(native_units_path, index=False)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    schema_path.write_text(json.dumps(prediction_schema_audit, indent=2), encoding="utf-8")

    audit = {
        "schema_version": "classification_v2_tabular_nonlinear_baseline_audit_v1",
        "baseline_id": "B2_nonlinear_tabular_whitelist",
        "config": _jsonable_config(config),
        "load_audit": load_audit,
        "train_audit": train_audit,
        "predictions_csv": str(predictions_path),
        "native_unit_predictions_csv": str(native_units_path),
        "metrics_json": str(metrics_path),
        "prediction_schema_audit_json": str(schema_path),
        "prediction_schema_valid": bool(prediction_schema_audit.get("valid")),
        "errors": [],
        "warnings": [
            "nonlinear tabular whitelist baseline only; use as B2 ablation control, not final multimodal model",
        ],
        "valid": bool(not predictions.empty and prediction_schema_audit.get("valid") is True),
    }
    if prediction_schema_audit.get("errors"):
        audit["errors"].append(f"prediction_schema_errors={prediction_schema_audit.get('errors')}")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if audit["errors"]:
        raise ValueError(f"tabular nonlinear baseline failed: {audit['errors']}")
    return {
        "predictions_csv": str(predictions_path),
        "native_unit_predictions_csv": str(native_units_path),
        "metrics_json": str(metrics_path),
        "audit_json": str(audit_path),
        "prediction_schema_audit_json": str(schema_path),
        "audit": audit,
    }


def _load_training_frame(config: TabularNonlinearBaselineConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load whitelisted X, labels, training mask, and native temporal fold ids."""

    trainer_contract = json.loads(config.trainer_contract_json.read_text(encoding="utf-8"))
    feature_cols = list(trainer_contract.get("tabular_feature_whitelist", []))
    if not feature_cols:
        raise ValueError("trainer_contract tabular_feature_whitelist is empty")
    x = pd.read_csv(config.root / "X_window_features.csv", usecols=feature_cols, low_memory=False)
    y = pd.read_csv(config.root / "y_behavior.csv").iloc[:, 0].fillna("").astype(str)
    train_mask = _read_bool(config.root / "train_mask.csv")
    split = pd.read_csv(
        config.root / "split_manifest.csv",
        usecols=["window_id", "window_valid_for_main_train", "window_sample_weight"],
        low_memory=False,
    )
    sequence = pd.read_csv(
        config.sequence_manifest_csv,
        usecols=["window_id", "temporal_unit_keys_window", "num_temporal_units_window"],
        low_memory=False,
    )
    folds = pd.read_csv(
        config.native_oof_fold_manifest_csv,
        usecols=["temporal_unit_key", "oof_fold_id", "native_unit_valid_for_main_eval"],
        low_memory=False,
    )
    if len(x) != len(y) or len(x) != len(split) or len(x) != len(train_mask):
        raise ValueError(f"row mismatch x={len(x)} y={len(y)} split={len(split)} mask={len(train_mask)}")

    frame = split.merge(sequence, on="window_id", how="left", validate="one_to_one")
    frame["num_temporal_units_window"] = pd.to_numeric(frame["num_temporal_units_window"], errors="coerce")
    frame = frame.rename(columns={"temporal_unit_keys_window": "temporal_unit_key"})
    frame = frame.merge(folds, on="temporal_unit_key", how="left")
    frame["behavior_true"] = y
    frame["eligible"] = (
        train_mask
        & frame["num_temporal_units_window"].eq(1)
        & _to_bool(frame["window_valid_for_main_train"])
        & _to_bool(frame["native_unit_valid_for_main_eval"])
        & frame["behavior_true"].isin(VALID_BEHAVIORS)
        & frame["oof_fold_id"].fillna("").astype(str).ne("")
    )
    data = pd.concat([frame.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    audit = {
        "rows": int(len(data)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "eligible_rows": int(data["eligible"].sum()),
        "fold_count": int(data.loc[data["eligible"], "oof_fold_id"].nunique()),
        "label_counts": data.loc[data["eligible"], "behavior_true"].value_counts().sort_index().to_dict(),
    }
    return data, audit


def _fit_oof_predictions(
    data: pd.DataFrame,
    config: TabularNonlinearBaselineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one nonlinear classifier per native OOF fold and emit probabilities."""

    feature_cols = json.loads(config.trainer_contract_json.read_text(encoding="utf-8"))["tabular_feature_whitelist"]
    eligible = data[data["eligible"]].copy()
    rows: list[pd.DataFrame] = []
    fold_audit: list[dict[str, Any]] = []
    label_order = list(VALID_BEHAVIORS)
    for fold_id, test in eligible.groupby("oof_fold_id", sort=True):
        train = eligible[eligible["oof_fold_id"].ne(fold_id)].copy()
        sample_weight = _balanced_sample_weight(
            train["behavior_true"],
            pd.to_numeric(train["window_sample_weight"], errors="coerce").fillna(1.0).to_numpy(),
        )
        model = HistGradientBoostingClassifier(
            max_iter=int(config.max_iter),
            learning_rate=float(config.learning_rate),
            l2_regularization=float(config.l2_regularization),
            max_leaf_nodes=int(config.max_leaf_nodes),
            random_state=int(config.random_state),
        )
        model.fit(train[feature_cols], train["behavior_true"], sample_weight=sample_weight)
        pred = model.predict(test[feature_cols])
        prob = model.predict_proba(test[feature_cols])
        classes = [str(label) for label in model.classes_]
        out = pd.DataFrame(
            {
                "temporal_unit_key": test["temporal_unit_key"].astype(str).to_numpy(),
                "window_id": test["window_id"].astype(str).to_numpy(),
                "behavior_true": test["behavior_true"].astype(str).to_numpy(),
                "behavior_pred": pred.astype(str),
                "window_sample_weight": pd.to_numeric(test["window_sample_weight"], errors="coerce")
                .fillna(1.0)
                .to_numpy(),
                "window_valid_for_main_train": True,
                "oof_fold_id": str(fold_id),
                "experiment_role": "B2_nonlinear_tabular_whitelist_native_oof",
            }
        )
        for label in label_order:
            out[f"prob_{label}"] = 0.0
        for class_index, label in enumerate(classes):
            out[f"prob_{label}"] = prob[:, class_index]
        rows.append(out)
        fold_audit.append(
            {
                "oof_fold_id": str(fold_id),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_label_count": int(train["behavior_true"].nunique()),
                "test_label_count": int(test["behavior_true"].nunique()),
            }
        )
    predictions = pd.concat(rows, ignore_index=True).sort_values(["oof_fold_id", "window_id"], kind="mergesort")
    audit = {
        "folds": fold_audit,
        "prediction_rows": int(len(predictions)),
        "native_temporal_units_predicted": int(predictions["temporal_unit_key"].nunique()),
    }
    return predictions.reset_index(drop=True), audit


def _balanced_sample_weight(labels: pd.Series, base_weight: np.ndarray) -> np.ndarray:
    """Combine window weights with inverse-frequency class weights for rare classes."""

    counts = labels.value_counts()
    class_count = max(int(counts.size), 1)
    total = max(int(counts.sum()), 1)
    weights = labels.map({label: total / (class_count * count) for label, count in counts.items()}).to_numpy()
    return np.asarray(base_weight, dtype=float) * np.asarray(weights, dtype=float)


def _read_bool(path: Path) -> pd.Series:
    """Read a one-column CSV mask and normalize common truthy strings."""

    series = pd.read_csv(path).iloc[:, 0]
    return _to_bool(series)


def _to_bool(series: pd.Series) -> pd.Series:
    """Convert mixed bool/string flags to a strict boolean series."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _jsonable_config(config: TabularNonlinearBaselineConfig) -> dict[str, Any]:
    """Serialize the dataclass config into JSON-safe primitive values."""

    return {
        "trainer_contract_json": str(config.trainer_contract_json),
        "root": str(config.root),
        "sequence_manifest_csv": str(config.sequence_manifest_csv),
        "native_oof_fold_manifest_csv": str(config.native_oof_fold_manifest_csv),
        "output_dir": str(config.output_dir),
        "max_iter": int(config.max_iter),
        "learning_rate": float(config.learning_rate),
        "l2_regularization": float(config.l2_regularization),
        "max_leaf_nodes": int(config.max_leaf_nodes),
        "random_state": int(config.random_state),
        "bootstrap_iterations": int(config.bootstrap_iterations),
    }
