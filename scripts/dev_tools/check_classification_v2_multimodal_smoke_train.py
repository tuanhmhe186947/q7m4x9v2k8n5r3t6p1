from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.prediction_schema_contract import check_prediction_schema_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Check classification_v2 multimodal smoke train artifacts.")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/multimodal_smoke_train/multimodal_smoke_train_audit.json"),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/multimodal_smoke_train/multimodal_smoke_predictions.csv"),
    )
    parser.add_argument(
        "--native-predictions-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/multimodal_smoke_train/multimodal_smoke_native_predictions.csv"
        ),
    )
    parser.add_argument(
        "--prediction-schema-audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/multimodal_smoke_train/"
            "multimodal_smoke_prediction_schema_audit.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/multimodal_smoke_train/multimodal_smoke_train.pt"),
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.audit_json.exists():
        errors.append(f"missing_audit={args.audit_json}")
        audit = {}
    else:
        audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        errors.extend(audit.get("errors", []))
    if not args.predictions_csv.exists():
        errors.append(f"missing_predictions={args.predictions_csv}")
        predictions = pd.DataFrame()
    else:
        predictions = pd.read_csv(args.predictions_csv, low_memory=False)
    if not args.native_predictions_csv.exists():
        errors.append(f"missing_native_predictions={args.native_predictions_csv}")
        native_schema_result = {"valid": False, "errors": ["missing_native_predictions"]}
    else:
        native_schema_result = check_prediction_schema_csv(args.native_predictions_csv)
        errors.extend(f"native_prediction_schema:{error}" for error in native_schema_result.get("errors", []))
    if not args.prediction_schema_audit_json.exists():
        errors.append(f"missing_prediction_schema_audit={args.prediction_schema_audit_json}")
    else:
        stored_schema_audit = json.loads(args.prediction_schema_audit_json.read_text(encoding="utf-8"))
        if stored_schema_audit.get("valid") is not True:
            errors.append(f"stored_prediction_schema_audit_invalid={stored_schema_audit.get('errors')}")
    if not args.checkpoint.exists():
        errors.append(f"missing_checkpoint={args.checkpoint}")

    required_prediction_cols = {
        "row_index",
        "window_id",
        "prediction_split",
        "source_split",
        "split_group_key",
        "y_true",
        "y_pred",
        "confidence",
        "correct",
    }
    missing_prediction_cols = sorted(required_prediction_cols.difference(predictions.columns))
    if missing_prediction_cols:
        errors.append(f"missing_prediction_cols={missing_prediction_cols}")
    if audit.get("loss_reduction", 0) <= 0:
        errors.append("loss_not_reduced")
    if int(audit.get("train_rows", 0)) <= 1:
        errors.append("train_rows_too_small")
    if int(audit.get("eval_rows", 0)) <= 1:
        errors.append("eval_rows_too_small")
    if "prediction_split" in predictions.columns:
        observed_splits = set(predictions["prediction_split"].astype(str))
        if not {"train_smoke", "val_smoke"}.issubset(observed_splits):
            errors.append(f"missing_prediction_splits={sorted(observed_splits)}")

    result = {
        "audit_json": str(args.audit_json),
        "predictions_csv": str(args.predictions_csv),
        "native_predictions_csv": str(args.native_predictions_csv),
        "prediction_schema_audit_json": str(args.prediction_schema_audit_json),
        "checkpoint": str(args.checkpoint),
        "train_rows": audit.get("train_rows"),
        "eval_rows": audit.get("eval_rows"),
        "loss_reduction": audit.get("loss_reduction"),
        "prediction_rows": int(len(predictions)),
        "native_prediction_rows": int(native_schema_result.get("prediction_rows", 0)),
        "native_prediction_schema_valid": bool(native_schema_result.get("valid")),
        "prediction_schema": list(predictions.columns),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
