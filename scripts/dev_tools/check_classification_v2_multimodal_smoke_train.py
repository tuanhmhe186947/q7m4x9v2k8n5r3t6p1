from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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
        "checkpoint": str(args.checkpoint),
        "train_rows": audit.get("train_rows"),
        "eval_rows": audit.get("eval_rows"),
        "loss_reduction": audit.get("loss_reduction"),
        "prediction_rows": int(len(predictions)),
        "prediction_schema": list(predictions.columns),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
