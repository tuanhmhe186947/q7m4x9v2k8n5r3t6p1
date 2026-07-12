from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_PREDICTION_COLUMNS = {
    "row_index",
    "window_id",
    "prediction_split",
    "source_split",
    "y_true",
    "y_pred",
    "confidence",
    "correct",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SpatialTCN smoke train artifacts.")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/spatial_tcn_smoke_train/spatial_tcn_smoke_train_audit.json"
        ),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/spatial_tcn_smoke_train/spatial_tcn_smoke_predictions.csv"),
    )
    args = parser.parse_args()

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.predictions_csv, low_memory=False)
    errors: list[str] = []
    missing_cols = sorted(REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing_cols:
        errors.append(f"missing_prediction_columns={missing_cols}")
    if audit.get("errors"):
        errors.extend([f"audit_error={err}" for err in audit["errors"]])
    if len(predictions) != int(audit.get("train_rows", 0)) + int(audit.get("eval_rows", 0)):
        errors.append("prediction_row_count_mismatch")
    confidence = pd.to_numeric(predictions.get("confidence"), errors="coerce")
    if confidence.isna().any() or confidence.lt(0).any() or confidence.gt(1).any():
        errors.append("invalid_confidence_values")
    if set(predictions.get("prediction_split", pd.Series(dtype=str)).astype(str)) != {"train_smoke", "val_smoke"}:
        errors.append("missing_train_or_val_smoke_predictions")

    result = {
        "audit_json": str(args.audit_json),
        "predictions_csv": str(args.predictions_csv),
        "prediction_rows": int(len(predictions)),
        "prediction_splits": predictions["prediction_split"].value_counts(dropna=False).to_dict()
        if "prediction_split" in predictions
        else {},
        "initial_loss": audit.get("initial_loss"),
        "final_loss": audit.get("final_loss"),
        "loss_reduction": audit.get("loss_reduction"),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
