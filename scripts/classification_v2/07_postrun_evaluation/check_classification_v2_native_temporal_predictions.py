from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "temporal_unit_key",
    "behavior_label",
    "y_true",
    "y_pred",
    "confidence",
    "supporting_window_count",
    "native_prediction_status",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check native temporal prediction collapse artifacts.")
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/native_temporal_predictions/native_temporal_predictions.csv"
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/native_temporal_predictions/native_temporal_prediction_audit.json"
        ),
    )
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions_csv, low_memory=False)
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    errors: list[str] = []
    missing = sorted(REQUIRED_COLUMNS.difference(predictions.columns))
    if missing:
        errors.append(f"missing_prediction_columns={missing}")
    if predictions["temporal_unit_key"].duplicated().any():
        errors.append(f"duplicate_temporal_unit_key={int(predictions['temporal_unit_key'].duplicated().sum())}")
    confidence = pd.to_numeric(predictions["confidence"], errors="coerce")
    if confidence.isna().any() or confidence.lt(0).any() or confidence.gt(1).any():
        errors.append("invalid_confidence_values")
    if audit.get("errors"):
        errors.extend([f"audit_error={err}" for err in audit["errors"]])
    result = {
        "prediction_rows": int(len(predictions)),
        "predicted_units": int(predictions["native_prediction_status"].astype(str).eq("predicted").sum()),
        "unpredicted_units": int(predictions["native_prediction_status"].astype(str).ne("predicted").sum()),
        "audit_warnings": audit.get("warnings", []),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
