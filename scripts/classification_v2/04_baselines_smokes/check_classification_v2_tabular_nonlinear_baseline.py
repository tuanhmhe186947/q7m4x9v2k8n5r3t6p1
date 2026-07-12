from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.metrics_payload_contract import check_paper_metrics_payload
from pig_behavior.classification_v2.evaluation.prediction_schema_contract import check_prediction_schema_csv


def main() -> None:
    """Validate B2 artifacts before accepting nonlinear tabular evidence."""

    parser = argparse.ArgumentParser(description="Check classification_v2 B2 nonlinear tabular baseline artifacts.")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/tabular_nonlinear_baseline/tabular_nonlinear_audit.json"),
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/tabular_nonlinear_baseline/tabular_nonlinear_metrics.json"),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/tabular_nonlinear_baseline/tabular_nonlinear_predictions.csv"
        ),
    )
    args = parser.parse_args()

    errors: list[str] = []
    audit = _read_json(args.audit_json, errors, "audit")
    metrics = _read_json(args.metrics_json, errors, "metrics")
    if audit.get("errors"):
        errors.extend(f"audit:{error}" for error in audit.get("errors", []))
    if audit.get("prediction_schema_valid") is not True:
        errors.append("audit_prediction_schema_invalid")
    schema_check = check_prediction_schema_csv(args.predictions_csv)
    errors.extend(f"prediction_schema:{error}" for error in schema_check.get("errors", []))
    metrics_check = check_paper_metrics_payload(metrics) if metrics else {"errors": ["missing_metrics"]}
    errors.extend(f"metrics_payload:{error}" for error in metrics_check.get("errors", []))

    result = {
        "audit_json": str(args.audit_json),
        "metrics_json": str(args.metrics_json),
        "predictions_csv": str(args.predictions_csv),
        "prediction_rows": schema_check.get("prediction_rows"),
        "native_temporal_rows": metrics.get("native_temporal_prediction_audit", {}).get("native_temporal_unit_rows")
        if metrics
        else None,
        "macro_f1_supported": metrics.get("native_temporal_metrics", {}).get("macro_f1_supported") if metrics else None,
        "prediction_schema_valid": bool(schema_check.get("valid")),
        "metrics_payload_valid": not metrics_check.get("errors"),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _read_json(path: Path, errors: list[str], name: str) -> dict:
    """Read JSON evidence while accumulating missing-file errors."""

    if not path.exists():
        errors.append(f"missing_{name}={path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
