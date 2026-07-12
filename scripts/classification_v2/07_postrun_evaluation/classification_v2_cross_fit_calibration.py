from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.calibration import cross_fit_temperature_scaling


def main() -> None:
    """Cross-fit temperature calibration over complete native-unit OOF predictions."""

    parser = argparse.ArgumentParser(description="Cross-fit classification_v2 native-unit calibration.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--expected-fold-count", type=int, default=None)
    args = parser.parse_args()

    predictions = pd.read_csv(args.input_csv, low_memory=False)
    calibrated, audit = cross_fit_temperature_scaling(
        predictions,
        ece_bins=args.ece_bins,
        expected_fold_count=args.expected_fold_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_path = args.output_dir / "cross_fitted_calibrated_native_predictions.csv"
    audit_path = args.output_dir / "cross_fitted_calibration_audit.json"
    calibrated.to_csv(calibrated_path, index=False)
    audit["input_csv"] = str(args.input_csv)
    audit["calibrated_predictions_csv"] = str(calibrated_path)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
