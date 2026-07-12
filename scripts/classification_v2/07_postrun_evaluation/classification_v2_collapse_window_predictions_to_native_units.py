from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.native_temporal_collapse import (
    collapse_window_predictions_to_native_units,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collapse classification_v2 window predictions to native units.")
    parser.add_argument(
        "--window-predictions-csv",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/spatial_tcn_smoke_train/spatial_tcn_smoke_predictions.csv"),
    )
    parser.add_argument(
        "--window-manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv"),
    )
    parser.add_argument(
        "--native-units-csv",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units/native_temporal_unit_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/native_temporal_predictions"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collapse_window_predictions_to_native_units(
        pd.read_csv(args.window_predictions_csv, low_memory=False),
        pd.read_csv(args.window_manifest_csv, usecols=["window_id", "temporal_unit_keys_window"], low_memory=False),
        pd.read_csv(args.native_units_csv, low_memory=False),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "native_temporal_predictions.csv"
    audit_path = args.output_dir / "native_temporal_prediction_audit.json"
    result.predictions.to_csv(predictions_path, index=False)
    audit = {
        "window_predictions_csv": str(args.window_predictions_csv),
        "window_manifest_csv": str(args.window_manifest_csv),
        "native_units_csv": str(args.native_units_csv),
        "predictions_csv": str(predictions_path),
        **result.audit,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
