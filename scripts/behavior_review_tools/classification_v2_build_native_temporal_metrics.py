from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate classification_v2 window predictions into native temporal-unit metrics."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unit-id-col", default="temporal_unit_key")
    parser.add_argument("--true-col", default="behavior_true")
    parser.add_argument("--pred-col", default="behavior_pred")
    parser.add_argument("--weight-col", default="window_sample_weight")
    parser.add_argument("--valid-col", default="window_valid_for_main_train")
    parser.add_argument("--window-id-col", default="window_id")
    parser.add_argument("--prob-prefix", default="prob_")
    parser.add_argument("--include-invalid-windows", action="store_true")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_csv)
    config = NativeTemporalMetricsConfig(
        unit_id_col=args.unit_id_col,
        true_col=args.true_col,
        pred_col=args.pred_col,
        weight_col=args.weight_col or None,
        valid_col=args.valid_col or None,
        window_id_col=args.window_id_col,
        prob_prefix=args.prob_prefix,
        include_invalid_windows=args.include_invalid_windows,
    )
    units, payload = build_native_temporal_metrics(predictions, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    units_path = args.output_dir / "native_temporal_unit_predictions.csv"
    metrics_path = args.output_dir / "native_temporal_metrics.json"
    units.to_csv(units_path, index=False)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"units_csv": str(units_path), "metrics_json": str(metrics_path)}, indent=2))

    audit = payload["native_temporal_prediction_audit"]
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
