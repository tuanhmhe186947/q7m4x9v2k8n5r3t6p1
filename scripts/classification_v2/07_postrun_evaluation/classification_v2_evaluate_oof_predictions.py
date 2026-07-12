"""Collapse and evaluate complete Q2 out-of-fold window predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.native_unit_metrics import evaluate_native_oof


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Q2 OOF predictions at native-unit level.")
    parser.add_argument("--window-predictions", type=Path, required=True)
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path("outputs/classification_v2/q2_grouped_folds/q2_outer_fold_assignments.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/q2_oof_evaluation"),
    )
    args = parser.parse_args()
    units, audit = evaluate_native_oof(
        pd.read_csv(args.window_predictions, low_memory=False),
        pd.read_csv(args.fold_assignments, low_memory=False),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.output_dir / "native_unit_predictions.csv", index=False)
    payload = {
        "window_predictions_csv": str(args.window_predictions),
        "fold_assignments_csv": str(args.fold_assignments),
        "native_predictions_csv": str(args.output_dir / "native_unit_predictions.csv"),
        **audit,
    }
    (args.output_dir / "q2_oof_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
