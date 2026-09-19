"""Collapse and evaluate complete Q2 out-of-fold window predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.evaluation.native_unit_metrics import (
    evaluate_native_oof,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Q2 OOF predictions at native-unit level."
    )
    parser.add_argument("--window-predictions", type=Path, required=True)
    parser.add_argument(
        "--fold-assignments",
        type=Path,
        default=Path(
            "outputs/classification_v2/q2_grouped_folds/"
            "q2_outer_fold_assignments.csv"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
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
    native_path = args.output_dir / "native_unit_predictions.csv"
    metrics_path = args.output_dir / "q2_oof_metrics.json"
    per_class_path = args.output_dir / "metrics_per_class.csv"
    confusion_path = args.output_dir / "confusion_matrix.csv"
    support_path = args.output_dir / "class_fold_support.csv"
    require_output_paths_available(
        [
            native_path,
            metrics_path,
            per_class_path,
            confusion_path,
            support_path,
        ],
        overwrite=args.overwrite,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(native_path, index=False)
    pooled = audit["pooled_metrics"]
    per_class = pd.DataFrame.from_dict(
        pooled.get("per_class", {}),
        orient="index",
    ).rename_axis("behavior_label").reset_index()
    confusion_payload = pooled.get("confusion_matrix", {})
    confusion = pd.DataFrame(
        confusion_payload.get("values", []),
        index=confusion_payload.get("index", []),
        columns=confusion_payload.get("columns", []),
    )
    per_class.to_csv(per_class_path, index=False)
    confusion.to_csv(confusion_path, index_label="true_label")
    pd.DataFrame(audit["class_fold_support"]).to_csv(
        support_path,
        index=False,
    )
    payload = {
        "window_predictions_csv": str(args.window_predictions),
        "fold_assignments_csv": str(args.fold_assignments),
        "native_predictions_csv": str(native_path),
        "metrics_per_class_csv": str(per_class_path),
        "confusion_matrix_csv": str(confusion_path),
        "class_fold_support_csv": str(support_path),
        **audit,
    }
    metrics_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
