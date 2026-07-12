from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.metrics import (
    DEFAULT_LABEL_ORDER,
    evaluate_predictions,
    evaluate_predictions_by_slice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classification_v2 predictions with confusion-focus metrics.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/confusion_focus_eval.json"),
    )
    parser.add_argument("--y-true-col", default="y_true")
    parser.add_argument("--y-pred-col", default="y_pred")
    parser.add_argument("--split-col", default="split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions_csv.exists():
        raise FileNotFoundError(args.predictions_csv)
    frame = pd.read_csv(args.predictions_csv, low_memory=False)
    missing = [c for c in [args.y_true_col, args.y_pred_col] if c not in frame.columns]
    if missing:
        raise SystemExit(f"Predictions CSV missing required columns: {missing}")

    result = {
        "predictions_csv": str(args.predictions_csv),
        "y_true_col": args.y_true_col,
        "y_pred_col": args.y_pred_col,
        "rows": int(len(frame)),
        "overall": evaluate_predictions(
            frame,
            y_true_col=args.y_true_col,
            y_pred_col=args.y_pred_col,
            label_order=DEFAULT_LABEL_ORDER,
        ),
    }
    if args.split_col in frame.columns:
        result["split_counts"] = frame[args.split_col].value_counts(dropna=False).to_dict()
        result["by_split"] = evaluate_predictions_by_slice(
            frame,
            y_true_col=args.y_true_col,
            y_pred_col=args.y_pred_col,
            slice_col=args.split_col,
            label_order=DEFAULT_LABEL_ORDER,
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
