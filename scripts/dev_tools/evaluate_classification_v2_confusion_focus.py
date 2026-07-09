from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FOCUS_PAIRS = [
    ("fight", "social-nose"),
    ("fight", "stand"),
    ("fight", "move"),
    ("eat", "stand"),
    ("eat", "explore"),
    ("drink", "stand"),
    ("drink", "explore"),
    ("playwithtoy", "explore"),
    ("playwithtoy", "stand"),
    ("playwithtoy", "move"),
    ("lying", "sitting"),
    ("move", "explore"),
    ("move", "stand"),
]


def evaluate_frame(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> dict:
    y_true = df[y_true_col].fillna("").astype(str)
    y_pred = df[y_pred_col].fillna("").astype(str)
    labels = sorted(set(y_true).union(y_pred))
    confusion = pd.crosstab(y_true, y_pred, dropna=False).reindex(index=labels, columns=labels, fill_value=0)

    focus = {}
    for a, b in FOCUS_PAIRS:
        a_to_b = int(confusion.loc[a, b]) if a in confusion.index and b in confusion.columns else 0
        b_to_a = int(confusion.loc[b, a]) if b in confusion.index and a in confusion.columns else 0
        focus[f"{a}__vs__{b}"] = {
            f"{a}_predicted_as_{b}": a_to_b,
            f"{b}_predicted_as_{a}": b_to_a,
            "total_pair_confusions": a_to_b + b_to_a,
        }

    per_class = {}
    for label in labels:
        tp = int(confusion.loc[label, label]) if label in confusion.index and label in confusion.columns else 0
        support = int(confusion.loc[label].sum()) if label in confusion.index else 0
        predicted = int(confusion[label].sum()) if label in confusion.columns else 0
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}

    return {
        "rows": int(len(df)),
        "accuracy": float((y_true == y_pred).mean()) if len(df) else 0.0,
        "per_class": per_class,
        "focus_pairs": focus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate confusion-focused classification_v2 predictions.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/classification_v2/train_ready_windows/confusion_focus_eval.json"))
    parser.add_argument("--y-true-col", default="y_true")
    parser.add_argument("--y-pred-col", default="y_pred")
    parser.add_argument("--split-col", default="split")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv, low_memory=False)
    missing = [c for c in [args.y_true_col, args.y_pred_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"Predictions CSV missing required columns: {missing}")

    result = {
        "predictions_csv": str(args.predictions_csv),
        **evaluate_frame(df, args.y_true_col, args.y_pred_col),
    }
    if args.split_col in df.columns:
        result["split_counts"] = df[args.split_col].value_counts(dropna=False).to_dict()
        result["by_split"] = {
            str(split): evaluate_frame(split_df, args.y_true_col, args.y_pred_col)
            for split, split_df in df.groupby(args.split_col, sort=True)
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
