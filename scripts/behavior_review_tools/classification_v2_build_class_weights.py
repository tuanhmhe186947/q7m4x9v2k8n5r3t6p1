from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("outputs/classification_v2/train_ready_windows")
CONFUSION_PAIRS = [
    ["fight", "social-nose"],
    ["fight", "stand"],
    ["fight", "move"],
    ["eat", "stand"],
    ["eat", "explore"],
    ["drink", "stand"],
    ["drink", "explore"],
    ["playwithtoy", "explore"],
    ["playwithtoy", "stand"],
    ["playwithtoy", "move"],
    ["lying", "sitting"],
    ["move", "explore"],
    ["move", "stand"],
]


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build class/sample-weight policy for classification_v2 training.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-weight", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    y = pd.read_csv(args.root / "y_behavior.csv", low_memory=False).iloc[:, 0].astype(str)
    mask = _to_bool(pd.read_csv(args.root / "train_mask.csv", low_memory=False).iloc[:, 0])
    split = pd.read_csv(args.root / "split_manifest.csv", low_memory=False)
    split_mask = split["split"].astype(str).eq(args.split)
    train_y = y[mask & split_mask]
    counts = train_y.value_counts().sort_index()
    if counts.empty:
        raise SystemExit(f"No valid rows for split={args.split}")

    median_count = float(counts.median())
    weights = (median_count / counts.astype(float)).pow(0.5).clip(lower=0.25, upper=args.max_weight)
    policy = {
        "root": str(args.root),
        "split": args.split,
        "valid_rows": int(len(train_y)),
        "class_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
        "class_weight_policy": "sqrt(median_class_count / class_count), clipped",
        "max_weight": float(args.max_weight),
        "class_weights": {str(k): float(v) for k, v in weights.to_dict().items()},
        "confusion_focus_pairs": CONFUSION_PAIRS,
    }

    output_json = args.output_json or (args.root / "class_weight_policy.json")
    output_json.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
