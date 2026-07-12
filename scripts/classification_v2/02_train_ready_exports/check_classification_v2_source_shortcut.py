from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether train-ready X predicts source_type shortcut.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/source_shortcut_audit.json"),
    )
    parser.add_argument("--max-iter", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = pd.read_csv(args.root / "X_window_features.csv", low_memory=False)
    split = pd.read_csv(args.root / "split_manifest.csv", low_memory=False)
    mask = _read_bool(args.root / "train_mask.csv")
    required = ["source_type", "split"]
    missing = [c for c in required if c not in split.columns]
    if missing:
        raise ValueError(f"split_manifest missing columns: {missing}")
    if len(x) != len(split) or len(x) != len(mask):
        raise ValueError(f"row mismatch x={len(x)} split={len(split)} mask={len(mask)}")

    train = split["split"].astype(str).eq("train") & mask
    test = split["split"].astype(str).eq("test") & mask
    if train.sum() == 0 or test.sum() == 0:
        raise ValueError("Need non-empty train/test valid rows")

    y_train = split.loc[train, "source_type"].fillna("").astype(str)
    y_test = split.loc[test, "source_type"].fillna("").astype(str)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=args.max_iter, class_weight="balanced", random_state=0),
    )
    model.fit(x.loc[train], y_train)
    pred = pd.Series(model.predict(x.loc[test]), index=y_test.index)
    labels = sorted(set(y_train).union(y_test).union(pred))
    cm = confusion_matrix(y_test, pred, labels=labels)
    accuracy = float(accuracy_score(y_test, pred))
    balanced_accuracy = float(balanced_accuracy_score(y_test, pred))
    audit = {
        "root": str(args.root),
        "rows": int(len(x)),
        "feature_count": int(len(x.columns)),
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "source_labels": labels,
        "train_source_counts": y_train.value_counts(dropna=False).to_dict(),
        "test_source_counts": y_test.value_counts(dropna=False).to_dict(),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": {"labels": labels, "values": cm.astype(int).tolist()},
        "warnings": [
            "High source predictability means features may encode source/domain; source_type itself is not in X."
        ],
        "errors": [],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def _read_bool(path: Path) -> pd.Series:
    series = pd.read_csv(path).iloc[:, 0]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()
