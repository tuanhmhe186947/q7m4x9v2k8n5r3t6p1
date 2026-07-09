from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.tf_sequence_dataset import load_train_ready_dataset

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    SGDClassifier = None  # type: ignore
    StandardScaler = None  # type: ignore
    accuracy_score = None  # type: ignore
    balanced_accuracy_score = None  # type: ignore
    classification_report = None  # type: ignore
    f1_score = None  # type: ignore
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


DEFAULT_ROOT = Path("outputs/classification_v2/train_ready_windows")


def _sample_indices_by_label(
    indices: np.ndarray,
    labels: pd.Series,
    *,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    if max_rows <= 0 or len(indices) <= max_rows:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = {}
    for idx in indices:
        by_label.setdefault(str(labels.iloc[idx]), []).append(int(idx))
    total = len(indices)
    selected: list[int] = []
    for label in sorted(by_label):
        group = np.asarray(by_label[label], dtype=np.int64)
        quota = max(1, int(round(max_rows * len(group) / total)))
        quota = min(quota, len(group))
        chosen = rng.choice(group, size=quota, replace=False)
        selected.extend(int(x) for x in chosen)
    if len(selected) > max_rows:
        selected = list(rng.choice(np.asarray(selected, dtype=np.int64), size=max_rows, replace=False))
    return np.sort(np.asarray(selected, dtype=np.int64))


def _load_class_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.get("class_weights", {}).items()}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small leakage-safe tabular smoke training check for classification_v2."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "smoke_tabular_baseline")
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--max-eval-rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--no-class-weight", action="store_true")
    args = parser.parse_args()

    if SGDClassifier is None or StandardScaler is None:
        raise SystemExit(f"scikit-learn is required for smoke training: {SKLEARN_IMPORT_ERROR!r}")

    ds = load_train_ready_dataset(args.root, load_spatial=False)
    train_idx = ds.split_indices("train", valid_only=True)
    val_idx = ds.split_indices("val", valid_only=True)
    test_idx = ds.split_indices("test", valid_only=True)

    train_idx = _sample_indices_by_label(train_idx, ds.y, max_rows=args.max_train_rows, seed=args.seed)
    val_idx = _sample_indices_by_label(val_idx, ds.y, max_rows=args.max_eval_rows, seed=args.seed + 1)
    test_idx = _sample_indices_by_label(test_idx, ds.y, max_rows=args.max_eval_rows, seed=args.seed + 2)

    x_train = ds.x_tabular.iloc[train_idx].to_numpy(dtype=np.float32, copy=True)
    y_train = ds.y.iloc[train_idx].to_numpy()
    w_train = ds.sample_weight.iloc[train_idx].to_numpy(dtype=np.float32, copy=True)

    class_weights = {} if args.no_class_weight else _load_class_weights(args.root / "class_weight_policy.json")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    clf = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=0.0005,
        l1_ratio=0.05,
        class_weight=class_weights or None,
        max_iter=args.max_iter,
        tol=1e-3,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(x_train_scaled, y_train, sample_weight=w_train)

    prediction_frames = []
    metrics: dict[str, Any] = {}
    for split_name, split_idx in [("val", val_idx), ("test", test_idx)]:
        x_eval = ds.x_tabular.iloc[split_idx].to_numpy(dtype=np.float32, copy=True)
        y_true = ds.y.iloc[split_idx].to_numpy()
        y_pred = clf.predict(scaler.transform(x_eval))
        metrics[split_name] = _metrics(y_true, y_pred)
        confidence = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(scaler.transform(x_eval))
            confidence = proba.max(axis=1)
        pred = pd.DataFrame(
            {
                "row_index": split_idx,
                "window_id": ds.split_manifest.iloc[split_idx]["window_id"].to_numpy(),
                "split": split_name,
                "y_true": y_true,
                "y_pred": y_pred,
                "confidence": confidence if confidence is not None else np.nan,
            }
        )
        prediction_frames.append(pred)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "smoke_predictions.csv"
    metrics_path = args.output_dir / "smoke_metrics.json"
    predictions.to_csv(predictions_path, index=False)

    audit = {
        "root": str(args.root),
        "output_dir": str(args.output_dir),
        "predictions_csv": str(predictions_path),
        "metrics_json": str(metrics_path),
        "mode": "small_tabular_smoke_not_full_training",
        "feature_count": int(ds.x_tabular.shape[1]),
        "feature_columns": list(ds.x_tabular.columns),
        "class_weights_used": class_weights,
        "sampled_rows": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "dataset_audit": ds.audit,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["mode", "feature_count", "sampled_rows", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
