from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pig_behavior.classification_v2.training.spatial_tcn_smoke import MODEL_GROUPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial control shortcut checks for classification_v2."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/spatial_control_shortcut_audit.json"),
    )
    parser.add_argument("--max-rows-per-split", type=int, default=5000)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = {name: value for name, value in np.load(args.root / "X_spatial_sequences.npz").items()}
    split = pd.read_csv(args.root / "split_manifest.csv", low_memory=False)
    mask = _read_bool(args.root / "train_mask.csv")
    missing = [
        name
        for name in [*MODEL_GROUPS, "length_mask", "observed_mask"]
        if name not in arrays
    ]
    if missing:
        raise ValueError(f"missing spatial arrays: {missing}")
    if "source_type" not in split.columns or "split" not in split.columns:
        raise ValueError("split_manifest must include source_type and split")

    train_idx = _sample_indices(split, mask, "train", args.max_rows_per_split, args.seed)
    test_idx = _sample_indices(split, mask, "test", args.max_rows_per_split, args.seed + 1)
    y_train = split.iloc[train_idx]["source_type"].fillna("").astype(str)
    y_test = split.iloc[test_idx]["source_type"].fillna("").astype(str)

    controls = {
        "real_sequence": _flatten_real,
        "repeat_first_frame": _flatten_repeat_first,
        "mean_only": _flatten_mean_only,
    }
    results = {}
    for name, builder in controls.items():
        x_train = builder(arrays, train_idx)
        x_test = builder(arrays, test_idx)
        results[name] = _fit_predict_source(
            x_train,
            y_train,
            x_test,
            y_test,
            max_iter=args.max_iter,
        )

    audit = {
        "root": str(args.root),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "source_counts_train": y_train.value_counts(dropna=False).to_dict(),
        "source_counts_test": y_test.value_counts(dropna=False).to_dict(),
        "controls": results,
        "interpretation": (
            "High repeat_first_frame or mean_only source accuracy means static "
            "geometry/domain cues remain sufficient to identify source_type; "
            "temporal models must report source-balanced "
            "and video/session-safe metrics."
        ),
        "errors": [],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


def _sample_indices(
    split: pd.DataFrame,
    mask: pd.Series,
    split_name: str,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    valid = np.flatnonzero((split["split"].astype(str).eq(split_name) & mask).to_numpy())
    if len(valid) <= max_rows:
        return valid
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(valid, size=max_rows, replace=False))


def _fit_predict_source(
    x_train: np.ndarray,
    y_train: pd.Series,
    x_test: np.ndarray,
    y_test: pd.Series,
    *,
    max_iter: int,
) -> dict[str, object]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=0),
    )
    model.fit(x_train, y_train)
    pred = pd.Series(model.predict(x_test), index=y_test.index)
    labels = sorted(set(y_train).union(set(y_test)).union(set(pred)))
    cm = confusion_matrix(y_test, pred, labels=labels)
    return {
        "feature_dim": int(x_train.shape[1]),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "labels": labels,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def _flatten_real(arrays: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    pieces = [arrays[name][indices].reshape(len(indices), -1) for name in MODEL_GROUPS]
    return np.concatenate(pieces, axis=1)


def _flatten_repeat_first(arrays: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    length_mask = arrays["length_mask"][indices]
    observed_mask = arrays["observed_mask"][indices]
    pieces = []
    for name in MODEL_GROUPS:
        values = arrays[name][indices].copy()
        first_positions = _first_observed_positions(length_mask, observed_mask)
        first_values = values[np.arange(len(indices)), first_positions]
        values = (
            np.repeat(first_values[:, None, :], values.shape[1], axis=1)
            * length_mask[:, :, None]
        )
        pieces.append(values.reshape(len(indices), -1))
    return np.concatenate(pieces, axis=1)


def _flatten_mean_only(arrays: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    observed = arrays["observed_mask"][indices]
    pieces = []
    denom = observed.sum(axis=1).clip(min=1.0)[:, None]
    for name in MODEL_GROUPS:
        values = arrays[name][indices]
        pieces.append((values * observed[:, :, None]).sum(axis=1) / denom)
    return np.concatenate(pieces, axis=1)


def _first_observed_positions(length_mask: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    usable = (length_mask * observed_mask) > 0
    has_observed = usable.any(axis=1)
    positions = usable.argmax(axis=1)
    positions[~has_observed] = 0
    return positions


def _read_bool(path: Path) -> pd.Series:
    series = pd.read_csv(path).iloc[:, 0]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()
