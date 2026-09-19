"""Grouped source probes over the exact spatial sequence groups used by the trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pig_behavior.classification_v2.features.spatial_schema import (
    load_current_spatial_tensor_bundle,
)
from pig_behavior.classification_v2.training.spatial_tcn_smoke import MODEL_GROUPS


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe source shortcuts in strict spatial sequence X.")
    parser.add_argument("--root", type=Path, default=Path("outputs/classification_v2/train_ready_windows"))
    parser.add_argument(
        "--grouped-roles",
        type=Path,
        default=Path("outputs/classification_v2/q2_grouped_folds/q2_outer_inner_roles.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/domain_controls/grouped_spatial_source_probe.json"),
    )
    parser.add_argument("--max-rows-per-source-role", type=int, default=1500)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    arrays, _ = load_current_spatial_tensor_bundle(
        args.root / "X_spatial_sequences.npz",
        args.root / "spatial_sequence_audit.json",
    )
    metadata = pd.read_csv(args.root / "split_manifest.csv", low_memory=False)
    events = pd.read_csv(args.root / "event_weight_manifest.csv", low_memory=False)
    roles = pd.read_csv(args.grouped_roles, low_memory=False)
    _validate_alignment(arrays, metadata, events)
    controls = {
        "real_sequence": _flatten_real,
        "repeat_first_frame": _flatten_repeat_first,
        "mean_only": _flatten_mean_only,
    }
    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in controls}
    fold_audits: list[dict[str, object]] = []
    base = metadata[["window_id", "source_type", "window_valid_for_main_train"]].copy()
    base["temporal_unit_key"] = events["temporal_unit_keys_window"].astype(str).to_numpy()
    base["row_index"] = np.arange(len(base), dtype=np.int64)
    for fold_number, fold_id in enumerate(sorted(roles["outer_fold_id"].astype(str).unique())):
        fold_roles = roles.loc[roles["outer_fold_id"].astype(str).eq(fold_id), ["temporal_unit_key", "role"]]
        merged = base.merge(fold_roles, on="temporal_unit_key", how="left", validate="many_to_one")
        valid = _to_bool(merged["window_valid_for_main_train"])
        train_idx = _stratified_indices(
            merged,
            valid & merged["role"].eq("train"),
            args.max_rows_per_source_role,
            args.seed + fold_number,
        )
        test_idx = _stratified_indices(
            merged,
            valid & merged["role"].eq("test"),
            args.max_rows_per_source_role,
            args.seed + 100 + fold_number,
        )
        y_train = metadata.iloc[train_idx]["source_type"].astype(str).reset_index(drop=True)
        y_test = metadata.iloc[test_idx]["source_type"].astype(str).reset_index(drop=True)
        if y_train.nunique() != 2:
            raise ValueError(f"spatial source probe train fold lacks two sources={fold_id}")
        fold_control: dict[str, object] = {}
        for name, flatten in controls.items():
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=args.max_iter, class_weight="balanced", random_state=0),
            )
            model.fit(flatten(arrays, train_idx), y_train)
            predicted = model.predict(flatten(arrays, test_idx))
            part = pd.DataFrame(
                {
                    "true": y_test,
                    "predicted": predicted,
                    "outer_fold_id": fold_id,
                }
            )
            pooled[name].append(part)
            fold_control[name] = {
                "balanced_accuracy_supported": _supported_balanced_accuracy(y_test, predicted),
                "test_source_count": int(y_test.nunique()),
            }
        fold_audits.append(
            {
                "outer_fold_id": fold_id,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_source_counts": y_train.value_counts().sort_index().to_dict(),
                "test_source_counts": y_test.value_counts().sort_index().to_dict(),
                "controls": fold_control,
                "scaler_fit_on_train_only": True,
            }
        )
    pooled_metrics = {}
    for name, parts in pooled.items():
        frame = pd.concat(parts, ignore_index=True)
        pooled_metrics[name] = {
            "rows": int(len(frame)),
            "balanced_accuracy": float(balanced_accuracy_score(frame["true"], frame["predicted"])),
        }
    audit = {
        "schema_version": "classification_v2_grouped_spatial_source_probe_v1",
        "model_input_groups": list(MODEL_GROUPS),
        "control_definition": {
            "real_sequence": "full strict spatial sequence",
            "repeat_first_frame": "static first-observed geometry repeated over time",
            "mean_only": "observed-frame temporal mean",
        },
        "fold_count": len(fold_audits),
        "folds": fold_audits,
        "pooled_controls": pooled_metrics,
        "source_type_in_model_x": False,
        "errors": [],
        "valid": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


def _validate_alignment(arrays: dict[str, np.ndarray], metadata: pd.DataFrame, events: pd.DataFrame) -> None:
    missing = [name for name in [*MODEL_GROUPS, "length_mask", "observed_mask"] if name not in arrays]
    if missing:
        raise ValueError(f"missing strict spatial arrays: {missing}")
    if (
        not metadata["window_id"]
        .astype(str)
        .reset_index(drop=True)
        .equals(events["window_id"].astype(str).reset_index(drop=True))
    ):
        raise ValueError("spatial source probe window/event alignment mismatch")
    if any(len(value) != len(metadata) for value in arrays.values()):
        raise ValueError("spatial source probe array row count mismatch")


def _stratified_indices(frame: pd.DataFrame, mask: pd.Series, max_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in frame.loc[mask].groupby("source_type", sort=True):
        indices = group["row_index"].to_numpy(dtype=np.int64)
        if len(indices) > max_rows:
            indices = rng.choice(indices, size=max_rows, replace=False)
        selected.extend(indices.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _flatten_real(arrays: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    pieces = [arrays[name][indices].reshape(len(indices), -1) for name in MODEL_GROUPS]
    return np.concatenate(pieces, axis=1)


def _flatten_repeat_first(arrays: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    length_mask = arrays["length_mask"][indices]
    observed_mask = arrays["observed_mask"][indices]
    usable = (length_mask * observed_mask) > 0
    positions = usable.argmax(axis=1)
    positions[~usable.any(axis=1)] = 0
    pieces = []
    for name in MODEL_GROUPS:
        values = arrays[name][indices]
        first = values[np.arange(len(indices)), positions]
        repeated = np.repeat(first[:, None, :], values.shape[1], axis=1)
        pieces.append((repeated * length_mask[:, :, None]).reshape(len(indices), -1))
    return np.concatenate(pieces, axis=1)


def _flatten_mean_only(arrays: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    observed = arrays["observed_mask"][indices]
    denominator = observed.sum(axis=1).clip(min=1.0)[:, None]
    return np.concatenate(
        [(arrays[name][indices] * observed[:, :, None]).sum(axis=1) / denominator for name in MODEL_GROUPS],
        axis=1,
    )


def _supported_balanced_accuracy(true: pd.Series, predicted: np.ndarray) -> float:
    recalls = []
    prediction = pd.Series(predicted, index=true.index)
    for label in sorted(true.unique()):
        mask = true.eq(label)
        recalls.append(float(prediction.loc[mask].eq(label).mean()))
    return float(np.mean(recalls))


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()
