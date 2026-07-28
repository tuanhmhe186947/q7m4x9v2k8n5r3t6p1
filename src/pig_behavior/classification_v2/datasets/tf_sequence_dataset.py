"""Train-ready dataset loader for classification_v2 sequence artifacts.

The loader stays framework-light. TensorFlow/PyTorch trainers can wrap the
returned Pandas/NumPy objects, but feature selection and split decisions must
already be encoded in the audited artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.features.spatial_schema import (
    load_current_spatial_tensor_bundle,
)

DEFAULT_TRAIN_READY_DIR = Path("outputs/classification_v2/train_ready_windows")


@dataclass(slots=True)
class ClassificationV2TrainReadyDataset:
    x_tabular: pd.DataFrame
    y: pd.Series
    train_mask: pd.Series
    sample_weight: pd.Series
    split_manifest: pd.DataFrame
    spatial_sequences: dict[str, np.ndarray] | None
    audit: dict[str, Any]
    event_weight_manifest: pd.DataFrame | None = None
    event_sample_weight: pd.Series | None = None

    def split_indices(self, split: str, *, valid_only: bool = True) -> np.ndarray:
        """Return row indices for one split, optionally restricted to train-valid rows."""
        if "split" not in self.split_manifest.columns:
            raise ValueError("split_manifest has no split column")
        mask = self.split_manifest["split"].astype(str).eq(split)
        if valid_only:
            mask &= _to_bool(self.train_mask)
        return np.flatnonzero(mask.to_numpy())

    def class_counts(self, split: str | None = None, *, valid_only: bool = True) -> dict[str, int]:
        """Return behavior-label counts for all rows or one split."""
        if split is None:
            mask = _to_bool(self.train_mask) if valid_only else pd.Series(True, index=self.y.index)
            labels = self.y[mask]
        else:
            labels = self.y.iloc[self.split_indices(split, valid_only=valid_only)]
        return {str(k): int(v) for k, v in labels.value_counts(dropna=False).to_dict().items()}

    def training_sample_weight(self, *, event_balanced: bool = False) -> pd.Series:
        """Return standard or event-balanced sample weights."""
        if event_balanced:
            if self.event_sample_weight is None:
                raise ValueError("event_weight_manifest.csv was not loaded")
            return self.event_sample_weight
        return self.sample_weight


def load_train_ready_dataset(
    root: Path = DEFAULT_TRAIN_READY_DIR,
    *,
    load_spatial: bool = True,
    load_event_weights: bool = True,
) -> ClassificationV2TrainReadyDataset:
    """Load and validate classification_v2 train-ready artifacts."""
    paths = {
        "x": root / "X_window_features.csv",
        "y": root / "y_behavior.csv",
        "mask": root / "train_mask.csv",
        "weight": root / "sample_weight.csv",
        "split": root / "split_manifest.csv",
        "spatial": root / "X_spatial_sequences.npz",
        "spatial_audit": root / "spatial_sequence_audit.json",
        "event_weight": root / "event_weight_manifest.csv",
    }
    required = ["x", "y", "mask", "weight", "split"]
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing train-ready artifacts: {missing}")

    x = pd.read_csv(paths["x"], low_memory=False)
    y_df = pd.read_csv(paths["y"], low_memory=False)
    mask_df = pd.read_csv(paths["mask"], low_memory=False)
    weight_df = pd.read_csv(paths["weight"], low_memory=False)
    split = pd.read_csv(paths["split"], low_memory=False)

    y = y_df.iloc[:, 0].fillna("").astype(str)
    train_mask = _to_bool(mask_df.iloc[:, 0])
    sample_weight = pd.to_numeric(weight_df.iloc[:, 0], errors="coerce").fillna(0.0).clip(lower=0.0)

    errors: list[str] = []
    row_counts = {
        "x": int(len(x)),
        "y": int(len(y)),
        "mask": int(len(train_mask)),
        "weight": int(len(sample_weight)),
        "split": int(len(split)),
    }
    if len(set(row_counts.values())) != 1:
        errors.append(f"row_count_mismatch={row_counts}")
    if "window_id" not in split.columns:
        errors.append("split_manifest_missing_window_id")
    if "split" not in split.columns:
        errors.append("split_manifest_missing_split")
    if "window_id" in split.columns and split["window_id"].duplicated().any():
        errors.append("duplicate_window_id_in_split_manifest")
    leakage_groups = None
    if "split_group_key" in split.columns and "split" in split.columns:
        leakage_groups = split.groupby("split_group_key")["split"].nunique().gt(1).sum()

    event_weight_manifest = None
    event_sample_weight = None
    event_weight_audit: dict[str, Any] = {"loaded": False}
    if load_event_weights and paths["event_weight"].exists():
        event_weight_manifest = pd.read_csv(paths["event_weight"], low_memory=False)
        event_weight_audit = _validate_event_weights(event_weight_manifest, split)
        errors.extend(event_weight_audit["errors"])
        event_sample_weight = pd.to_numeric(
            event_weight_manifest["event_balanced_sample_weight"], errors="coerce"
        ).fillna(0.0)
    elif load_event_weights:
        event_weight_audit = {"loaded": False, "errors": [f"missing_event_weight_manifest={paths['event_weight']}"]}
        errors.extend(event_weight_audit["errors"])

    spatial_sequences = None
    spatial_shapes: dict[str, list[int]] = {}
    spatial_audit: dict[str, Any] = {"loaded": False}
    if load_spatial and paths["spatial"].exists():
        spatial_sequences, _ = load_current_spatial_tensor_bundle(
            paths["spatial"],
            paths["spatial_audit"],
        )
        spatial_audit = _validate_spatial_sequences(spatial_sequences, expected_rows=len(x))
        spatial_shapes = spatial_audit["shapes"]
        errors.extend(spatial_audit["errors"])
    elif load_spatial:
        errors.append(f"missing_spatial_sequences={paths['spatial']}")

    audit = {
        "root": str(root),
        "paths": {k: str(v) for k, v in paths.items()},
        "row_counts": row_counts,
        "feature_count": int(len(x.columns)),
        "split_counts": split["split"].value_counts(dropna=False).to_dict() if "split" in split.columns else {},
        "train_mask_true": int(train_mask.sum()),
        "train_mask_false": int((~train_mask).sum()),
        "sample_weight_nonzero": int((sample_weight > 0).sum()),
        "event_weight": event_weight_audit,
        "event_sample_weight_nonzero": int((event_sample_weight > 0).sum())
        if event_sample_weight is not None
        else 0,
        "leakage_group_count": None if leakage_groups is None else int(leakage_groups),
        "spatial_shapes": spatial_shapes,
        "spatial": spatial_audit,
        "errors": errors,
    }
    if errors:
        raise ValueError(f"Invalid classification_v2 train-ready dataset: {errors}")
    return ClassificationV2TrainReadyDataset(
        x_tabular=x,
        y=y,
        train_mask=train_mask,
        sample_weight=sample_weight,
        split_manifest=split,
        spatial_sequences=spatial_sequences,
        event_weight_manifest=event_weight_manifest,
        event_sample_weight=event_sample_weight,
        audit=audit,
    )


def _validate_event_weights(event_weights: pd.DataFrame, split: pd.DataFrame) -> dict[str, Any]:
    required = ["window_id", "event_balanced_sample_weight", "window_valid_for_event_weight"]
    missing = [c for c in required if c not in event_weights.columns]
    errors: list[str] = []
    if missing:
        return {"loaded": True, "rows": int(len(event_weights)), "errors": [f"missing_event_weight_columns={missing}"]}
    if len(event_weights) != len(split):
        errors.append(f"event_weight_row_mismatch={len(event_weights)} expected={len(split)}")
    if "window_id" in split.columns:
        window_ids_aligned = event_weights["window_id"].astype(str).equals(split["window_id"].astype(str))
        if not window_ids_aligned:
            errors.append("event_weight_window_id_order_mismatch")
    if pd.to_numeric(event_weights["event_balanced_sample_weight"], errors="coerce").fillna(0.0).lt(0).any():
        errors.append("event_weight_negative_values")
    invalid_nonzero = (
        ~_to_bool(event_weights["window_valid_for_event_weight"])
        & pd.to_numeric(event_weights["event_balanced_sample_weight"], errors="coerce").fillna(0.0).ne(0.0)
    )
    if invalid_nonzero.any():
        errors.append(f"event_weight_invalid_nonzero={int(invalid_nonzero.sum())}")
    return {
        "loaded": True,
        "rows": int(len(event_weights)),
        "event_overlap_cluster_count": int(event_weights["event_overlap_cluster_id"].nunique())
        if "event_overlap_cluster_id" in event_weights
        else None,
        "event_balanced_weight_sum": float(
            pd.to_numeric(event_weights["event_balanced_sample_weight"], errors="coerce").fillna(0.0).sum()
        ),
        "errors": errors,
    }


def _validate_spatial_sequences(spatial_sequences: dict[str, np.ndarray], *, expected_rows: int) -> dict[str, Any]:
    errors: list[str] = []
    shapes = {name: list(arr.shape) for name, arr in spatial_sequences.items()}
    required = {"length_mask", "observed_mask", "frame_index_sequence"}
    missing = sorted(required.difference(spatial_sequences))
    if missing:
        errors.append(f"spatial_missing_arrays={missing}")
    for name, arr in spatial_sequences.items():
        if arr.shape[0] != expected_rows:
            errors.append(f"spatial_row_mismatch:{name}={arr.shape[0]} expected={expected_rows}")
        if name != "frame_index_sequence" and not np.isfinite(arr).all():
            errors.append(f"spatial_nonfinite:{name}")
    if "length_mask" in spatial_sequences and "observed_mask" in spatial_sequences:
        length = spatial_sequences["length_mask"]
        observed = spatial_sequences["observed_mask"]
        if length.shape != observed.shape:
            errors.append(f"spatial_mask_shape_mismatch length={length.shape} observed={observed.shape}")
        elif (observed > length).any():
            errors.append("spatial_observed_outside_length_mask")
    return {"loaded": True, "shapes": shapes, "errors": errors}


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})
