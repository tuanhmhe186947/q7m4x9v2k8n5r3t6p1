"""Reusable SpatialTCN smoke trainer and prediction writer."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pig_behavior.classification_v2.evaluation.metrics import (
    DEFAULT_LABEL_ORDER,
    evaluate_predictions,
)
from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_GROUP_NAMES,
    load_current_spatial_tensor_bundle,
)
from pig_behavior.classification_v2.models.spatial_tcn import (
    SpatialTCNClassifier,
    SpatialTCNConfig,
)

MODEL_GROUPS = SPATIAL_PREDICTIVE_GROUP_NAMES


@dataclass(frozen=True, slots=True)
class SpatialTCNSmokeTrainConfig:
    root: Path = Path("outputs/classification_v2/train_ready_windows")
    output_dir: Path = Path("outputs/classification_v2/model_smoke/spatial_tcn_smoke_train")
    hidden_dim: int = 96
    dropout: float = 0.0
    lr: float = 0.005
    weight_decay: float = 0.0
    steps: int = 20
    per_class_train: int = 8
    per_class_eval: int = 4
    seed: int = 123
    device: str = "auto"


@dataclass(slots=True)
class SpatialTCNSmokeTrainResult:
    audit: dict[str, Any]
    predictions: pd.DataFrame
    checkpoint_path: Path
    predictions_path: Path
    audit_path: Path


def run_spatial_tcn_smoke_train(config: SpatialTCNSmokeTrainConfig) -> SpatialTCNSmokeTrainResult:
    """Run a small split-safe smoke train/eval and write reproducible artifacts."""
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.per_class_train <= 0 or config.per_class_eval <= 0:
        raise ValueError("per-class sample counts must be positive")
    _set_seed(config.seed)
    device = _resolve_device(config.device)

    bundle = _load_bundle(config.root)
    label_order = _label_order(bundle.y)
    label_to_idx = {label: idx for idx, label in enumerate(label_order)}
    train_indices = _select_balanced_indices(
        bundle.y,
        bundle.train_mask,
        bundle.split,
        split_name="train",
        per_class=config.per_class_train,
    )
    eval_indices = _select_balanced_indices(
        bundle.y,
        bundle.train_mask,
        bundle.split,
        split_name="val",
        per_class=config.per_class_eval,
    )
    if len(train_indices) < 2:
        raise ValueError("not enough train rows for SpatialTCN smoke train")
    if len(eval_indices) < 2:
        raise ValueError("not enough val rows for SpatialTCN smoke eval")

    model = SpatialTCNClassifier(
        SpatialTCNConfig(
            input_dims={name: int(bundle.arrays[name].shape[-1]) for name in MODEL_GROUPS},
            num_classes=len(label_order),
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    train_batch = _batch_from_indices(bundle, train_indices, label_to_idx, device)
    losses: list[float] = []
    model.train()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            train_batch["features"],
            length_mask=train_batch["length_mask"],
            observed_mask=train_batch["observed_mask"],
        )
        loss = loss_fn(logits, train_batch["target"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    predictions = pd.concat(
        [
            predict_indices(
                model,
                bundle,
                train_indices,
                label_order,
                split_name="train_smoke",
                device=device,
            ),
            predict_indices(
                model,
                bundle,
                eval_indices,
                label_order,
                split_name="val_smoke",
                device=device,
            ),
        ],
        ignore_index=True,
    )
    metrics = {
        split: evaluate_predictions(
            group,
            y_true_col="y_true",
            y_pred_col="y_pred",
            label_order=label_order,
        )
        for split, group in predictions.groupby("prediction_split", sort=True)
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "spatial_tcn_smoke_train.pt"
    predictions_path = config.output_dir / "spatial_tcn_smoke_predictions.csv"
    audit_path = config.output_dir / "spatial_tcn_smoke_train_audit.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "label_order": label_order,
            "smoke_config": _jsonable_config(config),
        },
        checkpoint_path,
    )
    predictions.to_csv(predictions_path, index=False)
    audit = {
        "config": _jsonable_config(config),
        "root": str(config.root),
        "checkpoint_path": str(checkpoint_path),
        "predictions_path": str(predictions_path),
        "audit_path": str(audit_path),
        "device": str(device),
        "label_order": label_order,
        "train_rows": int(len(train_indices)),
        "eval_rows": int(len(eval_indices)),
        "train_label_counts": bundle.y.iloc[train_indices].value_counts(dropna=False).to_dict(),
        "eval_label_counts": bundle.y.iloc[eval_indices].value_counts(dropna=False).to_dict(),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_reduction": float(losses[0] - losses[-1]),
        "metrics": metrics,
        "prediction_schema": list(predictions.columns),
        "errors": [],
        "warnings": [
            "smoke subset only; do not report as full model training result",
            "window-level smoke predictions are not independent confirmatory event metrics",
        ],
    }
    if not np.isfinite(losses).all():
        audit["errors"].append("loss_nonfinite")
    if losses[-1] >= losses[0]:
        audit["errors"].append(f"loss_not_reduced initial={losses[0]:.6f} final={losses[-1]:.6f}")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["errors"]:
        raise ValueError(f"SpatialTCN smoke train failed: {audit['errors']}")
    return SpatialTCNSmokeTrainResult(
        audit=audit,
        predictions=predictions,
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
        audit_path=audit_path,
    )


def predict_indices(
    model: SpatialTCNClassifier,
    bundle: _SpatialBundle,
    indices: np.ndarray,
    label_order: list[str],
    *,
    split_name: str,
    device: torch.device,
) -> pd.DataFrame:
    """Write one prediction row per selected window index."""
    batch = _batch_from_indices(
        bundle,
        indices,
        {label: i for i, label in enumerate(label_order)},
        device,
    )
    model.eval()
    with torch.no_grad():
        logits = model(
            batch["features"],
            length_mask=batch["length_mask"],
            observed_mask=batch["observed_mask"],
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    rows = bundle.split.iloc[indices].reset_index(drop=True)
    return pd.DataFrame(
        {
            "row_index": indices.astype(int),
            "window_id": rows["window_id"].astype(str) if "window_id" in rows else "",
            "prediction_split": split_name,
            "source_split": rows["split"].astype(str) if "split" in rows else "",
            "split_group_key": (
                rows["split_group_key"].astype(str)
                if "split_group_key" in rows
                else ""
            ),
            "y_true": bundle.y.iloc[indices].to_numpy(dtype=str),
            "y_pred": [label_order[i] for i in pred_idx],
            "confidence": probs.max(axis=1),
            "correct": [
                label_order[i] == y
                for i, y in zip(
                    pred_idx,
                    bundle.y.iloc[indices],
                    strict=True,
                )
            ],
        }
    )


@dataclass(slots=True)
class _SpatialBundle:
    arrays: dict[str, np.ndarray]
    y: pd.Series
    train_mask: pd.Series
    split: pd.DataFrame


def _load_bundle(root: Path) -> _SpatialBundle:
    arrays, _ = load_current_spatial_tensor_bundle(
        root / "X_spatial_sequences.npz",
        root / "spatial_sequence_audit.json",
    )
    missing = [
        name
        for name in [*MODEL_GROUPS, "length_mask", "observed_mask"]
        if name not in arrays
    ]
    if missing:
        raise ValueError(f"missing spatial arrays: {missing}")
    y = pd.read_csv(root / "y_behavior.csv").iloc[:, 0].fillna("").astype(str)
    train_mask = _read_bool(root / "train_mask.csv")
    split = pd.read_csv(root / "split_manifest.csv", low_memory=False)
    expected = len(y)
    row_counts = {"y": len(y), "train_mask": len(train_mask), "split": len(split)}
    row_counts.update({name: int(arr.shape[0]) for name, arr in arrays.items()})
    mismatched = {name: count for name, count in row_counts.items() if count != expected}
    if mismatched:
        raise ValueError(f"row count mismatch against y={expected}: {mismatched}")
    return _SpatialBundle(arrays=arrays, y=y, train_mask=train_mask, split=split)


def _batch_from_indices(
    bundle: _SpatialBundle,
    indices: np.ndarray,
    label_to_idx: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    features = {
        name: torch.from_numpy(bundle.arrays[name][indices]).float().to(device)
        for name in MODEL_GROUPS
    }
    target_labels = bundle.y.iloc[indices].tolist()
    target = torch.tensor(
        [label_to_idx[label] for label in target_labels],
        dtype=torch.long,
    ).to(device)
    return {
        "features": features,
        "length_mask": torch.from_numpy(bundle.arrays["length_mask"][indices]).float().to(device),
        "observed_mask": torch.from_numpy(
            bundle.arrays["observed_mask"][indices]
        )
        .float()
        .to(device),
        "target": target,
    }


def _select_balanced_indices(
    y: pd.Series,
    train_mask: pd.Series,
    split: pd.DataFrame,
    *,
    split_name: str,
    per_class: int,
) -> np.ndarray:
    valid_split = split["split"].astype(str).eq(split_name) & train_mask
    selected: list[int] = []
    for label in _label_order(y):
        label_indices = np.flatnonzero((valid_split & y.eq(label)).to_numpy())
        selected.extend(label_indices[:per_class].tolist())
    return np.array(selected, dtype=np.int64)


def _label_order(y: pd.Series) -> list[str]:
    observed = set(y.dropna().astype(str).tolist())
    labels = [label for label in DEFAULT_LABEL_ORDER if label in observed]
    labels.extend(sorted(observed.difference(labels)))
    return labels


def _read_bool(path: Path) -> pd.Series:
    series = pd.read_csv(path).iloc[:, 0]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable_config(config: SpatialTCNSmokeTrainConfig) -> dict[str, Any]:
    out = asdict(config)
    out["root"] = str(config.root)
    out["output_dir"] = str(config.output_dir)
    return out
