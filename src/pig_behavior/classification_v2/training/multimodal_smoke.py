"""Tiny split-safe multimodal smoke trainer for classification_v2."""

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

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    image_sequence_collate,
)
from pig_behavior.classification_v2.evaluation.metrics import DEFAULT_LABEL_ORDER, evaluate_predictions
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.training.spatial_tcn_smoke import MODEL_GROUPS


@dataclass(frozen=True, slots=True)
class MultimodalSmokeTrainConfig:
    root: Path = Path("outputs/classification_v2/train_ready_windows")
    output_dir: Path = Path("outputs/classification_v2/model_smoke/multimodal_smoke_train")
    image_size: int = 64
    hidden_dim: int = 48
    dropout: float = 0.0
    lr: float = 0.005
    weight_decay: float = 0.0
    steps: int = 6
    per_class_train: int = 1
    per_class_eval: int = 1
    seed: int = 123
    device: str = "auto"


@dataclass(slots=True)
class MultimodalSmokeTrainResult:
    audit: dict[str, Any]
    predictions: pd.DataFrame
    checkpoint_path: Path
    predictions_path: Path
    audit_path: Path


def run_multimodal_smoke_train(config: MultimodalSmokeTrainConfig) -> MultimodalSmokeTrainResult:
    """Run a tiny image+spatial training smoke and write audit artifacts."""
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.per_class_train <= 0 or config.per_class_eval <= 0:
        raise ValueError("per-class sample counts must be positive")
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    _set_seed(config.seed)
    device = _resolve_device(config.device)

    bundle = _load_bundle(config.root)
    label_order = _label_order(bundle.y)
    label_to_idx = {label: idx for idx, label in enumerate(label_order)}
    train_indices = _select_balanced_indices(
        bundle,
        split_name="train",
        per_class=config.per_class_train,
    )
    eval_indices = _select_balanced_indices(
        bundle,
        split_name="val",
        per_class=config.per_class_eval,
    )
    if len(train_indices) < 2:
        raise ValueError("not enough train rows for multimodal smoke train")
    if len(eval_indices) < 2:
        raise ValueError("not enough val rows for multimodal smoke eval")

    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=config.root / "image_frame_context_manifest.csv",
            window_context_csv=config.root / "image_window_context_manifest.csv",
            image_size=config.image_size,
            require_complete=False,
        )
    )
    try:
        train_batch = _batch_from_indices(dataset, bundle, train_indices, label_to_idx, device)
        eval_batch = _batch_from_indices(dataset, bundle, eval_indices, label_to_idx, device)
    finally:
        dataset.close()

    model = MultimodalFusionClassifier(
        MultimodalFusionConfig(
            spatial_input_dims={name: int(bundle.arrays[name].shape[-1]) for name in MODEL_GROUPS},
            num_classes=len(label_order),
            image_embedding_dim=config.hidden_dim,
            spatial_embedding_dim=config.hidden_dim,
            fusion_hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    losses: list[float] = []
    model.train()
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            image=train_batch["image"],
            spatial_features=train_batch["spatial_features"],
            length_mask=train_batch["image_length_mask"],
            observed_mask=train_batch["image_observed_mask"],
            spatial_length_mask=train_batch["spatial_length_mask"],
            spatial_observed_mask=train_batch["spatial_observed_mask"],
        )
        loss = loss_fn(logits, train_batch["target"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    predictions = pd.concat(
        [
            _predict_batch(model, bundle, train_indices, train_batch, label_order, "train_smoke"),
            _predict_batch(model, bundle, eval_indices, eval_batch, label_order, "val_smoke"),
        ],
        ignore_index=True,
    )
    metrics = {
        split: evaluate_predictions(group, y_true_col="y_true", y_pred_col="y_pred", label_order=label_order)
        for split, group in predictions.groupby("prediction_split", sort=True)
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "multimodal_smoke_train.pt"
    predictions_path = config.output_dir / "multimodal_smoke_predictions.csv"
    audit_path = config.output_dir / "multimodal_smoke_train_audit.json"
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
        "train_source_counts": _source_counts(bundle.image_windows, train_indices),
        "eval_source_counts": _source_counts(bundle.image_windows, eval_indices),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_reduction": float(losses[0] - losses[-1]),
        "metrics": metrics,
        "prediction_schema": list(predictions.columns),
        "errors": [],
        "warnings": [
            "tiny smoke subset only; do not report as full model training result",
            "image branch uses actor crops only; interaction full-frame/partner branch remains future work",
        ],
    }
    if not np.isfinite(losses).all():
        audit["errors"].append("loss_nonfinite")
    if losses[-1] >= losses[0]:
        audit["errors"].append(f"loss_not_reduced initial={losses[0]:.6f} final={losses[-1]:.6f}")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["errors"]:
        raise ValueError(f"Multimodal smoke train failed: {audit['errors']}")
    return MultimodalSmokeTrainResult(
        audit=audit,
        predictions=predictions,
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
        audit_path=audit_path,
    )


@dataclass(slots=True)
class _MultimodalBundle:
    arrays: dict[str, np.ndarray]
    y: pd.Series
    train_mask: pd.Series
    split: pd.DataFrame
    image_windows: pd.DataFrame


def _load_bundle(root: Path) -> _MultimodalBundle:
    arrays = {name: value for name, value in np.load(root / "X_spatial_sequences.npz").items()}
    missing = [name for name in [*MODEL_GROUPS, "length_mask", "observed_mask"] if name not in arrays]
    if missing:
        raise ValueError(f"missing spatial arrays: {missing}")
    y = pd.read_csv(root / "y_behavior.csv").iloc[:, 0].fillna("").astype(str)
    train_mask = _read_bool(root / "train_mask.csv")
    split = pd.read_csv(root / "split_manifest.csv", low_memory=False)
    image_windows = pd.read_csv(root / "image_window_context_manifest.csv", low_memory=False)
    expected = len(y)
    row_counts = {
        "y": len(y),
        "train_mask": len(train_mask),
        "split": len(split),
        "image_windows": len(image_windows),
    }
    row_counts.update({name: int(arr.shape[0]) for name, arr in arrays.items()})
    mismatched = {name: count for name, count in row_counts.items() if count != expected}
    if mismatched:
        raise ValueError(f"row count mismatch against y={expected}: {mismatched}")
    return _MultimodalBundle(arrays=arrays, y=y, train_mask=train_mask, split=split, image_windows=image_windows)


def _batch_from_indices(
    dataset: ClassificationV2ImageSequenceDataset,
    bundle: _MultimodalBundle,
    indices: np.ndarray,
    label_to_idx: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    image_batch = image_sequence_collate([dataset[int(index)] for index in indices])
    image_errors = [err for item_errors in image_batch["errors"] for err in item_errors]
    if image_errors:
        raise ValueError(f"image load errors in multimodal smoke batch: {image_errors[:10]}")
    target_labels = bundle.y.iloc[indices].tolist()
    return {
        "image": image_batch["image"].float().to(device),
        "image_length_mask": image_batch["length_mask"].float().to(device),
        "image_observed_mask": image_batch["observed_mask"].float().to(device),
        "spatial_features": {
            name: torch.from_numpy(bundle.arrays[name][indices]).float().to(device) for name in MODEL_GROUPS
        },
        "spatial_length_mask": torch.from_numpy(bundle.arrays["length_mask"][indices]).float().to(device),
        "spatial_observed_mask": torch.from_numpy(bundle.arrays["observed_mask"][indices]).float().to(device),
        "target": torch.tensor([label_to_idx[label] for label in target_labels], dtype=torch.long).to(device),
    }


def _predict_batch(
    model: MultimodalFusionClassifier,
    bundle: _MultimodalBundle,
    indices: np.ndarray,
    batch: dict[str, Any],
    label_order: list[str],
    split_name: str,
) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        logits = model(
            image=batch["image"],
            spatial_features=batch["spatial_features"],
            length_mask=batch["image_length_mask"],
            observed_mask=batch["image_observed_mask"],
            spatial_length_mask=batch["spatial_length_mask"],
            spatial_observed_mask=batch["spatial_observed_mask"],
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
            "split_group_key": rows["split_group_key"].astype(str) if "split_group_key" in rows else "",
            "y_true": bundle.y.iloc[indices].to_numpy(dtype=str),
            "y_pred": [label_order[i] for i in pred_idx],
            "confidence": probs.max(axis=1),
            "correct": [label_order[i] == y for i, y in zip(pred_idx, bundle.y.iloc[indices], strict=True)],
        }
    )


def _select_balanced_indices(bundle: _MultimodalBundle, *, split_name: str, per_class: int) -> np.ndarray:
    complete = _to_bool(bundle.image_windows["window_image_context_complete"])
    valid = bundle.split["split"].astype(str).eq(split_name) & bundle.train_mask & complete
    selected: list[int] = []
    selected_source_counts: dict[str, int] = {}
    for label in _label_order(bundle.y):
        label_indices = np.flatnonzero((valid & bundle.y.eq(label)).to_numpy())
        selected_for_label = _source_diverse_label_indices(
            bundle.image_windows,
            label_indices,
            per_class,
            selected_source_counts,
        )
        selected.extend(selected_for_label)
    return np.array(selected, dtype=np.int64)


def _source_counts(image_windows: pd.DataFrame, indices: np.ndarray) -> dict[str, int]:
    return image_windows.iloc[indices]["source_type"].value_counts(dropna=False).to_dict()


def _source_diverse_label_indices(
    image_windows: pd.DataFrame,
    label_indices: np.ndarray,
    per_class: int,
    selected_source_counts: dict[str, int],
) -> list[int]:
    if len(label_indices) <= per_class:
        out = label_indices.tolist()
    else:
        candidates = image_windows.iloc[label_indices].copy()
        candidates["_row_index"] = label_indices
        out = []
        remaining = candidates.sort_values(["source_type", "video_key", "object_track_key", "window_start_frame"])
        while len(out) < per_class and not remaining.empty:
            source_order = sorted(
                remaining["source_type"].astype(str).unique().tolist(),
                key=lambda source: (selected_source_counts.get(source, 0), source),
            )
            chosen_source = source_order[0]
            chosen = remaining[remaining["source_type"].astype(str).eq(chosen_source)].iloc[0]
            row_index = int(chosen["_row_index"])
            out.append(row_index)
            remaining = remaining[remaining["_row_index"] != row_index]
    for row_index in out:
        source = str(image_windows.iloc[row_index]["source_type"])
        selected_source_counts[source] = selected_source_counts.get(source, 0) + 1
    return out


def _label_order(y: pd.Series) -> list[str]:
    observed = set(y.dropna().astype(str).tolist())
    labels = [label for label in DEFAULT_LABEL_ORDER if label in observed]
    labels.extend(sorted(observed.difference(labels)))
    return labels


def _read_bool(path: Path) -> pd.Series:
    return _to_bool(pd.read_csv(path).iloc[:, 0])


def _to_bool(series: pd.Series) -> pd.Series:
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


def _jsonable_config(config: MultimodalSmokeTrainConfig) -> dict[str, Any]:
    out = asdict(config)
    out["root"] = str(config.root)
    out["output_dir"] = str(config.output_dir)
    return out
