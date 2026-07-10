"""Learned multimodal native-OOF runner for classification_v2.

The default configuration is a small OOF pilot that exercises the same data
boundaries as the future full evaluation: actor crop images, whitelisted spatial
sequence tensors, interaction context tensors, native temporal fold splits, and
the S16 prediction schema. Full paper-facing evaluation must be launched
explicitly and registered separately; the pilot is engineering evidence only.
"""

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
from pig_behavior.classification_v2.datasets.interaction_context_loader import (
    INTERACTION_CONTEXT_FEATURE_COLUMNS,
    InteractionContextDatasetConfig,
    InteractionContextWindowDataset,
)
from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_metrics,
)
from pig_behavior.classification_v2.evaluation.prediction_schema_contract import check_prediction_schema
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.spatial_tcn_smoke import MODEL_GROUPS


@dataclass(frozen=True, slots=True)
class FullMultimodalOofConfig:
    """Deterministic options for pilot or full learned native-OOF runs."""

    root: Path = Path("outputs/classification_v2/train_ready_windows")
    sequence_manifest_csv: Path = Path(
        "outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv"
    )
    interaction_context_manifest_csv: Path = Path(
        "outputs/classification_v2/train_ready_windows/interaction_window_context_manifest.csv"
    )
    native_oof_fold_manifest_csv: Path = Path(
        "outputs/classification_v2/native_temporal_units_oof_folds/native_oof_fold_manifest.csv"
    )
    output_dir: Path = Path("outputs/classification_v2/model_smoke/full_multimodal_oof_pilot")
    image_size: int = 32
    hidden_dim: int = 32
    dropout: float = 0.1
    lr: float = 0.003
    weight_decay: float = 0.0
    steps_per_fold: int = 2
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_folds: int | None = 2
    train_per_class_per_fold: int | None = 2
    eval_per_class_per_fold: int | None = 1
    bootstrap_iterations: int = 30
    seed: int = 20260710
    device: str = "auto"
    run_mode: str = "pilot"
    resume: bool = True


def run_full_multimodal_oof(config: FullMultimodalOofConfig) -> dict[str, Any]:
    """Run learned multimodal native-OOF pilot/full evaluation and write artifacts."""

    _validate_config(config)
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    bundle = _load_bundle(config)
    label_order = list(VALID_BEHAVIORS)
    label_to_idx = {label: idx for idx, label in enumerate(label_order)}
    fold_ids = sorted(bundle.frame.loc[bundle.frame["eligible"], "oof_fold_id"].astype(str).unique())
    if config.max_folds is not None:
        fold_ids = fold_ids[: int(config.max_folds)]
    if not fold_ids:
        raise ValueError("no eligible OOF folds available")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    fold_artifact_dir = config.output_dir / "fold_artifacts"
    fold_artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=config.root / "image_frame_context_manifest.csv",
            window_context_csv=config.root / "image_window_context_manifest.csv",
            image_size=config.image_size,
            require_complete=False,
        )
    )
    predictions: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    try:
        for fold_id in fold_ids:
            fold_predictions, fold_audit = _load_or_run_one_fold(
                dataset,
                bundle,
                config,
                fold_id,
                label_order,
                label_to_idx,
                device,
                fold_artifact_dir,
            )
            predictions.append(fold_predictions)
            fold_audits.append(fold_audit)
    finally:
        dataset.close()

    prediction_frame = pd.concat(predictions, ignore_index=True).sort_values(
        ["oof_fold_id", "window_id"],
        kind="mergesort",
    )
    prediction_schema_audit = check_prediction_schema(prediction_frame)
    native_units, metrics_payload = build_native_temporal_metrics(
        prediction_frame,
        NativeTemporalMetricsConfig(bootstrap_iterations=int(config.bootstrap_iterations)),
    )

    predictions_path = config.output_dir / "full_multimodal_oof_predictions.csv"
    native_units_path = config.output_dir / "full_multimodal_oof_unit_predictions.csv"
    metrics_path = config.output_dir / "full_multimodal_oof_metrics.json"
    schema_audit_path = config.output_dir / "full_multimodal_oof_prediction_schema_audit.json"
    audit_path = config.output_dir / "full_multimodal_oof_audit.json"
    prediction_frame.to_csv(predictions_path, index=False)
    native_units.to_csv(native_units_path, index=False)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    schema_audit_path.write_text(json.dumps(prediction_schema_audit, indent=2), encoding="utf-8")

    audit = {
        "schema_version": "classification_v2_full_multimodal_oof_audit_v1",
        "run_mode": config.run_mode,
        "paper_facing_result": _is_full_run(config, bundle),
        "config": _jsonable_config(config),
        "device": str(device),
        "label_order": label_order,
        "load_audit": bundle.load_audit,
        "fold_artifact_dir": str(fold_artifact_dir),
        "fold_audits": fold_audits,
        "prediction_rows": int(len(prediction_frame)),
        "native_temporal_rows": int(
            metrics_payload.get("native_temporal_prediction_audit", {}).get("native_temporal_unit_rows", 0)
        ),
        "metrics_json": str(metrics_path),
        "predictions_csv": str(predictions_path),
        "native_unit_predictions_csv": str(native_units_path),
        "prediction_schema_audit_json": str(schema_audit_path),
        "prediction_schema_valid": bool(prediction_schema_audit.get("valid")),
        "errors": [],
        "warnings": _mode_warnings(config, bundle),
        "valid": bool(not prediction_frame.empty and prediction_schema_audit.get("valid") is True),
    }
    if prediction_schema_audit.get("errors"):
        audit["errors"].append(f"prediction_schema_errors={prediction_schema_audit.get('errors')}")
    if metrics_payload.get("native_temporal_prediction_audit", {}).get("errors"):
        audit["errors"].append(
            "native_temporal_prediction_errors="
            f"{metrics_payload.get('native_temporal_prediction_audit', {}).get('errors')}"
        )
    audit["valid"] = audit["valid"] and not audit["errors"]
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if audit["errors"]:
        raise ValueError(f"full multimodal OOF run failed: {audit['errors']}")
    return {
        "audit_json": str(audit_path),
        "predictions_csv": str(predictions_path),
        "native_unit_predictions_csv": str(native_units_path),
        "metrics_json": str(metrics_path),
        "prediction_schema_audit_json": str(schema_audit_path),
        "audit": audit,
    }


def build_full_multimodal_oof_run_plan(config: FullMultimodalOofConfig) -> dict[str, Any]:
    """Summarize full/pilot OOF workload before loading images or training."""

    _validate_config(config)
    bundle = _load_bundle(config)
    fold_ids = sorted(bundle.frame.loc[bundle.frame["eligible"], "oof_fold_id"].astype(str).unique())
    selected_fold_ids = fold_ids if config.max_folds is None else fold_ids[: int(config.max_folds)]
    fold_rows: list[dict[str, Any]] = []
    total_eval_rows = 0
    total_train_steps = 0
    for fold_id in selected_fold_ids:
        train_mask = bundle.frame["eligible"] & bundle.frame["oof_fold_id"].astype(str).ne(str(fold_id))
        eval_mask = bundle.frame["eligible"] & bundle.frame["oof_fold_id"].astype(str).eq(str(fold_id))
        train_indices = _sample_indices(
            bundle.frame,
            mask=train_mask,
            per_class=config.train_per_class_per_fold,
            seed=config.seed + int(_stable_fold_offset(fold_id)),
        )
        eval_indices = _sample_indices(
            bundle.frame,
            mask=eval_mask,
            per_class=config.eval_per_class_per_fold,
            seed=config.seed + 10_000 + int(_stable_fold_offset(fold_id)),
        )
        eval_batches = _ceil_div(len(eval_indices), int(config.eval_batch_size))
        fold_rows.append(
            {
                "oof_fold_id": str(fold_id),
                "train_rows": int(len(train_indices)),
                "eval_rows": int(len(eval_indices)),
                "steps_per_fold": int(config.steps_per_fold),
                "train_batch_size": int(config.train_batch_size),
                "eval_batch_size": int(config.eval_batch_size),
                "eval_batches": int(eval_batches),
                "train_label_counts": _label_counts(bundle.frame, train_indices),
                "eval_label_counts": _label_counts(bundle.frame, eval_indices),
            }
        )
        total_eval_rows += int(len(eval_indices))
        total_train_steps += int(config.steps_per_fold)
    full_like = _is_full_run(config, bundle)
    warnings = []
    if not full_like:
        warnings.append("bounded_or_pilot_plan_not_valid_for_full_paper_record")
    if len(selected_fold_ids) != bundle.load_audit.get("eligible_fold_count"):
        warnings.append("selected_fold_count_less_than_available_fold_count")
    return {
        "schema_version": "classification_v2_full_multimodal_oof_run_plan_v1",
        "config": _jsonable_config(config),
        "run_mode": config.run_mode,
        "paper_facing_candidate_plan": full_like,
        "load_audit": bundle.load_audit,
        "available_fold_count": int(bundle.load_audit.get("eligible_fold_count", 0)),
        "selected_fold_count": int(len(selected_fold_ids)),
        "total_eval_rows": int(total_eval_rows),
        "total_train_steps": int(total_train_steps),
        "folds": fold_rows,
        "warnings": warnings,
        "errors": [],
        "valid": bool(selected_fold_ids and total_eval_rows > 0),
    }


@dataclass(slots=True)
class _OofBundle:
    """Aligned tensors and metadata needed by the learned OOF runner."""

    arrays: dict[str, np.ndarray]
    interaction_context_features: np.ndarray
    interaction_context_available_mask: np.ndarray
    y: pd.Series
    frame: pd.DataFrame
    load_audit: dict[str, Any]


def _run_one_fold(
    dataset: ClassificationV2ImageSequenceDataset,
    bundle: _OofBundle,
    config: FullMultimodalOofConfig,
    fold_id: str,
    label_order: list[str],
    label_to_idx: dict[str, int],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train on all non-held-out native folds and predict the held-out fold."""

    train_indices = _sample_indices(
        bundle.frame,
        mask=bundle.frame["eligible"] & bundle.frame["oof_fold_id"].astype(str).ne(str(fold_id)),
        per_class=config.train_per_class_per_fold,
        seed=config.seed + int(_stable_fold_offset(fold_id)),
    )
    eval_indices = _sample_indices(
        bundle.frame,
        mask=bundle.frame["eligible"] & bundle.frame["oof_fold_id"].astype(str).eq(str(fold_id)),
        per_class=config.eval_per_class_per_fold,
        seed=config.seed + 10_000 + int(_stable_fold_offset(fold_id)),
    )
    if len(train_indices) < 2 or len(eval_indices) < 1:
        raise ValueError(f"fold {fold_id} has insufficient train/eval rows")
    model = MultimodalFusionClassifier(
        MultimodalFusionConfig(
            spatial_input_dims={name: int(bundle.arrays[name].shape[-1]) for name in MODEL_GROUPS},
            num_classes=len(label_order),
            interaction_context_dim=len(INTERACTION_CONTEXT_FEATURE_COLUMNS),
            image_embedding_dim=config.hidden_dim,
            spatial_embedding_dim=config.hidden_dim,
            interaction_embedding_dim=max(8, config.hidden_dim // 2),
            fusion_hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.seed + int(_stable_fold_offset(fold_id)))
    losses: list[float] = []
    model.train()
    for step_index in range(int(config.steps_per_fold)):
        batch_indices = _step_train_indices(train_indices, config.train_batch_size, rng, step_index)
        train_batch = _batch_from_indices(dataset, bundle, batch_indices, label_to_idx, device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_model(model, train_batch)
        loss = loss_fn(logits, train_batch["target"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    predictions = _predict_in_batches(
        dataset,
        model,
        bundle,
        eval_indices,
        label_to_idx,
        label_order,
        fold_id,
        config,
        device,
    )
    audit = {
        "oof_fold_id": str(fold_id),
        "train_rows": int(len(train_indices)),
        "eval_rows": int(len(eval_indices)),
        "train_batch_size": int(config.train_batch_size),
        "eval_batch_size": int(config.eval_batch_size),
        "train_label_counts": bundle.frame.iloc[train_indices]["behavior_true"].value_counts().sort_index().to_dict(),
        "eval_label_counts": bundle.frame.iloc[eval_indices]["behavior_true"].value_counts().sort_index().to_dict(),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_reduction": float(losses[0] - losses[-1]),
    }
    return predictions, audit


def _load_or_run_one_fold(
    dataset: ClassificationV2ImageSequenceDataset,
    bundle: _OofBundle,
    config: FullMultimodalOofConfig,
    fold_id: str,
    label_order: list[str],
    label_to_idx: dict[str, int],
    device: torch.device,
    fold_artifact_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resume a completed fold or train/predict it and save fold artifacts."""

    safe_fold_id = _safe_fold_id(fold_id)
    predictions_path = fold_artifact_dir / f"{safe_fold_id}_predictions.csv"
    audit_path = fold_artifact_dir / f"{safe_fold_id}_audit.json"
    if config.resume and predictions_path.exists() and audit_path.exists():
        predictions = pd.read_csv(predictions_path, low_memory=False)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["resumed_from_artifact"] = True
        return predictions, audit
    predictions, audit = _run_one_fold(
        dataset,
        bundle,
        config,
        fold_id,
        label_order,
        label_to_idx,
        device,
    )
    audit["resumed_from_artifact"] = False
    predictions.to_csv(predictions_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return predictions, audit


def _load_bundle(config: FullMultimodalOofConfig) -> _OofBundle:
    """Load train-ready rows and keep identity/source columns as metadata only."""

    arrays = {name: value for name, value in np.load(config.root / "X_spatial_sequences.npz").items()}
    missing_arrays = [name for name in [*MODEL_GROUPS, "length_mask", "observed_mask"] if name not in arrays]
    if missing_arrays:
        raise ValueError(f"missing spatial arrays: {missing_arrays}")
    y = pd.read_csv(config.root / "y_behavior.csv").iloc[:, 0].fillna("").astype(str)
    train_mask = _read_bool(config.root / "train_mask.csv")
    split = pd.read_csv(config.root / "split_manifest.csv", low_memory=False)
    image_windows = pd.read_csv(config.root / "image_window_context_manifest.csv", low_memory=False)
    sequence = pd.read_csv(
        config.sequence_manifest_csv,
        usecols=[
            "window_id",
            "temporal_unit_keys_window",
            "num_temporal_units_window",
            "window_valid_for_main_train",
            "window_sample_weight",
        ],
        low_memory=False,
    )
    folds = pd.read_csv(
        config.native_oof_fold_manifest_csv,
        usecols=["temporal_unit_key", "oof_fold_id", "native_unit_valid_for_main_eval"],
        low_memory=False,
    )
    interaction = InteractionContextWindowDataset(
        InteractionContextDatasetConfig(manifest_csv=config.interaction_context_manifest_csv)
    ).manifest
    expected = int(len(y))
    row_counts = {
        "y": int(len(y)),
        "train_mask": int(len(train_mask)),
        "split": int(len(split)),
        "image_windows": int(len(image_windows)),
        "interaction": int(len(interaction)),
    }
    row_counts.update({name: int(arr.shape[0]) for name, arr in arrays.items()})
    mismatched = {name: count for name, count in row_counts.items() if count != expected}
    if mismatched:
        raise ValueError(f"row count mismatch against y={expected}: {mismatched}")

    frame = split[["window_id", "split", "split_group_key"]].copy()
    frame["behavior_true"] = y
    frame["train_mask"] = train_mask
    frame = frame.merge(
        sequence,
        on="window_id",
        how="left",
        validate="one_to_one",
    ).rename(columns={"temporal_unit_keys_window": "temporal_unit_key"})
    frame = frame.merge(folds, on="temporal_unit_key", how="left")
    frame["window_image_context_complete"] = _to_bool(image_windows["window_image_context_complete"])
    frame["window_valid_for_main_train"] = _to_bool(frame["window_valid_for_main_train"])
    frame["native_unit_valid_for_main_eval"] = _to_bool(frame["native_unit_valid_for_main_eval"])
    frame["num_temporal_units_window"] = pd.to_numeric(frame["num_temporal_units_window"], errors="coerce")
    frame["eligible"] = (
        frame["train_mask"]
        & frame["window_valid_for_main_train"]
        & frame["native_unit_valid_for_main_eval"]
        & frame["window_image_context_complete"]
        & frame["num_temporal_units_window"].eq(1)
        & frame["behavior_true"].isin(VALID_BEHAVIORS)
        & frame["oof_fold_id"].fillna("").astype(str).ne("")
    )
    interaction_features = (
        interaction[list(INTERACTION_CONTEXT_FEATURE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    interaction_available = _to_bool(interaction["scene_partner_context_ready"]).to_numpy(dtype=np.float32)
    load_audit = {
        "row_counts": row_counts,
        "eligible_rows": int(frame["eligible"].sum()),
        "eligible_fold_count": int(frame.loc[frame["eligible"], "oof_fold_id"].nunique()),
        "eligible_label_counts": frame.loc[frame["eligible"], "behavior_true"].value_counts().sort_index().to_dict(),
        "complete_image_context_rows": int(frame["window_image_context_complete"].sum()),
        "interaction_context_ready_rows": int(interaction_available.sum()),
    }
    return _OofBundle(
        arrays=arrays,
        interaction_context_features=interaction_features,
        interaction_context_available_mask=interaction_available,
        y=y,
        frame=frame,
        load_audit=load_audit,
    )


def _batch_from_indices(
    dataset: ClassificationV2ImageSequenceDataset,
    bundle: _OofBundle,
    indices: np.ndarray,
    label_to_idx: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    """Build one multimodal batch from aligned global row indices."""

    image_batch = image_sequence_collate([dataset[int(index)] for index in indices])
    image_errors = [err for item_errors in image_batch["errors"] for err in item_errors]
    if image_errors:
        raise ValueError(f"image load errors: {image_errors[:10]}")
    target_labels = bundle.frame.iloc[indices]["behavior_true"].astype(str).tolist()
    return {
        "image": image_batch["image"].float().to(device),
        "image_length_mask": image_batch["length_mask"].float().to(device),
        "image_observed_mask": image_batch["observed_mask"].float().to(device),
        "spatial_features": {
            name: torch.from_numpy(bundle.arrays[name][indices]).float().to(device) for name in MODEL_GROUPS
        },
        "spatial_length_mask": torch.from_numpy(bundle.arrays["length_mask"][indices]).float().to(device),
        "spatial_observed_mask": torch.from_numpy(bundle.arrays["observed_mask"][indices]).float().to(device),
        "interaction_context_features": torch.from_numpy(bundle.interaction_context_features[indices])
        .float()
        .to(device),
        "interaction_context_available_mask": torch.from_numpy(bundle.interaction_context_available_mask[indices])
        .float()
        .to(device),
        "target": torch.tensor([label_to_idx[label] for label in target_labels], dtype=torch.long).to(device),
    }


def _forward_model(model: MultimodalFusionClassifier, batch: dict[str, Any]) -> torch.Tensor:
    """Run the multimodal forward path with branch-specific masks."""

    return model(
        image=batch["image"],
        spatial_features=batch["spatial_features"],
        length_mask=batch["image_length_mask"],
        image_length_mask=batch["image_length_mask"],
        image_observed_mask=batch["image_observed_mask"],
        spatial_length_mask=batch["spatial_length_mask"],
        spatial_observed_mask=batch["spatial_observed_mask"],
        interaction_context_features=batch["interaction_context_features"],
        interaction_context_available_mask=batch["interaction_context_available_mask"],
    )


def _predict(
    model: MultimodalFusionClassifier,
    bundle: _OofBundle,
    indices: np.ndarray,
    batch: dict[str, Any],
    label_order: list[str],
    fold_id: str,
    experiment_role: str,
) -> pd.DataFrame:
    """Emit S16-compatible window predictions for held-out native units."""

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(_forward_model(model, batch), dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    rows = bundle.frame.iloc[indices].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "temporal_unit_key": rows["temporal_unit_key"].astype(str),
            "window_id": rows["window_id"].astype(str),
            "behavior_true": rows["behavior_true"].astype(str),
            "behavior_pred": [label_order[index] for index in pred_idx],
            "window_sample_weight": pd.to_numeric(rows["window_sample_weight"], errors="coerce").fillna(1.0),
            "window_valid_for_main_train": rows["window_valid_for_main_train"].astype(bool),
            "oof_fold_id": str(fold_id),
            "experiment_role": experiment_role,
        }
    )
    for class_index, label in enumerate(label_order):
        out[f"prob_{label}"] = probs[:, class_index]
    return out


def _predict_in_batches(
    dataset: ClassificationV2ImageSequenceDataset,
    model: MultimodalFusionClassifier,
    bundle: _OofBundle,
    indices: np.ndarray,
    label_to_idx: dict[str, int],
    label_order: list[str],
    fold_id: str,
    config: FullMultimodalOofConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Predict a held-out fold in bounded chunks so full OOF does not exhaust RAM."""

    chunks: list[pd.DataFrame] = []
    role = "full_multimodal_oof" if _is_full_run(config, bundle) else "full_multimodal_oof_pilot"
    for start in range(0, len(indices), int(config.eval_batch_size)):
        chunk_indices = indices[start : start + int(config.eval_batch_size)]
        batch = _batch_from_indices(dataset, bundle, chunk_indices, label_to_idx, device)
        chunks.append(_predict(model, bundle, chunk_indices, batch, label_order, fold_id, role))
    return pd.concat(chunks, ignore_index=True)


def _step_train_indices(
    train_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    step_index: int,
) -> np.ndarray:
    """Select one deterministic mini-batch, cycling all rows when batch covers the fold."""

    if len(train_indices) <= batch_size:
        return train_indices
    replace = len(train_indices) < batch_size
    if not replace:
        # Mix a deterministic offset with RNG sampling so short pilot runs still
        # see different rows across steps without relying on global state.
        shifted = np.roll(train_indices, step_index * batch_size)
        candidate_pool = shifted[: max(batch_size * 4, batch_size)]
        if len(candidate_pool) >= batch_size:
            return np.sort(rng.choice(candidate_pool, size=batch_size, replace=False))
    return np.sort(rng.choice(train_indices, size=batch_size, replace=replace))


def _sample_indices(frame: pd.DataFrame, *, mask: pd.Series, per_class: int | None, seed: int) -> np.ndarray:
    """Return deterministic per-class sample indices or all rows when per_class is None."""

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in VALID_BEHAVIORS:
        label_indices = np.flatnonzero((mask & frame["behavior_true"].eq(label)).to_numpy())
        if per_class is None or len(label_indices) <= per_class:
            chosen = label_indices
        else:
            chosen = np.sort(rng.choice(label_indices, size=int(per_class), replace=False))
        selected.extend(int(index) for index in chosen)
    return np.array(sorted(selected), dtype=np.int64)


def _validate_config(config: FullMultimodalOofConfig) -> None:
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.steps_per_fold <= 0:
        raise ValueError("steps_per_fold must be positive")
    if config.train_batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.max_folds is not None and config.max_folds <= 0:
        raise ValueError("max_folds must be positive when provided")
    if config.run_mode not in {"pilot", "full"}:
        raise ValueError("run_mode must be pilot or full")
    if config.run_mode == "pilot" and (
        config.max_folds is None or config.train_per_class_per_fold is None or config.eval_per_class_per_fold is None
    ):
        raise ValueError("pilot mode requires bounded folds and per-class sample caps")


def _is_full_run(config: FullMultimodalOofConfig, bundle: _OofBundle) -> bool:
    """Return whether the config is allowed to be interpreted as full OOF evidence."""

    return (
        config.run_mode == "full"
        and config.max_folds is None
        and config.train_per_class_per_fold is None
        and config.eval_per_class_per_fold is None
        and bundle.load_audit.get("eligible_fold_count", 0) >= 2
    )


def _mode_warnings(config: FullMultimodalOofConfig, bundle: _OofBundle) -> list[str]:
    warnings: list[str] = []
    if not _is_full_run(config, bundle):
        warnings.append("bounded pilot run; do not register as full_multimodal_oof_record or cite as paper metric")
    warnings.append("full learned OOF claim also requires source-balanced reporting and ablation report review")
    return warnings


def _stable_fold_offset(fold_id: str) -> int:
    return sum(ord(char) for char in str(fold_id))


def _safe_fold_id(fold_id: str) -> str:
    """Make fold IDs safe for deterministic per-fold artifact filenames."""

    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(fold_id))


def _label_counts(frame: pd.DataFrame, indices: np.ndarray) -> dict[str, int]:
    return frame.iloc[indices]["behavior_true"].value_counts().sort_index().to_dict()


def _ceil_div(value: int, divisor: int) -> int:
    return int((int(value) + int(divisor) - 1) // int(divisor))


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


def _jsonable_config(config: FullMultimodalOofConfig) -> dict[str, Any]:
    out = asdict(config)
    for key in [
        "root",
        "sequence_manifest_csv",
        "interaction_context_manifest_csv",
        "native_oof_fold_manifest_csv",
        "output_dir",
    ]:
        out[key] = str(out[key])
    return out
