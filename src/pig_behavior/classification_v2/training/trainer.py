"""Strict, reproducible trainer for the audited classification_v2 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pig_behavior.classification_v2.models.multimodal_fusion import MultimodalFusionConfig
from pig_behavior.classification_v2.models.multitask_fusion import MultitaskFusionClassifier
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    training_config_to_jsonable,
)
from pig_behavior.classification_v2.training.data_module import (
    StrictTrainingBatch,
    StrictTrainingDataModule,
    validate_model_inputs,
)
from pig_behavior.classification_v2.training.multitask_loss import (
    AuxiliaryTaskSpec,
    build_fold_auxiliary_class_weights,
    hierarchy_consistency_loss,
    masked_multitask_loss,
)

PREDICTION_SCHEMA_VERSION = "classification_v2_training_predictions_v1"
RUN_AUDIT_SCHEMA_VERSION = "classification_v2_training_run_audit_v1"


def run_training(config: ClassificationV2TrainingConfig) -> dict[str, Any]:
    """Train one declared fold and emit checkpoints, predictions, and lineage audit."""

    _seed_everything(config.optimization.seed, deterministic=config.optimization.deterministic)
    device = _resolve_device(config.optimization.precision)
    output_dir = config.execution.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with StrictTrainingDataModule(config, device=device) as data:
        train_indices = (
            data.balanced_smoke_indices(train=True)
            if config.execution.mode == "smoke"
            else data.fold_indices(train=True)
        )
        eval_indices = (
            data.balanced_smoke_indices(train=False)
            if config.execution.mode == "smoke"
            else data.fold_indices(train=False)
        )
        _require_nonempty_split(train_indices, eval_indices)
        probe = data.batch(train_indices[: min(len(train_indices), 2)])
        model = _build_model(config, probe).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.optimization.precision == "amp")
        behavior_weights = _behavior_class_weights(data, train_indices, config, device)
        auxiliary_weights = build_fold_auxiliary_class_weights(
            data.auxiliary.iloc[train_indices],
            data.auxiliary_label_maps,
            power=config.loss.class_weight_power,
            max_weight=config.loss.class_weight_max,
            device=device,
        )
        task_specs = _task_specs(config)
        history: list[dict[str, Any]] = []
        global_step = 0
        start_epoch = 0
        resumed_from: str | None = None
        best_metric = float("-inf")
        best_epoch = -1
        stale_epochs = 0
        max_epochs = 1 if config.execution.mode == "smoke" else config.optimization.epochs
        last_checkpoint = output_dir / "last.pt"
        if config.execution.resume and last_checkpoint.exists():
            resumed = load_training_checkpoint(
                last_checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                map_location=device,
                restore_rng=True,
            )
            start_epoch = int(resumed["epoch"]) + 1
            global_step = int(resumed["global_step"])
            resumed_from = str(last_checkpoint)
            prior_audit = output_dir / "run_audit.json"
            if prior_audit.exists():
                prior = json.loads(prior_audit.read_text(encoding="utf-8"))
                history = list(prior.get("history", []))
                best_epoch = int(prior.get("best_epoch", -1))
                if history:
                    best_metric = max(float(row["validation_window_macro_f1"]) for row in history)
        for epoch in range(start_epoch, max_epochs):
            train_result, global_step = _train_epoch(
                model,
                optimizer,
                scaler,
                data,
                train_indices,
                behavior_weights,
                auxiliary_weights,
                task_specs,
                config,
                device,
                epoch=epoch,
                global_step=global_step,
            )
            predictions, eval_metrics = _evaluate(model, data, eval_indices, config, device)
            predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
            record = {"epoch": epoch, **train_result, **eval_metrics}
            history.append(record)
            save_training_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                metrics=record,
            )
            metric = float(eval_metrics["validation_window_macro_f1"])
            if metric > best_metric:
                best_metric, best_epoch, stale_epochs = metric, epoch, 0
                save_training_checkpoint(
                    output_dir / "best_validation.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=record,
                )
            else:
                stale_epochs += 1
            if config.execution.mode != "smoke" and stale_epochs >= config.optimization.early_stopping_patience:
                break
        audit = _run_audit(
            config,
            data,
            model,
            train_indices,
            eval_indices,
            behavior_weights,
            auxiliary_weights,
            history,
            best_epoch,
            device,
            resumed_from,
        )
    _write_json_atomic(output_dir / "run_audit.json", audit)
    _write_json_atomic(output_dir / "registry_entry.json", _registry_entry(audit, output_dir))
    return audit


def _train_epoch(
    model: MultitaskFusionClassifier,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    data: StrictTrainingDataModule,
    indices: np.ndarray,
    behavior_weights: torch.Tensor,
    auxiliary_weights: dict[str, torch.Tensor],
    task_specs: tuple[AuxiliaryTaskSpec, ...],
    config: ClassificationV2TrainingConfig,
    device: torch.device,
    *,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, Any], int]:
    """Run deterministic mini-batches and preserve sample weights as a mask/weight channel."""

    model.train()
    rng = np.random.default_rng(config.optimization.seed + epoch)
    ordered = rng.permutation(indices)
    if config.execution.mode == "smoke":
        ordered = np.resize(ordered, config.execution.smoke_steps * config.optimization.batch_size)
    losses: list[float] = []
    max_steps = config.execution.smoke_steps if config.execution.mode == "smoke" else None
    for step, start in enumerate(range(0, len(ordered), config.optimization.batch_size)):
        if max_steps is not None and step >= max_steps:
            break
        batch = data.batch(ordered[start : start + config.optimization.batch_size])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            output = model(**batch.model_inputs)
            behavior_per_row = F.cross_entropy(
                output.behavior, batch.behavior_target, weight=behavior_weights, reduction="none"
            )
            denominator = batch.sample_weight.sum().clamp_min(1e-8)
            behavior_loss = (behavior_per_row * batch.sample_weight).sum() / denominator
            auxiliary_loss, _ = masked_multitask_loss(
                output.auxiliary_logits(),
                batch.auxiliary_targets,
                batch.auxiliary_masks,
                task_specs=task_specs,
                class_weights_by_task=auxiliary_weights,
            )
            consistency = hierarchy_consistency_loss(output.behavior, output.auxiliary_logits())
            total = (
                config.loss.behavior_weight * behavior_loss
                + auxiliary_loss
                + config.loss.hierarchy_consistency_weight * consistency
            )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimization.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        losses.append(float(total.detach().cpu().item()))
    if not losses:
        raise ValueError("training produced no valid batches")
    return {
        "train_loss_mean": float(np.mean(losses)),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train_steps": len(losses),
    }, global_step


@torch.no_grad()
def _evaluate(
    model: MultitaskFusionClassifier,
    data: StrictTrainingDataModule,
    indices: np.ndarray,
    config: ClassificationV2TrainingConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate only the held-out fold and retain identifiers strictly as output metadata."""

    model.eval()
    rows: list[dict[str, Any]] = []
    labels = list(VALID_BEHAVIORS)
    for start in range(0, len(indices), config.optimization.eval_batch_size):
        batch = data.batch(indices[start : start + config.optimization.eval_batch_size])
        amp_enabled = device.type == "cuda" and config.optimization.precision == "amp"
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(**batch.model_inputs)
        probabilities = torch.softmax(output.behavior.float(), dim=1).cpu().numpy()
        true_values = batch.behavior_target.cpu().numpy()
        predicted = probabilities.argmax(axis=1)
        for row_index in range(len(predicted)):
            row = {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "window_id": batch.metadata["window_id"][row_index],
                "fold_id": config.execution.fold_id,
                "split": "validation",
                "source_type": batch.metadata["source_type"][row_index],
                "true_label": labels[int(true_values[row_index])],
                "predicted_label": labels[int(predicted[row_index])],
                "confidence": float(probabilities[row_index, predicted[row_index]]),
                "model_version": config.model.architecture_version,
                "snapshot_id": config.dataset.snapshot_json.stem,
            }
            row.update({f"prob_{label}": float(probabilities[row_index, i]) for i, label in enumerate(labels)})
            rows.append(row)
    predictions = pd.DataFrame(rows)
    metric = _macro_f1(predictions["true_label"], predictions["predicted_label"], labels)
    return predictions, {"validation_window_macro_f1": metric}


def _build_model(
    config: ClassificationV2TrainingConfig, probe: StrictTrainingBatch
) -> MultitaskFusionClassifier:
    """Derive tensor dimensions only from declared model branches, never arbitrary columns."""

    validate_model_inputs(probe.model_inputs)
    spatial = probe.model_inputs["spatial_features"]
    spatial_dims = {name: int(value.shape[-1]) for name, value in spatial.items()}
    if tuple(sorted(spatial_dims)) != tuple(sorted(config.model.spatial_feature_groups)):
        raise ValueError(
            f"spatial whitelist mismatch: config={config.model.spatial_feature_groups}, "
            f"data={sorted(spatial_dims)}"
        )
    interaction_dim = int(probe.model_inputs["interaction_context_features"].shape[-1])
    return MultitaskFusionClassifier(
        MultimodalFusionConfig(
            spatial_input_dims=spatial_dims,
            num_classes=len(VALID_BEHAVIORS),
            interaction_context_dim=interaction_dim,
            image_embedding_dim=config.model.hidden_dim,
            spatial_embedding_dim=config.model.hidden_dim,
            interaction_embedding_dim=max(8, config.model.hidden_dim // 2),
            visual_context_embedding_dim=config.model.hidden_dim,
            fusion_hidden_dim=config.model.hidden_dim * 2,
            dropout=config.model.dropout,
            enable_image=config.model.enable_image,
            enable_spatial=config.model.enable_spatial,
            enable_interaction_context=config.model.enable_interaction_context,
            enable_visual_context=config.model.enable_visual_context,
        )
    )


def _behavior_class_weights(
    data: StrictTrainingDataModule,
    indices: np.ndarray,
    config: ClassificationV2TrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    labels = data.bundle.y.iloc[indices].astype(str)
    counts = labels.value_counts().reindex(VALID_BEHAVIORS, fill_value=0).astype(float)
    if (counts <= 0).any():
        raise ValueError(f"training fold missing behavior classes: {counts[counts <= 0].index.tolist()}")
    inverse = (float(counts.max()) / counts) ** config.loss.class_weight_power
    weights = (inverse / float(inverse.mean())).clip(upper=config.loss.class_weight_max)
    return torch.tensor(weights.to_numpy(), dtype=torch.float32, device=device)


def _task_specs(config: ClassificationV2TrainingConfig) -> tuple[AuxiliaryTaskSpec, ...]:
    weights = {
        "posture": config.loss.posture_weight,
        "motion_context": config.loss.motion_context_weight,
        "roi_intent": config.loss.roi_intent_weight,
        "interaction": config.loss.interaction_weight,
    }
    columns = {
        "posture": ("posture_target", "has_posture_aux_target"),
        "motion_context": ("motion_context_target", "has_motion_context_aux_target"),
        "roi_intent": ("roi_intent_target", "has_roi_intent_aux_target"),
        "interaction": ("interaction_target", "has_interaction_aux_target"),
    }
    return tuple(AuxiliaryTaskSpec(name, *columns[name], weights[name]) for name in columns)


def _run_audit(
    config: ClassificationV2TrainingConfig,
    data: StrictTrainingDataModule,
    model: MultitaskFusionClassifier,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    behavior_weights: torch.Tensor,
    auxiliary_weights: dict[str, torch.Tensor],
    history: list[dict[str, Any]],
    best_epoch: int,
    device: torch.device,
    resumed_from: str | None,
) -> dict[str, Any]:
    config_payload = training_config_to_jsonable(config)
    git = _git_state()
    return {
        "schema_version": RUN_AUDIT_SCHEMA_VERSION,
        "valid": True,
        "config": config_payload,
        "config_sha256": hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest(),
        "snapshot_id": config.dataset.snapshot_json.stem,
        "snapshot_sha256": _file_sha256(config.dataset.snapshot_json),
        "split_manifest_sha256": _file_sha256(config.dataset.native_oof_fold_manifest),
        "trainer_contract_sha256": _file_sha256(config.dataset.trainer_contract_json),
        "git": git,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "model_architecture": config.model.architecture_version,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "label_order": list(VALID_BEHAVIORS),
        "feature_whitelist": list(config.model.spatial_feature_groups),
        "normalization_imputation": "bound_to_trainer_contract_and_snapshot",
        "train_selected_window_id_sha256": _selected_id_hash(data, train_indices),
        "validation_selected_window_id_sha256": _selected_id_hash(data, eval_indices),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(eval_indices)),
        "behavior_class_weights_train_fold_only": behavior_weights.detach().cpu().tolist(),
        "auxiliary_class_weights_train_fold_only": {
            name: value.detach().cpu().tolist() for name, value in auxiliary_weights.items()
        },
        "data_module_audit": data.audit(),
        "history": history,
        "best_epoch": best_epoch,
        "resume": {
            "enabled": config.execution.resume,
            "resumed_from": resumed_from,
            "optimizer_scaler_rng_restored": resumed_from is not None,
        },
        "artifacts": {
            "last_checkpoint": "last.pt",
            "best_checkpoint": "best_validation.pt",
            "predictions": "validation_predictions.csv",
        },
        "errors": [],
    }


def _seed_everything(seed: int, *, deterministic: bool) -> None:
    # CUDA >= 10.2 requires this workspace contract before its first cuBLAS call.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def _resolve_device(precision: str) -> torch.device:
    del precision
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_nonempty_split(train_indices: np.ndarray, eval_indices: np.ndarray) -> None:
    if len(train_indices) == 0 or len(eval_indices) == 0:
        raise ValueError(f"empty train/validation split: train={len(train_indices)}, validation={len(eval_indices)}")


def _macro_f1(true: pd.Series, predicted: pd.Series, labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = int(((true == label) & (predicted == label)).sum())
        fp = int(((true != label) & (predicted == label)).sum())
        fn = int(((true == label) & (predicted != label)).sum())
        denominator = 2 * tp + fp + fn
        scores.append((2.0 * tp / denominator) if denominator else 0.0)
    return float(np.mean(scores))


def _selected_id_hash(data: StrictTrainingDataModule, indices: np.ndarray) -> str:
    values = data.bundle.frame.iloc[indices]["window_id"].astype(str)
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _registry_entry(audit: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Create a self-contained experiment index without mutating a global registry."""

    return {
        "schema_version": "classification_v2_experiment_registry_entry_v1",
        "run_id": hashlib.sha256(
            f"{audit['config_sha256']}:{audit['git']['commit']}".encode()
        ).hexdigest()[:20],
        "snapshot_id": audit["snapshot_id"],
        "config_sha256": audit["config_sha256"],
        "split_manifest_sha256": audit["split_manifest_sha256"],
        "git": audit["git"],
        "fold_id": audit["config"]["execution"]["fold_id"],
        "model_architecture": audit["model_architecture"],
        "best_epoch": audit["best_epoch"],
        "history": audit["history"],
        "artifact_root": str(output_dir),
        "artifacts": audit["artifacts"],
        "valid": audit["valid"],
    }
