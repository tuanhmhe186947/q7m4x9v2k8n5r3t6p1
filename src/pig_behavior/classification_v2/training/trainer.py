"""Strict, reproducible trainer for the audited classification_v2 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pig_behavior.classification_v2.models.model_factory import (
    build_multimodal_model,
    model_mode_contract,
    model_parameter_report,
)
from pig_behavior.classification_v2.models.multitask_fusion import MultitaskFusionClassifier
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    training_config_sha256,
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
from pig_behavior.classification_v2.training.fold_preprocessing import (
    FoldPreprocessingState,
    ensure_fold_preprocessing_state,
)
from pig_behavior.classification_v2.training.multitask_loss import (
    AuxiliaryTaskSpec,
    build_fold_auxiliary_class_weights,
    hierarchy_consistency_loss,
    masked_multitask_loss,
)
from pig_behavior.classification_v2.training.run_lineage import (
    PredictionArtifact,
    RunLineageSession,
    finalize_run_lineage,
    initialize_run_lineage,
)
from pig_behavior.classification_v2.training.visual_freeze import (
    build_visual_optimizer_groups,
    configure_visual_train_stage,
    optimizer_group_report,
    visual_freeze_schedule_payload,
)

PREDICTION_SCHEMA_VERSION = "classification_v2_training_predictions_v1"
RUN_AUDIT_SCHEMA_VERSION = "classification_v2_training_run_audit_v2"


def training_run_dir(audit: Mapping[str, Any]) -> Path:
    """Return the lineage-owned artifact directory from a completed run audit."""

    lineage = audit.get("run_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("training audit is missing run_lineage")
    value = lineage.get("run_dir")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("training run lineage is missing run_dir")
    return Path(value)


def run_training(config: ClassificationV2TrainingConfig) -> dict[str, Any]:
    """Train one declared fold and emit checkpoints, predictions, and lineage audit."""

    _seed_everything(config.optimization.seed, deterministic=config.optimization.deterministic)
    device = _resolve_device(config.optimization.precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with initialize_run_lineage(config) as session:
        audit = _run_training_impl(config, device=device, session=session)
        peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        lineage = finalize_run_lineage(
            session,
            checkpoint_paths=[
                session.run_dir / "last.pt",
                session.run_dir / "best_validation.pt",
            ],
            predictions=[
                PredictionArtifact(
                    path=session.run_dir / "oof_test_predictions.csv",
                    checkpoint_path=session.run_dir / "best_validation.pt",
                    split="test",
                    expected_rows=int(audit["test_rows"]),
                )
            ],
            metric_paths=[session.run_dir / "run_audit.json"],
            peak_vram_bytes=peak_vram,
        )
    return {**audit, "run_lineage": lineage}


def _run_training_impl(
    config: ClassificationV2TrainingConfig,
    *,
    device: torch.device,
    session: RunLineageSession,
) -> dict[str, Any]:
    """Execute one already-initialized and lineage-bound fold run."""

    output_dir = session.run_dir
    with StrictTrainingDataModule(config, device=device) as data:
        train_indices = (
            data.balanced_smoke_indices(train=True)
            if config.execution.mode == "smoke"
            else data.fold_indices(train=True)
        )
        eval_indices = (
            data.balanced_smoke_indices(train=False)
            if config.execution.mode == "smoke"
            else data.split_indices("validation")
        )
        test_indices = (
            data.balanced_smoke_split("test")
            if config.execution.mode == "smoke"
            else data.split_indices("test")
        )
        _require_nonempty_split(train_indices, eval_indices, test_indices)
        preprocessing_state = data.fit_fold_preprocessor()
        preprocessing_status = ensure_fold_preprocessing_state(
            output_dir / "preprocessing.json",
            preprocessing_state,
        )
        probe = data.batch(train_indices[: min(len(train_indices), 2)])
        model = _build_model(config, probe).to(device)
        optimizer_groups, optimizer_group_contract = build_visual_optimizer_groups(
            model,
            learning_rate=config.optimization.learning_rate,
            backbone_lr_multiplier=(
                config.model.visual_backbone_lr_multiplier
            ),
            weight_decay=config.optimization.weight_decay,
        )
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda" and config.optimization.precision == "amp"),
        )
        behavior_weights = _behavior_class_weights(data, train_indices, config, device)
        auxiliary_weights = (
            build_fold_auxiliary_class_weights(
                data.auxiliary.iloc[train_indices],
                data.auxiliary_label_maps,
                power=config.loss.class_weight_power,
                max_weight=config.loss.class_weight_max,
                device=device,
            )
            if config.model.enable_multitask
            else {}
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
                run_identity=session.identity.to_payload(),
                preprocessing_sha256=preprocessing_state.state_sha256,
                train_window_id_sha256=(preprocessing_state.train_window_id_sha256),
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
            predictions, eval_metrics = _evaluate(
                model, data, eval_indices, config, device, split="validation"
            )
            predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
            record = {"epoch": epoch, **train_result, **eval_metrics}
            history.append(record)
            save_training_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                run_identity=session.identity.to_payload(),
                preprocessing_sha256=preprocessing_state.state_sha256,
                train_window_id_sha256=(preprocessing_state.train_window_id_sha256),
                epoch=epoch,
                global_step=global_step,
                metrics=record,
                visual_freeze_state=train_result["visual_freeze"],
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
                    run_identity=session.identity.to_payload(),
                    preprocessing_sha256=preprocessing_state.state_sha256,
                    train_window_id_sha256=(preprocessing_state.train_window_id_sha256),
                    epoch=epoch,
                    global_step=global_step,
                    metrics=record,
                    visual_freeze_state=train_result["visual_freeze"],
                )
            else:
                stale_epochs += 1
            if (
                config.execution.mode != "smoke"
                and stale_epochs >= config.optimization.early_stopping_patience
            ):
                break
        best_checkpoint = output_dir / "best_validation.pt"
        if not best_checkpoint.exists():
            raise ValueError("best-validation checkpoint missing before outer-test evaluation")
        load_training_checkpoint(
            best_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            run_identity=session.identity.to_payload(),
            preprocessing_sha256=preprocessing_state.state_sha256,
            train_window_id_sha256=preprocessing_state.train_window_id_sha256,
            map_location=device,
            restore_rng=False,
        )
        test_predictions, test_metrics = _evaluate(
            model, data, test_indices, config, device, split="test"
        )
        test_predictions.to_csv(output_dir / "oof_test_predictions.csv", index=False)
        audit = _run_audit(
            config,
            data,
            model,
            train_indices,
            eval_indices,
            test_indices,
            behavior_weights,
            auxiliary_weights,
            history,
            best_epoch,
            device,
            resumed_from,
            test_metrics,
            preprocessing_state,
            preprocessing_status,
            session.identity.to_payload(),
            optimizer_group_contract,
        )
    _write_json_atomic(output_dir / "run_audit.json", audit)
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
    visual_freeze = configure_visual_train_stage(
        model,
        config.model,
        epoch=epoch,
    )
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
            if config.model.enable_multitask:
                auxiliary_loss, _ = masked_multitask_loss(
                    output.auxiliary_logits(),
                    batch.auxiliary_targets,
                    batch.auxiliary_masks,
                    task_specs=task_specs,
                    class_weights_by_task=auxiliary_weights,
                )
                consistency = hierarchy_consistency_loss(output.behavior, output.auxiliary_logits())
            else:
                auxiliary_loss = behavior_loss.new_zeros(())
                consistency = behavior_loss.new_zeros(())
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
        "visual_freeze": visual_freeze,
        "optimizer_groups": optimizer_group_report(optimizer),
    }, global_step


@torch.no_grad()
def _evaluate(
    model: MultitaskFusionClassifier,
    data: StrictTrainingDataModule,
    indices: np.ndarray,
    config: ClassificationV2TrainingConfig,
    device: torch.device,
    *,
    split: str,
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
                "temporal_unit_key": batch.metadata["temporal_unit_key"][row_index],
                "fold_id": config.execution.fold_id,
                "split": split,
                "source_type": batch.metadata["source_type"][row_index],
                "true_label": labels[int(true_values[row_index])],
                "predicted_label": labels[int(predicted[row_index])],
                "confidence": float(probabilities[row_index, predicted[row_index]]),
                "model_version": config.model.architecture_version,
                "snapshot_id": config.dataset.snapshot_json.stem,
            }
            row.update(
                {
                    f"prob_{label}": float(probabilities[row_index, i])
                    for i, label in enumerate(labels)
                }
            )
            rows.append(row)
    predictions = pd.DataFrame(rows)
    metric = _macro_f1(predictions["true_label"], predictions["predicted_label"], labels)
    return predictions, {f"{split}_window_macro_f1": metric}


def _build_model(
    config: ClassificationV2TrainingConfig, probe: StrictTrainingBatch
) -> MultitaskFusionClassifier:
    """Derive tensor dimensions only from declared model branches, never arbitrary columns."""

    validate_model_inputs(probe.model_inputs)
    spatial = probe.model_inputs["spatial_features"]
    observed_spatial_dims = {name: int(value.shape[-1]) for name, value in spatial.items()}
    spatial_dims = observed_spatial_dims if config.model.enable_spatial else {}
    if config.model.enable_spatial and tuple(sorted(spatial_dims)) != tuple(
        sorted(config.model.spatial_feature_groups)
    ):
        raise ValueError(
            f"spatial whitelist mismatch: config={config.model.spatial_feature_groups}, "
            f"data={sorted(observed_spatial_dims)}"
        )
    interaction_dim = (
        int(probe.model_inputs["interaction_context_features"].shape[-1])
        if config.model.enable_interaction_context
        else None
    )
    return build_multimodal_model(
        config.model,
        spatial_input_dims=spatial_dims,
        interaction_context_dim=interaction_dim,
        num_classes=len(VALID_BEHAVIORS),
    )


def _behavior_class_weights(
    data: StrictTrainingDataModule,
    indices: np.ndarray,
    config: ClassificationV2TrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    labels = data.bundle.y.iloc[indices].astype(str)
    counts = labels.value_counts().reindex(VALID_BEHAVIORS, fill_value=0)
    if (counts <= 0).any():
        missing = counts[counts <= 0].index.tolist()
        raise ValueError(f"training fold missing behavior classes: {missing}")
    if config.loss.sample_weight_policy != "event_class":
        values = np.ones(len(VALID_BEHAVIORS), dtype=np.float32)
    else:
        values = np.asarray(
            [data.fold_class_weights[label] for label in VALID_BEHAVIORS],
            dtype=np.float32,
        )
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError("event-class policy requires positive train-fold class weights")
    return torch.tensor(values, dtype=torch.float32, device=device)


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
    test_indices: np.ndarray,
    behavior_weights: torch.Tensor,
    auxiliary_weights: dict[str, torch.Tensor],
    history: list[dict[str, Any]],
    best_epoch: int,
    device: torch.device,
    resumed_from: str | None,
    test_metrics: dict[str, float],
    preprocessing_state: FoldPreprocessingState,
    preprocessing_status: str,
    run_identity: dict[str, Any],
    optimizer_group_contract: dict[str, Any],
) -> dict[str, Any]:
    config_payload = training_config_to_jsonable(config)
    git = _git_state()
    return {
        "schema_version": RUN_AUDIT_SCHEMA_VERSION,
        "valid": True,
        "config": config_payload,
        "config_sha256": training_config_sha256(config),
        "run_identity": run_identity,
        "snapshot_id": config.dataset.snapshot_json.stem,
        "snapshot_sha256": _file_sha256(config.dataset.snapshot_json),
        "split_manifest_sha256": _file_sha256(config.dataset.native_oof_fold_manifest),
        "trainer_contract_sha256": _file_sha256(config.dataset.trainer_contract_json),
        "git": git,
        "device": str(device),
        "hardware": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "model_architecture": config.model.architecture_version,
        "model_mode_contract": model_mode_contract(config.model.model_mode),
        "model_parameters": model_parameter_report(model),
        "model_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "visual_freeze_schedule": visual_freeze_schedule_payload(
            config.model,
            total_epochs=config.optimization.epochs,
        ),
        "optimizer_group_contract": optimizer_group_contract,
        "label_order": list(VALID_BEHAVIORS),
        "feature_whitelist": list(config.model.spatial_feature_groups),
        "normalization_imputation": "bound_to_trainer_contract_and_snapshot",
        "preprocessing": {
            "path": "preprocessing.json",
            "state_sha256": preprocessing_state.state_sha256,
            "train_window_id_sha256": (preprocessing_state.train_window_id_sha256),
            "status": preprocessing_status,
        },
        "train_selected_window_id_sha256": _selected_id_hash(data, train_indices),
        "validation_selected_window_id_sha256": _selected_id_hash(data, eval_indices),
        "test_selected_window_id_sha256": _selected_id_hash(data, test_indices),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(eval_indices)),
        "test_rows": int(len(test_indices)),
        "behavior_class_weights_train_fold_only": behavior_weights.detach().cpu().tolist(),
        "auxiliary_class_weights_train_fold_only": {
            name: value.detach().cpu().tolist() for name, value in auxiliary_weights.items()
        },
        "data_module_audit": data.audit(),
        "history": history,
        "outer_test_metrics": test_metrics,
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
            "oof_test_predictions": "oof_test_predictions.csv",
            "preprocessing": "preprocessing.json",
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


def _require_nonempty_split(
    train_indices: np.ndarray, eval_indices: np.ndarray, test_indices: np.ndarray
) -> None:
    if len(train_indices) == 0 or len(eval_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            f"empty grouped split: train={len(train_indices)}, "
            f"validation={len(eval_indices)}, test={len(test_indices)}"
        )


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
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
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
