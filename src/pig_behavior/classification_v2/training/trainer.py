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
    validate_training_config,
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
from pig_behavior.classification_v2.training.validation_selection import (
    VALIDATION_PRIMARY_METRIC,
    VALIDATION_TIEBREAKER,
    ValidationSelectionScore,
    build_native_split_evaluation,
    selection_score_from_metrics,
    validation_score_is_better,
    validation_selection_policy,
)
from pig_behavior.classification_v2.training.visual_freeze import (
    build_visual_optimizer_groups,
    configure_visual_train_stage,
    optimizer_group_report,
    visual_freeze_schedule_payload,
)

PREDICTION_SCHEMA_VERSION = "classification_v2_training_predictions_v2"
RUN_AUDIT_SCHEMA_VERSION = "classification_v2_training_run_audit_v3"


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

    validate_training_config(config)
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
                    path=session.run_dir / "best_validation_predictions.csv",
                    checkpoint_path=session.run_dir / "best_validation.pt",
                    split="validation",
                    expected_rows=int(audit["validation_rows"]),
                ),
                PredictionArtifact(
                    path=(
                        session.run_dir
                        / "best_validation_native_unit_predictions.csv"
                    ),
                    checkpoint_path=session.run_dir / "best_validation.pt",
                    split="validation",
                    expected_rows=int(
                        audit["best_validation_native_unit_rows"]
                    ),
                    key_col="temporal_unit_key",
                    true_col="true_label",
                    pred_col="native_predicted_behavior",
                    split_col="prediction_split",
                    prediction_unit="native_temporal_unit",
                ),
                PredictionArtifact(
                    path=session.run_dir / "oof_test_predictions.csv",
                    checkpoint_path=session.run_dir / "best_validation.pt",
                    split="test",
                    expected_rows=int(audit["test_rows"]),
                ),
                PredictionArtifact(
                    path=(
                        session.run_dir
                        / "oof_test_native_unit_predictions.csv"
                    ),
                    checkpoint_path=session.run_dir / "best_validation.pt",
                    split="test",
                    expected_rows=int(audit["outer_test_native_unit_rows"]),
                    key_col="temporal_unit_key",
                    true_col="true_label",
                    pred_col="native_predicted_behavior",
                    split_col="prediction_split",
                    prediction_unit="native_temporal_unit",
                ),
            ],
            metric_paths=[
                session.run_dir / "run_audit.json",
                session.run_dir / "best_validation_aggregation_audit.json",
                session.run_dir / "oof_test_aggregation_audit.json",
            ],
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
        model = _build_model(config, probe, data).to(device)
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
        best_score: ValidationSelectionScore | None = None
        best_epoch = -1
        stale_epochs = 0
        max_epochs = 1 if config.execution.mode == "smoke" else config.optimization.epochs
        last_checkpoint = output_dir / "last.pt"
        best_validation_aggregation_audit: dict[str, Any] = {}
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
            history, best_score, best_epoch, stale_epochs = (
                _restore_validation_selection_state(
                    resumed,
                    config,
                )
            )
            best_validation_aggregation_audit = _read_json_object(
                output_dir / "best_validation_aggregation_audit.json"
            )
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
            predictions, native_predictions, eval_metrics, aggregation_audit = _evaluate(
                model, data, eval_indices, config, device, split="validation"
            )
            predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
            native_predictions.to_csv(
                output_dir / "validation_native_unit_predictions.csv",
                index=False,
            )
            _write_json_atomic(
                output_dir / "validation_aggregation_audit.json",
                aggregation_audit,
            )
            candidate_score = selection_score_from_metrics(eval_metrics)
            improved = validation_score_is_better(
                candidate_score,
                best_score,
                tolerance=(
                    config.optimization.early_stopping_tie_tolerance
                ),
            )
            if improved:
                best_score, best_epoch, stale_epochs = candidate_score, epoch, 0
            else:
                stale_epochs += 1
            record = {
                "epoch": epoch,
                **train_result,
                **eval_metrics,
                "selected_as_best_validation": improved,
            }
            history.append(record)
            checkpoint_metrics = _checkpoint_metrics(
                record,
                history,
                best_score,
                best_epoch,
                stale_epochs,
                config,
            )
            if improved:
                predictions.to_csv(
                    output_dir / "best_validation_predictions.csv",
                    index=False,
                )
                native_predictions.to_csv(
                    output_dir / "best_validation_native_unit_predictions.csv",
                    index=False,
                )
                _write_json_atomic(
                    output_dir / "best_validation_aggregation_audit.json",
                    aggregation_audit,
                )
                best_validation_aggregation_audit = aggregation_audit
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
                    metrics=checkpoint_metrics,
                    visual_freeze_state=train_result["visual_freeze"],
                )
            save_training_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                run_identity=session.identity.to_payload(),
                preprocessing_sha256=preprocessing_state.state_sha256,
                train_window_id_sha256=(
                    preprocessing_state.train_window_id_sha256
                ),
                epoch=epoch,
                global_step=global_step,
                metrics=checkpoint_metrics,
                visual_freeze_state=train_result["visual_freeze"],
            )
            if (
                config.execution.mode != "smoke"
                and stale_epochs >= config.optimization.early_stopping_patience
            ):
                break
        best_checkpoint = output_dir / "best_validation.pt"
        if not best_checkpoint.exists():
            raise ValueError("best-validation checkpoint missing before outer-test evaluation")
        loaded_best = load_training_checkpoint(
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
        _validate_best_validation_artifacts(
            loaded_best,
            best_validation_aggregation_audit,
            best_score,
            best_epoch,
        )
        (
            test_predictions,
            test_native_predictions,
            test_metrics,
            test_aggregation_audit,
        ) = _evaluate(
            model, data, test_indices, config, device, split="test"
        )
        test_predictions.to_csv(output_dir / "oof_test_predictions.csv", index=False)
        test_native_predictions.to_csv(
            output_dir / "oof_test_native_unit_predictions.csv",
            index=False,
        )
        _write_json_atomic(
            output_dir / "oof_test_aggregation_audit.json",
            test_aggregation_audit,
        )
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
            best_score,
            best_validation_aggregation_audit,
            device,
            resumed_from,
            test_metrics,
            test_aggregation_audit,
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
                consistency = hierarchy_consistency_loss(
                    output.behavior,
                    output.auxiliary_logits(),
                    batch.behavior_target,
                    batch.auxiliary_masks,
                )
            else:
                auxiliary_loss = behavior_loss.new_zeros(())
                consistency = behavior_loss.new_zeros(())
            if config.model.enable_date_adversarial:
                if output.domain is None:
                    raise ValueError("M1-DG1 training forward omitted domain logits")
                if batch.recording_date_target is None:
                    raise ValueError("M1-DG1 training batch omitted train date target")
                domain_loss = F.cross_entropy(
                    output.domain,
                    batch.recording_date_target,
                )
            else:
                domain_loss = behavior_loss.new_zeros(())
            total = (
                config.loss.behavior_weight * behavior_loss
                + auxiliary_loss
                + config.loss.hierarchy_consistency_weight * consistency
                + config.model.domain_loss_weight * domain_loss
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
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int],
    dict[str, Any],
]:
    """Evaluate windows and collapse them to one strict native-unit record."""

    model.eval()
    rows: list[dict[str, Any]] = []
    labels = list(VALID_BEHAVIORS)
    for start in range(0, len(indices), config.optimization.eval_batch_size):
        batch = data.batch(indices[start : start + config.optimization.eval_batch_size])
        amp_enabled = device.type == "cuda" and config.optimization.precision == "amp"
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(**batch.model_inputs)
        behavior_logits = output.behavior if hasattr(output, "behavior") else output
        probabilities = torch.softmax(behavior_logits.float(), dim=1).cpu().numpy()
        true_values = batch.behavior_target.cpu().numpy()
        predicted = probabilities.argmax(axis=1)
        for row_index in range(len(predicted)):
            row = {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "window_id": batch.metadata["window_id"][row_index],
                "temporal_unit_key": batch.metadata["temporal_unit_key"][row_index],
                "fold_id": config.execution.fold_id,
                "oof_fold_id": batch.metadata["oof_fold_id"][row_index],
                "split": split,
                "source_type": batch.metadata["source_type"][row_index],
                "split_group_key": batch.metadata["split_group_key"][row_index],
                "true_label": labels[int(true_values[row_index])],
                "predicted_label": labels[int(predicted[row_index])],
                "y_true": labels[int(true_values[row_index])],
                "y_pred": labels[int(predicted[row_index])],
                "prediction_split": split,
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
    min_supported_classes = (
        config.optimization.early_stopping_min_supported_classes
        if split == "validation"
        else 1
    )
    native_predictions, metrics, aggregation_audit = (
        build_native_split_evaluation(
            predictions,
            split=split,
            min_supported_classes=min_supported_classes,
            label_order=tuple(labels),
        )
    )
    return predictions, native_predictions, metrics, aggregation_audit


def _build_model(
    config: ClassificationV2TrainingConfig,
    probe: StrictTrainingBatch,
    data: StrictTrainingDataModule,
) -> MultitaskFusionClassifier:
    """Derive tensor dimensions only from declared model branches, never arbitrary columns."""

    validate_model_inputs(
        probe.model_inputs,
        expected_keys=data.expected_model_input_keys,
    )
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


def _checkpoint_metrics(
    current: dict[str, Any],
    history: list[dict[str, Any]],
    best_score: ValidationSelectionScore | None,
    best_epoch: int,
    stale_epochs: int,
    config: ClassificationV2TrainingConfig,
) -> dict[str, Any]:
    """Persist complete early-stopping state so interrupted runs resume exactly."""

    if best_score is None or best_epoch < 0 or stale_epochs < 0:
        raise ValueError("validation selection state is incomplete")
    policy = validation_selection_policy(
        tolerance=config.optimization.early_stopping_tie_tolerance,
        min_supported_classes=(
            config.optimization.early_stopping_min_supported_classes
        ),
    )
    return {
        "schema_version": "classification_v2.checkpoint_metrics.v1",
        "current_epoch_metrics": current,
        "history": history,
        "validation_selection": {
            **policy,
            "best_primary": best_score.primary,
            "best_tiebreaker": best_score.tiebreaker,
            "best_epoch": int(best_epoch),
            "stale_epochs": int(stale_epochs),
        },
    }


def _restore_validation_selection_state(
    resumed: dict[str, Any],
    config: ClassificationV2TrainingConfig,
) -> tuple[
    list[dict[str, Any]],
    ValidationSelectionScore,
    int,
    int,
]:
    """Restore history and native-unit selection state from the last checkpoint."""

    metrics = resumed.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("checkpoint metrics payload is missing")
    if metrics.get("schema_version") != "classification_v2.checkpoint_metrics.v1":
        raise ValueError("checkpoint metrics schema mismatch")
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("checkpoint validation history is missing")
    if any(not isinstance(row, dict) for row in history):
        raise ValueError("checkpoint validation history is malformed")
    current = metrics.get("current_epoch_metrics")
    if current != history[-1]:
        raise ValueError("checkpoint current metrics do not match history tail")
    if int(history[-1].get("epoch", -1)) != int(resumed["epoch"]):
        raise ValueError("checkpoint validation history epoch mismatch")

    state = metrics.get("validation_selection")
    if not isinstance(state, dict):
        raise ValueError("checkpoint validation selection state is missing")
    expected_policy = validation_selection_policy(
        tolerance=config.optimization.early_stopping_tie_tolerance,
        min_supported_classes=(
            config.optimization.early_stopping_min_supported_classes
        ),
    )
    policy_mismatches = {
        key: {"expected": value, "observed": state.get(key)}
        for key, value in expected_policy.items()
        if state.get(key) != value
    }
    if policy_mismatches:
        raise ValueError(
            "checkpoint validation selection policy mismatch="
            f"{policy_mismatches}"
        )
    score = selection_score_from_metrics(
        {
            VALIDATION_PRIMARY_METRIC: state.get("best_primary"),
            VALIDATION_TIEBREAKER: state.get("best_tiebreaker"),
        }
    )
    best_epoch = int(state.get("best_epoch", -1))
    stale_epochs = int(state.get("stale_epochs", -1))
    if best_epoch < 0 or stale_epochs < 0:
        raise ValueError("checkpoint validation selection counters are invalid")
    selected_epochs = [
        int(row["epoch"])
        for row in history
        if bool(row.get("selected_as_best_validation"))
    ]
    if not selected_epochs or selected_epochs[-1] != best_epoch:
        raise ValueError("checkpoint best epoch disagrees with validation history")
    if int(history[-1]["epoch"]) - best_epoch != stale_epochs:
        raise ValueError("checkpoint stale epoch count disagrees with history")
    return list(history), score, best_epoch, stale_epochs


def _validate_best_validation_artifacts(
    loaded_checkpoint: dict[str, Any],
    aggregation_audit: dict[str, Any],
    best_score: ValidationSelectionScore | None,
    best_epoch: int,
) -> None:
    """Prove checkpoint, prediction audit, and selected score name one epoch."""

    if best_score is None or best_epoch < 0:
        raise ValueError("best validation state is missing")
    metrics = loaded_checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("best checkpoint metrics are missing")
    state = metrics.get("validation_selection")
    if not isinstance(state, dict):
        raise ValueError("best checkpoint validation-selection state is missing")
    observed_score = selection_score_from_metrics(
        {
            VALIDATION_PRIMARY_METRIC: state.get("best_primary"),
            VALIDATION_TIEBREAKER: state.get("best_tiebreaker"),
        }
    )
    if observed_score != best_score or int(state.get("best_epoch", -1)) != best_epoch:
        raise ValueError("best checkpoint selection state mismatch")
    audit_metrics = aggregation_audit.get("metrics")
    if not isinstance(audit_metrics, dict):
        raise ValueError("best validation aggregation metrics are missing")
    if selection_score_from_metrics(audit_metrics) != best_score:
        raise ValueError("best validation prediction audit score mismatch")


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
    best_score: ValidationSelectionScore | None,
    best_validation_aggregation_audit: dict[str, Any],
    device: torch.device,
    resumed_from: str | None,
    test_metrics: dict[str, float | int],
    test_aggregation_audit: dict[str, Any],
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
        "validation_selection_policy": validation_selection_policy(
            tolerance=config.optimization.early_stopping_tie_tolerance,
            min_supported_classes=(
                config.optimization.early_stopping_min_supported_classes
            ),
        ),
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
        "best_validation_aggregation_audit": (
            best_validation_aggregation_audit
        ),
        "outer_test_metrics": test_metrics,
        "outer_test_aggregation_audit": test_aggregation_audit,
        "outer_predictions_used_for_model_selection": False,
        "best_epoch": best_epoch,
        "best_validation_native_unit_rows": int(
            best_validation_aggregation_audit.get(
                "output_native_unit_rows",
                0,
            )
        ),
        "outer_test_native_unit_rows": int(
            test_aggregation_audit.get("output_native_unit_rows", 0)
        ),
        "best_validation_score": (
            {
                "primary": best_score.primary,
                "tiebreaker": best_score.tiebreaker,
            }
            if best_score is not None
            else None
        ),
        "resume": {
            "enabled": config.execution.resume,
            "resumed_from": resumed_from,
            "optimizer_scaler_rng_restored": resumed_from is not None,
        },
        "artifacts": {
            "last_checkpoint": "last.pt",
            "best_checkpoint": "best_validation.pt",
            "predictions": "validation_predictions.csv",
            "validation_native_predictions": (
                "validation_native_unit_predictions.csv"
            ),
            "best_validation_predictions": "best_validation_predictions.csv",
            "best_validation_native_predictions": (
                "best_validation_native_unit_predictions.csv"
            ),
            "validation_aggregation_audit": (
                "validation_aggregation_audit.json"
            ),
            "best_validation_aggregation_audit": (
                "best_validation_aggregation_audit.json"
            ),
            "oof_test_predictions": "oof_test_predictions.csv",
            "oof_test_native_predictions": (
                "oof_test_native_unit_predictions.csv"
            ),
            "oof_test_aggregation_audit": "oof_test_aggregation_audit.json",
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


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required JSON artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload
