"""Atomic, lineage-validated checkpoints for classification_v2 training."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    training_config_to_jsonable,
)
from pig_behavior.classification_v2.training.visual_freeze import (
    optimizer_group_report,
    visual_freeze_parameter_report,
    visual_freeze_schedule_payload,
    visual_freeze_stage_for_epoch,
)

CHECKPOINT_SCHEMA_VERSION = "classification_v2_training_checkpoint_v5"
RUN_IDENTITY_REQUIRED_FIELDS = (
    "identity_schema_version",
    "run_id",
    "experiment_name",
    "execution_profile",
    "code_sha",
    "dirty_worktree",
    "worktree_state_sha256",
    "config_sha256",
    "dataset_snapshot_id",
    "dataset_snapshot_sha256",
    "cache_sha256",
    "fold_manifest_sha256",
    "feature_whitelist_sha256",
    "temporal_view_selection_sha256",
    "temporal_view_manifest_sha256",
    "fold_event_weight_sha256",
    "fold_id",
    "architecture_version",
    "model_mode",
    "backbone_name",
    "pretrained_weight_enum",
    "resolution",
    "visual_freeze_contract_version",
    "visual_freeze_policy",
    "visual_frozen_warmup_epochs",
    "visual_layer4_only_epochs",
    "visual_backbone_lr_multiplier",
    "temporal_view",
    "temporal_encoder_name",
    "modalities",
    "loss_name",
    "sampler_policy",
    "optimizer_name",
    "precision",
    "augmentation_policy",
)


def training_config_sha256(config: ClassificationV2TrainingConfig) -> str:
    payload = json.dumps(
        training_config_to_jsonable(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config: ClassificationV2TrainingConfig,
    run_identity: dict[str, Any],
    preprocessing_sha256: str,
    train_window_id_sha256: str,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    visual_freeze_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save complete training and random-number-generator state."""

    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step must be non-negative")
    lineage = _lineage(
        config,
        run_identity=run_identity,
        preprocessing_sha256=preprocessing_sha256,
        train_window_id_sha256=train_window_id_sha256,
    )
    if visual_freeze_state is None:
        resolved_visual_freeze_state = visual_freeze_parameter_report(
            model,
            config.model,
            epoch=epoch,
        )
    else:
        resolved_visual_freeze_state = _validate_visual_freeze_state(
            visual_freeze_state,
            config=config,
            epoch=epoch,
        )
    optimizer_groups = optimizer_group_report(optimizer)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "lineage": lineage,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_group_state": optimizer_groups,
        "visual_freeze_state": resolved_visual_freeze_state,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": _rng_state(),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    audit = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_path": str(path),
        "checkpoint_size_bytes": int(path.stat().st_size),
        "lineage": lineage,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "has_optimizer_state": True,
        "has_scaler_state": scaler is not None,
        "has_rng_state": True,
        "optimizer_group_state": optimizer_groups,
        "visual_freeze_state": resolved_visual_freeze_state,
        "metrics": metrics,
        "errors": [],
        "valid": True,
    }
    audit_path = path.with_suffix(path.suffix + ".audit.json")
    audit_temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    audit_temporary.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit_temporary.replace(audit_path)
    return audit


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config: ClassificationV2TrainingConfig,
    run_identity: dict[str, Any],
    preprocessing_sha256: str,
    train_window_id_sha256: str,
    map_location: torch.device | str = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint only when config/snapshot/fold/architecture lineage matches."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema mismatch: {payload.get('schema_version')}")
    expected = _lineage(
        config,
        run_identity=run_identity,
        preprocessing_sha256=preprocessing_sha256,
        train_window_id_sha256=train_window_id_sha256,
    )
    observed = payload.get("lineage", {})
    if set(observed) != set(expected):
        raise ValueError(
            "checkpoint lineage schema mismatch: "
            f"missing={sorted(set(expected).difference(observed))}, "
            f"unknown={sorted(set(observed).difference(expected))}"
        )
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint lineage mismatch: {mismatches}")
    visual_freeze_state = _validate_visual_freeze_state(
        payload.get("visual_freeze_state"),
        config=config,
        epoch=int(payload["epoch"]),
    )
    saved_optimizer_groups = payload.get("optimizer_group_state")
    if optimizer_group_report(optimizer) != saved_optimizer_groups:
        raise ValueError("checkpoint optimizer-group contract mismatch before load")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    optimizer_groups = optimizer_group_report(optimizer)
    if optimizer_groups != saved_optimizer_groups:
        raise ValueError("checkpoint optimizer-group contract mismatch")
    if scaler is not None:
        state = payload.get("scaler_state_dict")
        if state is None:
            raise ValueError("checkpoint missing requested AMP scaler state")
        scaler.load_state_dict(state)
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "metrics": payload.get("metrics", {}),
        "lineage": observed,
        "optimizer_group_state": optimizer_groups,
        "visual_freeze_state": visual_freeze_state,
    }


def _lineage(
    config: ClassificationV2TrainingConfig,
    *,
    run_identity: dict[str, Any],
    preprocessing_sha256: str,
    train_window_id_sha256: str,
) -> dict[str, Any]:
    if not preprocessing_sha256 or not train_window_id_sha256:
        raise ValueError("checkpoint preprocessing lineage must not be blank")
    missing = sorted(set(RUN_IDENTITY_REQUIRED_FIELDS).difference(run_identity))
    if missing:
        raise ValueError(f"checkpoint run identity missing fields={missing}")
    expected = {
        "config_sha256": training_config_sha256(config),
        "fold_id": config.execution.fold_id,
        "architecture_version": config.model.architecture_version,
        "model_mode": config.model.model_mode,
        "backbone_name": config.model.backbone_name,
        "pretrained_weight_enum": config.model.pretrained_weight_enum,
        "resolution": config.model.image_size,
        "visual_freeze_contract_version": (
            visual_freeze_schedule_payload(config.model)["contract_version"]
        ),
        "visual_freeze_policy": config.model.visual_freeze_policy,
        "visual_frozen_warmup_epochs": (
            config.model.visual_frozen_warmup_epochs
        ),
        "visual_layer4_only_epochs": config.model.visual_layer4_only_epochs,
        "visual_backbone_lr_multiplier": (
            float(config.model.visual_backbone_lr_multiplier)
        ),
        "temporal_view": config.model.temporal_view,
        "temporal_encoder_name": config.model.temporal_encoder_name,
        "optimizer_name": config.optimization.optimizer,
        "precision": config.optimization.precision,
        "augmentation_policy": config.dataset.augmentation_policy,
    }
    mismatches = {
        key: {"expected": value, "observed": run_identity.get(key)}
        for key, value in expected.items()
        if run_identity.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint run identity mismatch={mismatches}")
    lineage = {
        **{
            key: run_identity[key]
            for key in RUN_IDENTITY_REQUIRED_FIELDS
        },
        "snapshot_id": run_identity["dataset_snapshot_id"],
        "preprocessing_sha256": preprocessing_sha256,
        "train_window_id_sha256": train_window_id_sha256,
    }
    return lineage


def _validate_visual_freeze_state(
    payload: object,
    *,
    config: ClassificationV2TrainingConfig,
    epoch: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint visual-freeze state is missing")
    schedule = visual_freeze_schedule_payload(config.model)
    expected = {
        **schedule,
        "epoch": epoch,
        "stage": visual_freeze_stage_for_epoch(config.model, epoch),
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint visual-freeze mismatch={mismatches}")
    if payload.get("valid") is not True or payload.get("errors"):
        raise ValueError("checkpoint visual-freeze audit is invalid")
    return payload


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
