"""Atomic, lineage-validated checkpoints for classification_v2 training."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    training_config_to_jsonable,
)

CHECKPOINT_SCHEMA_VERSION = "classification_v2_training_checkpoint_v1"


def training_config_sha256(config: ClassificationV2TrainingConfig) -> str:
    payload = json.dumps(training_config_to_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    config: ClassificationV2TrainingConfig,
    preprocessing_sha256: str,
    train_window_id_sha256: str,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Atomically save complete training and random-number-generator state."""

    if epoch < 0 or global_step < 0:
        raise ValueError("epoch and global_step must be non-negative")
    git_state = _git_state()
    lineage = _lineage(
        config,
        git_state,
        preprocessing_sha256=preprocessing_sha256,
        train_window_id_sha256=train_window_id_sha256,
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "lineage": lineage,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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
        _git_state(),
        preprocessing_sha256=preprocessing_sha256,
        train_window_id_sha256=train_window_id_sha256,
        include_git=False,
    )
    observed = payload.get("lineage", {})
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint lineage mismatch: {mismatches}")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
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
    }


def _lineage(
    config: ClassificationV2TrainingConfig,
    git_state: dict[str, Any],
    *,
    preprocessing_sha256: str,
    train_window_id_sha256: str,
    include_git: bool = True,
) -> dict[str, Any]:
    if not preprocessing_sha256 or not train_window_id_sha256:
        raise ValueError("checkpoint preprocessing lineage must not be blank")
    lineage = {
        "config_sha256": training_config_sha256(config),
        "snapshot_id": config.dataset.snapshot_json.stem,
        "fold_id": config.execution.fold_id,
        "architecture_version": config.model.architecture_version,
        "preprocessing_sha256": preprocessing_sha256,
        "train_window_id_sha256": train_window_id_sha256,
    }
    if include_git:
        lineage.update(git_commit=git_state.get("commit"), git_dirty=git_state.get("dirty"))
    return lineage


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


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except Exception:
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}
