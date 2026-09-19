"""Immutable semantic identity for one classification_v2 fold run."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.checkpoint import (
    training_config_sha256,
)
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    resolve_temporal_view_manifest,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
    payload_sha256,
)
from pig_behavior.classification_v2.training.validation_selection import (
    VALIDATION_PRIMARY_METRIC,
    VALIDATION_SELECTION_CONTRACT_VERSION,
    VALIDATION_TIEBREAKER,
)
from pig_behavior.classification_v2.training.visual_freeze import (
    VISUAL_FREEZE_CONTRACT_VERSION,
    visual_freeze_schedule_payload,
)

RUN_IDENTITY_SCHEMA_VERSION = "classification_v2.run_identity.v3"
EXECUTION_PROFILES = frozenset(
    {"local_smoke", "remote_pilot", "remote_full_oof"}
)
IDENTITY_HASH_FIELDS = (
    "worktree_state_sha256",
    "config_sha256",
    "dataset_snapshot_sha256",
    "cache_sha256",
    "fold_manifest_sha256",
    "feature_whitelist_sha256",
    "temporal_view_selection_sha256",
    "temporal_view_manifest_sha256",
    "fold_event_weight_sha256",
)
CODE_SCOPE_PATHS = (
    "src/pig_behavior/classification_v2",
    "configs/classification_v2",
    "scripts/classification_v2",
    "tests",
)


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Semantic fields that must remain exact across resume and artifacts."""

    identity_schema_version: str
    run_id: str
    experiment_name: str
    execution_profile: str
    fold_id: str
    seed: int
    code_sha: str
    dirty_worktree: bool
    worktree_state_sha256: str
    config_sha256: str
    dataset_snapshot_id: str
    dataset_snapshot_sha256: str
    cache_sha256: str
    fold_manifest_sha256: str
    feature_whitelist_sha256: str
    temporal_view_selection_sha256: str
    temporal_view_manifest_sha256: str
    fold_event_weight_sha256: str
    architecture_version: str
    model_mode: str
    backbone_name: str
    pretrained_weight_enum: str
    resolution: int
    visual_freeze_contract_version: str
    visual_freeze_policy: str
    visual_frozen_warmup_epochs: int
    visual_layer4_only_epochs: int
    visual_backbone_lr_multiplier: float
    early_stopping_contract_version: str
    early_stopping_metric: str
    early_stopping_tiebreaker: str
    early_stopping_tie_tolerance: float
    early_stopping_min_supported_classes: int
    temporal_view: str
    temporal_encoder_name: str
    modalities: tuple[str, ...]
    loss_name: str
    sampler_policy: str
    optimizer_name: str
    precision: str
    augmentation_policy: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "identity_schema_version": self.identity_schema_version,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "execution_profile": self.execution_profile,
            "fold_id": self.fold_id,
            "seed": self.seed,
            "code_sha": self.code_sha,
            "dirty_worktree": self.dirty_worktree,
            "worktree_state_sha256": self.worktree_state_sha256,
            "config_sha256": self.config_sha256,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "dataset_snapshot_sha256": self.dataset_snapshot_sha256,
            "cache_sha256": self.cache_sha256,
            "fold_manifest_sha256": self.fold_manifest_sha256,
            "feature_whitelist_sha256": self.feature_whitelist_sha256,
            "temporal_view_selection_sha256": (
                self.temporal_view_selection_sha256
            ),
            "temporal_view_manifest_sha256": (
                self.temporal_view_manifest_sha256
            ),
            "fold_event_weight_sha256": self.fold_event_weight_sha256,
            "architecture_version": self.architecture_version,
            "model_mode": self.model_mode,
            "backbone_name": self.backbone_name,
            "pretrained_weight_enum": self.pretrained_weight_enum,
            "resolution": self.resolution,
            "visual_freeze_contract_version": (
                self.visual_freeze_contract_version
            ),
            "visual_freeze_policy": self.visual_freeze_policy,
            "visual_frozen_warmup_epochs": (
                self.visual_frozen_warmup_epochs
            ),
            "visual_layer4_only_epochs": self.visual_layer4_only_epochs,
            "visual_backbone_lr_multiplier": (
                self.visual_backbone_lr_multiplier
            ),
            "early_stopping_contract_version": (
                self.early_stopping_contract_version
            ),
            "early_stopping_metric": self.early_stopping_metric,
            "early_stopping_tiebreaker": self.early_stopping_tiebreaker,
            "early_stopping_tie_tolerance": (
                self.early_stopping_tie_tolerance
            ),
            "early_stopping_min_supported_classes": (
                self.early_stopping_min_supported_classes
            ),
            "temporal_view": self.temporal_view,
            "temporal_encoder_name": self.temporal_encoder_name,
            "modalities": list(self.modalities),
            "loss_name": self.loss_name,
            "sampler_policy": self.sampler_policy,
            "optimizer_name": self.optimizer_name,
            "precision": self.precision,
            "augmentation_policy": self.augmentation_policy,
        }

    @property
    def identity_sha256(self) -> str:
        return payload_sha256(self.to_payload())


def build_run_identity(
    config: ClassificationV2TrainingConfig,
    snapshot_check: dict[str, Any],
) -> RunIdentity:
    """Bind code, config, data, cache, fold, and model semantics."""

    if snapshot_check.get("valid") is not True:
        raise ValueError("cannot build run identity from an invalid snapshot")
    current = snapshot_check.get("current") or {}
    artifacts = current.get("artifacts") or {}
    config_hash = training_config_sha256(config)
    code_sha, dirty, worktree_state_hash = _git_state()
    if not code_sha:
        raise ValueError("run lineage requires a git code SHA")
    snapshot_hash = file_sha256(config.dataset.snapshot_json)
    cache_paths = (
        config.dataset.actor_packed_cache,
        config.dataset.actor_packed_index,
        config.dataset.visual_cache_manifest,
        config.dataset.visual_packed_cache,
        config.dataset.visual_packed_index,
    )
    cache_records = [
        _snapshot_artifact_by_path(artifacts, path) for path in cache_paths
    ]
    cache_hash = payload_sha256(
        [
            {
                "path": str(Path(str(item["path"])).resolve()),
                "sha256": _required_hash(item, "cache artifact"),
            }
            for item in cache_records
        ]
    )
    fold_hash = _required_hash(
        _snapshot_artifact_by_path(
            artifacts,
            config.dataset.grouped_fold_roles,
        ),
        "fold manifest",
    )
    whitelist_hash = _required_hash(
        _snapshot_artifact_by_path(
            artifacts,
            config.dataset.trainer_contract_json,
        ),
        "feature whitelist contract",
    )
    temporal_view_hash = _required_hash(
        _snapshot_artifact_by_path(
            artifacts,
            config.dataset.temporal_view_selection_manifest,
        ),
        "temporal-view selection",
    )
    temporal_manifest_hash = _required_hash(
        _snapshot_artifact_by_path(
            artifacts,
            resolve_temporal_view_manifest(config),
        ),
        "temporal-view tensor manifest",
    )
    if config.dataset.fold_event_weight_manifest is None:
        raise ValueError("run lineage requires fold event-weight manifest")
    event_weight_hash = _required_hash(
        _snapshot_artifact_by_path(
            artifacts,
            config.dataset.fold_event_weight_manifest,
        ),
        "fold event-weight manifest",
    )
    pretrained = config.model.pretrained_weight_enum.strip()
    if pretrained.lower() in {"", "auto", "default", "unknown"}:
        raise ValueError("pretrained weight enum is ambiguous")
    modalities = _enabled_modalities(config)
    freeze_schedule = visual_freeze_schedule_payload(
        config.model,
        total_epochs=config.optimization.epochs,
    )
    generated_run_id = _generated_run_id(
        experiment_name=config.execution.experiment_name,
        execution_profile=config.execution.execution_profile,
        fold_id=config.execution.fold_id,
        seed=config.optimization.seed,
        config_sha256=config_hash,
        code_sha=code_sha,
        worktree_state_sha256=worktree_state_hash,
        dataset_snapshot_sha256=snapshot_hash,
    )
    identity = RunIdentity(
        identity_schema_version=RUN_IDENTITY_SCHEMA_VERSION,
        run_id=config.execution.run_id or generated_run_id,
        experiment_name=config.execution.experiment_name,
        execution_profile=config.execution.execution_profile,
        fold_id=config.execution.fold_id,
        seed=config.optimization.seed,
        code_sha=code_sha,
        dirty_worktree=dirty,
        worktree_state_sha256=worktree_state_hash,
        config_sha256=config_hash,
        dataset_snapshot_id=str(snapshot_check["current_snapshot_id"]),
        dataset_snapshot_sha256=snapshot_hash,
        cache_sha256=cache_hash,
        fold_manifest_sha256=fold_hash,
        feature_whitelist_sha256=whitelist_hash,
        temporal_view_selection_sha256=temporal_view_hash,
        temporal_view_manifest_sha256=temporal_manifest_hash,
        fold_event_weight_sha256=event_weight_hash,
        architecture_version=config.model.architecture_version,
        model_mode=config.model.model_mode,
        backbone_name=config.model.backbone_name,
        pretrained_weight_enum=pretrained,
        resolution=config.model.image_size,
        visual_freeze_contract_version=str(freeze_schedule["contract_version"]),
        visual_freeze_policy=str(freeze_schedule["policy"]),
        visual_frozen_warmup_epochs=int(freeze_schedule["frozen_warmup_epochs"]),
        visual_layer4_only_epochs=int(freeze_schedule["layer4_only_epochs"]),
        visual_backbone_lr_multiplier=float(
            freeze_schedule["backbone_lr_multiplier"]
        ),
        early_stopping_contract_version=(
            config.optimization.early_stopping_contract_version
        ),
        early_stopping_metric=config.optimization.early_stopping_metric,
        early_stopping_tiebreaker=(
            config.optimization.early_stopping_tiebreaker
        ),
        early_stopping_tie_tolerance=float(
            config.optimization.early_stopping_tie_tolerance
        ),
        early_stopping_min_supported_classes=int(
            config.optimization.early_stopping_min_supported_classes
        ),
        temporal_view=config.model.temporal_view,
        temporal_encoder_name=config.model.temporal_encoder_name,
        modalities=modalities,
        loss_name=f"cross_entropy+{config.loss.sample_weight_policy}",
        sampler_policy=config.loss.sampler_policy,
        optimizer_name=config.optimization.optimizer,
        precision=config.optimization.precision,
        augmentation_policy=config.dataset.augmentation_policy,
    )
    _validate_identity(identity)
    if identity.execution_profile == "remote_full_oof" and dirty:
        raise ValueError("remote_full_oof requires a clean worktree")
    return identity


def _snapshot_artifact_by_path(
    artifacts: dict[str, Any],
    wanted_path: Path,
) -> dict[str, Any]:
    wanted = wanted_path.resolve()
    matches = [
        item
        for item in artifacts.values()
        if item.get("path")
        and Path(str(item["path"])).resolve() == wanted
    ]
    if len(matches) != 1:
        raise ValueError(
            f"snapshot artifact path match count={len(matches)} path={wanted}"
        )
    return matches[0]


def _required_hash(item: dict[str, Any], name: str) -> str:
    value = str(item.get("sha256", ""))
    if not is_sha256(value):
        raise ValueError(f"{name} lacks a valid sha256")
    return value


def _enabled_modalities(
    config: ClassificationV2TrainingConfig,
) -> tuple[str, ...]:
    flags = {
        "actor_rgb": config.model.enable_image,
        "spatial": config.model.enable_spatial,
        "interaction_numeric": config.model.enable_interaction_context,
        "partner_visual": config.model.enable_visual_context,
        "auxiliary_heads": config.model.enable_multitask,
    }
    modalities = tuple(name for name, enabled in flags.items() if enabled)
    if not modalities:
        raise ValueError("run identity has no enabled modalities")
    return modalities


def _validate_identity(identity: RunIdentity) -> None:
    if identity.identity_schema_version != RUN_IDENTITY_SCHEMA_VERSION:
        raise ValueError(
            "run identity schema mismatch="
            f"{identity.identity_schema_version}"
        )
    if identity.visual_freeze_contract_version != VISUAL_FREEZE_CONTRACT_VERSION:
        raise ValueError(
            "run identity visual-freeze contract mismatch="
            f"{identity.visual_freeze_contract_version}"
        )
    if (
        identity.early_stopping_contract_version
        != VALIDATION_SELECTION_CONTRACT_VERSION
        or identity.early_stopping_metric != VALIDATION_PRIMARY_METRIC
        or identity.early_stopping_tiebreaker != VALIDATION_TIEBREAKER
    ):
        raise ValueError("run identity validation-selection contract mismatch")
    if (
        identity.early_stopping_tie_tolerance < 0.0
        or identity.early_stopping_min_supported_classes <= 0
    ):
        raise ValueError("run identity validation-selection values are invalid")
    if identity.execution_profile not in EXECUTION_PROFILES:
        raise ValueError(
            f"unsupported execution profile={identity.execution_profile}"
        )
    for name in IDENTITY_HASH_FIELDS:
        if not is_sha256(str(getattr(identity, name))):
            raise ValueError(f"run identity has invalid {name}")
    for name in ["run_id", "fold_id", "experiment_name"]:
        value = str(getattr(identity, name))
        if not value or _safe_name(value) != value or value in {".", ".."}:
            raise ValueError(f"run identity has unsafe {name}={value!r}")
        if len(value) > 160:
            raise ValueError(f"run identity has overlong {name}")
    if not identity.code_sha or not identity.modalities or not identity.model_mode:
        raise ValueError("run identity has blank code or modalities")
    if not all(
        [
            identity.optimizer_name,
            identity.precision,
            identity.augmentation_policy,
        ]
    ):
        raise ValueError("run identity has blank execution policy fields")
    if identity.resolution <= 0:
        raise ValueError("run identity resolution must be positive")


def _generated_run_id(
    *,
    experiment_name: str,
    execution_profile: str,
    fold_id: str,
    seed: int,
    config_sha256: str,
    code_sha: str,
    worktree_state_sha256: str,
    dataset_snapshot_sha256: str,
) -> str:
    parts = [
        _safe_name(experiment_name),
        _safe_name(execution_profile),
        _safe_name(fold_id),
        f"seed{seed}",
        config_sha256[:12],
        code_sha[:10],
        worktree_state_sha256[:10],
        dataset_snapshot_sha256[:10],
    ]
    return "__".join(parts)


def _safe_name(value: str) -> str:
    stripped = value.strip()
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in stripped
    )
    return safe.strip("_")


def _git_state() -> tuple[str, bool, str]:
    """Return commit, global dirty flag, and scoped source-state digest."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        global_status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
        ).stdout
        scoped_status = subprocess.run(
            ["git", "status", "--short", "--", *CODE_SCOPE_PATHS],
            check=True,
            capture_output=True,
        ).stdout
        scoped_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", *CODE_SCOPE_PATHS],
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *CODE_SCOPE_PATHS,
            ],
            check=True,
            capture_output=True,
        ).stdout
        digest = hashlib.sha256()
        for payload in [commit.encode("ascii"), scoped_status, scoped_diff]:
            digest.update(payload)
        for raw_path in sorted(item for item in untracked.split(b"\0") if item):
            relative = Path(os.fsdecode(raw_path))
            digest.update(raw_path)
            if relative.is_file():
                digest.update(file_sha256(relative).encode("ascii"))
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot capture git state: {exc}") from exc
    return commit, bool(global_status.strip()), digest.hexdigest()


__all__ = [
    "RUN_IDENTITY_SCHEMA_VERSION",
    "RunIdentity",
    "build_run_identity",
]
