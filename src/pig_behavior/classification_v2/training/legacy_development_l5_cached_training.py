"""Crash-bounded short training over immutable legacy L5 frame features."""

from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5 import (
    LINEAGE_SCOPE,
    LegacyL5Config,
    git_state,
    load_legacy_l5_config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    DECLARED_LOCAL_GPU_VRAM_GIB,
    GPU_ALLOCATOR_FRACTION_CEILING,
    GPU_ALLOCATOR_LIMIT_BYTES,
    VALIDATED_LOCAL_GPU_VRAM_BYTES,
    LegacyL5CachedFeatureClassifier,
    LegacyL5CachedFeatureView,
    build_legacy_l5_cached_feature_view,
)
from pig_behavior.classification_v2.training.legacy_development_l5_feature_cache import (
    FEATURE_DIM,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CACHED_TRAINING_CONFIG_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_config.v1"
)
CACHED_TRAINING_SELECTION_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_selection.v1"
)
CACHED_TRAINING_METRICS_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_metrics.v1"
)
CACHED_TRAINING_RUN_MANIFEST_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_run_manifest.v1"
)
CACHED_TRAINING_RUN_RESULT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_run_result.v1"
)
CACHED_TRAINING_ENVIRONMENT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_environment.v1"
)
CACHED_TRAINING_ARTIFACT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_artifacts.v1"
)
CACHED_TRAINING_CHECKPOINT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_checkpoints.v1"
)
CACHED_TRAINING_PREDICTION_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_predictions.v1"
)
CACHED_TRAINING_REGISTRY_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_registry.v1"
)
CACHED_TRAINING_REPEAT_GATE_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_training_repeat_gate.v1"
)
EXPECTED_TEMPORAL_VIEW = "legacy_t16_centered_matched_observed_time"
MODEL_VISIBLE_ROLES = ("train", "validation")
SELECTION_FIELDS = (
    "selection_order",
    "position",
    "l5_role",
    "window_id",
    "temporal_unit_key",
    "recording_group_id",
    "video_key",
    "source_type",
    "dataset_id",
    "behavior_label",
    "target_index",
    "sample_weight",
    "selection_score",
    "lineage_scope",
    "human_review_complete",
)
REGISTRY_FIELDS = (
    "registry_schema_version",
    "run_id",
    "experiment_name",
    "execution_mode",
    "status",
    "failure_reason",
    "code_sha",
    "dirty_worktree",
    "config_hash",
    "dataset_snapshot_hash",
    "cache_hash",
    "fold_manifest_hash",
    "feature_whitelist_hash",
    "control_id",
    "temporal_view_name",
    "seed",
    "train_native_units",
    "validation_native_units",
    "optimizer_steps",
    "best_epoch",
    "validation_macro_f1",
    "validation_accuracy",
    "validation_nll",
    "outer_predictions_created",
    "source_media_reads",
    "peak_vram_bytes",
    "runtime_seconds",
    "manifest_path",
    "manifest_sha256",
    "completed_at_utc",
)
MAXIMUM_LOADED_BATCH_BYTES = 2_103_552
EXPECTED_CACHED_CLASSIFIER_PARAMETERS = 68_234
PROBABILITY_FIELDS = tuple(
    f"prob_{behavior.replace('-', '_')}" for behavior in VALID_BEHAVIORS
)
PREDICTION_FIELDS = (
    "prediction_order",
    "window_id",
    "temporal_unit_key",
    "recording_group_id",
    "video_key",
    "source_type",
    "dataset_id",
    "behavior_label",
    "target_index",
    "predicted_index",
    "predicted_label",
    *PROBABILITY_FIELDS,
    "lineage_scope",
    "human_review_complete",
)
EPOCH_METRIC_FIELDS = (
    "epoch",
    "optimizer_steps_cumulative",
    "train_native_units",
    "train_loss",
    "validation_native_units",
    "validation_macro_f1_global_10_class",
    "validation_macro_f1_supported_classes",
    "validation_accuracy",
    "validation_nll",
    "parameter_sha256",
    "prediction_sha256",
    "selected_checkpoint",
)
_RUN_EXECUTED_IN_PROCESS = False


@dataclass(frozen=True, slots=True)
class LegacyL5CachedTrainingConfig:
    """One immutable cached-feature short-training semantic contract."""

    path: Path
    payload: dict[str, Any]
    repo_root: Path

    @property
    def sha256(self) -> str:
        return file_sha256(self.path)

    @property
    def base_config_path(self) -> Path:
        return self.repo_root / str(self.payload["base_config"]["path"])

    @property
    def consumer_run_root(self) -> Path:
        base = load_legacy_l5_config(self.base_config_path)
        relative = self.payload["consumer_parent"][
            "consumer_run_relative_path"
        ]
        return base.primary_root / str(relative)

    @property
    def feature_run_root(self) -> Path:
        base = load_legacy_l5_config(self.base_config_path)
        relative = self.payload["consumer_parent"][
            "feature_run_relative_path"
        ]
        return base.primary_root / str(relative)

    @property
    def output_root(self) -> Path:
        base = load_legacy_l5_config(self.base_config_path)
        relative = self.payload["output"]["run_root_relative_path"]
        return base.primary_root / str(relative)


@dataclass(frozen=True, slots=True)
class LegacyL5CachedShortSelection:
    """Deterministic train subset plus complete development validation role."""

    manifest: pd.DataFrame
    train_positions: np.ndarray
    validation_positions: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LegacyL5CachedTrainingOutcome:
    """Deterministic core result with every tensor detached from the device."""

    epoch_metrics: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    per_class_metrics: pd.DataFrame
    confusion: pd.DataFrame
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    best_epoch: int
    optimizer_steps: int
    parameter_sha256: str
    prediction_sha256: str
    epoch_metrics_sha256: str
    maximum_loaded_batch_bytes: int


def load_legacy_l5_cached_training_config(
    path: Path,
) -> LegacyL5CachedTrainingConfig:
    """Load one exact short-training config and reject semantic drift."""

    resolved_path = path.resolve()
    payload = _read_json(resolved_path)
    required = {
        "schema_version",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "development_metrics_authorized",
        "execution_mode",
        "experiment_name",
        "experiment_contract",
        "base_config",
        "consumer_parent",
        "data",
        "model",
        "optimization",
        "execution_guard",
        "repeat_gate",
        "output",
    }
    _require_exact_keys(payload, required, name="cached training config")
    if payload["schema_version"] != CACHED_TRAINING_CONFIG_SCHEMA_VERSION:
        raise ValueError("cached training config schema mismatch")
    if payload["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("cached training lineage scope mismatch")
    false_claims = (
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
    )
    if any(payload[name] is not False for name in false_claims):
        raise ValueError("cached training config exceeds its claim boundary")
    if payload["development_metrics_authorized"] is not True:
        raise ValueError("cached training development metrics are not explicit")
    if payload["execution_mode"] != "local_smoke":
        raise ValueError("cached short training must use local_smoke mode")
    _validate_experiment_contract(payload["experiment_contract"])
    _validate_base_config(payload["base_config"])
    _validate_consumer_parent(payload["consumer_parent"])
    _validate_data_contract(payload["data"])
    _validate_model_contract(payload["model"])
    _validate_optimization_contract(payload["optimization"])
    _validate_execution_guard(payload["execution_guard"])
    _validate_repeat_contract(payload["repeat_gate"])
    _validate_output_contract(payload["output"])
    repo_root = resolved_path.parents[2]
    config = LegacyL5CachedTrainingConfig(
        path=resolved_path,
        payload=payload,
        repo_root=repo_root,
    )
    if not config.base_config_path.is_file():
        raise FileNotFoundError(
            f"cached training base config missing: {config.base_config_path}"
        )
    if file_sha256(config.base_config_path) != payload["base_config"]["sha256"]:
        raise ValueError("cached training base config hash drift")
    return config


def load_legacy_l5_cached_training_view(
    config: LegacyL5CachedTrainingConfig,
) -> tuple[LegacyL5Config, LegacyL5CachedFeatureView, dict[str, Any]]:
    """Validate every consumer parent before rebuilding the model-visible view."""

    base = load_legacy_l5_config(config.base_config_path)
    parent = config.payload["consumer_parent"]
    consumer_root = config.consumer_run_root
    feature_root = config.feature_run_root
    paths = {
        "consumer_run_manifest": consumer_root / "run_manifest.json",
        "consumer_cached_data_audit": consumer_root / "cached_data_audit.json",
        "consumer_environment": consumer_root / "environment.json",
        "consumer_checkpoint_manifest": consumer_root / "checkpoint_manifest.json",
        "consumer_prediction_manifest": consumer_root / "prediction_manifest.json",
        "feature_run_manifest": feature_root / "run_manifest.json",
        "feature_result": feature_root / "run_result.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"cached training parent missing={name}:{path}")
    manifest = _read_json(paths["consumer_run_manifest"])
    audit = _read_json(paths["consumer_cached_data_audit"])
    environment = _read_json(paths["consumer_environment"])
    checkpoints = _read_json(paths["consumer_checkpoint_manifest"])
    predictions = _read_json(paths["consumer_prediction_manifest"])
    feature_manifest = _read_json(paths["feature_run_manifest"])
    feature_result = _read_json(paths["feature_result"])
    exact_hashes = {
        "run_manifest_sha256": paths["consumer_run_manifest"],
        "cached_data_audit_sha256": paths["consumer_cached_data_audit"],
        "feature_manifest_sha256": paths["feature_run_manifest"],
        "feature_result_sha256": paths["feature_result"],
    }
    for field, path in exact_hashes.items():
        if file_sha256(path) != parent[field]:
            raise ValueError(f"cached training parent hash drift={field}")
    required_manifest = {
        "run_id": parent["run_id"],
        "code_sha": parent["code_sha"],
        "status": "completed",
        "config_hash": base.sha256,
        "cache_hash": parent["feature_tensor_sha256"],
        "dataset_snapshot_hash": parent["dataset_snapshot_sha256"],
        "fold_manifest_hash": parent["fold_manifest_sha256"],
        "feature_whitelist_sha256": parent["feature_whitelist_sha256"],
        "control_id": config.payload["data"]["control_id"],
        "temporal_view_name": config.payload["data"]["temporal_view_name"],
        "sequence_length": config.payload["data"]["sequence_length"],
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "source_media_reads": 0,
        "optimizer_steps": 0,
        "peak_vram_bytes": 0,
    }
    _validate_exact_values(
        manifest,
        required_manifest,
        name="cached consumer manifest",
    )
    required_audit = {
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHED_DATA",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "control_id": config.payload["data"]["control_id"],
        "temporal_view_name": config.payload["data"]["temporal_view_name"],
        "sequence_length": config.payload["data"]["sequence_length"],
        "feature_dim": FEATURE_DIM,
        "source_media_loads": 0,
        "valid": True,
    }
    _validate_exact_values(audit, required_audit, name="cached consumer audit")
    bounded = audit.get("bounded_batch_audit") or {}
    if (
        bounded.get("valid") is not True
        or bounded.get("outer_holdout_rows_loaded") != 0
        or bounded.get("cuda_runtime_initialized_before") is not False
        or bounded.get("cuda_runtime_initialized_after") is not False
    ):
        raise ValueError("cached training consumer bounded audit is invalid")
    memory = audit.get("memory_safety") or {}
    required_memory = {
        "declared_local_gpu_vram_gib": DECLARED_LOCAL_GPU_VRAM_GIB,
        "validated_local_gpu_vram_bytes": VALIDATED_LOCAL_GPU_VRAM_BYTES,
        "gpu_allocator_fraction_ceiling": GPU_ALLOCATOR_FRACTION_CEILING,
        "gpu_allocator_limit_bytes": GPU_ALLOCATOR_LIMIT_BYTES,
        "mmap_close_after_each_loaded_batch": True,
        "dataloader_num_workers": 0,
        "pin_memory": False,
        "oom_retry_allowed": False,
    }
    _validate_exact_values(memory, required_memory, name="cached memory audit")
    if environment.get("gpu_execution_performed") is not False:
        raise ValueError("cached training parent performed GPU execution")
    if checkpoints.get("checkpoints") != []:
        raise ValueError("cached training consumer contains checkpoints")
    if predictions.get("predictions") != []:
        raise ValueError("cached training consumer contains predictions")
    if feature_manifest.get("status") != "completed":
        raise ValueError("cached training feature parent is not completed")
    if feature_manifest.get("run_result_sha256") != parent[
        "feature_result_sha256"
    ]:
        raise ValueError("cached training feature result link drift")
    if feature_result.get("feature_tensor_sha256") != parent[
        "feature_tensor_sha256"
    ]:
        raise ValueError("cached training feature tensor hash drift")
    if feature_result.get("feature_index_sha256") != parent[
        "feature_index_sha256"
    ]:
        raise ValueError("cached training feature index hash drift")
    view = build_legacy_l5_cached_feature_view(
        base,
        feature_result_path=paths["feature_result"],
        temporal_view_name=config.payload["data"]["temporal_view_name"],
    )
    return base, view, {
        "paths": {
            name: str(path.resolve()) for name, path in paths.items()
        },
        "hashes": {
            name: file_sha256(path) for name, path in paths.items()
        },
        "consumer_manifest": manifest,
        "consumer_audit": audit,
        "feature_manifest": feature_manifest,
        "feature_result": feature_result,
        "errors": [],
        "valid": True,
    }


def build_legacy_l5_cached_short_selection(
    view: LegacyL5CachedFeatureView,
    config: LegacyL5CachedTrainingConfig,
) -> LegacyL5CachedShortSelection:
    """Select a deterministic class-balanced train smoke and all validation."""

    data = config.payload["data"]
    if view.control_id != data["control_id"]:
        raise ValueError("cached short selection control drift")
    if view.temporal_view_name != data["temporal_view_name"]:
        raise ValueError("cached short selection temporal view drift")
    if view.sequence_length != int(data["sequence_length"]):
        raise ValueError("cached short selection sequence length drift")
    windows = view.windows.copy().reset_index(drop=True)
    windows["position"] = np.arange(len(windows), dtype=np.int64)
    if not windows["l5_role"].isin(MODEL_VISIBLE_ROLES).all():
        raise ValueError("cached short selection contains a forbidden role")
    if windows["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("centered cached view is not one window per native unit")
    train = windows.loc[windows["l5_role"].eq("train")].copy()
    validation = windows.loc[windows["l5_role"].eq("validation")].copy()
    salt = str(data["train_selection_salt"])
    train["selection_score"] = train["temporal_unit_key"].map(
        lambda value: _selection_score(salt, str(value))
    )
    train = train.sort_values(
        ["behavior_label", "selection_score", "temporal_unit_key"],
        kind="mergesort",
    )
    per_class = int(data["train_native_units_per_class"])
    selected_train = train.groupby(
        "behavior_label",
        sort=False,
        group_keys=False,
    ).head(per_class)
    selected_train = selected_train.sort_values(
        ["behavior_label", "selection_score", "temporal_unit_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    train_counts = selected_train["behavior_label"].value_counts().to_dict()
    expected_train_counts = {label: per_class for label in VALID_BEHAVIORS}
    if train_counts != expected_train_counts:
        raise ValueError(
            f"cached short train class support={train_counts}!="
            f"{expected_train_counts}"
        )
    validation["selection_score"] = validation["temporal_unit_key"].map(
        lambda value: _selection_score("all_validation", str(value))
    )
    validation = validation.sort_values(
        ["temporal_unit_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    if set(validation["behavior_label"].astype(str)) != set(VALID_BEHAVIORS):
        raise ValueError("cached short validation lacks global class support")
    if len(selected_train) != int(data["expected_train_native_units"]):
        raise ValueError("cached short train native-unit count drift")
    if len(validation) != int(data["expected_validation_native_units"]):
        raise ValueError("cached short validation native-unit count drift")
    overlap = _selection_group_overlap(selected_train, validation)
    if overlap["errors"]:
        raise ValueError(f"cached short selection group overlap={overlap['errors']}")
    selected_train["selection_order"] = np.arange(
        len(selected_train),
        dtype=np.int64,
    )
    validation["selection_order"] = np.arange(
        len(selected_train),
        len(selected_train) + len(validation),
        dtype=np.int64,
    )
    manifest = pd.concat(
        [selected_train, validation],
        ignore_index=True,
    )
    manifest["target_index"] = view.targets[
        manifest["position"].to_numpy(dtype=np.int64)
    ]
    manifest["sample_weight"] = view.sample_weights[
        manifest["position"].to_numpy(dtype=np.int64)
    ]
    manifest["lineage_scope"] = LINEAGE_SCOPE
    manifest["human_review_complete"] = False
    manifest = manifest[list(SELECTION_FIELDS)].copy()
    if not np.allclose(manifest["sample_weight"].to_numpy(), 1.0):
        raise ValueError("centered cached short selection has non-unit weights")
    train_positions = selected_train["position"].to_numpy(dtype=np.int64)
    validation_positions = validation["position"].to_numpy(dtype=np.int64)
    audit = {
        "schema_version": CACHED_TRAINING_SELECTION_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHED_SHORT_SELECTION",
        "lineage_scope": LINEAGE_SCOPE,
        "selection_policy": data["train_selection_policy"],
        "selection_salt": salt,
        "train_native_units_per_class": per_class,
        "train_native_units": int(len(train_positions)),
        "validation_native_units": int(len(validation_positions)),
        "train_class_counts": train_counts,
        "validation_class_counts": {
            label: int(validation["behavior_label"].eq(label).sum())
            for label in VALID_BEHAVIORS
        },
        "train_unit_sha256": _ordered_sha256(
            selected_train["temporal_unit_key"]
        ),
        "validation_unit_sha256": _ordered_sha256(
            validation["temporal_unit_key"]
        ),
        "selection_content_sha256": _dataframe_sha256(manifest),
        "group_overlap": overlap,
        "outer_holdout_rows": 0,
        "source_media_reads": 0,
        "errors": [],
        "valid": True,
    }
    return LegacyL5CachedShortSelection(
        manifest=manifest,
        train_positions=train_positions,
        validation_positions=validation_positions,
        audit=audit,
    )


def preflight_legacy_l5_cached_short_training(
    config: LegacyL5CachedTrainingConfig,
) -> dict[str, Any]:
    """Run the exact real-parent gate without initializing CUDA or writing."""

    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    base: LegacyL5Config | None = None
    view: LegacyL5CachedFeatureView | None = None
    parent: dict[str, Any] | None = None
    selection: LegacyL5CachedShortSelection | None = None
    loaded_bytes = 0
    output_shape: list[int] = []
    parameter_count = 0
    try:
        base, view, parent = load_legacy_l5_cached_training_view(config)
        selection = build_legacy_l5_cached_short_selection(view, config)
        _validate_training_selection(view, selection, config)
        probe_positions = selection.validation_positions[
            : int(config.payload["optimization"]["evaluation_batch_size"])
        ]
        batch, loaded_bytes = _load_selected_batch(
            view,
            probe_positions,
            maximum_batch_bytes=int(
                config.payload["optimization"]["maximum_loaded_batch_bytes"]
            ),
        )
        _seed_all(
            int(config.payload["optimization"]["seed"]),
            seed_cuda=False,
        )
        model = _build_cached_classifier(config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model.eval()
        with torch.inference_mode():
            logits = model(
                torch.from_numpy(batch["features"]),
                torch.from_numpy(batch["observed_mask"]).float(),
                time_delta=torch.from_numpy(batch["time_delta"]).float(),
            )
        output_shape = list(logits.shape)
        if not torch.isfinite(logits).all():
            errors.append("cpu_preflight_nonfinite_logits")
        if output_shape != [len(probe_positions), len(VALID_BEHAVIORS)]:
            errors.append(f"cpu_preflight_logit_shape={output_shape}")
        if parameter_count != EXPECTED_CACHED_CLASSIFIER_PARAMETERS:
            errors.append(f"model_parameter_count={parameter_count}")
        del batch, logits, model
        gc.collect()
    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = _git_launch_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before:
        errors.append("cuda_runtime_was_initialized_before_cpu_preflight")
    if cuda_after:
        errors.append("cuda_runtime_initialized_by_cpu_preflight")
    valid = not errors
    return {
        "schema_version": (
            "classification_v2.legacy_development_l5."
            "cached_training_preflight.v1"
        ),
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_CACHED_TRAINING_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_CACHED_TRAINING_PREFLIGHT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "base_config_sha256": base.sha256 if base is not None else None,
        "consumer_parent_valid": parent is not None,
        "selection_content_sha256": (
            selection.audit["selection_content_sha256"]
            if selection is not None
            else None
        ),
        "train_native_units": (
            len(selection.train_positions) if selection is not None else 0
        ),
        "validation_native_units": (
            len(selection.validation_positions) if selection is not None else 0
        ),
        "outer_holdout_rows_loaded": 0,
        "maximum_loaded_batch_bytes": loaded_bytes,
        "maximum_loaded_batch_bytes_allowed": int(
            config.payload["optimization"]["maximum_loaded_batch_bytes"]
        ),
        "model_parameter_count": parameter_count,
        "cpu_forward_output_shape": output_shape,
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "git_guard": git_guard,
        "gpu_launch_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def run_legacy_l5_cached_short_training(
    config: LegacyL5CachedTrainingConfig,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Execute one and only one fresh-process CUDA short run."""

    global _RUN_EXECUTED_IN_PROCESS
    if _RUN_EXECUTED_IN_PROCESS:
        raise RuntimeError("cached short training permits one run per process")
    if not _safe_run_id(run_id):
        raise ValueError(f"unsafe cached training run ID: {run_id!r}")
    preflight = preflight_legacy_l5_cached_short_training(config)
    if not preflight["gpu_launch_authorized"]:
        raise RuntimeError(
            f"cached training preflight failed={preflight['errors']}"
        )
    _RUN_EXECUTED_IN_PROCESS = True
    _, view, parent = load_legacy_l5_cached_training_view(config)
    selection = build_legacy_l5_cached_short_selection(view, config)
    run_root = config.output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    paths = _cached_training_run_paths(run_root)
    git_guard = _git_launch_guard(config)
    started_at = _utc_now()
    started = time.perf_counter()
    planned = _planned_training_manifest(
        config,
        selection=selection,
        parent=parent,
        preflight=preflight,
        git_guard=git_guard,
        run_id=run_id,
        started_at=started_at,
    )
    _write_json_exclusive(paths["run_manifest"], planned)
    planned_sha256 = file_sha256(paths["run_manifest"])
    _write_json_exclusive(
        paths["environment"],
        _training_environment_payload(planned),
    )
    _write_json_exclusive(paths["preflight"], preflight)
    _write_dataframe_exclusive(paths["selection_manifest"], selection.manifest)
    _write_json_exclusive(paths["selection_audit"], selection.audit)
    outcome: LegacyL5CachedTrainingOutcome | None = None
    execution: dict[str, Any]
    failure: dict[str, Any] | None = None
    try:
        outcome, execution = _execute_cuda_short_training(
            view,
            selection,
            config,
        )
    except Exception as error:
        failure = {
            "schema_version": (
                "classification_v2.legacy_development_l5."
                "cached_training_failure.v1"
            ),
            "run_id": run_id,
            "process_id": os.getpid(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "oom_retry_performed": False,
            "captured_at_utc": _utc_now(),
        }
        _write_json_exclusive(paths["unexpected_failure"], failure)
        execution = _failed_execution_payload(config, error)
    runtime_seconds = time.perf_counter() - started
    result = _finalize_cached_training_run(
        paths,
        config=config,
        planned=planned,
        planned_sha256=planned_sha256,
        selection=selection,
        outcome=outcome,
        execution=execution,
        failure=failure,
        runtime_seconds=runtime_seconds,
    )
    return result


def _execute_cuda_short_training(
    view: LegacyL5CachedFeatureView,
    selection: LegacyL5CachedShortSelection,
    config: LegacyL5CachedTrainingConfig,
) -> tuple[LegacyL5CachedTrainingOutcome | None, dict[str, Any]]:
    optimization = _object(config.payload["optimization"], "optimization")
    errors: list[str] = []
    oom = False
    oom_message: str | None = None
    device_name = str(optimization["device"])
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("cached short production run requires CUDA")
    if torch.cuda.is_initialized():
        raise RuntimeError("cached short run did not start in a fresh process")
    if not torch.cuda.is_available():
        raise RuntimeError("cached short run requested unavailable CUDA")
    device_index = int(device.index) if device.index is not None else 0
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    actual_total = int(properties.total_memory)
    free_before, mem_info_total = (
        int(value) for value in torch.cuda.mem_get_info(device)
    )
    expected_total = int(optimization["validated_local_gpu_vram_bytes"])
    allocator_limit = int(optimization["allocator_limit_bytes"])
    allocator_fraction = allocator_limit / actual_total
    if actual_total != expected_total or mem_info_total != expected_total:
        errors.append(
            f"gpu_total_vram={actual_total},{mem_info_total}!={expected_total}"
        )
    if free_before < allocator_limit:
        errors.append(f"free_vram={free_before}<{allocator_limit}")
    if allocator_fraction > float(optimization["maximum_peak_vram_fraction"]):
        errors.append("allocator_fraction_exceeds_configured_ceiling")
    allocated_before = int(torch.cuda.memory_allocated(device))
    reserved_before = int(torch.cuda.memory_reserved(device))
    if allocated_before != 0 or reserved_before != 0:
        errors.append("cuda_memory_not_empty_before_model_creation")
    outcome: LegacyL5CachedTrainingOutcome | None = None
    peak_allocated = 0
    peak_reserved = 0
    cleanup_errors: list[str] = []
    if not errors:
        torch.cuda.set_per_process_memory_fraction(allocator_fraction, device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            outcome = train_legacy_l5_cached_short_core(
                view,
                selection,
                config,
                device=device,
            )
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError as error:
            oom = True
            oom_message = str(error)
            errors.append("cuda_out_of_memory_no_retry")
        finally:
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            gc.collect()
            try:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            except RuntimeError as error:
                cleanup_errors.append(f"cuda_cleanup_error={error}")
    post_allocated = int(torch.cuda.memory_allocated(device))
    post_reserved = int(torch.cuda.memory_reserved(device))
    if peak_allocated > allocator_limit or peak_reserved > allocator_limit:
        errors.append("cuda_peak_exceeds_allocator_limit")
    if post_allocated != 0 or post_reserved != 0:
        errors.append("cuda_memory_not_released_after_training")
    if outcome is not None and outcome.maximum_loaded_batch_bytes > int(
        optimization["maximum_loaded_batch_bytes"]
    ):
        errors.append("cpu_loaded_batch_exceeds_frozen_limit")
    errors.extend(cleanup_errors)
    valid = outcome is not None and not errors
    return outcome, {
        "device": str(device),
        "device_name": str(properties.name),
        "process_id": os.getpid(),
        "actual_total_vram_bytes": actual_total,
        "mem_info_total_vram_bytes": mem_info_total,
        "free_vram_before_bytes": free_before,
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "allocator_fraction": allocator_fraction,
        "allocator_limit_bytes": allocator_limit,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "post_cleanup_allocated_bytes": post_allocated,
        "post_cleanup_reserved_bytes": post_reserved,
        "precision": "float32",
        "autocast_enabled": False,
        "oom": oom,
        "oom_message": oom_message,
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": errors,
        "valid": valid,
    }


def _failed_execution_payload(
    config: LegacyL5CachedTrainingConfig,
    error: Exception,
) -> dict[str, Any]:
    optimization = config.payload["optimization"]
    return {
        "device": optimization["device"],
        "device_name": "execution_failed_before_complete_report",
        "process_id": os.getpid(),
        "actual_total_vram_bytes": None,
        "mem_info_total_vram_bytes": None,
        "free_vram_before_bytes": None,
        "allocated_before_bytes": None,
        "reserved_before_bytes": None,
        "allocator_fraction": optimization["maximum_peak_vram_fraction"],
        "allocator_limit_bytes": optimization["allocator_limit_bytes"],
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "post_cleanup_allocated_bytes": None,
        "post_cleanup_reserved_bytes": None,
        "precision": "float32",
        "autocast_enabled": False,
        "oom": isinstance(error, torch.cuda.OutOfMemoryError),
        "oom_message": str(error),
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [f"{type(error).__name__}: {error}"],
        "valid": False,
    }


def audit_legacy_l5_cached_training_repeat_gate(
    config: LegacyL5CachedTrainingConfig,
    *,
    primary_result_path: Path,
    repeat_result_path: Path,
) -> dict[str, Any]:
    """Require two isolated, non-overlapping and bit-identical short runs."""

    cuda_before = torch.cuda.is_initialized()
    reports: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, path in (
        ("primary", primary_result_path),
        ("repeat", repeat_result_path),
    ):
        try:
            report = _audit_cached_training_result_packet(config, path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report = {
                "result_path": str(path.resolve()),
                "errors": [f"{type(error).__name__}: {error}"],
                "valid": False,
            }
        reports[name] = report
        errors.extend(f"{name}:{value}" for value in report["errors"])
    primary = reports["primary"].get("result") or {}
    repeat = reports["repeat"].get("result") or {}
    equality_fields = (
        "code_sha",
        "config_sha256",
        "semantic_identity_sha256",
        "selection_content_sha256",
        "train_native_units",
        "validation_native_units",
        "optimizer_steps",
        "best_epoch",
        "parameter_sha256",
        "prediction_content_sha256",
        "epoch_metrics_content_sha256",
        "maximum_loaded_batch_bytes",
        "validation_metrics",
    )
    equality = {
        field: primary.get(field) == repeat.get(field)
        for field in equality_fields
    }
    errors.extend(
        f"repeat_mismatch={field}"
        for field, matches in equality.items()
        if not matches
    )
    if primary.get("run_id") == repeat.get("run_id"):
        errors.append("repeat_run_ids_are_not_distinct")
    if primary.get("process_id") == repeat.get("process_id"):
        errors.append("repeat_process_ids_are_not_distinct")
    intervals = _non_overlapping_intervals(primary, repeat)
    errors.extend(intervals["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("repeat_gate_initialized_cuda")
    valid = not errors
    return {
        "schema_version": CACHED_TRAINING_REPEAT_GATE_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_CACHED_TRAINING_SHORT_GATE"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_CACHED_TRAINING_SHORT_GATE"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "required_runs": 2,
        "reports": reports,
        "equality": equality,
        "non_overlapping_execution": intervals,
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "exact_full_v0_t16_centered_expansion_authorized": valid,
        "other_visual_or_temporal_controls_authorized": False,
        "errors": errors,
        "valid": valid,
    }


def write_legacy_l5_cached_training_repeat_gate(
    config: LegacyL5CachedTrainingConfig,
    *,
    primary_result_path: Path,
    repeat_result_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Persist only a passing immutable two-process short gate."""

    audit = audit_legacy_l5_cached_training_repeat_gate(
        config,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
    )
    if not audit["valid"]:
        raise ValueError(f"cached training repeat gate failed={audit['errors']}")
    output = config.output_root / str(
        config.payload["output"]["short_gate_filename"]
    )
    _write_json_exclusive(output, audit)
    return output, audit


def _audit_cached_training_result_packet(
    config: LegacyL5CachedTrainingConfig,
    path: Path,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.name != "run_result.json":
        raise FileNotFoundError(f"cached training result missing: {resolved}")
    result = _read_json(resolved)
    root = resolved.parent
    paths = _cached_training_run_paths(root)
    required_files = (
        "run_manifest",
        "selection_manifest",
        "selection_audit",
        "epoch_metrics",
        "validation_predictions",
        "validation_metrics",
        "checkpoint",
        "checkpoint_manifest",
        "prediction_manifest",
        "artifact_manifest",
        "registry_entry",
        "runs_registry",
    )
    missing = [name for name in required_files if not paths[name].is_file()]
    errors = [f"missing_artifacts={missing}"] if missing else []
    expected = {
        "schema_version": CACHED_TRAINING_RUN_RESULT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHED_SHORT_TRAINING",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_sha256": config.sha256,
        "train_native_units": config.payload["data"][
            "expected_train_native_units"
        ],
        "validation_native_units": config.payload["data"][
            "expected_validation_native_units"
        ],
        "outer_holdout_rows_loaded": 0,
        "optimizer_steps": config.payload["optimization"][
            "maximum_optimizer_steps"
        ],
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "valid": True,
    }
    errors.extend(
        f"result_mismatch={name}:{result.get(name)!r}!={value!r}"
        for name, value in expected.items()
        if result.get(name) != value
    )
    if missing:
        return {
            "result_path": str(resolved),
            "result_sha256": file_sha256(resolved),
            "result": result,
            "errors": errors,
            "valid": False,
        }
    manifest = _read_json(paths["run_manifest"])
    selection_audit = _read_json(paths["selection_audit"])
    checkpoint_manifest = _read_json(paths["checkpoint_manifest"])
    prediction_manifest = _read_json(paths["prediction_manifest"])
    if manifest.get("status") != "completed":
        errors.append("run_manifest_is_not_completed")
    if manifest.get("run_result_sha256") != file_sha256(resolved):
        errors.append("run_manifest_result_hash_mismatch")
    if selection_audit.get("selection_content_sha256") != result.get(
        "selection_content_sha256"
    ):
        errors.append("selection_audit_hash_mismatch")
    if file_sha256(paths["selection_manifest"]) != result.get(
        "selection_content_sha256"
    ):
        errors.append("selection_manifest_content_hash_mismatch")
    if file_sha256(paths["epoch_metrics"]) != result.get(
        "epoch_metrics_content_sha256"
    ):
        errors.append("epoch_metrics_content_hash_mismatch")
    if file_sha256(paths["validation_predictions"]) != result.get(
        "prediction_content_sha256"
    ):
        errors.append("prediction_content_hash_mismatch")
    checkpoint = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    if _state_dict_sha256(checkpoint["model_state_dict"]) != result.get(
        "parameter_sha256"
    ):
        errors.append("checkpoint_parameter_hash_mismatch")
    if len(checkpoint_manifest.get("checkpoints") or []) != 1:
        errors.append("checkpoint_manifest_count_mismatch")
    if prediction_manifest.get("validation_predictions_created") != int(
        config.payload["data"]["expected_validation_native_units"]
    ):
        errors.append("prediction_manifest_validation_count_mismatch")
    if prediction_manifest.get("outer_holdout_predictions_created") != 0:
        errors.append("prediction_manifest_contains_outer_predictions")
    predictions = pd.read_csv(paths["validation_predictions"])
    if tuple(predictions.columns) != PREDICTION_FIELDS:
        errors.append("validation_prediction_schema_mismatch")
    if len(predictions) != config.payload["data"][
        "expected_validation_native_units"
    ]:
        errors.append("validation_prediction_row_count_mismatch")
    execution_errors = _cached_training_execution_errors(
        result.get("execution") or {},
        config,
    )
    errors.extend(execution_errors)
    return {
        "result_path": str(resolved),
        "result_sha256": file_sha256(resolved),
        "run_manifest_path": str(paths["run_manifest"].resolve()),
        "run_manifest_sha256": file_sha256(paths["run_manifest"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "prediction_file_sha256": file_sha256(
            paths["validation_predictions"]
        ),
        "result": result,
        "errors": errors,
        "valid": not errors,
    }


def _cached_training_execution_errors(
    execution: dict[str, Any],
    config: LegacyL5CachedTrainingConfig,
) -> list[str]:
    optimization = config.payload["optimization"]
    expected = {
        "actual_total_vram_bytes": optimization[
            "validated_local_gpu_vram_bytes"
        ],
        "mem_info_total_vram_bytes": optimization[
            "validated_local_gpu_vram_bytes"
        ],
        "allocator_limit_bytes": optimization["allocator_limit_bytes"],
        "precision": "float32",
        "autocast_enabled": False,
        "oom": False,
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "valid": True,
    }
    errors = [
        f"execution_mismatch={name}:{execution.get(name)!r}!={value!r}"
        for name, value in expected.items()
        if execution.get(name) != value
    ]
    limit = int(optimization["allocator_limit_bytes"])
    if int(execution.get("peak_allocated_bytes", limit + 1)) > limit:
        errors.append("execution_peak_allocated_exceeds_limit")
    if int(execution.get("peak_reserved_bytes", limit + 1)) > limit:
        errors.append("execution_peak_reserved_exceeds_limit")
    return errors


def _non_overlapping_intervals(
    primary: dict[str, Any],
    repeat: dict[str, Any],
) -> dict[str, Any]:
    try:
        primary_start = datetime.fromisoformat(
            str(primary["started_at_utc"])
        )
        primary_end = datetime.fromisoformat(
            str(primary["completed_at_utc"])
        )
        repeat_start = datetime.fromisoformat(str(repeat["started_at_utc"]))
        repeat_end = datetime.fromisoformat(str(repeat["completed_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        return {
            "primary_interval": None,
            "repeat_interval": None,
            "errors": [f"invalid_execution_interval={error}"],
            "valid": False,
        }
    ordered = primary_end <= repeat_start or repeat_end <= primary_start
    errors = [] if ordered else ["training_run_intervals_overlap"]
    return {
        "primary_interval": [primary_start.isoformat(), primary_end.isoformat()],
        "repeat_interval": [repeat_start.isoformat(), repeat_end.isoformat()],
        "errors": errors,
        "valid": not errors,
    }


def compute_legacy_l5_native_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    temporal_unit_keys: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compute strict global ten-class metrics for unique native units."""

    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    keys = temporal_unit_keys.fillna("").astype(str).reset_index(drop=True)
    if probs.ndim != 2 or probs.shape[1] != len(VALID_BEHAVIORS):
        raise ValueError("native metric probabilities must be [N,10]")
    if len(probs) != len(labels) or len(probs) != len(keys):
        raise ValueError("native metric row counts differ")
    if len(probs) == 0 or keys.str.strip().eq("").any():
        raise ValueError("native metric keys are empty")
    if keys.duplicated().any():
        raise ValueError("native metrics contain duplicate temporal units")
    if not np.isfinite(probs).all() or (probs < 0.0).any():
        raise ValueError("native metric probabilities are invalid")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("native metric probabilities do not sum to one")
    if (labels < 0).any() or (labels >= len(VALID_BEHAVIORS)).any():
        raise ValueError("native metric targets are invalid")
    predicted = probs.argmax(axis=1).astype(np.int64)
    confusion = np.zeros(
        (len(VALID_BEHAVIORS), len(VALID_BEHAVIORS)),
        dtype=np.int64,
    )
    np.add.at(confusion, (labels, predicted), 1)
    per_class_records: list[dict[str, Any]] = []
    class_f1: list[float] = []
    supported_f1: list[float] = []
    for index, behavior in enumerate(VALID_BEHAVIORS):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positive / recall_denominator if recall_denominator else 0.0
        )
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
        class_f1.append(float(f1))
        if support:
            supported_f1.append(float(f1))
        per_class_records.append(
            {
                "behavior_label": behavior,
                "class_index": index,
                "support": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    metrics = {
        "schema_version": CACHED_TRAINING_METRICS_SCHEMA_VERSION,
        "native_unit_rows": int(len(labels)),
        "global_class_count": len(VALID_BEHAVIORS),
        "supported_class_count": int(len(supported_f1)),
        "macro_f1_global_10_class": float(np.mean(class_f1)),
        "macro_f1_supported_classes": float(np.mean(supported_f1)),
        "accuracy": float(np.mean(predicted == labels)),
        "nll": float(-np.mean(np.log(clipped))),
        "class_order": list(VALID_BEHAVIORS),
        "aggregation": "one_prediction_per_native_temporal_unit",
        "errors": [],
        "valid": True,
    }
    per_class = pd.DataFrame.from_records(per_class_records)
    confusion_frame = pd.DataFrame(
        confusion,
        index=VALID_BEHAVIORS,
        columns=VALID_BEHAVIORS,
    ).reset_index(names="true_behavior")
    return metrics, per_class, confusion_frame


def train_legacy_l5_cached_short_core(
    view: LegacyL5CachedFeatureView,
    selection: LegacyL5CachedShortSelection,
    config: LegacyL5CachedTrainingConfig,
    *,
    device: torch.device | str,
) -> LegacyL5CachedTrainingOutcome:
    """Train the exact short head while opening one feature mmap per batch."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cached short training requested unavailable CUDA")
    optimization = _object(config.payload["optimization"], "optimization")
    maximum_batch_bytes = int(optimization["maximum_loaded_batch_bytes"])
    _validate_training_selection(view, selection, config)
    seed = int(optimization["seed"])
    _seed_all(seed, seed_cuda=resolved_device.type == "cuda")
    model: LegacyL5CachedFeatureClassifier | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = _build_cached_classifier(config).to(resolved_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
        )
        optimizer_steps = 0
        maximum_observed_batch_bytes = 0
        best_epoch = 0
        best_score: tuple[float, float] | None = None
        best_model_state: dict[str, torch.Tensor] | None = None
        best_optimizer_state: dict[str, Any] | None = None
        best_predictions: pd.DataFrame | None = None
        best_metrics: dict[str, Any] | None = None
        best_per_class: pd.DataFrame | None = None
        best_confusion: pd.DataFrame | None = None
        epoch_records: list[dict[str, Any]] = []
        epochs = int(optimization["epochs"])
        for epoch in range(1, epochs + 1):
            train_positions = selection.train_positions.copy()
            np.random.default_rng(seed + epoch).shuffle(train_positions)
            train_loss_mass = 0.0
            train_weight_mass = 0.0
            model.train()
            for batch_positions in _position_batches(
                train_positions,
                batch_size=int(optimization["batch_size"]),
            ):
                batch, loaded_bytes = _load_selected_batch(
                    view,
                    batch_positions,
                    maximum_batch_bytes=maximum_batch_bytes,
                )
                maximum_observed_batch_bytes = max(
                    maximum_observed_batch_bytes,
                    loaded_bytes,
                )
                loss_value, weight_mass = _cached_training_step(
                    model,
                    optimizer,
                    batch,
                    device=resolved_device,
                    gradient_clip_norm=float(
                        optimization["gradient_clip_norm"]
                    ),
                )
                optimizer_steps += 1
                if optimizer_steps > int(
                    optimization["maximum_optimizer_steps"]
                ):
                    raise RuntimeError("cached short optimizer-step cap exceeded")
                train_loss_mass += loss_value * weight_mass
                train_weight_mass += weight_mass
                del batch
            if train_weight_mass <= 0.0:
                raise RuntimeError("cached short train weight mass is empty")
            evaluation = _evaluate_cached_classifier(
                model,
                view,
                selection.validation_positions,
                batch_size=int(optimization["evaluation_batch_size"]),
                maximum_batch_bytes=maximum_batch_bytes,
                device=resolved_device,
            )
            maximum_observed_batch_bytes = max(
                maximum_observed_batch_bytes,
                int(evaluation["maximum_loaded_batch_bytes"]),
            )
            predictions = _cached_prediction_frame(
                view,
                selection.validation_positions,
                probabilities=evaluation["probabilities"],
                targets=evaluation["targets"],
            )
            metrics, per_class, confusion = compute_legacy_l5_native_metrics(
                evaluation["probabilities"],
                evaluation["targets"],
                predictions["temporal_unit_key"],
            )
            parameter_sha256 = _state_dict_sha256(model.state_dict())
            prediction_sha256 = _dataframe_sha256(predictions)
            score = (
                float(metrics["macro_f1_global_10_class"]),
                -float(metrics["nll"]),
            )
            selected = best_score is None or score > best_score
            if selected:
                best_score = score
                best_epoch = epoch
                best_model_state = _clone_state_dict(model.state_dict())
                best_optimizer_state = _clone_to_cpu(optimizer.state_dict())
                best_predictions = predictions.copy(deep=True)
                best_metrics = copy.deepcopy(metrics)
                best_per_class = per_class.copy(deep=True)
                best_confusion = confusion.copy(deep=True)
            epoch_records.append(
                {
                    "epoch": epoch,
                    "optimizer_steps_cumulative": optimizer_steps,
                    "train_native_units": len(selection.train_positions),
                    "train_loss": train_loss_mass / train_weight_mass,
                    "validation_native_units": len(
                        selection.validation_positions
                    ),
                    "validation_macro_f1_global_10_class": metrics[
                        "macro_f1_global_10_class"
                    ],
                    "validation_macro_f1_supported_classes": metrics[
                        "macro_f1_supported_classes"
                    ],
                    "validation_accuracy": metrics["accuracy"],
                    "validation_nll": metrics["nll"],
                    "parameter_sha256": parameter_sha256,
                    "prediction_sha256": prediction_sha256,
                    "selected_checkpoint": False,
                }
            )
        expected_steps = int(optimization["maximum_optimizer_steps"])
        if optimizer_steps != expected_steps:
            raise RuntimeError(
                f"cached short optimizer steps={optimizer_steps}!={expected_steps}"
            )
        if (
            best_model_state is None
            or best_optimizer_state is None
            or best_predictions is None
            or best_metrics is None
            or best_per_class is None
            or best_confusion is None
        ):
            raise RuntimeError("cached short checkpoint selection is empty")
        epoch_records[best_epoch - 1]["selected_checkpoint"] = True
        epoch_metrics = pd.DataFrame.from_records(
            epoch_records,
            columns=list(EPOCH_METRIC_FIELDS),
        )
        parameter_sha256 = _state_dict_sha256(best_model_state)
        prediction_sha256 = _dataframe_sha256(best_predictions)
        return LegacyL5CachedTrainingOutcome(
            epoch_metrics=epoch_metrics,
            predictions=best_predictions,
            metrics=best_metrics,
            per_class_metrics=best_per_class,
            confusion=best_confusion,
            model_state=best_model_state,
            optimizer_state=best_optimizer_state,
            best_epoch=best_epoch,
            optimizer_steps=optimizer_steps,
            parameter_sha256=parameter_sha256,
            prediction_sha256=prediction_sha256,
            epoch_metrics_sha256=_dataframe_sha256(epoch_metrics),
            maximum_loaded_batch_bytes=maximum_observed_batch_bytes,
        )
    finally:
        if model is not None:
            model.to("cpu")
        del model, optimizer
        gc.collect()


def _validate_training_selection(
    view: LegacyL5CachedFeatureView,
    selection: LegacyL5CachedShortSelection,
    config: LegacyL5CachedTrainingConfig,
) -> None:
    expected_hash = _dataframe_sha256(selection.manifest)
    if selection.audit.get("selection_content_sha256") != expected_hash:
        raise ValueError("cached short selection content hash drift")
    if selection.audit.get("outer_holdout_rows") != 0:
        raise ValueError("cached short selection exposes outer holdout")
    maximum_position = len(view.windows) - 1
    for name, positions in (
        ("train", selection.train_positions),
        ("validation", selection.validation_positions),
    ):
        values = np.asarray(positions, dtype=np.int64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(f"cached short {name} positions are invalid")
        if values.min() < 0 or values.max() > maximum_position:
            raise ValueError(f"cached short {name} positions are out of bounds")
        roles = set(view.windows.iloc[values]["l5_role"].astype(str))
        if roles != {name}:
            raise ValueError(f"cached short {name} role routing drift={roles}")
    expected = config.payload["data"]
    if len(selection.train_positions) != expected["expected_train_native_units"]:
        raise ValueError("cached short train selection count drift")
    if len(selection.validation_positions) != expected[
        "expected_validation_native_units"
    ]:
        raise ValueError("cached short validation selection count drift")


def _build_cached_classifier(
    config: LegacyL5CachedTrainingConfig,
) -> LegacyL5CachedFeatureClassifier:
    model = _object(config.payload["model"], "model")
    return LegacyL5CachedFeatureClassifier(
        temporal_encoder_name=str(model["temporal_encoder_name"]),
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        transformer_layers=int(model["transformer_layers"]),
        transformer_heads=int(model["transformer_heads"]),
    )


def _position_batches(
    positions: np.ndarray,
    *,
    batch_size: int,
) -> list[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("cached training batch size must be positive")
    values = np.asarray(positions, dtype=np.int64)
    return [
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    ]


def _load_selected_batch(
    view: LegacyL5CachedFeatureView,
    positions: np.ndarray,
    *,
    maximum_batch_bytes: int,
) -> tuple[dict[str, np.ndarray], int]:
    batch_positions = np.asarray(positions, dtype=np.int64).copy()
    batch = {
        "positions": batch_positions,
        "features": view.load_sequences(batch_positions),
        "observed_mask": view.observed_mask[batch_positions].copy(),
        "time_delta": view.time_delta[batch_positions].copy(),
        "targets": view.targets[batch_positions].copy(),
        "sample_weights": view.sample_weights[batch_positions].copy(),
    }
    loaded_bytes = sum(int(value.nbytes) for value in batch.values())
    if loaded_bytes > maximum_batch_bytes:
        raise MemoryError(
            f"cached training loaded batch={loaded_bytes}>"
            f"{maximum_batch_bytes}"
        )
    if not np.isfinite(batch["sample_weights"]).all():
        raise ValueError("cached training sample weights are nonfinite")
    if (batch["sample_weights"] <= 0.0).any():
        raise ValueError("cached training sample weights are not positive")
    return batch, loaded_bytes


def _cached_training_step(
    model: LegacyL5CachedFeatureClassifier,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, np.ndarray],
    *,
    device: torch.device,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    features = torch.from_numpy(batch["features"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )
    observed_mask = torch.from_numpy(batch["observed_mask"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )
    time_delta = torch.from_numpy(batch["time_delta"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )
    targets = torch.from_numpy(batch["targets"]).to(
        device=device,
        dtype=torch.long,
        non_blocking=False,
    )
    weights = torch.from_numpy(batch["sample_weights"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(features, observed_mask, time_delta=time_delta)
    losses = torch.nn.functional.cross_entropy(
        logits,
        targets,
        reduction="none",
    )
    weight_mass = weights.sum()
    loss = (losses * weights).sum() / weight_mass
    if not torch.isfinite(loss):
        raise FloatingPointError("cached short training loss is nonfinite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=gradient_clip_norm,
        error_if_nonfinite=True,
    )
    if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0.0:
        raise FloatingPointError("cached short gradients are invalid")
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    weight_value = float(weight_mass.detach().cpu())
    del features, observed_mask, time_delta, targets, weights, logits, losses
    del weight_mass, loss, gradient_norm
    return loss_value, weight_value


def _evaluate_cached_classifier(
    model: LegacyL5CachedFeatureClassifier,
    view: LegacyL5CachedFeatureView,
    positions: np.ndarray,
    *,
    batch_size: int,
    maximum_batch_bytes: int,
    device: torch.device,
) -> dict[str, Any]:
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    maximum_observed_bytes = 0
    model.eval()
    with torch.inference_mode():
        for batch_positions in _position_batches(
            positions,
            batch_size=batch_size,
        ):
            batch, loaded_bytes = _load_selected_batch(
                view,
                batch_positions,
                maximum_batch_bytes=maximum_batch_bytes,
            )
            maximum_observed_bytes = max(maximum_observed_bytes, loaded_bytes)
            features = torch.from_numpy(batch["features"]).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )
            observed_mask = torch.from_numpy(batch["observed_mask"]).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )
            time_delta = torch.from_numpy(batch["time_delta"]).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )
            logits = model(features, observed_mask, time_delta=time_delta)
            probs = torch.softmax(logits, dim=1)
            if not torch.isfinite(probs).all():
                raise FloatingPointError(
                    "cached short validation probabilities are nonfinite"
                )
            probabilities.append(probs.cpu().numpy().astype(np.float64))
            targets.append(batch["targets"].astype(np.int64, copy=True))
            del batch, features, observed_mask, time_delta, logits, probs
    return {
        "probabilities": np.concatenate(probabilities, axis=0),
        "targets": np.concatenate(targets, axis=0),
        "maximum_loaded_batch_bytes": maximum_observed_bytes,
    }


def _cached_prediction_frame(
    view: LegacyL5CachedFeatureView,
    positions: np.ndarray,
    *,
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> pd.DataFrame:
    metadata = view.windows.iloc[np.asarray(positions, dtype=np.int64)].copy()
    metadata = metadata.reset_index(drop=True)
    if len(metadata) != len(probabilities) or len(metadata) != len(targets):
        raise ValueError("cached short prediction row counts differ")
    predicted = probabilities.argmax(axis=1).astype(np.int64)
    frame = pd.DataFrame(
        {
            "prediction_order": np.arange(len(metadata), dtype=np.int64),
            "window_id": metadata["window_id"].astype(str),
            "temporal_unit_key": metadata["temporal_unit_key"].astype(str),
            "recording_group_id": metadata["recording_group_id"].astype(str),
            "video_key": metadata["video_key"].astype(str),
            "source_type": metadata["source_type"].astype(str),
            "dataset_id": metadata["dataset_id"].astype(str),
            "behavior_label": metadata["behavior_label"].astype(str),
            "target_index": targets.astype(np.int64),
            "predicted_index": predicted,
            "predicted_label": [VALID_BEHAVIORS[index] for index in predicted],
        }
    )
    for index, field in enumerate(PROBABILITY_FIELDS):
        frame[field] = probabilities[:, index].astype(np.float64)
    frame["lineage_scope"] = LINEAGE_SCOPE
    frame["human_review_complete"] = False
    frame = frame[list(PREDICTION_FIELDS)]
    expected_targets = metadata["behavior_label"].map(
        {label: index for index, label in enumerate(VALID_BEHAVIORS)}
    )
    if expected_targets.isna().any() or not np.array_equal(
        expected_targets.to_numpy(dtype=np.int64),
        targets,
    ):
        raise ValueError("cached short prediction targets drift from labels")
    return frame


def _seed_all(seed: int, *, seed_cuda: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)


def _clone_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in state.items()
    }


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_base_config(payload: object) -> None:
    value = _object(payload, "base_config")
    _require_exact_keys(value, {"path", "sha256"}, name="base_config")
    _validate_sha256(value["sha256"], name="base_config.sha256")


def _validate_experiment_contract(payload: object) -> None:
    value = _object(payload, "experiment_contract")
    expected = {
        "experiment_id": "L5_V0_T16_SHORT",
        "parent_id": "cfd_v0_t16_b425c86",
        "scientific_role": "foundational_baseline_gate",
        "changed_family": "cached_temporal_head_training",
        "hypothesis": (
            "V0 cached T16 head is deterministic and finite in nine steps."
        ),
        "compute_cap": (
            "two isolated runs, three epochs and nine steps per run"
        ),
        "stop_rule": (
            "Stop on lineage, memory, finite, repeat, or outer-access failure."
        ),
    }
    _require_exact_keys(
        value,
        set(expected),
        name="cached training experiment contract",
    )
    _validate_exact_values(
        value,
        expected,
        name="cached training experiment contract",
    )


def _validate_consumer_parent(payload: object) -> None:
    value = _object(payload, "consumer_parent")
    required = {
        "run_id",
        "code_sha",
        "consumer_run_relative_path",
        "run_manifest_sha256",
        "cached_data_audit_sha256",
        "feature_run_relative_path",
        "feature_manifest_sha256",
        "feature_result_sha256",
        "feature_tensor_sha256",
        "feature_index_sha256",
        "dataset_snapshot_sha256",
        "fold_manifest_sha256",
        "feature_whitelist_sha256",
    }
    _require_exact_keys(value, required, name="consumer_parent")
    for name in required:
        if name.endswith("sha256"):
            _validate_sha256(value[name], name=f"consumer_parent.{name}")
    if not str(value["run_id"]).strip() or not str(value["code_sha"]).strip():
        raise ValueError("cached training consumer identity is blank")


def _validate_data_contract(payload: object) -> None:
    value = _object(payload, "data")
    required = {
        "control_id",
        "temporal_view_name",
        "sampling_protocol",
        "sequence_length",
        "feature_dim",
        "model_visible_roles",
        "outer_holdout_access",
        "train_selection_policy",
        "train_selection_salt",
        "train_native_units_per_class",
        "expected_train_native_units",
        "validation_selection_policy",
        "expected_validation_native_units",
        "native_prediction_aggregation",
    }
    _require_exact_keys(value, required, name="cached training data")
    expected = {
        "control_id": "V0",
        "temporal_view_name": EXPECTED_TEMPORAL_VIEW,
        "sampling_protocol": "one_centered_window_matched",
        "sequence_length": 16,
        "feature_dim": FEATURE_DIM,
        "model_visible_roles": list(MODEL_VISIBLE_ROLES),
        "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
        "train_selection_policy": "sha256_rank_per_class_v1",
        "train_native_units_per_class": 8,
        "expected_train_native_units": 80,
        "validation_selection_policy": "all_validation_native_units_v1",
        "expected_validation_native_units": 245,
        "native_prediction_aggregation": (
            "one_centered_window_per_native_unit"
        ),
    }
    _validate_exact_values(value, expected, name="cached training data")
    if not str(value["train_selection_salt"]).strip():
        raise ValueError("cached training selection salt is blank")


def _validate_model_contract(payload: object) -> None:
    value = _object(payload, "model")
    required = {
        "architecture",
        "temporal_encoder_name",
        "hidden_dim",
        "dropout",
        "transformer_layers",
        "transformer_heads",
        "direct_ten_class_supervision",
        "data_derived_normalization",
        "learned_input_transform",
    }
    _require_exact_keys(value, required, name="cached training model")
    expected = {
        "architecture": "cached_frame_feature_temporal_classifier_v1",
        "temporal_encoder_name": "masked_mean",
        "hidden_dim": 128,
        "dropout": 0.1,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "direct_ten_class_supervision": True,
        "data_derived_normalization": "none",
        "learned_input_transform": "layer_norm_and_linear_projection",
    }
    _validate_exact_values(value, expected, name="cached training model")


def _validate_optimization_contract(payload: object) -> None:
    value = _object(payload, "optimization")
    required = {
        "seed",
        "epochs",
        "batch_size",
        "evaluation_batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "loss",
        "sampler",
        "checkpoint_selection",
        "precision",
        "autocast_enabled",
        "deterministic_algorithms",
        "dataloader_num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "device",
        "declared_local_gpu_vram_gib",
        "validated_local_gpu_vram_bytes",
        "maximum_peak_vram_fraction",
        "allocator_limit_bytes",
        "oom_retry_allowed",
        "maximum_optimizer_steps",
        "maximum_loaded_batch_bytes",
    }
    _require_exact_keys(value, required, name="cached training optimization")
    expected = {
        "seed": 20260714,
        "epochs": 3,
        "batch_size": 32,
        "evaluation_batch_size": 64,
        "learning_rate": 0.003,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "loss": "event_mass_balanced_cross_entropy",
        "sampler": "deterministic_seeded_shuffle",
        "checkpoint_selection": (
            "validation_native_global_10_class_macro_f1_then_nll"
        ),
        "precision": "float32",
        "autocast_enabled": False,
        "deterministic_algorithms": True,
        "dataloader_num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "device": "cuda:0",
        "declared_local_gpu_vram_gib": DECLARED_LOCAL_GPU_VRAM_GIB,
        "validated_local_gpu_vram_bytes": VALIDATED_LOCAL_GPU_VRAM_BYTES,
        "maximum_peak_vram_fraction": GPU_ALLOCATOR_FRACTION_CEILING,
        "allocator_limit_bytes": GPU_ALLOCATOR_LIMIT_BYTES,
        "oom_retry_allowed": False,
        "maximum_optimizer_steps": 9,
        "maximum_loaded_batch_bytes": 2_103_552,
    }
    _validate_exact_values(value, expected, name="cached training optimization")


def _validate_execution_guard(payload: object) -> None:
    value = _object(payload, "execution_guard")
    required = {
        "require_fresh_process",
        "require_committed_training_source",
        "allowed_dirty_paths",
    }
    _require_exact_keys(value, required, name="cached training execution guard")
    if value["require_fresh_process"] is not True:
        raise ValueError("cached training fresh-process guard is disabled")
    if value["require_committed_training_source"] is not True:
        raise ValueError("cached training committed-source guard is disabled")
    expected_dirty = [
        ".tokensave/config.json",
        (
            "outputs/classification_v2/train_ready_windows/"
            "feature_semantics_audit.json"
        ),
        "scripts/diagnostics/detect_single_frame.py",
    ]
    if value["allowed_dirty_paths"] != expected_dirty:
        raise ValueError("cached training dirty-worktree allowlist drift")


def _validate_repeat_contract(payload: object) -> None:
    value = _object(payload, "repeat_gate")
    required = {
        "required_runs",
        "require_distinct_run_ids",
        "require_distinct_process_ids",
        "require_non_overlapping_execution",
        "require_identical_subset_hash",
        "require_identical_parameter_hash",
        "require_identical_prediction_hash",
        "require_identical_epoch_metric_hash",
    }
    _require_exact_keys(value, required, name="cached training repeat gate")
    if value["required_runs"] != 2:
        raise ValueError("cached training repeat gate requires two runs")
    if any(value[name] is not True for name in required if name != "required_runs"):
        raise ValueError("cached training repeat gate weakened")


def _validate_output_contract(payload: object) -> None:
    value = _object(payload, "output")
    required = {
        "run_root_relative_path",
        "registry_filename",
        "short_gate_filename",
    }
    _require_exact_keys(value, required, name="cached training output")
    expected = {
        "run_root_relative_path": "15_l5_core_baselines",
        "registry_filename": "runs_registry.csv",
        "short_gate_filename": "legacy_l5_cached_training_short_gate_v1.json",
    }
    _validate_exact_values(value, expected, name="cached training output")


def _cached_training_run_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "run_manifest": root / "run_manifest.json",
        "environment": root / "environment.json",
        "preflight": root / "preflight.json",
        "selection_manifest": root / "short_selection_manifest.csv",
        "selection_audit": root / "short_selection_audit.json",
        "epoch_metrics": root / "epoch_metrics.csv",
        "validation_predictions": root / "validation_predictions.csv",
        "validation_metrics": root / "validation_metrics.json",
        "validation_per_class": root / "validation_per_class.csv",
        "validation_confusion": root / "validation_confusion.csv",
        "checkpoint": root / "best_validation_checkpoint.pt",
        "checkpoint_manifest": root / "checkpoint_manifest.json",
        "prediction_manifest": root / "prediction_manifest.json",
        "run_result": root / "run_result.json",
        "artifact_manifest": root / "artifact_manifest.json",
        "registry_entry": root / "registry_entry.json",
        "runs_registry": root / "runs_registry.csv",
        "unexpected_failure": root / "unexpected_failure.json",
    }


def _safe_run_id(value: str) -> bool:
    if not value or len(value) > 64 or value in {".", ".."}:
        return False
    if Path(value).name != value:
        return False
    return all(
        character.isascii()
        and (character.isalnum() or character in {"-", "_"})
        for character in value
    )


def _planned_training_manifest(
    config: LegacyL5CachedTrainingConfig,
    *,
    selection: LegacyL5CachedShortSelection,
    parent: dict[str, Any],
    preflight: dict[str, Any],
    git_guard: dict[str, Any],
    run_id: str,
    started_at: str,
) -> dict[str, Any]:
    parent_config = config.payload["consumer_parent"]
    base = load_legacy_l5_config(config.base_config_path)
    split = base.payload["split_contract"]
    feature_manifest = parent["feature_manifest"]
    identity = {
        "config_sha256": config.sha256,
        "code_sha": git_guard["code_sha"],
        "consumer_run_manifest_sha256": parent_config[
            "run_manifest_sha256"
        ],
        "cached_data_audit_sha256": parent_config[
            "cached_data_audit_sha256"
        ],
        "feature_tensor_sha256": parent_config["feature_tensor_sha256"],
        "feature_index_sha256": parent_config["feature_index_sha256"],
        "backbone_name": feature_manifest["backbone_name"],
        "pretrained_weight_enum": feature_manifest[
            "pretrained_weight_enum"
        ],
        "normalization_name": feature_manifest["normalization_name"],
        "resolution": feature_manifest["image_size"],
        "selection_content_sha256": selection.audit[
            "selection_content_sha256"
        ],
        "experiment_contract": config.payload["experiment_contract"],
        "model": config.payload["model"],
        "optimization": config.payload["optimization"],
    }
    return {
        "schema_version": CACHED_TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_name": config.payload["experiment_name"],
        "experiment_contract": config.payload["experiment_contract"],
        "execution_mode": config.payload["execution_mode"],
        "status": "planned",
        "failure_reason": "",
        "process_id": os.getpid(),
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "development_metrics_authorized": True,
        "code_sha": git_guard["code_sha"],
        "dirty_worktree": git_guard["dirty_worktree"],
        "dirty_entries": git_guard["dirty_entries"],
        "config_path": str(config.path),
        "config_hash": config.sha256,
        "base_config_hash": config.payload["base_config"]["sha256"],
        "consumer_run_manifest_hash": parent_config[
            "run_manifest_sha256"
        ],
        "cached_data_audit_hash": parent_config[
            "cached_data_audit_sha256"
        ],
        "dataset_snapshot_hash": parent_config["dataset_snapshot_sha256"],
        "cache_hash": parent_config["feature_tensor_sha256"],
        "feature_index_hash": parent_config["feature_index_sha256"],
        "fold_manifest_hash": parent_config["fold_manifest_sha256"],
        "feature_whitelist_hash": parent_config[
            "feature_whitelist_sha256"
        ],
        "consumer_parent_valid": parent["valid"],
        "fold": split["development_validation_fold_id"],
        "outer_holdout_fold": split["outer_holdout_fold_id"],
        "preflight_valid": preflight["valid"],
        "semantic_identity_sha256": _canonical_json_sha256(identity),
        "selection_content_sha256": selection.audit[
            "selection_content_sha256"
        ],
        "train_native_units": len(selection.train_positions),
        "validation_native_units": len(selection.validation_positions),
        "outer_holdout_native_units_loaded": 0,
        "control_id": config.payload["data"]["control_id"],
        "backbone_name": feature_manifest["backbone_name"],
        "pretrained_weight_enum": feature_manifest[
            "pretrained_weight_enum"
        ],
        "pretrained_weight_sha256": feature_manifest[
            "pretrained_weight_sha256"
        ],
        "resolution": feature_manifest["image_size"],
        "normalization_name": feature_manifest["normalization_name"],
        "image_preprocessing": "aspect_preserving_letterbox",
        "temporal_view_name": config.payload["data"][
            "temporal_view_name"
        ],
        "sequence_length": config.payload["data"]["sequence_length"],
        "temporal_encoder_name": config.payload["model"][
            "temporal_encoder_name"
        ],
        "seed": config.payload["optimization"]["seed"],
        "epochs": config.payload["optimization"]["epochs"],
        "batch_size": config.payload["optimization"]["batch_size"],
        "evaluation_batch_size": config.payload["optimization"][
            "evaluation_batch_size"
        ],
        "maximum_optimizer_steps": config.payload["optimization"][
            "maximum_optimizer_steps"
        ],
        "precision": "float32",
        "autocast_enabled": False,
        "oom_retry_allowed": False,
        "allocator_limit_bytes": config.payload["optimization"][
            "allocator_limit_bytes"
        ],
        "maximum_loaded_batch_bytes": config.payload["optimization"][
            "maximum_loaded_batch_bytes"
        ],
        "source_media_reads": 0,
        "outer_predictions_created": 0,
    }


def _training_environment_payload(
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CACHED_TRAINING_ENVIRONMENT_SCHEMA_VERSION,
        "captured_at_utc": _utc_now(),
        "process_id": os.getpid(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "torch_version": str(torch.__version__),
        "torchvision_version": _package_version("torchvision"),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_runtime_initialized_at_capture": torch.cuda.is_initialized(),
        "execution_mode": run_manifest["execution_mode"],
        "code_sha": run_manifest["code_sha"],
        "config_hash": run_manifest["config_hash"],
        "precision": "float32",
        "autocast_enabled": False,
        "deterministic_algorithms_required": True,
        "dataloader_num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": None,
        "oom_retry_allowed": False,
    }


def _finalize_cached_training_run(
    paths: dict[str, Path],
    *,
    config: LegacyL5CachedTrainingConfig,
    planned: dict[str, Any],
    planned_sha256: str,
    selection: LegacyL5CachedShortSelection,
    outcome: LegacyL5CachedTrainingOutcome | None,
    execution: dict[str, Any],
    failure: dict[str, Any] | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    errors = [str(value) for value in execution["errors"]]
    if failure is not None:
        errors.append(
            f"{failure['error_type']}: {failure['error_message']}"
        )
    valid = outcome is not None and execution["valid"] and not errors
    terminal_status = "completed" if valid else "failed"
    failure_reason = ";".join(errors)
    checkpoint_record: dict[str, Any] | None = None
    prediction_record: dict[str, Any] | None = None
    if outcome is not None:
        checkpoint_record, prediction_record = _write_training_outcome(
            paths,
            config=config,
            planned=planned,
            selection=selection,
            outcome=outcome,
        )
    checkpoint_manifest = _checkpoint_manifest_payload(
        planned,
        status=terminal_status,
        checkpoint_record=checkpoint_record,
        failure_reason=failure_reason,
    )
    prediction_manifest = _prediction_manifest_payload(
        planned,
        status=terminal_status,
        prediction_record=prediction_record,
        failure_reason=failure_reason,
    )
    _write_json_exclusive(paths["checkpoint_manifest"], checkpoint_manifest)
    _write_json_exclusive(paths["prediction_manifest"], prediction_manifest)
    completed_at = _utc_now()
    _finalize_training_environment(
        paths["environment"],
        execution=execution,
        status=terminal_status,
        completed_at=completed_at,
        failure_reason=failure_reason,
    )
    result = _training_result_payload(
        paths,
        config=config,
        planned=planned,
        planned_sha256=planned_sha256,
        selection=selection,
        outcome=outcome,
        execution=execution,
        valid=valid,
        errors=errors,
        completed_at=completed_at,
        runtime_seconds=runtime_seconds,
    )
    _write_json_exclusive(paths["run_result"], result)
    artifact_manifest = {
        "schema_version": CACHED_TRAINING_ARTIFACT_SCHEMA_VERSION,
        "run_id": planned["run_id"],
        "semantic_identity_sha256": planned["semantic_identity_sha256"],
        "status": terminal_status,
        "artifacts": _training_artifact_records(paths),
        "failure_reason": failure_reason,
    }
    _write_json_exclusive(paths["artifact_manifest"], artifact_manifest)
    if file_sha256(paths["run_manifest"]) != planned_sha256:
        raise ValueError("cached training planned manifest changed during run")
    final_manifest = {
        **planned,
        "status": terminal_status,
        "failure_reason": failure_reason,
        "completed_at_utc": completed_at,
        "runtime_seconds": runtime_seconds,
        "optimizer_steps": outcome.optimizer_steps if outcome else 0,
        "best_epoch": outcome.best_epoch if outcome else None,
        "peak_vram_bytes": execution["peak_reserved_bytes"],
        "planned_run_manifest_sha256": planned_sha256,
        "run_result_sha256": file_sha256(paths["run_result"]),
        "artifact_manifest_sha256": file_sha256(
            paths["artifact_manifest"]
        ),
        "checkpoint_manifest_sha256": file_sha256(
            paths["checkpoint_manifest"]
        ),
        "prediction_manifest_sha256": file_sha256(
            paths["prediction_manifest"]
        ),
        "runs_registry_path": str(paths["runs_registry"].resolve()),
    }
    _write_json_atomic(paths["run_manifest"], final_manifest)
    entry = _training_registry_entry(
        paths,
        config=config,
        manifest=final_manifest,
        result=result,
        failure_reason=failure_reason,
    )
    _write_json_exclusive(paths["registry_entry"], entry)
    _write_registry(paths["runs_registry"], entry)
    return result


def _finalize_training_environment(
    path: Path,
    *,
    execution: dict[str, Any],
    status: str,
    completed_at: str,
    failure_reason: str,
) -> None:
    payload = _read_json(path)
    payload.update(
        {
            "status": status,
            "completed_at_utc": completed_at,
            "gpu_inventory_source": "live_torch_cuda_runtime",
            "gpu_model": execution["device_name"],
            "gpu_vram_bytes": execution["actual_total_vram_bytes"],
            "free_vram_before_bytes": execution["free_vram_before_bytes"],
            "allocator_limit_bytes": execution["allocator_limit_bytes"],
            "peak_allocated_bytes": execution["peak_allocated_bytes"],
            "peak_reserved_bytes": execution["peak_reserved_bytes"],
            "post_cleanup_allocated_bytes": execution[
                "post_cleanup_allocated_bytes"
            ],
            "post_cleanup_reserved_bytes": execution[
                "post_cleanup_reserved_bytes"
            ],
            "failure_reason": failure_reason,
        }
    )
    _write_json_atomic(path, payload)


def _write_training_outcome(
    paths: dict[str, Path],
    *,
    config: LegacyL5CachedTrainingConfig,
    planned: dict[str, Any],
    selection: LegacyL5CachedShortSelection,
    outcome: LegacyL5CachedTrainingOutcome,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _write_dataframe_exclusive(paths["epoch_metrics"], outcome.epoch_metrics)
    _write_dataframe_exclusive(
        paths["validation_predictions"],
        outcome.predictions,
    )
    _write_json_exclusive(paths["validation_metrics"], outcome.metrics)
    _write_dataframe_exclusive(
        paths["validation_per_class"],
        outcome.per_class_metrics,
    )
    _write_dataframe_exclusive(
        paths["validation_confusion"],
        outcome.confusion,
    )
    checkpoint = {
        "schema_version": CACHED_TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "run_id": planned["run_id"],
        "semantic_identity_sha256": planned["semantic_identity_sha256"],
        "config_sha256": config.sha256,
        "selection_content_sha256": selection.audit[
            "selection_content_sha256"
        ],
        "best_epoch": outcome.best_epoch,
        "optimizer_steps": outcome.optimizer_steps,
        "checkpoint_selection": config.payload["optimization"][
            "checkpoint_selection"
        ],
        "validation_metrics": outcome.metrics,
        "parameter_sha256": outcome.parameter_sha256,
        "model_state_dict": outcome.model_state,
        "optimizer_state_dict": outcome.optimizer_state,
    }
    _write_torch_exclusive(paths["checkpoint"], checkpoint)
    loaded = torch.load(
        paths["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    if _state_dict_sha256(loaded["model_state_dict"]) != (
        outcome.parameter_sha256
    ):
        raise ValueError("cached training checkpoint reload hash mismatch")
    checkpoint_record = {
        "path": str(paths["checkpoint"].resolve()),
        "sha256": file_sha256(paths["checkpoint"]),
        "size_bytes": int(paths["checkpoint"].stat().st_size),
        "best_epoch": outcome.best_epoch,
        "optimizer_steps": outcome.optimizer_steps,
        "parameter_sha256": outcome.parameter_sha256,
        "selection_metric": (
            "validation_native_global_10_class_macro_f1_then_nll"
        ),
    }
    prediction_record = {
        "path": str(paths["validation_predictions"].resolve()),
        "sha256": file_sha256(paths["validation_predictions"]),
        "content_sha256": outcome.prediction_sha256,
        "rows": len(outcome.predictions),
        "role": "validation",
        "native_unit_grain": True,
        "outer_holdout_rows": 0,
    }
    return checkpoint_record, prediction_record


def _checkpoint_manifest_payload(
    planned: dict[str, Any],
    *,
    status: str,
    checkpoint_record: dict[str, Any] | None,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHED_TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "run_id": planned["run_id"],
        "semantic_identity_sha256": planned["semantic_identity_sha256"],
        "status": status,
        "checkpoints": [checkpoint_record] if checkpoint_record else [],
        "selection_uses_validation_only": True,
        "outer_holdout_selection_allowed": False,
        "failure_reason": failure_reason,
    }


def _prediction_manifest_payload(
    planned: dict[str, Any],
    *,
    status: str,
    prediction_record: dict[str, Any] | None,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHED_TRAINING_PREDICTION_SCHEMA_VERSION,
        "run_id": planned["run_id"],
        "semantic_identity_sha256": planned["semantic_identity_sha256"],
        "status": status,
        "predictions": [prediction_record] if prediction_record else [],
        "validation_predictions_created": (
            prediction_record["rows"] if prediction_record else 0
        ),
        "outer_holdout_predictions_created": 0,
        "outer_holdout_predictions_authorized": False,
        "failure_reason": failure_reason,
    }


def _training_result_payload(
    paths: dict[str, Path],
    *,
    config: LegacyL5CachedTrainingConfig,
    planned: dict[str, Any],
    planned_sha256: str,
    selection: LegacyL5CachedShortSelection,
    outcome: LegacyL5CachedTrainingOutcome | None,
    execution: dict[str, Any],
    valid: bool,
    errors: list[str],
    completed_at: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    metrics = outcome.metrics if outcome is not None else None
    return {
        "schema_version": CACHED_TRAINING_RUN_RESULT_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_CACHED_SHORT_TRAINING"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_CACHED_SHORT_TRAINING"
        ),
        "run_id": planned["run_id"],
        "process_id": planned["process_id"],
        "started_at_utc": planned["started_at_utc"],
        "completed_at_utc": completed_at,
        "runtime_seconds": runtime_seconds,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "development_metrics_authorized": True,
        "code_sha": planned["code_sha"],
        "config_sha256": config.sha256,
        "planned_run_manifest_sha256": planned_sha256,
        "semantic_identity_sha256": planned["semantic_identity_sha256"],
        "selection_content_sha256": selection.audit[
            "selection_content_sha256"
        ],
        "train_native_units": len(selection.train_positions),
        "validation_native_units": len(selection.validation_positions),
        "outer_holdout_rows_loaded": 0,
        "optimizer_steps": outcome.optimizer_steps if outcome else 0,
        "best_epoch": outcome.best_epoch if outcome else None,
        "validation_metrics": metrics,
        "parameter_sha256": (
            outcome.parameter_sha256 if outcome is not None else None
        ),
        "prediction_content_sha256": (
            outcome.prediction_sha256 if outcome is not None else None
        ),
        "epoch_metrics_content_sha256": (
            outcome.epoch_metrics_sha256 if outcome is not None else None
        ),
        "maximum_loaded_batch_bytes": (
            outcome.maximum_loaded_batch_bytes if outcome is not None else 0
        ),
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "checkpoint_manifest_sha256": file_sha256(
            paths["checkpoint_manifest"]
        ),
        "prediction_manifest_sha256": file_sha256(
            paths["prediction_manifest"]
        ),
        "execution": execution,
        "errors": errors,
        "valid": valid,
    }


def _training_artifact_records(
    paths: dict[str, Path],
) -> list[dict[str, Any]]:
    excluded = {
        "root",
        "run_manifest",
        "artifact_manifest",
        "registry_entry",
        "runs_registry",
    }
    records: list[dict[str, Any]] = []
    for name, path in paths.items():
        if name in excluded or not path.is_file():
            continue
        records.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
                "direction": "output",
            }
        )
    return records


def _training_registry_entry(
    paths: dict[str, Path],
    *,
    config: LegacyL5CachedTrainingConfig,
    manifest: dict[str, Any],
    result: dict[str, Any],
    failure_reason: str,
) -> dict[str, Any]:
    metrics = result["validation_metrics"] or {}
    return {
        "registry_schema_version": CACHED_TRAINING_REGISTRY_SCHEMA_VERSION,
        "run_id": result["run_id"],
        "experiment_name": manifest["experiment_name"],
        "execution_mode": manifest["execution_mode"],
        "status": manifest["status"],
        "failure_reason": failure_reason,
        "code_sha": result["code_sha"],
        "dirty_worktree": manifest["dirty_worktree"],
        "config_hash": result["config_sha256"],
        "dataset_snapshot_hash": manifest["dataset_snapshot_hash"],
        "cache_hash": manifest["cache_hash"],
        "fold_manifest_hash": manifest["fold_manifest_hash"],
        "feature_whitelist_hash": manifest["feature_whitelist_hash"],
        "control_id": config.payload["data"]["control_id"],
        "temporal_view_name": config.payload["data"][
            "temporal_view_name"
        ],
        "seed": config.payload["optimization"]["seed"],
        "train_native_units": result["train_native_units"],
        "validation_native_units": result["validation_native_units"],
        "optimizer_steps": result["optimizer_steps"],
        "best_epoch": result["best_epoch"] or "",
        "validation_macro_f1": metrics.get(
            "macro_f1_global_10_class",
            "",
        ),
        "validation_accuracy": metrics.get("accuracy", ""),
        "validation_nll": metrics.get("nll", ""),
        "outer_predictions_created": 0,
        "source_media_reads": 0,
        "peak_vram_bytes": result["execution"]["peak_reserved_bytes"],
        "runtime_seconds": result["runtime_seconds"],
        "manifest_path": str(paths["run_manifest"].resolve()),
        "manifest_sha256": file_sha256(paths["run_manifest"]),
        "completed_at_utc": result["completed_at_utc"],
    }


def _selection_score(salt: str, temporal_unit_key: str) -> str:
    return hashlib.sha256(f"{salt}\0{temporal_unit_key}".encode()).hexdigest()


def _git_launch_guard(
    config: LegacyL5CachedTrainingConfig,
) -> dict[str, Any]:
    state = git_state()
    allowed = set(config.payload["execution_guard"]["allowed_dirty_paths"])
    observed_paths: list[str] = []
    parse_errors: list[str] = []
    for entry in state["dirty_entries"]:
        if len(entry) < 4 or " -> " in entry:
            parse_errors.append(f"unsupported_git_status_entry={entry}")
            continue
        path = entry[3:].strip().strip('"').replace("\\", "/")
        observed_paths.append(path)
    unexpected = sorted(set(observed_paths).difference(allowed))
    tracked_paths = [
        str(config.path.relative_to(config.repo_root)).replace("\\", "/"),
        str(Path(__file__).resolve().relative_to(config.repo_root)).replace(
            "\\",
            "/",
        ),
    ]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *tracked_paths],
        cwd=config.repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    errors = list(parse_errors)
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if tracked.returncode != 0:
        errors.append("training_config_or_source_not_committed_at_head")
    return {
        "code_sha": state["commit"],
        "dirty_worktree": bool(state["dirty"]),
        "dirty_entries": list(state["dirty_entries"]),
        "allowed_dirty_paths": sorted(allowed),
        "observed_dirty_paths": sorted(observed_paths),
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": tracked_paths,
        "required_paths_tracked": tracked.returncode == 0,
        "errors": errors,
        "valid": not errors,
    }


def _selection_group_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, Any]:
    errors: list[str] = []
    overlaps: dict[str, int] = {}
    for column in (
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
    ):
        overlap = set(train[column].astype(str)).intersection(
            validation[column].astype(str)
        )
        overlaps[f"{column}_overlap"] = len(overlap)
        if overlap:
            errors.append(f"{column}:{len(overlap)}")
    return {"overlaps": overlaps, "errors": errors, "valid": not errors}


def _ordered_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        if not value.strip():
            raise ValueError("ordered hash contains a blank value")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_canonical_dataframe_bytes(frame)).hexdigest()


def _canonical_dataframe_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        _write_json_exclusive(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_dataframe_exclusive(path: Path, frame: pd.DataFrame) -> None:
    if frame.columns.duplicated().any():
        raise ValueError(f"dataframe contains duplicate columns: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_dataframe_bytes(frame)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_torch_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_registry(path: Path, entry: dict[str, Any]) -> None:
    if tuple(entry) != REGISTRY_FIELDS:
        raise ValueError("cached training registry entry schema drift")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(REGISTRY_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(entry)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_exact_values(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    name: str,
) -> None:
    errors = [
        f"{field}:{observed.get(field)!r}!={value!r}"
        for field, value in expected.items()
        if observed.get(field) != value
    ]
    if errors:
        raise ValueError(f"{name} drift={errors}")


def _validate_sha256(value: object, *, name: str) -> None:
    normalized = str(value)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} is not lowercase SHA256")


def _require_exact_keys(
    payload: dict[str, Any],
    required: set[str],
    *,
    name: str,
) -> None:
    observed = set(payload)
    if observed != required:
        raise ValueError(
            f"{name} keys drift: missing={sorted(required - observed)},"
            f"extra={sorted(observed - required)}"
        )


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


__all__ = (
    "CACHED_TRAINING_CONFIG_SCHEMA_VERSION",
    "CACHED_TRAINING_METRICS_SCHEMA_VERSION",
    "CACHED_TRAINING_REPEAT_GATE_SCHEMA_VERSION",
    "CACHED_TRAINING_SELECTION_SCHEMA_VERSION",
    "LegacyL5CachedShortSelection",
    "LegacyL5CachedTrainingConfig",
    "LegacyL5CachedTrainingOutcome",
    "audit_legacy_l5_cached_training_repeat_gate",
    "build_legacy_l5_cached_short_selection",
    "compute_legacy_l5_native_metrics",
    "load_legacy_l5_cached_training_config",
    "load_legacy_l5_cached_training_view",
    "preflight_legacy_l5_cached_short_training",
    "run_legacy_l5_cached_short_training",
    "train_legacy_l5_cached_short_core",
    "write_legacy_l5_cached_training_repeat_gate",
)
