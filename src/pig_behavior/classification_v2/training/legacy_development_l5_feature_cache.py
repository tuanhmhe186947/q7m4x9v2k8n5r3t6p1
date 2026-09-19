"""Lineage-safe pretrained frame-feature caches for legacy L5."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn

from pig_behavior.classification_v2.models.visual_backbones import (
    build_visual_frame_encoder,
    visual_backbone_contract,
)
from pig_behavior.classification_v2.training.legacy_development_l5 import (
    LINEAGE_SCOPE,
    LegacyL5Config,
    git_state,
)
from pig_behavior.classification_v2.training.legacy_development_l5_visual import (
    L5_VRAM_PROBE_SCHEMA_VERSION,
    LegacyVisualProbeControl,
    _cache_lineage,
    _close_memmap,
    _device_preflight_errors,
    _live_weight_errors,
    _object,
    _read_json,
    _seed_cuda,
    _validate_full_cache_parent,
    _validate_readiness_parent,
    _validate_weights_parent,
    _vram_budget_bytes,
    legacy_l5_visual_probe_controls,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
)

FEATURE_RUN_MANIFEST_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_run_manifest.v1"
)
FEATURE_PROGRESS_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_progress.v1"
)
FEATURE_RUN_RESULT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_run_result.v1"
)
FEATURE_ARTIFACT_MANIFEST_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_artifacts.v1"
)
FEATURE_ENVIRONMENT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_environment.v1"
)
FEATURE_PREFLIGHT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_preflight.v1"
)
FEATURE_CHECKPOINT_MANIFEST_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_checkpoints.v1"
)
FEATURE_PREDICTION_MANIFEST_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_predictions.v1"
)
FEATURE_REGISTRY_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_registry.v1"
)
FEATURE_SHORT_GATE_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.feature_short_gate.v1"
)
FEATURE_DIM = 512
FEATURE_DTYPE = np.dtype(np.float32)
DEFAULT_CHECKPOINT_EVERY_ROWS = 2048
FEATURE_SCOPES = ("short", "full")
FEATURE_CONTROL_IDS = ("V0", "V1", "V2")
SOURCE_INDEX_FIELDS = (
    "image_context_id",
    "packed_row",
    "lineage_scope",
    "human_review_complete",
)
FEATURE_INDEX_FIELDS = (
    "image_context_id",
    "feature_row",
    "control_id",
    "backbone_name",
    "pretrained_weight_enum",
    "image_size",
    "feature_dim",
    "feature_dtype",
    "lineage_scope",
    "human_review_complete",
)
FEATURE_REGISTRY_FIELDS = (
    "registry_schema_version",
    "run_id",
    "experiment_name",
    "execution_mode",
    "scope",
    "control_id",
    "seed",
    "status",
    "failure_reason",
    "code_sha",
    "dirty_worktree",
    "config_hash",
    "dataset_snapshot_hash",
    "cache_hash",
    "fold_manifest_hash",
    "feature_whitelist_hash",
    "backbone_name",
    "pretrained_weight_enum",
    "resolution",
    "frame_batch_size",
    "precision",
    "gpu_model",
    "gpu_vram_bytes",
    "python_version",
    "torch_version",
    "torchvision_version",
    "runtime_seconds",
    "peak_vram_bytes",
    "feature_tensor_path",
    "feature_tensor_sha256",
    "feature_index_path",
    "feature_index_sha256",
    "checkpoint_manifest_path",
    "prediction_manifest_path",
    "metric_path",
    "run_manifest_path",
    "run_manifest_sha256",
    "completed_at_utc",
)


@dataclass(frozen=True, slots=True)
class FeatureCacheSource:
    """One packed RGB source bound to immutable parent evidence."""

    scope: str
    root: Path
    tensor_path: Path
    index_path: Path
    rows: int
    image_size: int
    tensor_sha256: str
    index_sha256: str
    parent_audit_hashes: dict[str, str]


def resolve_legacy_l5_feature_source(
    config: LegacyL5Config,
    *,
    control: LegacyVisualProbeControl,
    scope: str,
    full_cache: dict[str, Any],
    readiness: dict[str, Any],
    short_cache: dict[str, Any],
) -> FeatureCacheSource:
    """Resolve exact short/full packed inputs without source-media fallback."""

    if scope not in FEATURE_SCOPES:
        raise ValueError(f"unsupported legacy L5 feature scope: {scope}")
    if scope == "full":
        lineage = _cache_lineage(
            config,
            image_size=control.image_size,
            full_cache=full_cache,
            readiness=readiness,
        )
        tensor_path = Path(str(lineage["packed_tensor_path"]))
        index_path = Path(str(lineage["packed_index_path"]))
        source = FeatureCacheSource(
            scope=scope,
            root=tensor_path.parent,
            tensor_path=tensor_path,
            index_path=index_path,
            rows=int(lineage["expected_rows"]),
            image_size=control.image_size,
            tensor_sha256=str(lineage["packed_tensor_sha256"]),
            index_sha256=str(lineage["packed_index_sha256"]),
            parent_audit_hashes={
                "full_cache": str(full_cache["_audit_sha256"]),
                "readiness": str(readiness["_audit_sha256"]),
            },
        )
        _validate_source_files(source, rehash=False)
        return source
    expected_rows = int(config.payload["cache_contract"]["short_context_rows"])
    if control.image_size == 224:
        root = Path(str(short_cache["cache_root"]))
        hashes = _object(short_cache["cache_artifact_hashes"], "short hashes")
        source = FeatureCacheSource(
            scope=scope,
            root=root,
            tensor_path=root / "packed_rgb_224_letterbox.npy",
            index_path=root / "packed_image_cache_index.csv",
            rows=expected_rows,
            image_size=224,
            tensor_sha256=str(hashes["packed_tensor"]),
            index_sha256=str(hashes["packed_index"]),
            parent_audit_hashes={
                "short_cache": str(short_cache["_audit_sha256"]),
            },
        )
        _validate_source_files(source, rehash=True)
        return source
    short_reference = config.short_cache_224_reference_root
    if short_reference.name != "09_actor_cache_224":
        raise ValueError("legacy L5 short 224 reference name drift")
    root = short_reference.parent / "09_actor_cache_160"
    cache_audit_path = root / "cache_audit.json"
    packed_audit_path = root / "packed_image_cache_audit.json"
    cache_audit = _read_json(cache_audit_path)
    packed_audit = _read_json(packed_audit_path)
    _validate_short_160_audits(
        cache_audit,
        packed_audit,
        root=root,
        expected_rows=expected_rows,
    )
    source = FeatureCacheSource(
        scope=scope,
        root=root,
        tensor_path=root / "packed_rgb_160_letterbox.npy",
        index_path=root / "packed_image_cache_index.csv",
        rows=expected_rows,
        image_size=160,
        tensor_sha256=file_sha256(root / "packed_rgb_160_letterbox.npy"),
        index_sha256=file_sha256(root / "packed_image_cache_index.csv"),
        parent_audit_hashes={
            "cache_audit": file_sha256(cache_audit_path),
            "packed_audit": file_sha256(packed_audit_path),
        },
    )
    _validate_source_files(source, rehash=False)
    return source


def _validate_short_160_audits(
    cache_audit: dict[str, Any],
    packed_audit: dict[str, Any],
    *,
    root: Path,
    expected_rows: int,
) -> None:
    expected_cache = {
        "output_dir": str(root),
        "image_size": 160,
        "selected_context_rows": expected_rows,
        "manifest_rows": expected_rows,
        "missing_context_rows": 0,
        "duplicate_context_rows": 0,
        "failed_rows": 0,
        "resize_policy": "letterbox_preserve_aspect_rgb_pad_black_v1",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "valid": True,
    }
    expected_packed = {
        "shape": [expected_rows, 160, 160, 3],
        "dtype": "uint8",
        "source_rows": expected_rows,
        "packed_rows": expected_rows,
        "index_rows": expected_rows,
        "failed_rows": 0,
        "duplicate_index_ids": 0,
        "verification_mismatches": 0,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "valid": True,
    }
    _require_mapping(cache_audit, expected_cache, "short 160 cache")
    _require_mapping(packed_audit, expected_packed, "short 160 packed")


def _validate_source_files(
    source: FeatureCacheSource,
    *,
    rehash: bool,
) -> None:
    if not source.tensor_path.is_file() or not source.index_path.is_file():
        raise FileNotFoundError("legacy L5 packed feature source is incomplete")
    tensor = np.load(source.tensor_path, mmap_mode="r")
    try:
        expected = (source.rows, source.image_size, source.image_size, 3)
        if tuple(tensor.shape) != expected or tensor.dtype != np.uint8:
            raise ValueError(
                "legacy L5 feature source tensor drift: "
                f"shape={tuple(tensor.shape)},dtype={tensor.dtype}"
            )
    finally:
        _close_memmap(tensor)
    if rehash:
        if file_sha256(source.tensor_path) != source.tensor_sha256:
            raise ValueError("legacy L5 feature source tensor hash drift")
        if file_sha256(source.index_path) != source.index_sha256:
            raise ValueError("legacy L5 feature source index hash drift")


def load_legacy_l5_feature_parents(
    config: LegacyL5Config,
    *,
    readiness_audit_path: Path,
    short_cache_audit_path: Path,
    full_cache_audit_path: Path,
    weights_audit_path: Path,
    vram_probe_audit_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load and validate every parent before creating a run directory."""

    readiness = _read_json(readiness_audit_path)
    short_cache = _read_json(short_cache_audit_path)
    full_cache = _read_json(full_cache_audit_path)
    weights = _read_json(weights_audit_path)
    vram = _read_json(vram_probe_audit_path)
    _validate_readiness_parent(config, readiness)
    _validate_short_cache_parent(config, short_cache)
    _validate_full_cache_parent(config, full_cache)
    _validate_weights_parent(config, weights)
    _validate_vram_parent(config, vram)
    if _live_weight_errors(weights):
        raise ValueError("legacy L5 live pretrained-weight hash drift")
    paths = {
        "readiness": readiness_audit_path,
        "short_cache": short_cache_audit_path,
        "full_cache": full_cache_audit_path,
        "weights": weights_audit_path,
        "vram": vram_probe_audit_path,
    }
    parents = {
        "readiness": readiness,
        "short_cache": short_cache,
        "full_cache": full_cache,
        "weights": weights,
        "vram": vram,
    }
    for name, payload in parents.items():
        payload["_audit_path"] = str(paths[name])
        payload["_audit_sha256"] = file_sha256(paths[name])
    return parents


def audit_legacy_l5_feature_preflight(
    config: LegacyL5Config,
    *,
    parents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve all six cache sources without creating a CUDA context."""

    cuda_initialized_before = bool(torch.cuda.is_initialized())
    errors = ["cuda_initialized_before_feature_preflight"] if (
        cuda_initialized_before
    ) else []
    sources: dict[str, dict[str, Any]] = {}
    for control in legacy_l5_visual_probe_controls(config):
        for scope in FEATURE_SCOPES:
            key = f"{control.control_id}_{scope}"
            try:
                source = resolve_legacy_l5_feature_source(
                    config,
                    control=control,
                    scope=scope,
                    full_cache=parents["full_cache"],
                    readiness=parents["readiness"],
                    short_cache=parents["short_cache"],
                )
                sources[key] = {
                    "control_id": control.control_id,
                    "scope": scope,
                    "image_size": source.image_size,
                    "rows": source.rows,
                    "tensor_path": str(source.tensor_path),
                    "tensor_sha256": source.tensor_sha256,
                    "index_path": str(source.index_path),
                    "index_sha256": source.index_sha256,
                    "valid": True,
                }
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"{key}:{type(error).__name__}:{error}")
                sources[key] = {
                    "control_id": control.control_id,
                    "scope": scope,
                    "errors": [str(error)],
                    "valid": False,
                }
    cuda_initialized_after = bool(torch.cuda.is_initialized())
    if cuda_initialized_after:
        errors.append("cuda_initialized_during_feature_preflight")
    valid = not errors
    return {
        "schema_version": FEATURE_PREFLIGHT_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_FEATURE_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_FEATURE_PREFLIGHT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_sha256": config.sha256,
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": cuda_initialized_after,
        "sources": sources,
        "short_feature_cache_runs_authorized": valid,
        "full_feature_cache_expansion_authorized": False,
        "errors": errors,
        "valid": valid,
    }


def _validate_short_cache_parent(
    config: LegacyL5Config,
    short_cache: dict[str, Any],
) -> None:
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHE_SHORT",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "mode": "short",
        "config_sha256": config.sha256,
        "full_224_cache_build_authorized": True,
        "valid": True,
    }
    _require_mapping(short_cache, expected, "short cache parent")
    if Path(str(short_cache["cache_root"])).resolve() != (
        config.short_cache_224_root.resolve()
    ):
        raise ValueError("legacy L5 short feature-cache root drift")


def _validate_vram_parent(
    config: LegacyL5Config,
    vram: dict[str, Any],
) -> None:
    expected = {
        "schema_version": L5_VRAM_PROBE_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_VRAM_PROBE",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_sha256": config.sha256,
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "feature_cache_expansion_authorized": True,
        "valid": True,
    }
    _require_mapping(vram, expected, "VRAM parent")


def _require_mapping(
    payload: dict[str, Any],
    expected: dict[str, Any],
    name: str,
) -> None:
    errors = [
        f"{field}:{payload.get(field)!r}!={value!r}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]
    if payload.get("errors"):
        errors.append(f"declared_errors={payload['errors']}")
    if errors:
        raise ValueError(f"legacy L5 {name} mismatch: {errors}")


def build_legacy_l5_feature_cache(
    config: LegacyL5Config,
    *,
    control_id: str,
    scope: str,
    run_id: str,
    output_dir: Path,
    parents: dict[str, dict[str, Any]],
    device_name: str,
    checkpoint_every_rows: int = DEFAULT_CHECKPOINT_EVERY_ROWS,
    resume: bool = False,
    short_gate_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Build one isolated float32 cache with fail-closed resume lineage."""

    if checkpoint_every_rows <= 0:
        raise ValueError("legacy L5 feature checkpoint interval must be positive")
    controls = {
        control.control_id: control
        for control in legacy_l5_visual_probe_controls(config)
    }
    if control_id not in controls:
        raise ValueError(f"unknown legacy L5 GPU control: {control_id}")
    control = controls[control_id]
    if checkpoint_every_rows % control.frame_batch_size != 0:
        raise ValueError(
            "legacy L5 feature checkpoint interval must be batch-aligned"
        )
    if scope not in FEATURE_SCOPES:
        raise ValueError(f"unsupported legacy L5 feature scope: {scope}")
    output = output_dir.resolve()
    if not output.is_relative_to(config.l5_output_root.resolve()):
        raise ValueError("legacy L5 feature run escaped its output root")
    if not run_id.strip() or Path(run_id).name != run_id:
        raise ValueError("legacy L5 feature run_id must be one path-safe name")
    if output.name != run_id:
        raise ValueError("legacy L5 feature output directory must equal run_id")
    readiness = parents["readiness"]
    source = resolve_legacy_l5_feature_source(
        config,
        control=control,
        scope=scope,
        full_cache=parents["full_cache"],
        readiness=readiness,
        short_cache=parents["short_cache"],
    )
    short_gate = _load_short_gate_parent(
        config,
        scope=scope,
        path=short_gate_audit_path,
    )
    semantic = _feature_run_semantic_identity(
        config,
        control=control,
        scope=scope,
        run_id=run_id,
        output_dir=output,
        source=source,
        parents=parents,
        readiness=readiness,
        short_gate=short_gate,
        checkpoint_every_rows=checkpoint_every_rows,
    )
    run_manifest = {
        "schema_version": FEATURE_RUN_MANIFEST_SCHEMA_VERSION,
        **semantic,
        "semantic_identity_sha256": _canonical_json_sha256(semantic),
        "status": "planned",
        "created_at_utc": _utc_now(),
    }
    paths = _feature_run_paths(output)
    if resume:
        stored_manifest = _read_json(paths["run_manifest"])
        stored_semantic = {
            field: stored_manifest.get(field) for field in semantic
        }
        if stored_semantic != semantic:
            raise ValueError("legacy L5 feature resume semantic drift")
        run_manifest = stored_manifest
    start_row = _prepare_feature_run(
        paths,
        run_manifest=run_manifest,
        source=source,
        resume=resume,
    )
    try:
        return _execute_feature_run(
            config,
            control=control,
            scope=scope,
            source=source,
            parents=parents,
            short_gate=short_gate,
            paths=paths,
            run_manifest=run_manifest,
            start_row=start_row,
            checkpoint_every_rows=checkpoint_every_rows,
            device_name=device_name,
            resumed=resume,
        )
    except Exception as error:
        _write_json_atomic(
            paths["unexpected_failure"],
            {
                "schema_version": (
                    "classification_v2.legacy_development_l5."
                    "feature_unexpected_failure.v1"
                ),
                "run_id": run_id,
                "semantic_identity_sha256": run_manifest[
                    "semantic_identity_sha256"
                ],
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "failure_hidden": False,
                "recorded_at_utc": _utc_now(),
            },
        )
        raise


def _feature_run_semantic_identity(
    config: LegacyL5Config,
    *,
    control: LegacyVisualProbeControl,
    scope: str,
    run_id: str,
    output_dir: Path,
    source: FeatureCacheSource,
    parents: dict[str, dict[str, Any]],
    readiness: dict[str, Any],
    short_gate: dict[str, Any] | None,
    checkpoint_every_rows: int,
) -> dict[str, Any]:
    state = git_state()
    input_artifacts = _object(
        readiness["input_hash_audit"]["artifacts"],
        "readiness input artifacts",
    )
    fold_artifact = _object(input_artifacts["window_folds"], "window folds")
    feature_artifact = _object(
        input_artifacts["feature_contract"],
        "feature contract",
    )
    weight = _weight_report(parents["weights"], control)
    vram_control = _vram_control_report(parents["vram"], control)
    semantic = {
        "run_id": run_id,
        "execution_mode": "local_smoke",
        "experiment_name": "legacy_l5_pretrained_frame_feature_cache",
        "code_sha": str(state["commit"]),
        "dirty_worktree": bool(state["dirty"]),
        "dirty_entries": list(state["dirty_entries"]),
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "config_path": str(config.path),
        "config_hash": config.sha256,
        "dataset_snapshot_path": str(readiness["l3_audit_path"]),
        "dataset_snapshot_hash": str(readiness["l3_audit_sha256"]),
        "cache_hash": source.tensor_sha256,
        "fold_manifest_path": str(fold_artifact["path"]),
        "fold_manifest_hash": str(fold_artifact["sha256"]),
        "feature_whitelist_path": str(feature_artifact["path"]),
        "feature_whitelist_hash": str(feature_artifact["sha256"]),
        "fold": "not_applicable_frame_feature_cache",
        "seed": int(config.payload["optimization"]["seeds"][0]),
        "scope": scope,
        "control_id": control.control_id,
        "backbone_name": control.backbone_name,
        "pretrained_weight_enum": control.pretrained_weight_enum,
        "pretrained_weight_path": str(weight["cache_path"]),
        "pretrained_weight_sha256": str(weight["sha256"]),
        "image_size": control.image_size,
        "frame_batch_size": control.frame_batch_size,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": str(FEATURE_DTYPE),
        "declared_gpu_vram_gib": int(
            config.payload["optimization"]["declared_local_gpu_vram_gib"]
        ),
        "maximum_peak_vram_fraction": float(
            config.payload["optimization"]["maximum_peak_vram_fraction"]
        ),
        "normalization_name": visual_backbone_contract(
            control.backbone_name,
            control.pretrained_weight_enum,
        ).normalization_name,
        "source_tensor_path": str(source.tensor_path),
        "source_tensor_sha256": source.tensor_sha256,
        "source_index_path": str(source.index_path),
        "source_index_sha256": source.index_sha256,
        "source_rows": source.rows,
        "source_parent_audit_hashes": dict(source.parent_audit_hashes),
        "readiness_audit_path": str(readiness["_audit_path"]),
        "readiness_audit_sha256": str(readiness["_audit_sha256"]),
        "short_cache_audit_path": str(parents["short_cache"]["_audit_path"]),
        "short_cache_audit_sha256": str(
            parents["short_cache"]["_audit_sha256"]
        ),
        "full_cache_audit_path": str(parents["full_cache"]["_audit_path"]),
        "full_cache_audit_sha256": str(
            parents["full_cache"]["_audit_sha256"]
        ),
        "weights_audit_path": str(parents["weights"]["_audit_path"]),
        "weights_audit_sha256": str(parents["weights"]["_audit_sha256"]),
        "vram_probe_audit_path": str(parents["vram"]["_audit_path"]),
        "vram_probe_audit_sha256": str(parents["vram"]["_audit_sha256"]),
        "vram_probe_feature_sha256": str(
            vram_control["repeat_pass_1"]["feature_sha256"]
        ),
        "short_gate_audit_sha256": (
            str(short_gate["_audit_sha256"])
            if short_gate is not None
            else None
        ),
        "short_gate_audit_path": (
            str(short_gate["_audit_path"])
            if short_gate is not None
            else None
        ),
        "gpu_model": str(parents["vram"]["device_name"]),
        "gpu_vram_bytes": int(parents["vram"]["actual_total_vram_bytes"]),
        "checkpoint_every_rows": checkpoint_every_rows,
        "working_set_release_policy": (
            "flush_close_reopen_input_output_each_checkpoint_v1"
        ),
        "output_dir": str(output_dir),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "oom_retry_allowed": False,
    }
    paired_semantic = {
        key: value
        for key, value in semantic.items()
        if key not in {"run_id", "output_dir"}
    }
    semantic["scientific_identity_sha256"] = _canonical_json_sha256(
        paired_semantic
    )
    return semantic


def _weight_report(
    weights: dict[str, Any],
    control: LegacyVisualProbeControl,
) -> dict[str, Any]:
    artifacts = _object(weights["artifacts"], "pretrained artifacts")
    report = _object(
        artifacts[control.pretrained_weight_enum],
        control.pretrained_weight_enum,
    )
    expected = {
        "backbone_name": control.backbone_name,
        "pretrained_weight_enum": control.pretrained_weight_enum,
        "output_dim": FEATURE_DIM,
        "valid": True,
    }
    _require_mapping(report, expected, "pretrained weight control")
    return report


def _vram_control_report(
    vram: dict[str, Any],
    control: LegacyVisualProbeControl,
) -> dict[str, Any]:
    controls = _object(vram["controls"], "VRAM controls")
    report = _object(controls[control.control_id], control.control_id)
    expected = {
        "backbone_name": control.backbone_name,
        "pretrained_weight_enum": control.pretrained_weight_enum,
        "image_size": control.image_size,
        "frame_batch_size": control.frame_batch_size,
        "oom": False,
        "oom_retry_count": 0,
        "valid": True,
    }
    _require_mapping(report, expected, "VRAM control")
    return report


def _load_short_gate_parent(
    config: LegacyL5Config,
    *,
    scope: str,
    path: Path | None,
) -> dict[str, Any] | None:
    if scope == "short":
        if path is not None:
            raise ValueError("short feature runs do not accept a short gate")
        return None
    if path is None:
        raise ValueError("full feature runs require the exact short gate")
    payload = _read_json(path)
    expected = {
        "schema_version": FEATURE_SHORT_GATE_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_FEATURE_SHORT_GATE",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "config_sha256": config.sha256,
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "full_feature_cache_expansion_authorized": True,
        "valid": True,
    }
    _require_mapping(payload, expected, "feature short gate")
    payload["_audit_path"] = str(path)
    payload["_audit_sha256"] = file_sha256(path)
    return payload


def _prepare_feature_run(
    paths: dict[str, Path],
    *,
    run_manifest: dict[str, Any],
    source: FeatureCacheSource,
    resume: bool,
) -> int:
    output_dir = paths["root"]
    if resume:
        if not output_dir.is_dir():
            raise FileNotFoundError("legacy L5 resume run directory is missing")
        if paths["unexpected_failure"].exists():
            raise ValueError("legacy L5 failed run cannot be resumed implicitly")
        stored = _read_json(paths["run_manifest"])
        if stored != run_manifest:
            raise ValueError("legacy L5 feature resume manifest drift")
        progress = _read_json(paths["progress"])
        expected = {
            "schema_version": FEATURE_PROGRESS_SCHEMA_VERSION,
            "semantic_identity_sha256": run_manifest[
                "semantic_identity_sha256"
            ],
            "planned_run_manifest_sha256": file_sha256(
                paths["run_manifest"]
            ),
            "source_tensor_sha256": source.tensor_sha256,
            "source_index_sha256": source.index_sha256,
            "expected_rows": source.rows,
            "feature_dim": FEATURE_DIM,
            "feature_dtype": str(FEATURE_DTYPE),
        }
        _require_mapping(progress, expected, "feature resume progress")
        if paths["run_result"].exists():
            raise FileExistsError("legacy L5 feature run is already complete")
        if progress.get("status") not in {"planned", "running", "complete"}:
            raise ValueError("legacy L5 failed progress cannot be resumed")
        completed_rows = int(progress["completed_rows"])
        if not 0 <= completed_rows <= source.rows:
            raise ValueError("legacy L5 resume row is outside source bounds")
        output = np.load(paths["feature_tensor"], mmap_mode="r")
        try:
            _validate_feature_output_mapping(output, source.rows)
        finally:
            _close_memmap(output)
        return completed_rows
    if output_dir.exists():
        raise FileExistsError(f"legacy L5 feature run already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json_exclusive(paths["run_manifest"], run_manifest)
    _write_json_exclusive(paths["environment"], _environment_payload(run_manifest))
    feature_tensor = np.lib.format.open_memmap(
        paths["feature_tensor"],
        mode="w+",
        dtype=FEATURE_DTYPE,
        shape=(source.rows, FEATURE_DIM),
    )
    feature_tensor.flush()
    _close_memmap(feature_tensor)
    _write_progress(
        paths,
        run_manifest=run_manifest,
        source=source,
        completed_rows=0,
        input_mapping_open_count=0,
        output_mapping_open_count=1,
        status="planned",
    )
    return 0


def _execute_feature_run(
    config: LegacyL5Config,
    *,
    control: LegacyVisualProbeControl,
    scope: str,
    source: FeatureCacheSource,
    parents: dict[str, dict[str, Any]],
    short_gate: dict[str, Any] | None,
    paths: dict[str, Path],
    run_manifest: dict[str, Any],
    start_row: int,
    checkpoint_every_rows: int,
    device_name: str,
    resumed: bool,
) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("legacy L5 feature cache requires available CUDA")
    device_index = (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    actual_total = int(properties.total_memory)
    free_before, mem_info_total = (
        int(value) for value in torch.cuda.mem_get_info(device)
    )
    optimization = _object(config.payload["optimization"], "optimization")
    declared_gib = int(optimization["declared_local_gpu_vram_gib"])
    maximum_fraction = float(optimization["maximum_peak_vram_fraction"])
    budget_bytes = _vram_budget_bytes(
        declared_bytes=declared_gib * 1024**3,
        actual_total_bytes=actual_total,
        maximum_fraction=maximum_fraction,
    )
    allocator_fraction = budget_bytes / actual_total
    errors = _device_preflight_errors(
        declared_gib=declared_gib,
        actual_total_bytes=actual_total,
        mem_info_total_bytes=mem_info_total,
        free_bytes=free_before,
        budget_bytes=budget_bytes,
    )
    vram_parent = parents["vram"]
    if int(vram_parent["actual_total_vram_bytes"]) != actual_total:
        errors.append("feature_cache_gpu_total_differs_from_vram_probe")
    progress = _read_json(paths["progress"])
    input_mapping_open_count = int(progress["input_mapping_open_count"])
    output_mapping_open_count = int(progress["output_mapping_open_count"])
    completed_rows = start_row
    nonfinite_values = 0
    peak_allocated = 0
    peak_reserved = 0
    oom = False
    oom_message: str | None = None
    encoder: nn.Module | None = None
    input_tensor: np.ndarray | None = None
    output_tensor: np.ndarray | None = None
    started = time.perf_counter()
    if not errors:
        torch.cuda.set_per_process_memory_fraction(allocator_fraction, device)
        _seed_cuda(int(run_manifest["seed"]))
        torch.hub.set_dir(str(parents["weights"]["torch_hub_dir"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            encoder, contract = build_visual_frame_encoder(
                control.backbone_name,
                control.pretrained_weight_enum,
            )
            encoder.eval()
            encoder.to(device)
            input_tensor = np.load(source.tensor_path, mmap_mode="r")
            input_mapping_open_count += 1
            output_tensor = np.load(paths["feature_tensor"], mmap_mode="r+")
            output_mapping_open_count += 1
            _validate_feature_output_mapping(output_tensor, source.rows)
            with torch.inference_mode():
                for start in range(
                    start_row,
                    source.rows,
                    control.frame_batch_size,
                ):
                    end = min(start + control.frame_batch_size, source.rows)
                    images = np.array(
                        input_tensor[start:end],
                        dtype=np.uint8,
                        copy=True,
                    )
                    features = _encode_feature_batch(
                        encoder,
                        images=images,
                        control=control,
                        device=device,
                    )
                    if features.shape != (end - start, contract.output_dim):
                        raise RuntimeError(
                            "legacy L5 feature batch shape drift: "
                            f"{features.shape}"
                        )
                    nonfinite_values += int((~np.isfinite(features)).sum())
                    output_tensor[start:end] = features
                    completed_rows = end
                    del images, features
                    if (
                        completed_rows % checkpoint_every_rows == 0
                        or completed_rows == source.rows
                    ):
                        input_tensor, output_tensor = _checkpoint_mappings(
                            input_tensor,
                            output_tensor,
                        )
                        _write_progress(
                            paths,
                            run_manifest=run_manifest,
                            source=source,
                            completed_rows=completed_rows,
                            input_mapping_open_count=input_mapping_open_count,
                            output_mapping_open_count=output_mapping_open_count,
                            status=(
                                "complete"
                                if completed_rows == source.rows
                                else "running"
                            ),
                        )
                        if completed_rows < source.rows:
                            input_tensor = np.load(
                                source.tensor_path,
                                mmap_mode="r",
                            )
                            output_tensor = np.load(
                                paths["feature_tensor"],
                                mmap_mode="r+",
                            )
                            input_mapping_open_count += 1
                            output_mapping_open_count += 1
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError as error:
            oom = True
            oom_message = str(error)
            errors.append("cuda_out_of_memory_no_retry")
        finally:
            if input_tensor is not None:
                _close_memmap(input_tensor)
            if output_tensor is not None:
                output_tensor.flush()
                _close_memmap(output_tensor)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            if encoder is not None:
                encoder.to("cpu")
            del encoder
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    post_allocated = int(torch.cuda.memory_allocated(device))
    post_reserved = int(torch.cuda.memory_reserved(device))
    if completed_rows != source.rows:
        errors.append(f"incomplete_feature_rows={completed_rows}!={source.rows}")
    if nonfinite_values:
        errors.append(f"nonfinite_feature_values={nonfinite_values}")
    if peak_allocated > budget_bytes:
        errors.append("feature_peak_allocated_exceeds_allocator_limit")
    if peak_reserved > budget_bytes:
        errors.append("feature_peak_reserved_exceeds_allocator_limit")
    if post_allocated != 0 or post_reserved != 0:
        errors.append("feature_cuda_memory_not_released")
    if errors:
        _write_progress(
            paths,
            run_manifest=run_manifest,
            source=source,
            completed_rows=completed_rows,
            input_mapping_open_count=input_mapping_open_count,
            output_mapping_open_count=output_mapping_open_count,
            status="failed",
        )
    feature_sha256: str | None = None
    index_sha256: str | None = None
    vram_sample_equivalence: dict[str, Any] | None = None
    if not errors:
        if paths["feature_index"].exists():
            if not resumed:
                raise FileExistsError(
                    "legacy L5 feature index appeared during a fresh run"
                )
            _validate_feature_index_contract(
                paths["feature_index"],
                control=control,
                expected_rows=source.rows,
            )
        else:
            _write_feature_index(
                source.index_path,
                paths["feature_index"],
                control=control,
                expected_rows=source.rows,
            )
        feature_sha256 = file_sha256(paths["feature_tensor"])
        index_sha256 = file_sha256(paths["feature_index"])
        if scope == "full":
            vram_sample_equivalence = _vram_sample_equivalence(
                paths["feature_tensor"],
                parents["vram"],
                control=control,
            )
            if not vram_sample_equivalence["valid"]:
                errors.extend(vram_sample_equivalence["errors"])
    valid = not errors
    result = {
        "schema_version": FEATURE_RUN_RESULT_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_FEATURE_CACHE"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_FEATURE_CACHE"
        ),
        "run_id": run_manifest["run_id"],
        "semantic_identity_sha256": run_manifest[
            "semantic_identity_sha256"
        ],
        "scientific_identity_sha256": run_manifest[
            "scientific_identity_sha256"
        ],
        "planned_run_manifest_sha256": file_sha256(paths["run_manifest"]),
        "environment_sha256": file_sha256(paths["environment"]),
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "config_sha256": config.sha256,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "scope": scope,
        "control_id": control.control_id,
        "backbone_name": control.backbone_name,
        "pretrained_weight_enum": control.pretrained_weight_enum,
        "image_size": control.image_size,
        "frame_batch_size": control.frame_batch_size,
        "source_rows": source.rows,
        "completed_rows": completed_rows,
        "feature_shape": [source.rows, FEATURE_DIM],
        "feature_dtype": str(FEATURE_DTYPE),
        "feature_tensor_path": str(paths["feature_tensor"]),
        "feature_tensor_sha256": feature_sha256,
        "feature_index_path": str(paths["feature_index"]),
        "feature_index_sha256": index_sha256,
        "source_tensor_sha256": source.tensor_sha256,
        "source_index_sha256": source.index_sha256,
        "short_gate_audit_sha256": (
            str(short_gate["_audit_sha256"])
            if short_gate is not None
            else None
        ),
        "input_mapping_open_count": input_mapping_open_count,
        "output_mapping_open_count": output_mapping_open_count,
        "working_set_release_policy": (
            "flush_close_reopen_input_output_each_checkpoint_v1"
        ),
        "device": str(device),
        "device_name": str(properties.name),
        "actual_total_vram_bytes": actual_total,
        "free_vram_before_bytes": free_before,
        "allocator_limit_bytes": budget_bytes,
        "configured_allocator_fraction": allocator_fraction,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "post_cleanup_allocated_bytes": post_allocated,
        "post_cleanup_reserved_bytes": post_reserved,
        "precision": "float32",
        "autocast_enabled": False,
        "gradient_enabled": False,
        "oom": oom,
        "oom_message": oom_message,
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "nonfinite_feature_values": nonfinite_values,
        "vram_probe_sample_equivalence": vram_sample_equivalence,
        "source_media_loads": 0,
        "video_decode_count": 0,
        "video_seek_count": 0,
        "resumed": resumed,
        "runtime_sec": float(time.perf_counter() - started),
        "short_repeat_gate_eligible": scope == "short" and valid,
        "full_control_complete": scope == "full" and valid,
        "baseline_metrics_authorized": False,
        "accuracy_f1_computed": False,
        "optimizer_steps": 0,
        "errors": errors,
        "valid": valid,
    }
    _finalize_feature_run(paths, result=result)
    return result


def _encode_feature_batch(
    encoder: nn.Module,
    *,
    images: np.ndarray,
    control: LegacyVisualProbeControl,
    device: torch.device,
) -> np.ndarray:
    contract = visual_backbone_contract(
        control.backbone_name,
        control.pretrained_weight_enum,
    )
    raw = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    batch = raw.to(torch.float32) / 255.0
    mean = torch.tensor(contract.input_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(contract.input_std, dtype=torch.float32).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    features = encoder(batch.to(device, non_blocking=False))
    output = features.detach().to("cpu").contiguous().numpy()
    output = output.astype(FEATURE_DTYPE, copy=False)
    del raw, batch, features
    return output


def _checkpoint_mappings(
    input_tensor: np.ndarray,
    output_tensor: np.ndarray,
) -> tuple[None, None]:
    output_tensor.flush()
    _close_memmap(input_tensor)
    _close_memmap(output_tensor)
    return None, None


def _validate_feature_output_mapping(
    output: np.ndarray,
    expected_rows: int,
) -> None:
    if tuple(output.shape) != (expected_rows, FEATURE_DIM):
        raise ValueError(f"legacy L5 output feature shape drift: {output.shape}")
    if output.dtype != FEATURE_DTYPE:
        raise ValueError(f"legacy L5 output feature dtype drift: {output.dtype}")


def _vram_sample_equivalence(
    feature_path: Path,
    vram: dict[str, Any],
    *,
    control: LegacyVisualProbeControl,
) -> dict[str, Any]:
    report = _vram_control_report(vram, control)
    rows = np.asarray(report["sample_packed_rows"], dtype=np.int64)
    features = np.load(feature_path, mmap_mode="r")
    try:
        sample = np.array(features[rows], dtype=FEATURE_DTYPE, copy=True)
    finally:
        _close_memmap(features)
    observed = hashlib.sha256(sample.tobytes()).hexdigest()
    expected = str(report["repeat_pass_1"]["feature_sha256"])
    errors = [] if observed == expected else [
        f"vram_probe_feature_sha256={observed}!={expected}"
    ]
    return {
        "sample_rows": rows.tolist(),
        "observed_feature_sha256": observed,
        "expected_feature_sha256": expected,
        "errors": errors,
        "valid": not errors,
    }


def _feature_run_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "run_manifest": root / "run_manifest.json",
        "environment": root / "environment.json",
        "progress": root / "feature_progress.json",
        "feature_tensor": root / "frame_features_f32.npy",
        "feature_index": root / "frame_feature_index.csv",
        "run_result": root / "run_result.json",
        "artifact_manifest": root / "artifact_manifest.json",
        "checkpoint_manifest": root / "checkpoint_manifest.json",
        "prediction_manifest": root / "prediction_manifest.json",
        "registry_entry": root / "registry_entry.json",
        "runs_registry": root / "runs_registry.csv",
        "unexpected_failure": root / "unexpected_failure.json",
    }


def _environment_payload(run_manifest: dict[str, Any]) -> dict[str, Any]:
    cudnn_version = torch.backends.cudnn.version()
    return {
        "schema_version": FEATURE_ENVIRONMENT_SCHEMA_VERSION,
        "captured_at_utc": _utc_now(),
        "execution_mode": run_manifest["execution_mode"],
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "gpu_inventory_source": "validated_vram_probe_parent",
        "gpu_model": run_manifest["gpu_model"],
        "gpu_vram_bytes": run_manifest["gpu_vram_bytes"],
        "cuda_runtime_probe_deferred_until_allocator_gate": True,
        "declared_gpu_vram_gib": run_manifest["declared_gpu_vram_gib"],
        "maximum_peak_vram_fraction": run_manifest[
            "maximum_peak_vram_fraction"
        ],
        "precision": "float32",
        "autocast_enabled": False,
        "deterministic_algorithms_required": True,
        "oom_retry_allowed": False,
    }


def _write_progress(
    paths: dict[str, Path],
    *,
    run_manifest: dict[str, Any],
    source: FeatureCacheSource,
    completed_rows: int,
    input_mapping_open_count: int,
    output_mapping_open_count: int,
    status: str,
) -> None:
    payload = {
        "schema_version": FEATURE_PROGRESS_SCHEMA_VERSION,
        "run_id": run_manifest["run_id"],
        "semantic_identity_sha256": run_manifest[
            "semantic_identity_sha256"
        ],
        "planned_run_manifest_sha256": file_sha256(paths["run_manifest"]),
        "source_tensor_sha256": source.tensor_sha256,
        "source_index_sha256": source.index_sha256,
        "expected_rows": source.rows,
        "completed_rows": completed_rows,
        "feature_dim": FEATURE_DIM,
        "feature_dtype": str(FEATURE_DTYPE),
        "input_mapping_open_count": input_mapping_open_count,
        "output_mapping_open_count": output_mapping_open_count,
        "source_media_loads": 0,
        "video_decode_count": 0,
        "video_seek_count": 0,
        "status": status,
        "updated_at_utc": _utc_now(),
    }
    _write_json_atomic(paths["progress"], payload)


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


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def _write_feature_index(
    source_path: Path,
    output_path: Path,
    *,
    control: LegacyVisualProbeControl,
    expected_rows: int,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"legacy L5 feature index exists: {output_path}")
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    seen_ids: set[str] = set()
    rows = 0
    try:
        with source_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != SOURCE_INDEX_FIELDS:
                raise ValueError("legacy L5 source packed-index schema drift")
            with temporary.open("x", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=list(FEATURE_INDEX_FIELDS),
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in reader:
                    _write_feature_index_row(
                        writer,
                        row,
                        row_number=rows,
                        seen_ids=seen_ids,
                        control=control,
                    )
                    rows += 1
                target.flush()
                os.fsync(target.fileno())
        if rows != expected_rows:
            raise ValueError(
                f"legacy L5 feature index rows={rows}!={expected_rows}"
            )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_feature_index_row(
    writer: csv.DictWriter,
    row: dict[str, str],
    *,
    row_number: int,
    seen_ids: set[str],
    control: LegacyVisualProbeControl,
) -> None:
    context_id = str(row["image_context_id"])
    if not context_id or context_id in seen_ids:
        raise ValueError("legacy L5 source index has blank or duplicate ID")
    if int(row["packed_row"]) != row_number:
        raise ValueError("legacy L5 source packed rows are not contiguous")
    if row["lineage_scope"] != LINEAGE_SCOPE:
        raise ValueError("legacy L5 source index lineage scope drift")
    if str(row["human_review_complete"]).lower() != "false":
        raise ValueError("legacy L5 source index exceeds review boundary")
    seen_ids.add(context_id)
    writer.writerow(
        {
            "image_context_id": context_id,
            "feature_row": row_number,
            "control_id": control.control_id,
            "backbone_name": control.backbone_name,
            "pretrained_weight_enum": control.pretrained_weight_enum,
            "image_size": control.image_size,
            "feature_dim": FEATURE_DIM,
            "feature_dtype": str(FEATURE_DTYPE),
            "lineage_scope": LINEAGE_SCOPE,
            "human_review_complete": False,
        }
    )


def _finalize_feature_run(
    paths: dict[str, Path],
    *,
    result: dict[str, Any],
) -> None:
    terminal_status = "completed" if result["valid"] else "failed"
    failure_reason = ";".join(str(value) for value in result["errors"])
    _write_json_exclusive(paths["run_result"], result)
    _write_json_exclusive(
        paths["checkpoint_manifest"],
        _empty_checkpoint_manifest(
            result,
            status=terminal_status,
            failure_reason=failure_reason,
        ),
    )
    _write_json_exclusive(
        paths["prediction_manifest"],
        _empty_prediction_manifest(
            result,
            status=terminal_status,
            failure_reason=failure_reason,
        ),
    )
    planned = _read_json(paths["run_manifest"])
    if file_sha256(paths["run_manifest"]) != result[
        "planned_run_manifest_sha256"
    ]:
        raise ValueError("legacy L5 planned run manifest changed during execution")
    artifacts = _feature_artifact_records(paths, planned=planned, result=result)
    _write_json_exclusive(
        paths["artifact_manifest"],
        {
            "schema_version": FEATURE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "run_id": result["run_id"],
            "semantic_identity_sha256": result[
                "semantic_identity_sha256"
            ],
            "status": terminal_status,
            "artifacts": artifacts,
            "failure_reason": failure_reason,
        },
    )
    completed_at = _utc_now()
    final_manifest = {
        **planned,
        "status": terminal_status,
        "completed_at_utc": completed_at,
        "runtime_seconds": result["runtime_sec"],
        "peak_vram_bytes": result["peak_reserved_bytes"],
        "failure_reason": failure_reason,
        "resumed": result["resumed"],
        "planned_run_manifest_sha256": result[
            "planned_run_manifest_sha256"
        ],
        "run_result_sha256": file_sha256(paths["run_result"]),
        "artifact_manifest_sha256": file_sha256(paths["artifact_manifest"]),
        "checkpoint_manifest_sha256": file_sha256(
            paths["checkpoint_manifest"]
        ),
        "prediction_manifest_sha256": file_sha256(
            paths["prediction_manifest"]
        ),
        "runs_registry_path": str(paths["runs_registry"]),
    }
    _write_json_atomic(paths["run_manifest"], final_manifest)
    entry = _feature_registry_entry(
        paths,
        run_manifest=final_manifest,
        result=result,
        failure_reason=failure_reason,
        completed_at=completed_at,
    )
    _write_json_exclusive(paths["registry_entry"], entry)
    _write_feature_registry(paths["runs_registry"], entry)


def _empty_checkpoint_manifest(
    result: dict[str, Any],
    *,
    status: str,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "run_id": result["run_id"],
        "semantic_identity_sha256": result["semantic_identity_sha256"],
        "status": status,
        "checkpoints": [],
        "checkpoint_creation_authorized": False,
        "reason": "deterministic_frozen_frame_feature_cache_only",
        "failure_reason": failure_reason,
    }


def _empty_prediction_manifest(
    result: dict[str, Any],
    *,
    status: str,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_PREDICTION_MANIFEST_SCHEMA_VERSION,
        "run_id": result["run_id"],
        "semantic_identity_sha256": result["semantic_identity_sha256"],
        "status": status,
        "predictions": [],
        "prediction_creation_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "reason": "frame_feature_cache_has_no_predictions",
        "failure_reason": failure_reason,
    }


def _feature_artifact_records(
    paths: dict[str, Path],
    *,
    planned: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    records = [
        _declared_artifact(name, path, digest, direction="input", kind=kind)
        for name, path, digest, kind in _feature_input_specs(planned)
    ]
    output_specs: list[tuple[str, Path, str | None, str]] = [
        (
            "feature_tensor",
            paths["feature_tensor"],
            result["feature_tensor_sha256"],
            "float32_frame_features",
        ),
        (
            "environment",
            paths["environment"],
            result["environment_sha256"],
            "environment",
        ),
        ("progress", paths["progress"], None, "resume_progress"),
        ("run_result", paths["run_result"], None, "terminal_result"),
        (
            "checkpoint_manifest",
            paths["checkpoint_manifest"],
            None,
            "empty_checkpoint_manifest",
        ),
        (
            "prediction_manifest",
            paths["prediction_manifest"],
            None,
            "empty_prediction_manifest",
        ),
    ]
    if paths["feature_index"].is_file():
        output_specs.append(
            (
                "feature_index",
                paths["feature_index"],
                result["feature_index_sha256"],
                "feature_index",
            )
        )
    records.extend(
        _declared_artifact(
            name,
            path,
            digest or file_sha256(path),
            direction="output",
            kind=kind,
        )
        for name, path, digest, kind in output_specs
    )
    return records


def _feature_input_specs(
    planned: dict[str, Any],
) -> list[tuple[str, Path, str, str]]:
    fields = [
        ("config", "config_path", "config_hash", "semantic_config"),
        (
            "dataset_snapshot",
            "dataset_snapshot_path",
            "dataset_snapshot_hash",
            "dataset_snapshot_audit",
        ),
        ("fold_manifest", "fold_manifest_path", "fold_manifest_hash", "folds"),
        (
            "feature_whitelist",
            "feature_whitelist_path",
            "feature_whitelist_hash",
            "feature_contract",
        ),
        ("source_tensor", "source_tensor_path", "source_tensor_sha256", "rgb"),
        ("source_index", "source_index_path", "source_index_sha256", "index"),
        (
            "pretrained_weight",
            "pretrained_weight_path",
            "pretrained_weight_sha256",
            "model_weight",
        ),
        ("readiness_audit", "readiness_audit_path", "readiness_audit_sha256", "audit"),
        ("short_cache_audit", "short_cache_audit_path", "short_cache_audit_sha256", "audit"),
        ("full_cache_audit", "full_cache_audit_path", "full_cache_audit_sha256", "audit"),
        ("weights_audit", "weights_audit_path", "weights_audit_sha256", "audit"),
        ("vram_probe_audit", "vram_probe_audit_path", "vram_probe_audit_sha256", "audit"),
    ]
    if planned.get("short_gate_audit_path"):
        fields.append(
            (
                "short_gate_audit",
                "short_gate_audit_path",
                "short_gate_audit_sha256",
                "authorization_gate",
            )
        )
    return [
        (name, Path(str(planned[path_field])), str(planned[hash_field]), kind)
        for name, path_field, hash_field, kind in fields
    ]


def _declared_artifact(
    name: str,
    path: Path,
    digest: str,
    *,
    direction: str,
    kind: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"legacy L5 lineage artifact missing: {name}")
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError(f"legacy L5 lineage artifact hash invalid: {name}")
    return {
        "name": name,
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": int(resolved.stat().st_size),
        "direction": direction,
        "kind": kind,
        "hash_source": (
            "computed_from_output"
            if direction == "output"
            else "validated_parent_lineage"
        ),
    }


def _feature_registry_entry(
    paths: dict[str, Path],
    *,
    run_manifest: dict[str, Any],
    result: dict[str, Any],
    failure_reason: str,
    completed_at: str,
) -> dict[str, Any]:
    environment = _read_json(paths["environment"])
    status = "completed" if result["valid"] else "failed"
    return {
        "registry_schema_version": FEATURE_REGISTRY_SCHEMA_VERSION,
        "run_id": result["run_id"],
        "experiment_name": run_manifest["experiment_name"],
        "execution_mode": run_manifest["execution_mode"],
        "scope": result["scope"],
        "control_id": result["control_id"],
        "seed": run_manifest["seed"],
        "status": status,
        "failure_reason": failure_reason,
        "code_sha": run_manifest["code_sha"],
        "dirty_worktree": run_manifest["dirty_worktree"],
        "config_hash": run_manifest["config_hash"],
        "dataset_snapshot_hash": run_manifest["dataset_snapshot_hash"],
        "cache_hash": run_manifest["cache_hash"],
        "fold_manifest_hash": run_manifest["fold_manifest_hash"],
        "feature_whitelist_hash": run_manifest["feature_whitelist_hash"],
        "backbone_name": result["backbone_name"],
        "pretrained_weight_enum": result["pretrained_weight_enum"],
        "resolution": result["image_size"],
        "frame_batch_size": result["frame_batch_size"],
        "precision": result["precision"],
        "gpu_model": result["device_name"],
        "gpu_vram_bytes": result["actual_total_vram_bytes"],
        "python_version": environment["python_version"],
        "torch_version": environment["torch_version"],
        "torchvision_version": environment["torchvision_version"],
        "runtime_seconds": result["runtime_sec"],
        "peak_vram_bytes": result["peak_reserved_bytes"],
        "feature_tensor_path": str(paths["feature_tensor"].resolve()),
        "feature_tensor_sha256": result["feature_tensor_sha256"] or "",
        "feature_index_path": (
            str(paths["feature_index"].resolve())
            if paths["feature_index"].is_file()
            else ""
        ),
        "feature_index_sha256": result["feature_index_sha256"] or "",
        "checkpoint_manifest_path": str(
            paths["checkpoint_manifest"].resolve()
        ),
        "prediction_manifest_path": str(
            paths["prediction_manifest"].resolve()
        ),
        "metric_path": "",
        "run_manifest_path": str(paths["run_manifest"].resolve()),
        "run_manifest_sha256": file_sha256(paths["run_manifest"]),
        "completed_at_utc": completed_at,
    }


def _write_feature_registry(path: Path, entry: dict[str, Any]) -> None:
    if tuple(entry) != FEATURE_REGISTRY_FIELDS:
        raise ValueError("legacy L5 feature registry entry schema drift")
    if path.exists():
        raise FileExistsError(f"legacy L5 feature registry exists: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(FEATURE_REGISTRY_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(entry)
        handle.flush()
        os.fsync(handle.fileno())


def audit_legacy_l5_feature_short_gate(
    config: LegacyL5Config,
    *,
    primary_result_paths: dict[str, Path],
    repeat_result_paths: dict[str, Path],
) -> dict[str, Any]:
    """Compare six isolated short caches before any full extraction."""

    if tuple(primary_result_paths) != FEATURE_CONTROL_IDS:
        raise ValueError("legacy L5 primary short-result control order drift")
    if tuple(repeat_result_paths) != FEATURE_CONTROL_IDS:
        raise ValueError("legacy L5 repeat short-result control order drift")
    controls = {
        control.control_id: control
        for control in legacy_l5_visual_probe_controls(config)
    }
    reports: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for control_id in FEATURE_CONTROL_IDS:
        try:
            report = _compare_short_feature_pair(
                config,
                control=controls[control_id],
                primary_path=primary_result_paths[control_id],
                repeat_path=repeat_result_paths[control_id],
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            report = {
                "control_id": control_id,
                "errors": [f"{type(error).__name__}: {error}"],
                "valid": False,
            }
        reports[control_id] = report
        errors.extend(
            f"{control_id}:{message}" for message in report["errors"]
        )
    sequence = _short_run_sequence_audit(reports)
    errors.extend(sequence["errors"])
    valid = not errors
    state = git_state()
    return {
        "schema_version": FEATURE_SHORT_GATE_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_FEATURE_SHORT_GATE"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_FEATURE_SHORT_GATE"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "git_state": state,
        "expected_controls": list(FEATURE_CONTROL_IDS),
        "short_rows_per_control": int(
            config.payload["cache_contract"]["short_context_rows"]
        ),
        "controls": reports,
        "sequential_execution_audit": sequence,
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "controls_run_in_isolated_processes_required": True,
        "full_feature_cache_expansion_authorized": valid,
        "baseline_metrics_authorized": False,
        "accuracy_f1_computed": False,
        "errors": errors,
        "valid": valid,
    }


def _compare_short_feature_pair(
    config: LegacyL5Config,
    *,
    control: LegacyVisualProbeControl,
    primary_path: Path,
    repeat_path: Path,
) -> dict[str, Any]:
    primary = _load_short_feature_run(
        config,
        control=control,
        result_path=primary_path,
    )
    repeat = _load_short_feature_run(
        config,
        control=control,
        result_path=repeat_path,
    )
    errors: list[str] = []
    if primary["run_id"] == repeat["run_id"]:
        errors.append("primary_and_repeat_run_id_match")
    if primary["run_root"] == repeat["run_root"]:
        errors.append("primary_and_repeat_run_root_match")
    compared = (
        "scientific_identity_sha256",
        "source_tensor_sha256",
        "source_index_sha256",
        "feature_tensor_sha256",
        "feature_index_sha256",
    )
    exact_matches = {
        field: primary[field] == repeat[field] for field in compared
    }
    errors.extend(
        f"primary_repeat_{field}_mismatch"
        for field, matches in exact_matches.items()
        if not matches
    )
    return {
        "control_id": control.control_id,
        "primary": primary,
        "repeat": repeat,
        "exact_matches": exact_matches,
        "feature_bytes_identical": exact_matches["feature_tensor_sha256"],
        "index_bytes_identical": exact_matches["feature_index_sha256"],
        "post_cleanup_vram_zero": (
            primary["post_cleanup_vram_zero"]
            and repeat["post_cleanup_vram_zero"]
        ),
        "errors": errors,
        "valid": not errors,
    }


def _load_short_feature_run(
    config: LegacyL5Config,
    *,
    control: LegacyVisualProbeControl,
    result_path: Path,
) -> dict[str, Any]:
    result = _read_json(result_path)
    expected_rows = int(config.payload["cache_contract"]["short_context_rows"])
    expected = {
        "schema_version": FEATURE_RUN_RESULT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_FEATURE_CACHE",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "scope": "short",
        "control_id": control.control_id,
        "backbone_name": control.backbone_name,
        "pretrained_weight_enum": control.pretrained_weight_enum,
        "image_size": control.image_size,
        "frame_batch_size": control.frame_batch_size,
        "source_rows": expected_rows,
        "completed_rows": expected_rows,
        "feature_shape": [expected_rows, FEATURE_DIM],
        "feature_dtype": str(FEATURE_DTYPE),
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "config_sha256": config.sha256,
        "oom": False,
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "nonfinite_feature_values": 0,
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "source_media_loads": 0,
        "video_decode_count": 0,
        "video_seek_count": 0,
        "short_repeat_gate_eligible": True,
        "accuracy_f1_computed": False,
        "optimizer_steps": 0,
        "valid": True,
    }
    _require_mapping(result, expected, "short feature result")
    if not result_path.resolve().is_relative_to(config.l5_output_root.resolve()):
        raise ValueError("legacy L5 short result escaped its L5 output root")
    return _validate_short_feature_artifacts(
        result_path,
        result=result,
        control=control,
    )


def _validate_short_feature_artifacts(
    result_path: Path,
    *,
    result: dict[str, Any],
    control: LegacyVisualProbeControl,
) -> dict[str, Any]:
    root = result_path.parent.resolve()
    paths = _feature_run_paths(root)
    if result_path.resolve() != paths["run_result"]:
        raise ValueError("legacy L5 short result filename drift")
    tensor_path = Path(str(result["feature_tensor_path"])).resolve()
    index_path = Path(str(result["feature_index_path"])).resolve()
    if tensor_path != paths["feature_tensor"]:
        raise ValueError("legacy L5 short feature tensor path drift")
    if index_path != paths["feature_index"]:
        raise ValueError("legacy L5 short feature index path drift")
    if file_sha256(tensor_path) != result["feature_tensor_sha256"]:
        raise ValueError("legacy L5 short feature tensor hash drift")
    if file_sha256(index_path) != result["feature_index_sha256"]:
        raise ValueError("legacy L5 short feature index hash drift")
    manifest = _read_json(paths["run_manifest"])
    manifest_expected = {
        "run_id": result["run_id"],
        "status": "completed",
        "scope": "short",
        "control_id": control.control_id,
        "scientific_identity_sha256": result[
            "scientific_identity_sha256"
        ],
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "oom_retry_allowed": False,
    }
    _require_mapping(manifest, manifest_expected, "short run manifest")
    _validate_short_lineage_files(paths, manifest=manifest, result=result)
    _validate_feature_index_contract(
        index_path,
        control=control,
        expected_rows=int(result["source_rows"]),
    )
    return {
        "run_id": result["run_id"],
        "run_root": str(root),
        "result_path": str(result_path.resolve()),
        "result_sha256": file_sha256(result_path),
        "created_at_utc": manifest["created_at_utc"],
        "completed_at_utc": manifest["completed_at_utc"],
        "scientific_identity_sha256": result[
            "scientific_identity_sha256"
        ],
        "source_tensor_sha256": result["source_tensor_sha256"],
        "source_index_sha256": result["source_index_sha256"],
        "feature_tensor_sha256": result["feature_tensor_sha256"],
        "feature_index_sha256": result["feature_index_sha256"],
        "feature_rows": result["completed_rows"],
        "peak_reserved_bytes": result["peak_reserved_bytes"],
        "allocator_limit_bytes": result["allocator_limit_bytes"],
        "post_cleanup_vram_zero": True,
        "oom": False,
        "oom_retry_count": 0,
    }


def _validate_short_lineage_files(
    paths: dict[str, Path],
    *,
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> None:
    hashes = {
        "run_result": ("run_result_sha256", paths["run_result"]),
        "artifacts": ("artifact_manifest_sha256", paths["artifact_manifest"]),
        "checkpoints": (
            "checkpoint_manifest_sha256",
            paths["checkpoint_manifest"],
        ),
        "predictions": (
            "prediction_manifest_sha256",
            paths["prediction_manifest"],
        ),
    }
    for name, (field, path) in hashes.items():
        if file_sha256(path) != manifest.get(field):
            raise ValueError(f"legacy L5 short {name} manifest hash drift")
    environment = _read_json(paths["environment"])
    _require_mapping(
        environment,
        {
            "schema_version": FEATURE_ENVIRONMENT_SCHEMA_VERSION,
            "oom_retry_allowed": False,
        },
        "short environment",
    )
    artifact_manifest = _read_json(paths["artifact_manifest"])
    _require_mapping(
        artifact_manifest,
        {
            "schema_version": FEATURE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "run_id": result["run_id"],
            "status": "completed",
        },
        "short artifact manifest",
    )
    artifact_names = {
        str(item.get("name"))
        for item in artifact_manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    required_names = {
        "config",
        "source_tensor",
        "source_index",
        "pretrained_weight",
        "feature_tensor",
        "feature_index",
        "environment",
        "progress",
        "run_result",
        "checkpoint_manifest",
        "prediction_manifest",
    }
    if not required_names.issubset(artifact_names):
        raise ValueError("legacy L5 short artifact manifest is incomplete")
    _validate_empty_output_manifest(
        paths["checkpoint_manifest"],
        schema=FEATURE_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        field="checkpoints",
        run_id=str(result["run_id"]),
    )
    _validate_empty_output_manifest(
        paths["prediction_manifest"],
        schema=FEATURE_PREDICTION_MANIFEST_SCHEMA_VERSION,
        field="predictions",
        run_id=str(result["run_id"]),
    )
    _validate_single_registry_row(paths["runs_registry"], result=result)


def _validate_empty_output_manifest(
    path: Path,
    *,
    schema: str,
    field: str,
    run_id: str,
) -> None:
    payload = _read_json(path)
    expected = {
        "schema_version": schema,
        "run_id": run_id,
        "status": "completed",
        field: [],
    }
    _require_mapping(payload, expected, f"short empty {field} manifest")


def _validate_single_registry_row(
    path: Path,
    *,
    result: dict[str, Any],
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FEATURE_REGISTRY_FIELDS:
            raise ValueError("legacy L5 short feature registry schema drift")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError("legacy L5 short feature registry row count drift")
    expected = {
        "registry_schema_version": FEATURE_REGISTRY_SCHEMA_VERSION,
        "run_id": str(result["run_id"]),
        "status": "completed",
        "scope": "short",
        "control_id": str(result["control_id"]),
        "feature_tensor_sha256": str(result["feature_tensor_sha256"]),
        "feature_index_sha256": str(result["feature_index_sha256"]),
    }
    errors = [
        f"{field}:{rows[0].get(field)!r}!={value!r}"
        for field, value in expected.items()
        if rows[0].get(field) != value
    ]
    if errors:
        raise ValueError(f"legacy L5 short feature registry mismatch: {errors}")


def _validate_feature_index_contract(
    path: Path,
    *,
    control: LegacyVisualProbeControl,
    expected_rows: int,
) -> None:
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FEATURE_INDEX_FIELDS:
            raise ValueError("legacy L5 feature-index schema drift")
        for row in reader:
            expected = {
                "feature_row": str(rows),
                "control_id": control.control_id,
                "backbone_name": control.backbone_name,
                "pretrained_weight_enum": control.pretrained_weight_enum,
                "image_size": str(control.image_size),
                "feature_dim": str(FEATURE_DIM),
                "feature_dtype": str(FEATURE_DTYPE),
                "lineage_scope": LINEAGE_SCOPE,
                "human_review_complete": "False",
            }
            if any(row.get(field) != value for field, value in expected.items()):
                raise ValueError("legacy L5 feature-index row contract drift")
            if not str(row.get("image_context_id", "")):
                raise ValueError("legacy L5 feature-index has blank context ID")
            rows += 1
    if rows != expected_rows:
        raise ValueError(
            f"legacy L5 feature-index rows={rows}!={expected_rows}"
        )


def _short_run_sequence_audit(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    entries: list[dict[str, Any]] = []
    for control_id in FEATURE_CONTROL_IDS:
        report = reports.get(control_id, {})
        if report.get("valid") is not True:
            continue
        for role in ("primary", "repeat"):
            run = _object(report[role], f"short {control_id} {role}")
            try:
                created = datetime.fromisoformat(str(run["created_at_utc"]))
                completed = datetime.fromisoformat(
                    str(run["completed_at_utc"])
                )
            except ValueError:
                errors.append(f"invalid_short_run_timestamp={control_id}:{role}")
                continue
            if completed < created:
                errors.append(f"negative_short_run_interval={control_id}:{role}")
            entries.append(
                {
                    "control_id": control_id,
                    "role": role,
                    "run_id": run["run_id"],
                    "created_at_utc": run["created_at_utc"],
                    "completed_at_utc": run["completed_at_utc"],
                    "created": created,
                    "completed": completed,
                    "post_cleanup_vram_zero": run[
                        "post_cleanup_vram_zero"
                    ],
                }
            )
    if len(entries) != 2 * len(FEATURE_CONTROL_IDS):
        errors.append(f"short_run_interval_count={len(entries)}!=6")
    run_ids = [str(entry["run_id"]) for entry in entries]
    if len(run_ids) != len(set(run_ids)):
        errors.append("short_run_ids_are_not_unique")
    ordered = sorted(entries, key=lambda value: value["created"])
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current["created"] < previous["completed"]:
            errors.append(
                "short_run_intervals_overlap="
                f"{previous['run_id']}:{current['run_id']}"
            )
    public_order = [
        {
            key: value
            for key, value in entry.items()
            if key not in {"created", "completed"}
        }
        for entry in ordered
    ]
    all_zero = bool(entries) and all(
        bool(entry["post_cleanup_vram_zero"]) for entry in entries
    )
    if not all_zero:
        errors.append("short_run_post_cleanup_vram_not_zero")
    return {
        "required_execution": "one_control_per_process_sequential_v1",
        "execution_order": public_order,
        "run_count": len(entries),
        "interval_overlap_count": sum(
            message.startswith("short_run_intervals_overlap=")
            for message in errors
        ),
        "all_post_cleanup_vram_zero": all_zero,
        "errors": errors,
        "valid": not errors,
    }


def write_legacy_l5_feature_short_gate(
    config: LegacyL5Config,
    *,
    output_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write one immutable short gate inside the scoped L5 output root."""

    resolved = output_path.resolve()
    if not resolved.is_relative_to(config.l5_output_root.resolve()):
        raise ValueError("legacy L5 feature short gate escaped its output root")
    expected = {
        "schema_version": FEATURE_SHORT_GATE_SCHEMA_VERSION,
        "config_sha256": config.sha256,
        "implementation_source_sha256": file_sha256(Path(__file__)),
    }
    mismatches = [
        field
        for field, value in expected.items()
        if payload.get(field) != value
    ]
    if mismatches:
        raise ValueError(
            f"legacy L5 feature short gate output drift: {mismatches}"
        )
    _write_json_exclusive(resolved, payload)


__all__ = [
    "DEFAULT_CHECKPOINT_EVERY_ROWS",
    "FEATURE_CONTROL_IDS",
    "FEATURE_RUN_RESULT_SCHEMA_VERSION",
    "FEATURE_SHORT_GATE_SCHEMA_VERSION",
    "FeatureCacheSource",
    "audit_legacy_l5_feature_preflight",
    "audit_legacy_l5_feature_short_gate",
    "build_legacy_l5_feature_cache",
    "load_legacy_l5_feature_parents",
    "resolve_legacy_l5_feature_source",
    "write_legacy_l5_feature_short_gate",
]
