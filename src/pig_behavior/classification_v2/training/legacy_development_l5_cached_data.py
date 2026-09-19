"""Leakage-safe cached-frame sequences for legacy L5 development."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.models.temporal_encoders import (
    build_temporal_encoder,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5 import (
    LINEAGE_SCOPE,
    LegacyL5Config,
    git_state,
)
from pig_behavior.classification_v2.training.legacy_development_l5_feature_cache import (
    FEATURE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    FEATURE_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    FEATURE_CONTROL_IDS,
    FEATURE_DIM,
    FEATURE_DTYPE,
    FEATURE_ENVIRONMENT_SCHEMA_VERSION,
    FEATURE_INDEX_FIELDS,
    FEATURE_PREDICTION_MANIFEST_SCHEMA_VERSION,
    FEATURE_RUN_MANIFEST_SCHEMA_VERSION,
    FEATURE_RUN_RESULT_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
)

CACHED_DATA_AUDIT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_data_audit.v1"
)
CACHED_DATA_MANIFEST_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_data_manifest.v1"
)
CACHED_DATA_ENVIRONMENT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_data_environment.v1"
)
CACHED_DATA_ARTIFACT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_data_artifacts.v1"
)
CACHED_DATA_CHECKPOINT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_data_checkpoints.v1"
)
CACHED_DATA_PREDICTION_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_data_predictions.v1"
)
CACHED_DATA_REGISTRY_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_data_registry.v1"
)
FEATURE_WHITELIST_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_feature_whitelist.v1"
)
FEATURE_BLACKLIST_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.cached_feature_blacklist.v1"
)
MODEL_ACCESS_ROLES = ("train", "validation")
ALL_ROUTING_ROLES = (
    "train",
    "validation",
    "outer_holdout",
    "policy_invalid",
)
ROUTING_ONLY_FIELDS = (
    "window_id",
    "temporal_unit_key",
    "recording_group_id",
    "video_key",
    "source_type",
    "dataset_id",
    "behavior_label",
    "oof_fold_id",
    "l5_role",
    "image_context_id",
    "feature_row",
    "slot_index",
    "item_order",
)
FORBIDDEN_X_PATTERNS = (
    "*behavior*",
    "*label*",
    "manual_*",
    "review_*",
    "*corrected*",
    "target_*",
    "*policy*",
    "*_path",
    "*fold*",
    "source_type",
    "video_key",
    "dataset_id",
    "pig_id",
    "track_id",
    "review_unit_id",
    "window_id",
    "temporal_unit_key",
    "frame_uid",
)
REGISTRY_FIELDS = (
    "registry_schema_version",
    "run_id",
    "experiment_name",
    "execution_mode",
    "status",
    "code_sha",
    "dirty_worktree",
    "config_hash",
    "dataset_snapshot_hash",
    "cache_hash",
    "fold_manifest_hash",
    "feature_whitelist_hash",
    "control_id",
    "temporal_view_name",
    "sequence_length",
    "train_native_units",
    "validation_native_units",
    "outer_holdout_native_units",
    "source_media_reads",
    "outer_predictions_created",
    "runtime_seconds",
    "peak_vram_bytes",
    "manifest_path",
    "manifest_sha256",
    "completed_at_utc",
)
DECLARED_LOCAL_GPU_VRAM_GIB = 4
VALIDATED_LOCAL_GPU_VRAM_BYTES = 4_294_443_008
GPU_ALLOCATOR_FRACTION_CEILING = 0.7
GPU_ALLOCATOR_LIMIT_BYTES = 3_006_110_105
MAX_CACHED_AUDIT_BATCH_SIZE = 256
MAX_CACHED_AUDIT_BATCHES_PER_ROLE = 2


@dataclass(frozen=True, slots=True)
class LegacyL5CachedFeatureView:
    """Train/validation-only frame-feature routing plus immutable audit data."""

    feature_tensor_path: Path
    feature_tensor_sha256: str
    control_id: str
    temporal_view_name: str
    sequence_length: int
    windows: pd.DataFrame
    fold_manifest: pd.DataFrame
    feature_rows: np.ndarray
    observed_mask: np.ndarray
    time_delta: np.ndarray
    targets: np.ndarray
    sample_weights: np.ndarray
    audit: dict[str, Any]

    def indices_for_role(self, role: str) -> np.ndarray:
        """Return positions for one model-visible role and reject outer access."""

        if role not in MODEL_ACCESS_ROLES:
            raise ValueError(f"cached feature model access forbidden for role={role}")
        values = self.windows["l5_role"].astype(str).to_numpy()
        return np.flatnonzero(values == role).astype(np.int64, copy=False)

    def load_sequences(self, positions: np.ndarray) -> np.ndarray:
        """Copy one bounded batch and close its mmap before returning."""

        indices = np.asarray(positions, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("cached feature positions must be one-dimensional")
        if len(indices) == 0:
            return np.empty(
                (0, self.sequence_length, FEATURE_DIM),
                dtype=FEATURE_DTYPE,
            )
        if indices.min() < 0 or indices.max() >= len(self.windows):
            raise IndexError("cached feature positions are out of bounds")
        mapping = np.load(self.feature_tensor_path, mmap_mode="r")
        try:
            values = np.asarray(
                mapping[self.feature_rows[indices]],
                dtype=FEATURE_DTYPE,
            ).copy()
        finally:
            _close_memmap(mapping)
        mask = self.observed_mask[indices]
        if not np.isfinite(values[mask]).all():
            raise ValueError("observed cached features contain nonfinite values")
        values[~mask] = 0.0
        return values

    def iter_role_batches(
        self,
        role: str,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield crash-bounded batches without workers, prefetch, or pinning."""

        if batch_size <= 0:
            raise ValueError("cached feature batch size must be positive")
        positions = self.indices_for_role(role).copy()
        if shuffle:
            np.random.default_rng(seed).shuffle(positions)
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start : start + batch_size]
            yield {
                "positions": batch_positions,
                "features": self.load_sequences(batch_positions),
                "observed_mask": self.observed_mask[batch_positions].copy(),
                "time_delta": self.time_delta[batch_positions].copy(),
                "targets": self.targets[batch_positions].copy(),
                "sample_weights": self.sample_weights[batch_positions].copy(),
            }


class LegacyL5CachedFeatureClassifier(nn.Module):
    """Canonical temporal head over frozen 512-dimensional frame features."""

    def __init__(
        self,
        *,
        temporal_encoder_name: str,
        hidden_dim: int,
        dropout: float,
        transformer_layers: int,
        transformer_heads: int,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("cached feature hidden dimension must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("cached feature dropout must be in [0,1)")
        self.input_norm = nn.LayerNorm(FEATURE_DIM)
        self.projection = nn.Linear(FEATURE_DIM, hidden_dim)
        self.temporal_encoder = build_temporal_encoder(
            temporal_encoder_name,
            embedding_dim=hidden_dim,
            dropout=dropout,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.behavior_head = nn.Linear(hidden_dim, len(VALID_BEHAVIORS))

    def forward(
        self,
        features: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        time_delta: torch.Tensor,
    ) -> torch.Tensor:
        if (
            features.ndim != 3
            or observed_mask.ndim != 2
            or features.shape[:2] != observed_mask.shape
            or features.shape[-1] != FEATURE_DIM
        ):
            raise ValueError(
                "cached features/mask must be [B,T,512] and [B,T]"
            )
        if not torch.isfinite(observed_mask).all():
            raise ValueError("cached feature mask contains nonfinite values")
        if not torch.all((observed_mask == 0) | (observed_mask == 1)):
            raise ValueError("cached feature mask must be binary")
        valid = observed_mask.bool()
        if not torch.isfinite(features[valid]).all():
            raise ValueError("observed cached features contain nonfinite values")
        clean = torch.where(
            valid.unsqueeze(-1),
            features,
            torch.zeros_like(features),
        )
        projected = torch.nn.functional.gelu(
            self.projection(self.input_norm(clean))
        )
        pooled = self.temporal_encoder(
            projected,
            observed_mask,
            time_delta=time_delta,
        )
        return self.behavior_head(
            self.dropout(self.output_norm(pooled))
        )


def cached_feature_whitelist_payload() -> dict[str, Any]:
    """Return the only 512 values allowed into model X."""

    features = [f"cached_frame_feature_{index:03d}" for index in range(FEATURE_DIM)]
    return {
        "schema_version": FEATURE_WHITELIST_SCHEMA_VERSION,
        "lineage_scope": LINEAGE_SCOPE,
        "selection_policy": "explicit_512_cached_frame_features_only",
        "features": features,
        "feature_dtype": str(FEATURE_DTYPE),
        "feature_dim": FEATURE_DIM,
        "routing_only_fields": list(ROUTING_ONLY_FIELDS),
        "data_derived_normalization": "none",
        "learned_transform": "train_only_layer_norm_and_linear_projection_v1",
    }


def cached_feature_blacklist_payload() -> dict[str, Any]:
    """Record forbidden names separately from the model-X whitelist."""

    return {
        "schema_version": FEATURE_BLACKLIST_SCHEMA_VERSION,
        "lineage_scope": LINEAGE_SCOPE,
        "forbidden_patterns": list(FORBIDDEN_X_PATTERNS),
        "routing_only_fields": list(ROUTING_ONLY_FIELDS),
        "availability_is_behavior_evidence": False,
        "identifiers_are_model_features": False,
    }


def build_legacy_l5_cached_feature_view(
    config: LegacyL5Config,
    *,
    feature_result_path: Path,
    temporal_view_name: str,
) -> LegacyL5CachedFeatureView:
    """Join one immutable feature packet to train/validation slots only."""

    if temporal_view_name not in LEGACY_TEMPORAL_MODEL_VIEW_SPECS:
        raise ValueError(f"unknown legacy temporal view={temporal_view_name}")
    spec = LEGACY_TEMPORAL_MODEL_VIEW_SPECS[temporal_view_name]
    sequence_length = int(spec["sequence_length"])
    selection_column = str(spec["selection_column"])
    paths = _cached_data_input_paths(
        config,
        feature_result_path=feature_result_path,
        slot_manifest_filename=str(spec["slot_manifest_filename"]),
    )
    feature_parent = _validate_feature_parent(
        config,
        paths=paths,
    )
    native = _read_native_manifest(paths["native_units"])
    folds = _read_fold_manifest(
        paths["window_folds"],
        selection_column=selection_column,
    )
    routing, fold_manifest, role_audit = _build_routing_roles(
        config,
        native=native,
        folds=folds,
        selection_column=selection_column,
        sequence_length=sequence_length,
        sampling_protocol=str(spec["sampling_view"]),
    )
    model_routing = routing.loc[
        routing["l5_role"].isin(MODEL_ACCESS_ROLES)
    ].copy()
    image_windows = _read_model_image_windows(
        paths["image_windows"],
        model_routing=model_routing,
        sequence_length=sequence_length,
    )
    slots = _read_model_slots(
        paths["slot_manifest"],
        model_routing=model_routing,
        temporal_view_name=temporal_view_name,
        sequence_length=sequence_length,
    )
    feature_index = _read_feature_index(
        paths["feature_index"],
        feature_parent=feature_parent,
    )
    joined, join_audit = _join_slots_to_features(
        model_routing=model_routing,
        image_windows=image_windows,
        slots=slots,
        feature_index=feature_index,
        sequence_length=sequence_length,
    )
    windows, feature_rows, mask, time_delta = _reshape_model_view(
        joined,
        model_routing=model_routing,
        sequence_length=sequence_length,
    )
    targets = np.asarray(
        [VALID_BEHAVIORS.index(str(value)) for value in windows["behavior_label"]],
        dtype=np.int64,
    )
    sample_weights, event_mass_audit = _event_mass_weights(windows)
    whitelist = cached_feature_whitelist_payload()
    leakage = _leakage_audit(whitelist)
    inputs = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in paths.items()
        if name not in {"feature_tensor", "feature_index"}
    }
    inputs["feature_tensor"] = {
        "path": str(paths["feature_tensor"]),
        "sha256": feature_parent["feature_tensor_sha256"],
        "size_bytes": int(paths["feature_tensor"].stat().st_size),
    }
    inputs["feature_index"] = {
        "path": str(paths["feature_index"]),
        "sha256": feature_parent["feature_index_sha256"],
        "size_bytes": int(paths["feature_index"].stat().st_size),
    }
    audit = {
        "schema_version": CACHED_DATA_AUDIT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHED_DATA",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "git_state": git_state(),
        "control_id": feature_parent["control_id"],
        "temporal_view_name": temporal_view_name,
        "selection_column": selection_column,
        "sequence_length": sequence_length,
        "model_visible_roles": list(MODEL_ACCESS_ROLES),
        "model_window_rows": int(len(windows)),
        "model_slot_rows": int(len(joined)),
        "feature_dim": FEATURE_DIM,
        "feature_dtype": str(FEATURE_DTYPE),
        "feature_parent": feature_parent,
        "role_audit": role_audit,
        "join_audit": join_audit,
        "event_mass_audit": event_mass_audit,
        "leakage_audit": leakage,
        "temporal_unit_audit": _temporal_unit_audit(
            fold_manifest,
            role_audit=role_audit,
        ),
        "memory_safety": {
            "declared_local_gpu_vram_gib": DECLARED_LOCAL_GPU_VRAM_GIB,
            "validated_local_gpu_vram_bytes": VALIDATED_LOCAL_GPU_VRAM_BYTES,
            "gpu_allocator_fraction_ceiling": (
                GPU_ALLOCATOR_FRACTION_CEILING
            ),
            "gpu_allocator_limit_bytes": GPU_ALLOCATOR_LIMIT_BYTES,
            "feature_tensor_mmap": True,
            "mmap_close_after_each_loaded_batch": True,
            "dataloader_num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "one_control_per_process": True,
            "oom_retry_allowed": False,
        },
        "outer_holdout_access": {
            "routing_rows_audited": int(
                role_audit["window_counts"]["outer_holdout"]
            ),
            "feature_slots_materialized": 0,
            "predictions_created": 0,
            "metrics_computed": 0,
            "access_policy": "FORBIDDEN_DURING_MODEL_SELECTION",
        },
        "source_media_loads": 0,
        "video_decode_count": 0,
        "video_seek_count": 0,
        "bounded_batch_audit": {
            "status": "NOT_RUN",
            "valid": False,
        },
        "inputs": inputs,
        "errors": [],
        "valid": True,
    }
    return LegacyL5CachedFeatureView(
        feature_tensor_path=paths["feature_tensor"],
        feature_tensor_sha256=feature_parent["feature_tensor_sha256"],
        control_id=str(feature_parent["control_id"]),
        temporal_view_name=temporal_view_name,
        sequence_length=sequence_length,
        windows=windows,
        fold_manifest=fold_manifest,
        feature_rows=feature_rows,
        observed_mask=mask,
        time_delta=time_delta,
        targets=targets,
        sample_weights=sample_weights,
        audit=audit,
    )


def _cached_data_input_paths(
    config: LegacyL5Config,
    *,
    feature_result_path: Path,
    slot_manifest_filename: str,
) -> dict[str, Path]:
    result_path = feature_result_path.resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"cached feature result missing: {result_path}")
    result = _read_json(result_path)
    run_dir = result_path.parent
    feature_tensor = Path(str(result.get("feature_tensor_path", ""))).resolve()
    feature_index = Path(str(result.get("feature_index_path", ""))).resolve()
    expected_tensor = (run_dir / "frame_features_f32.npy").resolve()
    expected_index = (run_dir / "frame_feature_index.csv").resolve()
    if feature_tensor != expected_tensor or feature_index != expected_index:
        raise ValueError("cached feature packet paths escape their run directory")
    root = config.primary_root.resolve()
    return {
        "feature_result": result_path,
        "feature_run_manifest": run_dir / "run_manifest.json",
        "feature_artifact_manifest": run_dir / "artifact_manifest.json",
        "feature_checkpoint_manifest": run_dir / "checkpoint_manifest.json",
        "feature_prediction_manifest": run_dir / "prediction_manifest.json",
        "feature_environment": run_dir / "environment.json",
        "feature_tensor": feature_tensor,
        "feature_index": feature_index,
        "native_units": (
            root
            / "06_temporal_tier_contract"
            / "native_temporal_unit_manifest.csv"
        ),
        "window_folds": root / "11_folds" / "window_oof_fold_manifest.csv",
        "image_windows": (
            root / "09_image_context" / "image_window_context_manifest.csv"
        ),
        "slot_manifest": (
            root / "06_temporal_tier_contract" / slot_manifest_filename
        ),
    }


def _validate_feature_parent(
    config: LegacyL5Config,
    *,
    paths: dict[str, Path],
) -> dict[str, Any]:
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"cached data input missing={name}:{path}")
    result = _read_json(paths["feature_result"])
    manifest = _read_json(paths["feature_run_manifest"])
    artifact_manifest = _read_json(paths["feature_artifact_manifest"])
    checkpoint_manifest = _read_json(paths["feature_checkpoint_manifest"])
    prediction_manifest = _read_json(paths["feature_prediction_manifest"])
    environment = _read_json(paths["feature_environment"])
    required_result = {
        "schema_version": FEATURE_RUN_RESULT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_FEATURE_CACHE",
        "scope": "full",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "feature_dtype": str(FEATURE_DTYPE),
        "precision": "float32",
        "autocast_enabled": False,
        "oom_retry_allowed": False,
        "gradient_enabled": False,
        "optimizer_steps": 0,
        "accuracy_f1_computed": False,
        "baseline_metrics_authorized": False,
        "nonfinite_feature_values": 0,
        "video_decode_count": 0,
        "video_seek_count": 0,
        "full_control_complete": True,
        "working_set_release_policy": (
            "flush_close_reopen_input_output_each_checkpoint_v1"
        ),
        "valid": True,
    }
    for name, expected in required_result.items():
        if result.get(name) != expected:
            raise ValueError(
                f"cached feature result mismatch={name}:"
                f"{result.get(name)!r}!={expected!r}"
            )
    if manifest.get("schema_version") != FEATURE_RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("cached feature run-manifest schema mismatch")
    if manifest.get("status") != "completed":
        raise ValueError("cached feature run is not terminal-complete")
    run_id = str(result.get("run_id", ""))
    if not run_id or manifest.get("run_id") != run_id:
        raise ValueError("cached feature run ID differs across packet")
    if result.get("config_sha256") != config.sha256:
        raise ValueError("cached feature result config hash drift")
    if manifest.get("config_hash") != config.sha256:
        raise ValueError("cached feature config hash differs from active L5 config")
    if manifest.get("run_result_sha256") != file_sha256(
        paths["feature_result"]
    ):
        raise ValueError("cached feature run-result hash drift")
    manifest_hash_fields = {
        "artifact_manifest_sha256": "feature_artifact_manifest",
        "checkpoint_manifest_sha256": "feature_checkpoint_manifest",
        "prediction_manifest_sha256": "feature_prediction_manifest",
    }
    for field, path_name in manifest_hash_fields.items():
        if manifest.get(field) != file_sha256(paths[path_name]):
            raise ValueError(f"cached feature parent hash drift={field}")
    packet_contracts = (
        (
            artifact_manifest,
            FEATURE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "artifact manifest",
        ),
        (
            checkpoint_manifest,
            FEATURE_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "checkpoint manifest",
        ),
        (
            prediction_manifest,
            FEATURE_PREDICTION_MANIFEST_SCHEMA_VERSION,
            "prediction manifest",
        ),
        (
            environment,
            FEATURE_ENVIRONMENT_SCHEMA_VERSION,
            "environment",
        ),
    )
    for payload, schema_version, name in packet_contracts:
        if payload.get("schema_version") != schema_version:
            raise ValueError(f"cached feature {name} schema mismatch")
    for payload, name in (
        (artifact_manifest, "artifact manifest"),
        (checkpoint_manifest, "checkpoint manifest"),
        (prediction_manifest, "prediction manifest"),
    ):
        if payload.get("run_id") != run_id or payload.get("status") != "completed":
            raise ValueError(f"cached feature {name} identity/status drift")
    if (
        checkpoint_manifest.get("checkpoints") != []
        or checkpoint_manifest.get("checkpoint_creation_authorized") is not False
    ):
        raise ValueError("cached feature parent contains checkpoint evidence")
    if (
        prediction_manifest.get("predictions") != []
        or prediction_manifest.get("prediction_creation_authorized") is not False
        or prediction_manifest.get("outer_holdout_predictions_authorized")
        is not False
    ):
        raise ValueError("cached feature parent contains prediction evidence")
    environment_hash = file_sha256(paths["feature_environment"])
    if result.get("environment_sha256") != environment_hash:
        raise ValueError("cached feature environment hash drift")
    required_environment = {
        "declared_gpu_vram_gib": DECLARED_LOCAL_GPU_VRAM_GIB,
        "gpu_vram_bytes": VALIDATED_LOCAL_GPU_VRAM_BYTES,
        "maximum_peak_vram_fraction": GPU_ALLOCATOR_FRACTION_CEILING,
        "precision": "float32",
        "autocast_enabled": False,
        "oom_retry_allowed": False,
    }
    for name, expected in required_environment.items():
        if environment.get(name) != expected:
            raise ValueError(
                f"cached feature environment mismatch={name}:"
                f"{environment.get(name)!r}!={expected!r}"
            )
    if (
        int(result.get("actual_total_vram_bytes", -1))
        != VALIDATED_LOCAL_GPU_VRAM_BYTES
        or int(result.get("allocator_limit_bytes", -1))
        != GPU_ALLOCATOR_LIMIT_BYTES
    ):
        raise ValueError("cached feature parent exceeds the frozen 4 GB contract")
    control_id = str(result.get("control_id", ""))
    if control_id not in FEATURE_CONTROL_IDS:
        raise ValueError(f"unsupported cached feature control={control_id}")
    if result.get("control_id") != manifest.get("control_id"):
        raise ValueError("cached feature control differs across packet")
    if result.get("backbone_name") != manifest.get("backbone_name"):
        raise ValueError("cached feature backbone differs across packet")
    if result.get("pretrained_weight_enum") != manifest.get(
        "pretrained_weight_enum"
    ):
        raise ValueError("cached feature pretrained enum differs across packet")
    expected_rows = int(config.payload["expected_counts"]["image_context_rows"])
    if result.get("feature_shape") != [expected_rows, FEATURE_DIM]:
        raise ValueError("cached feature tensor shape contract drift")
    if int(result.get("completed_rows", -1)) != expected_rows:
        raise ValueError("cached feature packet is incomplete")
    if result.get("oom") is not False or int(result.get("oom_retry_count", -1)):
        raise ValueError("cached feature packet contains OOM or retry evidence")
    if int(result.get("source_media_loads", -1)) != 0:
        raise ValueError("cached feature packet used source-media fallback")
    if (
        int(result.get("post_cleanup_allocated_bytes", -1)) != 0
        or int(result.get("post_cleanup_reserved_bytes", -1)) != 0
    ):
        raise ValueError("cached feature packet retained CUDA memory")
    tensor_hash = file_sha256(paths["feature_tensor"])
    index_hash = file_sha256(paths["feature_index"])
    if tensor_hash != result.get("feature_tensor_sha256"):
        raise ValueError("cached feature tensor hash drift")
    if index_hash != result.get("feature_index_sha256"):
        raise ValueError("cached feature index hash drift")
    mapping = np.load(paths["feature_tensor"], mmap_mode="r")
    try:
        shape = tuple(int(value) for value in mapping.shape)
        dtype = mapping.dtype
    finally:
        _close_memmap(mapping)
    if shape != (expected_rows, FEATURE_DIM) or dtype != FEATURE_DTYPE:
        raise ValueError(
            f"cached feature mmap contract drift=shape:{shape},dtype:{dtype}"
        )
    return {
        "run_id": str(result["run_id"]),
        "control_id": control_id,
        "backbone_name": str(result["backbone_name"]),
        "pretrained_weight_enum": str(result["pretrained_weight_enum"]),
        "image_size": int(result["image_size"]),
        "feature_rows": expected_rows,
        "feature_tensor_sha256": tensor_hash,
        "feature_index_sha256": index_hash,
        "feature_run_result_sha256": file_sha256(paths["feature_result"]),
        "feature_run_manifest_sha256": file_sha256(
            paths["feature_run_manifest"]
        ),
        "source_media_loads": 0,
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "errors": [],
        "valid": True,
    }


def _read_native_manifest(path: Path) -> pd.DataFrame:
    columns = [
        "temporal_unit_key",
        "source_type",
        "dataset_id",
        "video_key",
        "label_frame_count",
        "behavior_label",
        "native_unit_valid_for_development",
        "lineage_scope",
        "human_review_complete",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if frame["temporal_unit_key"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("native temporal-unit manifest contains blank keys")
    if frame["temporal_unit_key"].astype(str).duplicated().any():
        raise ValueError("native temporal-unit manifest contains duplicate keys")
    if set(frame["source_type"].astype(str)) != {"legacy_recovered"}:
        raise ValueError("legacy L5 native source type drift")
    if not frame["label_frame_count"].astype(int).eq(16).all():
        raise ValueError("legacy L5 native units are not exact 16-frame bursts")
    if set(frame["behavior_label"].astype(str)) != set(VALID_BEHAVIORS):
        raise ValueError("legacy L5 native behavior support drift")
    _validate_claim_columns(frame, name="native temporal units")
    return frame


def _read_fold_manifest(
    path: Path,
    *,
    selection_column: str,
) -> pd.DataFrame:
    columns = [
        "window_id",
        "temporal_unit_key",
        "window_length_frames",
        "tier_window_valid",
        selection_column,
        "recording_group_id",
        "oof_fold_id",
        "behavior_label",
        "source_type",
        "dataset_id",
        "video_key",
        "lineage_scope",
        "human_review_complete",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    for column in (
        "window_id",
        "temporal_unit_key",
        "recording_group_id",
        "oof_fold_id",
        "behavior_label",
        "source_type",
        "dataset_id",
        "video_key",
    ):
        blank = frame[column].fillna("").astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"window fold manifest blank {column} rows={blank.sum()}")
    if frame["window_id"].astype(str).duplicated().any():
        raise ValueError("window fold manifest contains duplicate window IDs")
    _strict_bool(frame["tier_window_valid"], name="tier_window_valid")
    _strict_bool(frame[selection_column], name=selection_column)
    _validate_claim_columns(frame, name="window folds")
    return frame


def _build_routing_roles(
    config: LegacyL5Config,
    *,
    native: pd.DataFrame,
    folds: pd.DataFrame,
    selection_column: str,
    sequence_length: int,
    sampling_protocol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected = folds.loc[
        _strict_bool(folds[selection_column], name=selection_column)
    ].copy()
    if not selected["window_length_frames"].astype(int).eq(sequence_length).all():
        raise ValueError("selected window length differs from temporal view")
    native_lookup = native[
        [
            "temporal_unit_key",
            "native_unit_valid_for_development",
            "label_frame_count",
            "behavior_label",
            "source_type",
            "video_key",
        ]
    ].rename(
        columns={
            "behavior_label": "native_behavior_label",
            "source_type": "native_source_type",
            "video_key": "native_video_key",
        }
    )
    selected = selected.merge(
        native_lookup,
        on="temporal_unit_key",
        how="left",
        validate="many_to_one",
    )
    if selected["native_unit_valid_for_development"].isna().any():
        raise ValueError("selected windows contain unknown native units")
    mismatch = (
        selected["behavior_label"].astype(str)
        != selected["native_behavior_label"].astype(str)
    )
    mismatch |= (
        selected["source_type"].astype(str)
        != selected["native_source_type"].astype(str)
    )
    mismatch |= (
        selected["video_key"].astype(str)
        != selected["native_video_key"].astype(str)
    )
    if mismatch.any():
        raise ValueError(f"native-to-window routing mismatch rows={mismatch.sum()}")
    valid = _strict_bool(
        selected["native_unit_valid_for_development"],
        name="native_unit_valid_for_development",
    ).to_numpy()
    tier_valid = _strict_bool(
        selected["tier_window_valid"],
        name="selected tier_window_valid",
    ).to_numpy()
    eligible_invalid_tier = valid & ~tier_valid
    if eligible_invalid_tier.any():
        raise ValueError("selected eligible windows include invalid tier rows")
    split = config.payload["split_contract"]
    outer_fold = str(split["outer_holdout_fold_id"])
    validation_fold = str(split["development_validation_fold_id"])
    fold_ids = selected["oof_fold_id"].astype(str).to_numpy()
    roles = np.full(len(selected), "policy_invalid", dtype=object)
    roles[valid & (fold_ids == outer_fold)] = "outer_holdout"
    roles[valid & (fold_ids == validation_fold)] = "validation"
    roles[valid & (fold_ids != outer_fold) & (fold_ids != validation_fold)] = (
        "train"
    )
    selected["l5_role"] = roles
    per_unit = selected.groupby("temporal_unit_key", sort=False)
    consistency_columns = [
        "recording_group_id",
        "oof_fold_id",
        "behavior_label",
        "source_type",
        "dataset_id",
        "video_key",
        "l5_role",
    ]
    for column in consistency_columns:
        conflicts = int(per_unit[column].nunique(dropna=False).gt(1).sum())
        if conflicts:
            raise ValueError(
                f"native routing conflicts={column}:{conflicts}"
            )
    expected_placements = (
        ((16 - sequence_length) // 3) + 1
        if sampling_protocol == "all_sliding_event_balanced"
        else 1
    )
    placements = per_unit.size()
    if not placements.eq(expected_placements).all():
        bad = int(placements.ne(expected_placements).sum())
        raise ValueError(f"temporal placement count mismatch units={bad}")
    fold_manifest = selected.drop_duplicates(
        "temporal_unit_key",
        keep="first",
    )[
        [
            "temporal_unit_key",
            "recording_group_id",
            "video_key",
            "source_type",
            "dataset_id",
            "behavior_label",
            "oof_fold_id",
            "l5_role",
            "label_frame_count",
        ]
    ].reset_index(drop=True)
    fold_manifest["outer_fold_id"] = outer_fold
    fold_manifest["role"] = fold_manifest["l5_role"]
    expected = config.payload["expected_counts"]
    native_counts = {
        role: int(fold_manifest["l5_role"].eq(role).sum())
        for role in ALL_ROUTING_ROLES
    }
    expected_counts = {
        "train": int(expected["train_native_units"]),
        "validation": int(expected["validation_native_units"]),
        "outer_holdout": int(expected["outer_holdout_native_units"]),
        "policy_invalid": int(expected["policy_invalid_native_units"]),
    }
    if native_counts != expected_counts:
        raise ValueError(
            f"cached feature native role counts={native_counts}!={expected_counts}"
        )
    if len(fold_manifest) != int(expected["selected_native_units"]):
        raise ValueError("cached feature selected native-unit count drift")
    overlap = _routing_overlap_audit(fold_manifest)
    if overlap["errors"]:
        raise ValueError(f"cached feature role overlap={overlap['errors']}")
    class_support = _class_support_frame(fold_manifest)
    supported = class_support.loc[
        class_support["l5_role"].isin(
            ("train", "validation", "outer_holdout")
        )
    ]
    if supported["native_units"].le(0).any():
        missing = supported.loc[
            supported["native_units"].le(0),
            ["l5_role", "behavior_label"],
        ].to_dict("records")
        raise ValueError(f"cached feature role class support missing={missing}")
    window_counts = {
        role: int(selected["l5_role"].eq(role).sum())
        for role in ALL_ROUTING_ROLES
    }
    role_hashes = {
        role: _ordered_sha256(
            fold_manifest.loc[
                fold_manifest["l5_role"].eq(role),
                "temporal_unit_key",
            ]
        )
        for role in ALL_ROUTING_ROLES
    }
    audit = {
        "outer_holdout_fold_id": outer_fold,
        "development_validation_fold_id": validation_fold,
        "native_counts": native_counts,
        "window_counts": window_counts,
        "windows_per_native_unit": expected_placements,
        "eligible_invalid_tier_rows": int(eligible_invalid_tier.sum()),
        "policy_invalid_tier_rows": int((~valid & ~tier_valid).sum()),
        "role_unit_sha256": role_hashes,
        "role_overlap": overlap,
        "class_support": class_support.to_dict("records"),
        "outer_holdout_prediction_access": "FORBIDDEN",
        "outer_holdout_metric_access": "FORBIDDEN",
        "errors": [],
        "valid": True,
    }
    return selected, fold_manifest, audit


def _read_model_image_windows(
    path: Path,
    *,
    model_routing: pd.DataFrame,
    sequence_length: int,
) -> pd.DataFrame:
    columns = [
        "window_id",
        "window_length_frames",
        "expected_frame_indices",
        "image_context_id_sequence",
        "observed_image_context_rows",
        "loadable_image_context_rows",
        "missing_image_context_slots",
        "window_image_context_complete",
        "lineage_scope",
        "human_review_complete",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    wanted = set(model_routing["window_id"].astype(str))
    frame = frame.loc[frame["window_id"].astype(str).isin(wanted)].copy()
    if len(frame) != len(wanted):
        raise ValueError(
            f"model image-window rows={len(frame)}!={len(wanted)}"
        )
    if frame["window_id"].astype(str).duplicated().any():
        raise ValueError("model image-window manifest contains duplicate IDs")
    if not frame["window_length_frames"].astype(int).eq(sequence_length).all():
        raise ValueError("model image-window sequence length drift")
    if not _strict_bool(
        frame["window_image_context_complete"],
        name="window_image_context_complete",
    ).all():
        raise ValueError("model image windows contain incomplete context")
    expected_counts = frame["observed_image_context_rows"].astype(int)
    loadable_counts = frame["loadable_image_context_rows"].astype(int)
    if (
        not expected_counts.eq(sequence_length).all()
        or not loadable_counts.eq(sequence_length).all()
        or not frame["missing_image_context_slots"].astype(int).eq(0).all()
    ):
        raise ValueError("model image-window context count drift")
    _validate_claim_columns(frame, name="model image windows")
    return frame


def _read_model_slots(
    path: Path,
    *,
    model_routing: pd.DataFrame,
    temporal_view_name: str,
    sequence_length: int,
) -> pd.DataFrame:
    columns = [
        "temporal_view_name",
        "view_item_id",
        "parent_window_id",
        "temporal_unit_key",
        "item_order",
        "slot_index",
        "declared_sequence_length",
        "frame_index_expected_audit",
        "time_delta",
        "length_mask",
        "observed_mask",
        "timing_valid_mask",
        "padding_mask",
        "lineage_scope",
        "human_review_complete",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if set(frame["temporal_view_name"].astype(str)) != {temporal_view_name}:
        raise ValueError("temporal slot manifest view-name drift")
    wanted = set(model_routing["window_id"].astype(str))
    frame = frame.loc[
        frame["parent_window_id"].astype(str).isin(wanted)
    ].copy()
    expected_rows = len(wanted) * sequence_length
    if len(frame) != expected_rows:
        raise ValueError(f"model temporal slots={len(frame)}!={expected_rows}")
    if frame[["parent_window_id", "slot_index"]].astype(str).duplicated().any():
        raise ValueError("model temporal slots contain duplicate window slots")
    if not frame["declared_sequence_length"].astype(int).eq(
        sequence_length
    ).all():
        raise ValueError("model temporal slot declared length drift")
    slot_sets = frame.groupby("parent_window_id", sort=False)["slot_index"].apply(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    expected_slots = tuple(range(sequence_length))
    if not slot_sets.map(lambda value: value == expected_slots).all():
        raise ValueError("model temporal slot indices are incomplete or reordered")
    item_orders = frame.groupby("parent_window_id", sort=False)["item_order"].nunique()
    if not item_orders.eq(1).all():
        raise ValueError("model temporal windows have conflicting item order")
    unique_item_orders = frame.drop_duplicates("parent_window_id")["item_order"]
    if unique_item_orders.astype(int).duplicated().any():
        raise ValueError("model temporal item order is not unique")
    for column in ("length_mask", "observed_mask", "timing_valid_mask"):
        if not _strict_bool(frame[column], name=column).all():
            raise ValueError(f"model temporal slots contain false {column}")
    if _strict_bool(frame["padding_mask"], name="padding_mask").any():
        raise ValueError("model temporal slots contain padding")
    deltas = pd.to_numeric(frame["time_delta"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(deltas).all() or (deltas < 0.0).any():
        raise ValueError("model temporal slots contain invalid time deltas")
    _validate_claim_columns(frame, name="model temporal slots")
    return frame


def _read_feature_index(
    path: Path,
    *,
    feature_parent: dict[str, Any],
) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    if header != FEATURE_INDEX_FIELDS:
        raise ValueError("cached feature index schema drift")
    frame = pd.read_csv(path, low_memory=False)
    if len(frame) != int(feature_parent["feature_rows"]):
        raise ValueError("cached feature index row count drift")
    if frame["image_context_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("cached feature index contains blank context IDs")
    if frame["image_context_id"].astype(str).duplicated().any():
        raise ValueError("cached feature index contains duplicate context IDs")
    expected_rows = np.arange(len(frame), dtype=np.int64)
    observed_rows = frame["feature_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_rows, expected_rows):
        raise ValueError("cached feature rows are not ordered and contiguous")
    exact_values = {
        "control_id": str(feature_parent["control_id"]),
        "backbone_name": str(feature_parent["backbone_name"]),
        "pretrained_weight_enum": str(feature_parent["pretrained_weight_enum"]),
        "image_size": int(feature_parent["image_size"]),
        "feature_dim": FEATURE_DIM,
        "feature_dtype": str(FEATURE_DTYPE),
        "lineage_scope": LINEAGE_SCOPE,
    }
    for column, expected in exact_values.items():
        values = set(frame[column].tolist())
        if values != {expected}:
            raise ValueError(
                f"cached feature index metadata drift={column}:{values}"
            )
    if _strict_bool(
        frame["human_review_complete"],
        name="feature index human_review_complete",
    ).any():
        raise ValueError("cached feature index claims human review")
    return frame[["image_context_id", "feature_row"]].copy()


def _join_slots_to_features(
    *,
    model_routing: pd.DataFrame,
    image_windows: pd.DataFrame,
    slots: pd.DataFrame,
    feature_index: pd.DataFrame,
    sequence_length: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    context_records: list[dict[str, Any]] = []
    for row in image_windows.itertuples(index=False):
        context_ids = str(row.image_context_id_sequence).split(";;")
        frame_indices = str(row.expected_frame_indices).split("|")
        if len(context_ids) != sequence_length or len(frame_indices) != sequence_length:
            raise ValueError(
                f"image context sequence length drift window={row.window_id}"
            )
        if any(not value.strip() for value in context_ids):
            raise ValueError(f"blank image context ID window={row.window_id}")
        for slot_index, (context_id, frame_index) in enumerate(
            zip(context_ids, frame_indices, strict=True)
        ):
            context_records.append(
                {
                    "parent_window_id": str(row.window_id),
                    "slot_index": slot_index,
                    "image_context_id": context_id,
                    "image_frame_index_expected": int(frame_index),
                }
            )
    context_long = pd.DataFrame.from_records(context_records)
    if len(context_long) != len(slots):
        raise ValueError("image-context expansion lost temporal slots")
    joined = slots.merge(
        context_long,
        on=["parent_window_id", "slot_index"],
        how="left",
        validate="one_to_one",
    )
    if joined["image_context_id"].isna().any():
        raise ValueError("temporal slots are missing image-context IDs")
    frame_mismatch = joined["frame_index_expected_audit"].astype(int).ne(
        joined["image_frame_index_expected"].astype(int)
    )
    if frame_mismatch.any():
        raise ValueError(
            f"slot-to-image frame order mismatches={int(frame_mismatch.sum())}"
        )
    joined = joined.merge(
        feature_index,
        on="image_context_id",
        how="left",
        validate="many_to_one",
    )
    if joined["feature_row"].isna().any():
        raise ValueError(
            "temporal image contexts missing cached features="
            f"{int(joined['feature_row'].isna().sum())}"
        )
    route_columns = [
        "window_id",
        "temporal_unit_key",
        "recording_group_id",
        "video_key",
        "source_type",
        "dataset_id",
        "behavior_label",
        "oof_fold_id",
        "l5_role",
    ]
    routing = model_routing[route_columns].rename(
        columns={"window_id": "parent_window_id"}
    )
    joined = joined.merge(
        routing,
        on="parent_window_id",
        how="left",
        validate="many_to_one",
        suffixes=("_slot", ""),
    )
    if joined["l5_role"].isna().any():
        raise ValueError("feature slots are missing model routing")
    if not joined["l5_role"].isin(MODEL_ACCESS_ROLES).all():
        raise ValueError("outer or policy-invalid feature slots were materialized")
    unit_mismatch = joined["temporal_unit_key_slot"].astype(str).ne(
        joined["temporal_unit_key"].astype(str)
    )
    if unit_mismatch.any():
        raise ValueError(
            f"slot-to-routing native-unit mismatches={int(unit_mismatch.sum())}"
        )
    duplicate_context_slots = int(
        joined[["parent_window_id", "slot_index"]].duplicated().sum()
    )
    if duplicate_context_slots:
        raise ValueError(f"duplicate joined feature slots={duplicate_context_slots}")
    return joined, {
        "model_window_rows": int(len(model_routing)),
        "expanded_image_context_rows": int(len(context_long)),
        "temporal_slot_rows": int(len(slots)),
        "joined_feature_slot_rows": int(len(joined)),
        "missing_feature_rows": 0,
        "duplicate_joined_slots": 0,
        "frame_order_mismatches": 0,
        "outer_holdout_feature_slots": 0,
        "source_media_reads": 0,
        "errors": [],
        "valid": True,
    }


def _reshape_model_view(
    joined: pd.DataFrame,
    *,
    model_routing: pd.DataFrame,
    sequence_length: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    ordered = joined.sort_values(
        ["item_order", "slot_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(ordered) % sequence_length:
        raise ValueError("joined feature slots cannot form exact sequences")
    window_order = ordered.drop_duplicates(
        "parent_window_id",
        keep="first",
    )[["parent_window_id", "item_order"]]
    if len(window_order) != len(model_routing):
        raise ValueError("joined feature view lost model windows")
    routing = model_routing.rename(columns={"window_id": "parent_window_id"})
    windows = window_order.merge(
        routing,
        on="parent_window_id",
        how="left",
        validate="one_to_one",
    ).sort_values("item_order", kind="mergesort")
    windows = windows.rename(columns={"parent_window_id": "window_id"})
    feature_rows = ordered["feature_row"].to_numpy(dtype=np.int64).reshape(
        len(windows),
        sequence_length,
    )
    observed_mask = _strict_bool(
        ordered["observed_mask"],
        name="observed_mask",
    ).to_numpy(dtype=np.bool_).reshape(len(windows), sequence_length)
    time_delta = ordered["time_delta"].to_numpy(dtype=np.float32).reshape(
        len(windows),
        sequence_length,
    )
    if not observed_mask.all():
        raise ValueError("legacy cached feature view contains unobserved slots")
    if not np.isfinite(time_delta).all() or (time_delta < 0.0).any():
        raise ValueError("legacy cached feature view contains invalid timing")
    expected_window_order = windows["window_id"].astype(str).tolist()
    observed_window_order = (
        ordered.drop_duplicates("parent_window_id")["parent_window_id"]
        .astype(str)
        .tolist()
    )
    if expected_window_order != observed_window_order:
        raise ValueError("cached feature window reshape order drift")
    return (
        windows.reset_index(drop=True),
        feature_rows,
        observed_mask,
        time_delta,
    )


def _event_mass_weights(
    windows: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    placements = windows.groupby("temporal_unit_key", sort=False)[
        "temporal_unit_key"
    ].transform("size")
    weights = 1.0 / placements.to_numpy(dtype=np.float64)
    mass = pd.Series(weights).groupby(
        windows["temporal_unit_key"].astype(str).reset_index(drop=True),
        sort=False,
    ).sum()
    maximum_error = float(np.max(np.abs(mass.to_numpy() - 1.0)))
    if maximum_error > 1e-12:
        raise ValueError(f"native event-mass error={maximum_error}")
    return weights.astype(np.float32), {
        "policy": "one_total_loss_mass_per_native_unit_and_tier_v1",
        "model_window_rows": int(len(windows)),
        "native_unit_rows": int(windows["temporal_unit_key"].nunique()),
        "minimum_window_weight": float(weights.min()),
        "maximum_window_weight": float(weights.max()),
        "maximum_native_mass_error": maximum_error,
        "fit_scope": "per_native_unit_structural_no_cross_role_statistics",
        "errors": [],
        "valid": True,
    }


def _leakage_audit(whitelist: dict[str, Any]) -> dict[str, Any]:
    features = [str(value) for value in whitelist["features"]]
    forbidden = sorted(
        feature
        for feature in features
        if any(
            fnmatch.fnmatch(feature.lower(), pattern)
            for pattern in FORBIDDEN_X_PATTERNS
        )
    )
    if len(features) != FEATURE_DIM or len(features) != len(set(features)):
        raise ValueError("cached feature whitelist is incomplete or duplicated")
    if forbidden:
        raise ValueError(f"cached feature whitelist contains leakage={forbidden}")
    return {
        "selection_policy": "explicit_only",
        "feature_count": len(features),
        "forbidden_features": forbidden,
        "routing_only_fields": list(ROUTING_ONLY_FIELDS),
        "labels_paths_ids_folds_in_model_x": False,
        "normalization_fit_scope": "none_data_derived",
        "learned_transform_fit_scope": "train_role_optimizer_steps_only",
        "event_weight_fit_scope": (
            "per_native_unit_structural_no_cross_role_statistics"
        ),
        "threshold_fit_scope": "validation_only_if_predeclared_later",
        "outer_holdout_statistics_read": False,
        "errors": [],
        "valid": True,
    }


def _temporal_unit_audit(
    fold_manifest: pd.DataFrame,
    *,
    role_audit: dict[str, Any],
) -> dict[str, Any]:
    duplicates = int(fold_manifest["temporal_unit_key"].astype(str).duplicated().sum())
    bad_lengths = int(fold_manifest["label_frame_count"].astype(int).ne(16).sum())
    if duplicates or bad_lengths:
        raise ValueError(
            f"cached native-unit contract errors=duplicates:{duplicates},"
            f"bad_lengths:{bad_lengths}"
        )
    return {
        "native_unit_definition": "one_complete_legacy_16_frame_burst",
        "native_unit_rows": int(len(fold_manifest)),
        "duplicate_native_unit_rows": duplicates,
        "bad_native_length_rows": bad_lengths,
        "role_native_counts": role_audit["native_counts"],
        "recording_video_overlap": role_audit["role_overlap"],
        "pig_id_used_for_grouping": False,
        "errors": [],
        "valid": True,
    }


def _routing_overlap_audit(fold_manifest: pd.DataFrame) -> dict[str, Any]:
    roles = ("train", "validation", "outer_holdout")
    errors: list[str] = []
    pairs: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(roles):
        left_frame = fold_manifest.loc[fold_manifest["l5_role"].eq(left)]
        for right in roles[left_index + 1 :]:
            right_frame = fold_manifest.loc[fold_manifest["l5_role"].eq(right)]
            pair = f"{left}_vs_{right}"
            pairs[pair] = {}
            for column in (
                "temporal_unit_key",
                "recording_group_id",
                "video_key",
            ):
                overlap = set(left_frame[column].astype(str)).intersection(
                    right_frame[column].astype(str)
                )
                pairs[pair][f"{column}_overlap"] = len(overlap)
                if overlap:
                    errors.append(f"{pair}:{column}:{len(overlap)}")
    return {"pairs": pairs, "errors": errors, "valid": not errors}


def _class_support_frame(fold_manifest: pd.DataFrame) -> pd.DataFrame:
    counts = fold_manifest.groupby(
        ["l5_role", "behavior_label"],
        sort=False,
    ).size()
    records = []
    for role in ALL_ROUTING_ROLES:
        for label in VALID_BEHAVIORS:
            records.append(
                {
                    "l5_role": role,
                    "behavior_label": label,
                    "native_units": int(counts.get((role, label), 0)),
                }
            )
    return pd.DataFrame.from_records(records)


def _source_support_frame(fold_manifest: pd.DataFrame) -> pd.DataFrame:
    counts = fold_manifest.groupby(
        ["l5_role", "source_type"],
        sort=False,
    ).size()
    records = []
    sources = sorted(fold_manifest["source_type"].astype(str).unique())
    for role in ALL_ROUTING_ROLES:
        for source in sources:
            records.append(
                {
                    "l5_role": role,
                    "source_type": source,
                    "native_units": int(counts.get((role, source), 0)),
                }
            )
    return pd.DataFrame.from_records(records)


def audit_legacy_l5_cached_feature_batches(
    view: LegacyL5CachedFeatureView,
    *,
    batch_size: int,
    max_batches_per_role: int,
) -> LegacyL5CachedFeatureView:
    """Read a bounded CPU sample while proving mmap and CUDA constraints."""

    if not 1 <= batch_size <= MAX_CACHED_AUDIT_BATCH_SIZE:
        raise ValueError(
            "cached audit batch size must be within "
            f"[1,{MAX_CACHED_AUDIT_BATCH_SIZE}]"
        )
    if not 1 <= max_batches_per_role <= MAX_CACHED_AUDIT_BATCHES_PER_ROLE:
        raise ValueError(
            "cached audit batches per role must be within "
            f"[1,{MAX_CACHED_AUDIT_BATCHES_PER_ROLE}]"
        )
    if torch.cuda.is_initialized():
        raise ValueError("cached batch audit requires a fresh CPU-only process")
    role_records: dict[str, dict[str, Any]] = {}
    total_loaded_windows = 0
    maximum_loaded_batch_bytes = 0
    for role in MODEL_ACCESS_ROLES:
        role_positions = view.indices_for_role(role)
        available_batches = int(
            np.ceil(len(role_positions) / float(batch_size))
        )
        requested_batches = min(max_batches_per_role, available_batches)
        iterator = view.iter_role_batches(
            role,
            batch_size=batch_size,
            seed=0,
            shuffle=False,
        )
        loaded_positions: list[int] = []
        loaded_batch_sizes: list[int] = []
        try:
            for _ in range(requested_batches):
                batch = next(iterator)
                features = batch["features"]
                observed_mask = batch["observed_mask"]
                time_delta = batch["time_delta"]
                targets = batch["targets"]
                sample_weights = batch["sample_weights"]
                if features.shape != (
                    len(batch["positions"]),
                    view.sequence_length,
                    FEATURE_DIM,
                ):
                    raise ValueError("cached audit feature batch shape drift")
                if features.dtype != FEATURE_DTYPE:
                    raise ValueError("cached audit feature dtype drift")
                if not np.isfinite(features).all():
                    raise ValueError("cached audit feature batch is nonfinite")
                if not observed_mask.all():
                    raise ValueError("cached audit batch has unobserved slots")
                if not np.isfinite(time_delta).all() or (time_delta < 0).any():
                    raise ValueError("cached audit timing batch is invalid")
                if (targets < 0).any() or (targets >= len(VALID_BEHAVIORS)).any():
                    raise ValueError("cached audit target indices are invalid")
                if (
                    not np.isfinite(sample_weights).all()
                    or (sample_weights <= 0).any()
                ):
                    raise ValueError("cached audit sample weights are invalid")
                loaded_positions.extend(
                    int(value) for value in batch["positions"]
                )
                loaded_batch_sizes.append(len(batch["positions"]))
                batch_bytes = sum(
                    int(batch[name].nbytes)
                    for name in (
                        "positions",
                        "features",
                        "observed_mask",
                        "time_delta",
                        "targets",
                        "sample_weights",
                    )
                )
                maximum_loaded_batch_bytes = max(
                    maximum_loaded_batch_bytes,
                    batch_bytes,
                )
                del batch
        finally:
            iterator.close()
        if not loaded_positions:
            raise ValueError(f"cached audit role has no model rows={role}")
        total_loaded_windows += len(loaded_positions)
        window_ids = view.windows.iloc[loaded_positions]["window_id"]
        role_records[role] = {
            "available_windows": int(len(role_positions)),
            "loaded_windows": len(loaded_positions),
            "loaded_batches": len(loaded_batch_sizes),
            "loaded_batch_sizes": loaded_batch_sizes,
            "loaded_window_sha256": _ordered_sha256(window_ids),
        }
    if torch.cuda.is_initialized():
        raise ValueError("cached batch audit unexpectedly initialized CUDA")
    bounded_audit = {
        "status": "PASS_LEGACY_DEVELOPMENT_L5_BOUNDED_BATCH_AUDIT",
        "execution_device": "cpu",
        "batch_size": batch_size,
        "max_batches_per_role": max_batches_per_role,
        "maximum_allowed_batch_size": MAX_CACHED_AUDIT_BATCH_SIZE,
        "maximum_allowed_batches_per_role": (
            MAX_CACHED_AUDIT_BATCHES_PER_ROLE
        ),
        "total_loaded_windows": total_loaded_windows,
        "maximum_loaded_batch_bytes": maximum_loaded_batch_bytes,
        "roles": role_records,
        "mmap_close_after_each_loaded_batch": True,
        "dataloader_num_workers": 0,
        "pin_memory": False,
        "prefetch_factor": None,
        "source_media_reads": 0,
        "outer_holdout_rows_loaded": 0,
        "cuda_runtime_initialized_before": False,
        "cuda_runtime_initialized_after": False,
        "errors": [],
        "valid": True,
    }
    return replace(
        view,
        audit={**view.audit, "bounded_batch_audit": bounded_audit},
    )


def write_legacy_l5_cached_data_packet(
    view: LegacyL5CachedFeatureView,
    *,
    output_dir: Path,
    run_id: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Write one isolated audit packet with no checkpoint or prediction."""

    if not _safe_run_id(run_id):
        raise ValueError(f"unsafe cached data run ID={run_id!r}")
    if output_dir.name != run_id:
        raise ValueError("cached data output directory must equal run ID")
    if not np.isfinite(runtime_seconds) or runtime_seconds < 0.0:
        raise ValueError("cached data runtime must be finite and non-negative")
    bounded_audit = view.audit.get("bounded_batch_audit") or {}
    if bounded_audit.get("valid") is not True:
        raise ValueError("cached data packet requires a passing bounded audit")
    if bounded_audit.get("outer_holdout_rows_loaded") != 0:
        raise ValueError("cached data bounded audit accessed outer holdout")
    if output_dir.exists():
        raise FileExistsError(f"cached data output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = _cached_packet_paths(output_dir)
    whitelist = cached_feature_whitelist_payload()
    blacklist = cached_feature_blacklist_payload()
    native_routing = view.fold_manifest.copy()
    fold_manifest = native_routing.loc[
        native_routing["l5_role"].isin(
            ("train", "validation", "outer_holdout")
        )
    ].copy()
    excluded_rows = len(native_routing) - len(fold_manifest)
    expected_excluded = view.audit["role_audit"]["native_counts"][
        "policy_invalid"
    ]
    if excluded_rows != expected_excluded:
        raise ValueError("cached data packet exclusion count drift")
    class_support = _class_by_fold_support_frame(view.fold_manifest)
    source_support = _source_by_fold_support_frame(view.fold_manifest)
    hidden_review = {
        "schema_version": (
            "classification_v2.legacy_development_l5.hidden_review_waiver.v1"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "review_policy": "explicitly_waived_for_unreviewed_development",
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "errors": [],
        "valid_for_declared_unreviewed_scope": True,
    }
    code = git_state()
    scientific_identity = _payload_sha256(
        {
            "config_sha256": view.audit["config_sha256"],
            "control_id": view.control_id,
            "feature_tensor_sha256": view.feature_tensor_sha256,
            "temporal_view_name": view.temporal_view_name,
            "sequence_length": view.sequence_length,
            "role_unit_sha256": view.audit["role_audit"]["role_unit_sha256"],
            "feature_whitelist": whitelist,
            "model_visible_roles": list(MODEL_ACCESS_ROLES),
            "outer_holdout_access": "FORBIDDEN_DURING_MODEL_SELECTION",
        }
    )
    started_at = _utc_now()
    planned_manifest = {
        "schema_version": CACHED_DATA_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_name": "legacy_l5_cached_feature_consumer_audit",
        "execution_mode": "local_smoke",
        "execution_kind": "cache_only_join_and_leakage_audit",
        "status": "planned",
        "created_at_utc": started_at,
        "code_sha": code["commit"],
        "dirty_worktree": code["dirty"],
        "dirty_entries": code["dirty_entries"],
        "config_path": view.audit["config_path"],
        "config_hash": view.audit["config_sha256"],
        "dataset_snapshot_hash": view.audit["inputs"]["native_units"]["sha256"],
        "cache_hash": view.feature_tensor_sha256,
        "fold_manifest_hash": view.audit["inputs"]["window_folds"]["sha256"],
        "feature_whitelist_hash": _payload_sha256(whitelist),
        "control_id": view.control_id,
        "temporal_view_name": view.temporal_view_name,
        "sequence_length": view.sequence_length,
        "scientific_identity_sha256": scientific_identity,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "oom_retry_allowed": False,
        "source_media_reads": 0,
        "optimizer_steps": 0,
    }
    _write_json_exclusive(paths["run_manifest"], planned_manifest)
    planned_hash = file_sha256(paths["run_manifest"])
    _write_json_exclusive(paths["feature_whitelist"], whitelist)
    _write_json_exclusive(paths["feature_blacklist"], blacklist)
    _write_dataframe_exclusive(paths["fold_manifest"], fold_manifest)
    _write_dataframe_exclusive(
        paths["native_routing_manifest"],
        native_routing,
    )
    _write_dataframe_exclusive(paths["class_support"], class_support)
    _write_dataframe_exclusive(paths["source_support"], source_support)
    _write_json_exclusive(
        paths["leakage_audit"],
        view.audit["leakage_audit"],
    )
    _write_json_exclusive(
        paths["temporal_unit_audit"],
        view.audit["temporal_unit_audit"],
    )
    _write_json_exclusive(paths["hidden_review_audit"], hidden_review)
    packet_audit = {
        **view.audit,
        "packet_manifest_audit": {
            "eligible_fold_manifest_rows": int(len(fold_manifest)),
            "native_routing_manifest_rows": int(len(native_routing)),
            "policy_invalid_rows_preserved": int(excluded_rows),
            "eligible_roles": [
                "train",
                "validation",
                "outer_holdout",
            ],
            "fold_manifest_sha256": file_sha256(paths["fold_manifest"]),
            "native_routing_manifest_sha256": file_sha256(
                paths["native_routing_manifest"]
            ),
            "silent_row_drop": False,
            "errors": [],
            "valid": True,
        },
    }
    _write_json_exclusive(paths["cached_data_audit"], packet_audit)
    environment = {
        "schema_version": CACHED_DATA_ENVIRONMENT_SCHEMA_VERSION,
        "captured_at_utc": _utc_now(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "cuda_runtime_initialized": bool(torch.cuda.is_initialized()),
        "gpu_execution_performed": False,
        "declared_local_gpu_vram_gib": DECLARED_LOCAL_GPU_VRAM_GIB,
        "validated_local_gpu_vram_bytes": VALIDATED_LOCAL_GPU_VRAM_BYTES,
        "maximum_peak_vram_fraction": GPU_ALLOCATOR_FRACTION_CEILING,
        "gpu_allocator_limit_bytes": GPU_ALLOCATOR_LIMIT_BYTES,
        "maximum_loaded_batch_bytes": bounded_audit[
            "maximum_loaded_batch_bytes"
        ],
        "dataloader_num_workers": 0,
        "pin_memory": False,
        "oom_retry_allowed": False,
    }
    if environment["cuda_runtime_initialized"]:
        raise ValueError("cached data packet unexpectedly initialized CUDA")
    _write_json_exclusive(paths["environment"], environment)
    checkpoint_manifest = {
        "schema_version": CACHED_DATA_CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "scientific_identity_sha256": scientific_identity,
        "checkpoints": [],
        "checkpoint_creation_authorized": False,
        "reason": "cached_data_consumer_audit_has_no_optimizer_steps",
        "errors": [],
    }
    prediction_manifest = {
        "schema_version": CACHED_DATA_PREDICTION_SCHEMA_VERSION,
        "run_id": run_id,
        "scientific_identity_sha256": scientific_identity,
        "predictions": [],
        "prediction_creation_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "reason": "cached_data_consumer_audit_has_no_predictions",
        "errors": [],
    }
    _write_json_exclusive(paths["checkpoint_manifest"], checkpoint_manifest)
    _write_json_exclusive(paths["prediction_manifest"], prediction_manifest)
    output_artifact_names = (
        "feature_whitelist",
        "feature_blacklist",
        "fold_manifest",
        "native_routing_manifest",
        "class_support",
        "source_support",
        "leakage_audit",
        "temporal_unit_audit",
        "hidden_review_audit",
        "cached_data_audit",
        "environment",
        "checkpoint_manifest",
        "prediction_manifest",
    )
    artifact_manifest = {
        "schema_version": CACHED_DATA_ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "scientific_identity_sha256": scientific_identity,
        "status": "completed",
        "artifacts": [
            {
                "name": name,
                "path": str(Path(record["path"]).resolve()),
                "sha256": str(record["sha256"]),
                "size_bytes": int(record["size_bytes"]),
                "direction": "input",
            }
            for name, record in view.audit["inputs"].items()
        ]
        + [
            {
                "name": name,
                "path": str(paths[name].resolve()),
                "sha256": file_sha256(paths[name]),
                "size_bytes": int(paths[name].stat().st_size),
                "direction": "output",
            }
            for name in output_artifact_names
        ],
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(paths["artifact_manifest"], artifact_manifest)
    completed_at = _utc_now()
    final_manifest = {
        **planned_manifest,
        "status": "completed",
        "completed_at_utc": completed_at,
        "runtime_seconds": float(runtime_seconds),
        "peak_vram_bytes": 0,
        "planned_run_manifest_sha256": planned_hash,
        "cached_data_audit_sha256": file_sha256(paths["cached_data_audit"]),
        "artifact_manifest_sha256": file_sha256(paths["artifact_manifest"]),
        "checkpoint_manifest_sha256": file_sha256(paths["checkpoint_manifest"]),
        "prediction_manifest_sha256": file_sha256(paths["prediction_manifest"]),
        "feature_whitelist_sha256": file_sha256(paths["feature_whitelist"]),
        "failure_reason": "",
    }
    _write_json_atomic(paths["run_manifest"], final_manifest)
    registry_entry = {
        "registry_schema_version": CACHED_DATA_REGISTRY_SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_name": planned_manifest["experiment_name"],
        "execution_mode": planned_manifest["execution_mode"],
        "status": "completed",
        "code_sha": code["commit"],
        "dirty_worktree": code["dirty"],
        "config_hash": view.audit["config_sha256"],
        "dataset_snapshot_hash": planned_manifest["dataset_snapshot_hash"],
        "cache_hash": view.feature_tensor_sha256,
        "fold_manifest_hash": planned_manifest["fold_manifest_hash"],
        "feature_whitelist_hash": final_manifest["feature_whitelist_sha256"],
        "control_id": view.control_id,
        "temporal_view_name": view.temporal_view_name,
        "sequence_length": view.sequence_length,
        "train_native_units": view.audit["role_audit"]["native_counts"]["train"],
        "validation_native_units": view.audit["role_audit"]["native_counts"][
            "validation"
        ],
        "outer_holdout_native_units": view.audit["role_audit"]["native_counts"][
            "outer_holdout"
        ],
        "source_media_reads": 0,
        "outer_predictions_created": 0,
        "runtime_seconds": float(runtime_seconds),
        "peak_vram_bytes": 0,
        "manifest_path": str(paths["run_manifest"].resolve()),
        "manifest_sha256": file_sha256(paths["run_manifest"]),
        "completed_at_utc": completed_at,
    }
    _write_json_exclusive(paths["registry_entry"], registry_entry)
    _write_registry(paths["runs_registry"], registry_entry)
    return final_manifest


def _class_by_fold_support_frame(fold_manifest: pd.DataFrame) -> pd.DataFrame:
    counts = fold_manifest.groupby(
        ["oof_fold_id", "l5_role", "behavior_label"],
        sort=False,
    ).size()
    records = []
    fold_roles = fold_manifest[["oof_fold_id", "l5_role"]].drop_duplicates()
    for row in fold_roles.itertuples(index=False):
        for label in VALID_BEHAVIORS:
            key = (str(row.oof_fold_id), str(row.l5_role), label)
            records.append(
                {
                    "oof_fold_id": key[0],
                    "l5_role": key[1],
                    "behavior_label": label,
                    "native_units": int(counts.get(key, 0)),
                }
            )
    return pd.DataFrame.from_records(records)


def _source_by_fold_support_frame(fold_manifest: pd.DataFrame) -> pd.DataFrame:
    counts = fold_manifest.groupby(
        ["oof_fold_id", "l5_role", "source_type"],
        sort=False,
    ).size()
    records = []
    fold_roles = fold_manifest[["oof_fold_id", "l5_role"]].drop_duplicates()
    sources = sorted(fold_manifest["source_type"].astype(str).unique())
    for row in fold_roles.itertuples(index=False):
        for source in sources:
            key = (str(row.oof_fold_id), str(row.l5_role), source)
            records.append(
                {
                    "oof_fold_id": key[0],
                    "l5_role": key[1],
                    "source_type": source,
                    "native_units": int(counts.get(key, 0)),
                }
            )
    return pd.DataFrame.from_records(records)


def _cached_packet_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "run_manifest": root / "run_manifest.json",
        "feature_whitelist": root / "feature_whitelist.json",
        "feature_blacklist": root / "feature_blacklist.json",
        "fold_manifest": root / "fold_manifest.csv",
        "native_routing_manifest": root / "native_routing_manifest.csv",
        "class_support": root / "class_by_fold_support.csv",
        "source_support": root / "source_by_fold_support.csv",
        "leakage_audit": root / "leakage_audit.json",
        "temporal_unit_audit": root / "temporal_unit_audit.json",
        "hidden_review_audit": root / "hidden_review_audit.json",
        "cached_data_audit": root / "cached_data_audit.json",
        "environment": root / "environment.json",
        "checkpoint_manifest": root / "checkpoint_manifest.json",
        "prediction_manifest": root / "prediction_manifest.json",
        "artifact_manifest": root / "artifact_manifest.json",
        "registry_entry": root / "registry_entry.json",
        "runs_registry": root / "runs_registry.csv",
    }


def _safe_run_id(value: str) -> bool:
    if not value or len(value) > 128 or value in {".", ".."}:
        return False
    if Path(value).name != value:
        return False
    return all(
        character.isascii()
        and (character.isalnum() or character in {"-", "_"})
        for character in value
    )


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_dataframe_exclusive(path: Path, frame: pd.DataFrame) -> None:
    if frame.columns.duplicated().any():
        raise ValueError(f"dataframe contains duplicate columns: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
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


def _write_registry(path: Path, entry: dict[str, Any]) -> None:
    if tuple(entry) != REGISTRY_FIELDS:
        raise ValueError("legacy L5 cached-data registry entry schema drift")
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


def _validate_claim_columns(frame: pd.DataFrame, *, name: str) -> None:
    required = {"lineage_scope", "human_review_complete"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing claim columns={missing}")
    scopes = set(frame["lineage_scope"].fillna("").astype(str))
    if scopes != {LINEAGE_SCOPE}:
        raise ValueError(f"{name} lineage scope drift={sorted(scopes)}")
    claim_columns = (
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
    )
    for column in claim_columns:
        if column in frame and _strict_bool(frame[column], name=column).any():
            raise ValueError(f"{name} exceeds unreviewed claim boundary={column}")


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"missing boolean values in {name}")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}
    valid = normalized.isin(truthy | falsy)
    if not valid.all():
        raise ValueError(f"invalid boolean values in {name}")
    return normalized.isin(truthy)


def _ordered_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        if not value.strip():
            raise ValueError("ordered hash contains a blank value")
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _close_memmap(array: np.ndarray) -> None:
    mmap_handle = getattr(array, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


__all__ = (
    "ALL_ROUTING_ROLES",
    "CACHED_DATA_AUDIT_SCHEMA_VERSION",
    "CACHED_DATA_MANIFEST_SCHEMA_VERSION",
    "DECLARED_LOCAL_GPU_VRAM_GIB",
    "GPU_ALLOCATOR_FRACTION_CEILING",
    "GPU_ALLOCATOR_LIMIT_BYTES",
    "LegacyL5CachedFeatureClassifier",
    "LegacyL5CachedFeatureView",
    "MAX_CACHED_AUDIT_BATCHES_PER_ROLE",
    "MAX_CACHED_AUDIT_BATCH_SIZE",
    "MODEL_ACCESS_ROLES",
    "VALIDATED_LOCAL_GPU_VRAM_BYTES",
    "audit_legacy_l5_cached_feature_batches",
    "build_legacy_l5_cached_feature_view",
    "cached_feature_blacklist_payload",
    "cached_feature_whitelist_payload",
    "write_legacy_l5_cached_data_packet",
)
