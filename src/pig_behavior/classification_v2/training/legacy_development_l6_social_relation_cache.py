"""Immutable window-local numeric-social cache for legacy L6 development."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.spatial_sequence_export import (
    SPATIAL_FRAME_FEATURES,
    SpatialSequenceExport,
    export_spatial_sequences,
)
from pig_behavior.classification_v2.training.legacy_development_l6_geometry_cache import (
    CANONICAL_SOURCE_NAME,
    DATASET_ID,
    EXPECTED_MODEL_SLOTS,
    EXPECTED_MODEL_WINDOWS,
    EXPECTED_RAW_ROWS,
    LINEAGE_SCOPE,
    SEQUENCE_LENGTH,
    SOURCE_TYPE,
    TEMPORAL_VIEW_NAME,
    VIEW_ID,
    LegacyL6GeometryCache,
    _close_memmap,
    _object,
    _ordered_sha256,
    _read_json,
    _require_exact_keys,
    _require_inside,
    _require_sha,
    _resolve_inside,
    _validate_bound_file,
    _write_json_exclusive,
    geometry_cache_git_guard,
    load_geometry_cache,
    load_geometry_cache_config,
    single_source_probe_audit,
)
from pig_behavior.classification_v2.training.legacy_development_l6_motion_cache import (
    _dataframe_sha256,
    _strict_bool,
    _validate_claim_columns,
    _validate_order_frames,
    _windows_from_order,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

SOCIAL_RELATION_FEATURE_NAMES = tuple(
    SPATIAL_FRAME_FEATURES["social_relation"]
)
SOCIAL_RELATION_DIM = len(SOCIAL_RELATION_FEATURE_NAMES)
SOCIAL_RELATION_DTYPE = np.dtype(np.float32)
AVAILABILITY_DTYPE = np.dtype(np.bool_)
SOCIAL_HELPER_FIELDS = (
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "speed_n_per_frame",
)
SOCIAL_QUALITY_FIELDS = (
    "bbox_valid",
    "actor_bbox_valid",
    "geometry_feature_valid",
    "spatiotemporal_feature_valid",
)
SOCIAL_FRAME_COLUMNS = (
    "frame_uid",
    "object_track_key",
    "frame_index",
    "source_type",
    "dataset_id",
    "lineage_scope",
    "human_review_complete",
    "nearest_pig_id",
    "nearest_track_id",
    *SOCIAL_HELPER_FIELDS,
    *SOCIAL_RELATION_FEATURE_NAMES,
    *SOCIAL_QUALITY_FIELDS,
)

CACHE_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.social_relation_cache_config.v1"
)
CACHE_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.social_relation_cache_manifest.v1"
)
CACHE_AUDIT_SCHEMA = (
    "classification_v2.legacy_development_l6.social_relation_cache_audit.v1"
)
REPEAT_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "social_relation_cache_repeat_config.v1"
)
REPEAT_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "social_relation_cache_repeat_gate.v1"
)

CACHE_FILES = {
    "social_relation": "social_relation_raw_f32.npy",
    "availability": "social_relation_available_bool.npy",
    "window_index": "social_relation_window_index.csv",
    "slot_index": "social_relation_slot_index.csv",
    "manifest": "social_relation_cache_manifest.json",
}


@dataclass(frozen=True, slots=True)
class LegacyL6SocialRelationCacheConfig:
    """Hash-bound numeric-social cache specification."""

    path: Path
    payload: dict[str, Any]
    repo_root: Path

    @property
    def sha256(self) -> str:
        return file_sha256(self.path)

    @property
    def output_root(self) -> Path:
        relative = str(self.payload["output"]["cache_root_relative_path"])
        return _resolve_inside(self.repo_root, relative)

    def bound_path(self, section: str, name: str) -> Path:
        value = _object(self.payload[section], section)[name]
        spec = _object(value, f"{section}.{name}")
        return _resolve_inside(self.repo_root, str(spec["path"]))


@dataclass(frozen=True, slots=True)
class LegacyL6SocialRelationCache:
    """Audited numeric-social arrays aligned to frozen L5 T6 windows."""

    root: Path
    social_relation_path: Path
    availability_path: Path
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def load_social_relation(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        mapping = np.load(self.social_relation_path, mmap_mode="r")
        try:
            if rows is None:
                values = np.asarray(
                    mapping,
                    dtype=SOCIAL_RELATION_DTYPE,
                ).copy()
            else:
                indices = _validated_rows(rows, len(self.window_index))
                values = np.asarray(
                    mapping[indices],
                    dtype=SOCIAL_RELATION_DTYPE,
                ).copy()
        finally:
            _close_memmap(mapping)
        return values

    def load_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        mapping = np.load(self.availability_path, mmap_mode="r")
        try:
            if rows is None:
                values = np.asarray(
                    mapping,
                    dtype=AVAILABILITY_DTYPE,
                ).copy()
            else:
                indices = _validated_rows(rows, len(self.window_index))
                values = np.asarray(
                    mapping[indices],
                    dtype=AVAILABILITY_DTYPE,
                ).copy()
        finally:
            _close_memmap(mapping)
        return values


def load_social_relation_cache_config(
    path: Path,
) -> LegacyL6SocialRelationCacheConfig:
    """Load one cache config and verify every immutable dependency."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_cache_config_payload(payload)
    config = LegacyL6SocialRelationCacheConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    source = _object(payload["source_identity"], "source_identity")
    _validate_bound_file(
        _resolve_inside(config.repo_root, str(source["raw_authority_path"])),
        str(source["raw_sha256"]),
        "raw authority",
    )
    for section in ("parents", "inputs"):
        for name, value in _object(payload[section], section).items():
            spec = _object(value, f"{section}.{name}")
            _validate_bound_file(
                _resolve_inside(config.repo_root, str(spec["path"])),
                str(spec["sha256"]),
                f"{section}.{name}",
            )
    order = _object(payload["order_authority"], "order_authority")
    for name in ("config", "manifest"):
        spec = _object(order[name], f"order_authority.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"order_authority.{name}",
        )
    implementation = _object(payload["implementation"], "implementation")
    for name, value in implementation.items():
        spec = _object(value, f"implementation.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"social cache implementation.{name}",
        )
    _load_order_authority(config)
    _validate_parent_decision(config)
    return config


def preflight_social_relation_cache(
    config: LegacyL6SocialRelationCacheConfig,
) -> dict[str, Any]:
    """Run the CPU-only source, schema, output, and Git gate."""

    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    windows = 0
    slots = 0
    frame_rows = 0
    neighbor_rows = 0
    try:
        order = _load_order_authority(config)
        windows = len(order.window_index)
        slots = len(order.slot_index)
        frames = _load_social_frames(config)
        frame_rows = len(frames)
        neighbor_rows = int(_partner_keys(frames).ne("").sum())
        if config.output_root.exists():
            errors.append(f"cache_output_exists={config.output_root}")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = geometry_cache_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("social relation cache preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "social_relation_cache_preflight.v1"
        ),
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_CACHE_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_CACHE_PREFLIGHT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "output_root": str(config.output_root),
        "model_window_rows": windows,
        "model_slot_rows": slots,
        "social_frame_rows": frame_rows,
        "social_neighbor_frame_rows": neighbor_rows,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "git_guard": git_guard,
        "build_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def build_social_relation_cache(
    config: LegacyL6SocialRelationCacheConfig,
) -> tuple[Path, dict[str, Any]]:
    """Build an immutable social cache after the preflight passes."""

    preflight = preflight_social_relation_cache(config)
    if not preflight["build_authorized"]:
        raise RuntimeError(
            f"social cache preflight failed={preflight['errors']}"
        )
    order = _load_order_authority(config)
    frames = _load_social_frames(config)
    packet = materialize_social_relation_cache(order, frames)
    root = config.output_root
    temporary = root.with_name(f"{root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"social cache temporary exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    np.save(
        temporary / CACHE_FILES["social_relation"],
        packet["social_relation"],
        allow_pickle=False,
    )
    np.save(
        temporary / CACHE_FILES["availability"],
        packet["availability"],
        allow_pickle=False,
    )
    packet["window_index"].to_csv(
        temporary / CACHE_FILES["window_index"],
        index=False,
        lineterminator="\n",
    )
    packet["slot_index"].to_csv(
        temporary / CACHE_FILES["slot_index"],
        index=False,
        lineterminator="\n",
    )
    artifacts = {
        name: {
            "filename": CACHE_FILES[name],
            "sha256": file_sha256(temporary / CACHE_FILES[name]),
            "size_bytes": int((temporary / CACHE_FILES[name]).stat().st_size),
        }
        for name in (
            "social_relation",
            "availability",
            "window_index",
            "slot_index",
        )
    }
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_CACHE",
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_hashes": {
            name: str(spec["sha256"])
            for name, spec in config.payload["implementation"].items()
        },
        "parent_view": {
            "view_id": VIEW_ID,
            "temporal_view_name": TEMPORAL_VIEW_NAME,
            "sequence_length": SEQUENCE_LENGTH,
            "model_window_rows": EXPECTED_MODEL_WINDOWS,
            "model_slot_rows": EXPECTED_MODEL_SLOTS,
            "ordered_window_id_sha256": _ordered_sha256(
                packet["window_index"]["window_id"]
            ),
        },
        "source_bindings": _source_bindings(config),
        "feature_contract": {
            "feature_names": list(SOCIAL_RELATION_FEATURE_NAMES),
            "feature_dim": SOCIAL_RELATION_DIM,
            "feature_dtype": str(SOCIAL_RELATION_DTYPE),
            "availability_dtype": str(AVAILABILITY_DTYPE),
            "window_local_rebase": True,
            "numeric_social_only": True,
            "partner_identity_values_used": False,
            "top_k_partner_features_used": False,
            "unit_aggregate_features_used": False,
            "geometry_values_in_model_x": False,
            "motion_values_in_model_x": False,
            "roi_values_in_model_x": False,
            "normalization": "fold_train_window_slots_only_at_consumer",
            "availability_is_behavior_evidence": False,
            "labels_ids_paths_or_folds_in_model_x": False,
        },
        "cache_scope": {
            "roles": ["train", "validation"],
            "outer_holdout_slots_materialized": 0,
            "source_media_reads": 0,
            "video_decodes": 0,
        },
        "content_audit": packet["content_audit"],
        "artifacts": artifacts,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(temporary / CACHE_FILES["manifest"], manifest)
    temporary.replace(root)
    audit = audit_social_relation_cache(config, cache_root=root)
    if not audit["valid"]:
        raise RuntimeError(f"written social cache failed={audit['errors']}")
    return root / CACHE_FILES["manifest"], audit


def materialize_social_relation_cache(
    order: LegacyL6GeometryCache,
    frames: pd.DataFrame,
) -> dict[str, Any]:
    """Export canonical window-local numeric social relations."""

    windows, ordered_slots = _windows_from_order(order)
    _validate_social_frames(frames)
    export_frames = frames[
        [
            "object_track_key",
            "frame_index",
            "nearest_pig_id",
            "nearest_track_id",
            *SOCIAL_HELPER_FIELDS,
            *SOCIAL_RELATION_FEATURE_NAMES,
            *SOCIAL_QUALITY_FIELDS,
        ]
    ].copy()
    exported = export_spatial_sequences(
        windows,
        export_frames,
        max_window_length=SEQUENCE_LENGTH,
    )
    _validate_spatial_export(exported)
    social = np.asarray(
        exported.arrays["social_relation"],
        dtype=SOCIAL_RELATION_DTYPE,
    ).copy()
    observed = np.asarray(exported.arrays["observed_mask"], dtype=bool)
    quality = np.asarray(exported.arrays["quality_mask"], dtype=np.float32)
    quality_names = list(exported.feature_names["quality_mask"])
    neighbor_index = quality_names.index("social_neighbor_available")
    availability = observed & (quality[..., neighbor_index] > 0.5)
    availability &= np.isfinite(social).all(axis=2)
    social[~availability] = 0.0
    if not np.isfinite(social).all():
        raise ValueError("social relation cache contains nonfinite values")
    window_index = order.window_index.copy().reset_index(drop=True)
    slot_index = _build_social_slot_index(ordered_slots, availability, frames)
    source_probe = single_source_probe_audit(slot_index)
    content = _cache_content_audit(
        social,
        availability,
        window_index,
        slot_index,
        exported,
        source_probe,
    )
    return {
        "social_relation": social,
        "availability": availability,
        "window_index": window_index,
        "slot_index": slot_index,
        "content_audit": content,
    }


def load_social_relation_cache(
    config: LegacyL6SocialRelationCacheConfig,
    *,
    cache_root: Path | None = None,
) -> LegacyL6SocialRelationCache:
    """Load and audit one numeric-social cache."""

    root = (cache_root or config.output_root).resolve()
    audit = audit_social_relation_cache(config, cache_root=root)
    if not audit["valid"]:
        raise ValueError(f"social relation cache audit failed={audit['errors']}")
    return LegacyL6SocialRelationCache(
        root=root,
        social_relation_path=root / CACHE_FILES["social_relation"],
        availability_path=root / CACHE_FILES["availability"],
        window_index=pd.read_csv(root / CACHE_FILES["window_index"]),
        slot_index=pd.read_csv(root / CACHE_FILES["slot_index"]),
        manifest=_read_json(root / CACHE_FILES["manifest"]),
        audit=audit,
    )


def audit_social_relation_cache(
    config: LegacyL6SocialRelationCacheConfig,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Re-hash and structurally verify a written social cache."""

    root = (cache_root or config.output_root).resolve()
    errors: list[str] = []
    verified = 0
    social_shape: list[int] = []
    availability_shape: list[int] = []
    available_slots = 0
    try:
        _require_inside(config.repo_root, root)
        manifest = _read_json(root / CACHE_FILES["manifest"])
        _validate_written_manifest(config, manifest)
        artifacts = _object(manifest["artifacts"], "artifacts")
        for name in (
            "social_relation",
            "availability",
            "window_index",
            "slot_index",
        ):
            spec = _object(artifacts[name], f"artifacts.{name}")
            path = root / str(spec["filename"])
            if file_sha256(path) != str(spec["sha256"]):
                errors.append(f"artifact_hash_mismatch={name}")
            elif int(path.stat().st_size) != int(spec["size_bytes"]):
                errors.append(f"artifact_size_mismatch={name}")
            else:
                verified += 1
        social = np.load(
            root / CACHE_FILES["social_relation"],
            mmap_mode="r",
        )
        availability = np.load(
            root / CACHE_FILES["availability"],
            mmap_mode="r",
        )
        try:
            social_shape = list(social.shape)
            availability_shape = list(availability.shape)
            expected_social = (
                EXPECTED_MODEL_WINDOWS,
                SEQUENCE_LENGTH,
                SOCIAL_RELATION_DIM,
            )
            expected_available = (EXPECTED_MODEL_WINDOWS, SEQUENCE_LENGTH)
            if social.shape != expected_social:
                errors.append(f"social_relation_shape={social_shape}")
            if availability.shape != expected_available:
                errors.append(f"availability_shape={availability_shape}")
            if social.dtype != SOCIAL_RELATION_DTYPE:
                errors.append(f"social_relation_dtype={social.dtype}")
            if availability.dtype != AVAILABILITY_DTYPE:
                errors.append(f"availability_dtype={availability.dtype}")
            if not np.isfinite(social).all():
                errors.append("social_relation_contains_nonfinite")
            if availability.shape == expected_available:
                available_slots = int(availability.sum())
                if np.any(social[~availability] != 0.0):
                    errors.append("unavailable_social_relation_is_nonzero")
        finally:
            _close_memmap(social)
            _close_memmap(availability)
        window_index = pd.read_csv(root / CACHE_FILES["window_index"])
        slot_index = pd.read_csv(root / CACHE_FILES["slot_index"])
        _validate_social_indexes(window_index, slot_index)
        observed = _ordered_sha256(window_index["window_id"])
        expected = str(manifest["parent_view"]["ordered_window_id_sha256"])
        if observed != expected:
            errors.append("ordered_window_id_sha256_mismatch")
        if int(manifest["content_audit"]["available_slots"]) != available_slots:
            errors.append("available_social_slot_count_mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    valid = not errors
    return {
        "schema_version": CACHE_AUDIT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_CACHE_AUDIT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_CACHE_AUDIT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "cache_root": str(root),
        "config_sha256": config.sha256,
        "manifest_sha256": (
            file_sha256(root / CACHE_FILES["manifest"])
            if (root / CACHE_FILES["manifest"]).is_file()
            else None
        ),
        "verified_artifacts": verified,
        "social_relation_shape": social_shape,
        "availability_shape": availability_shape,
        "available_slots": available_slots,
        "outer_holdout_slots_materialized": 0,
        "source_media_reads": 0,
        "errors": errors,
        "valid": valid,
    }


def evaluate_social_relation_cache_repeat(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Audit two caches and require exact semantic and artifact equality."""

    root = project_root.resolve()
    resolved = config_path.resolve()
    payload = _read_json(resolved)
    _validate_repeat_config(payload)
    packets: dict[str, dict[str, Any]] = {}
    for name in ("primary", "repeat"):
        spec = _object(payload[name], name)
        config_file = _resolve_inside(root, str(spec["config_path"]))
        manifest_file = _resolve_inside(root, str(spec["manifest_path"]))
        _validate_bound_file(
            config_file,
            str(spec["config_sha256"]),
            f"{name} social cache config",
        )
        _validate_bound_file(
            manifest_file,
            str(spec["manifest_sha256"]),
            f"{name} social cache manifest",
        )
        cache_config = load_social_relation_cache_config(config_file)
        audit = audit_social_relation_cache(
            cache_config,
            cache_root=manifest_file.parent,
        )
        if not audit["valid"]:
            raise ValueError(f"{name} social cache invalid={audit['errors']}")
        packets[name] = {
            "config": cache_config,
            "manifest_path": manifest_file,
            "manifest": _read_json(manifest_file),
            "audit": audit,
        }
    left = packets["primary"]
    right = packets["repeat"]
    semantic_fields = (
        "schema_version",
        "lineage_scope",
        "source_identity",
        "parents",
        "inputs",
        "order_authority",
        "features",
        "implementation",
    )
    semantic_mismatch = [
        name
        for name in semantic_fields
        if left["config"].payload[name] != right["config"].payload[name]
    ]
    artifact_rows: dict[str, Any] = {}
    artifact_errors: list[str] = []
    for name in (
        "social_relation",
        "availability",
        "window_index",
        "slot_index",
    ):
        first = left["manifest"]["artifacts"][name]
        second = right["manifest"]["artifacts"][name]
        equal = (
            first["sha256"] == second["sha256"]
            and int(first["size_bytes"]) == int(second["size_bytes"])
        )
        artifact_rows[name] = {
            "primary_sha256": first["sha256"],
            "repeat_sha256": second["sha256"],
            "equal": equal,
        }
        if not equal:
            artifact_errors.append(f"social_cache_artifact_diff={name}")
    content_equal = (
        left["manifest"]["content_audit"]
        == right["manifest"]["content_audit"]
    )
    errors = [
        *(f"semantic_config_diff={name}" for name in semantic_mismatch),
        *artifact_errors,
    ]
    if not content_equal:
        errors.append("social_cache_content_audit_diff")
    if left["manifest_path"].parent == right["manifest_path"].parent:
        errors.append("social_cache_repeat_output_roots_equal")
    valid = not errors
    return {
        "schema_version": REPEAT_GATE_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_CACHE_REPEAT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_CACHE_REPEAT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(resolved),
        "config_sha256": file_sha256(resolved),
        "primary": _repeat_packet_summary(left),
        "repeat": _repeat_packet_summary(right),
        "semantic_config_comparison": {
            "different_sections": semantic_mismatch,
            "valid": not semantic_mismatch,
        },
        "artifact_comparison": {
            "artifacts": artifact_rows,
            "all_artifact_sha256_equal": not artifact_errors,
            "valid": not artifact_errors,
        },
        "content_comparison": {
            "content_audit_equal": content_equal,
            "valid": content_equal,
        },
        "separate_output_roots": (
            left["manifest_path"].parent != right["manifest_path"].parent
        ),
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "errors": errors,
        "valid": valid,
    }


def configured_repeat_output_path(
    config_path: Path,
    project_root: Path,
) -> Path:
    payload = _read_json(config_path.resolve())
    _validate_repeat_config(payload)
    return _resolve_inside(project_root.resolve(), str(payload["output_path"]))


def _load_order_authority(
    config: LegacyL6SocialRelationCacheConfig,
) -> LegacyL6GeometryCache:
    order = _object(config.payload["order_authority"], "order_authority")
    cache_config = load_geometry_cache_config(
        config.bound_path("order_authority", "config")
    )
    root = _resolve_inside(
        config.repo_root,
        str(order["root_relative_path"]),
    )
    cache = load_geometry_cache(cache_config, cache_root=root)
    manifest = config.bound_path("order_authority", "manifest")
    if manifest != root / "geometry_cache_manifest.json":
        raise ValueError("social order-authority manifest path drift")
    if cache.audit.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("social order-authority manifest hash drift")
    return cache


def _load_social_frames(
    config: LegacyL6SocialRelationCacheConfig,
) -> pd.DataFrame:
    path = config.bound_path("inputs", "harmonized_frames")
    try:
        frames = pd.read_csv(path, usecols=list(SOCIAL_FRAME_COLUMNS))
    except ValueError as error:
        raise ValueError(f"social columns unavailable: {error}") from error
    _validate_social_frames(frames)
    return frames


def _validate_social_frames(frames: pd.DataFrame) -> None:
    missing = sorted(set(SOCIAL_FRAME_COLUMNS) - set(frames.columns))
    if missing:
        raise ValueError(f"social frames missing columns={missing}")
    if len(frames) != EXPECTED_RAW_ROWS:
        raise ValueError(f"social frame rows={len(frames)}")
    if set(frames["source_type"].astype(str)) != {SOURCE_TYPE}:
        raise ValueError("social frame source_type drift")
    if set(frames["dataset_id"].astype(str)) != {DATASET_ID}:
        raise ValueError("social frame dataset_id drift")
    _validate_claim_columns(frames, "social frames")
    if frames["frame_uid"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("social frames contain blank frame_uid")
    if frames["frame_uid"].astype(str).duplicated().any():
        raise ValueError("social frames contain duplicate frame_uid")
    keys = frames[["object_track_key", "frame_index"]].copy()
    keys["object_track_key"] = (
        keys["object_track_key"].fillna("").astype(str).str.strip()
    )
    keys["frame_index"] = pd.to_numeric(
        keys["frame_index"],
        errors="coerce",
    )
    invalid = (
        keys["object_track_key"].eq("")
        | keys["frame_index"].isna()
        | keys["frame_index"].mod(1).ne(0)
    )
    if invalid.any() or keys.duplicated().any():
        raise ValueError("social frame key contract failed")


def _validate_spatial_export(exported: SpatialSequenceExport) -> None:
    expected = {
        "rows": EXPECTED_MODEL_WINDOWS,
        "input_window_rows": EXPECTED_MODEL_WINDOWS,
        "aligned_window_rows": EXPECTED_MODEL_WINDOWS,
        "input_frame_rows": EXPECTED_RAW_ROWS,
        "aligned_frame_rows": EXPECTED_RAW_ROWS,
        "missing_frame_slots": 0,
        "observed_frame_slots": EXPECTED_MODEL_SLOTS,
        "padding_slots": 0,
        "missing_observed_slots_within_length": 0,
        "truncated_windows": 0,
        "social_rebased_windows": EXPECTED_MODEL_WINDOWS,
    }
    for field, value in expected.items():
        if exported.audit.get(field) != value:
            raise ValueError(
                f"social spatial export {field}="
                f"{exported.audit.get(field)!r}!={value!r}"
            )
    observed = tuple(exported.feature_names.get("social_relation", []))
    if observed != SOCIAL_RELATION_FEATURE_NAMES:
        raise ValueError(f"social feature order drift={observed}")
    model_x = [*observed, "social_relation_available"]
    forbidden = [
        name
        for name in model_x
        if name.endswith("_unit")
        or any(
            token in name.lower()
            for token in (
                "behavior",
                "label",
                "review",
                "path",
                "source",
                "dataset",
                "window",
                "frame_uid",
                "track",
                "pig_id",
                "fold",
            )
        )
    ]
    if forbidden:
        raise ValueError(f"forbidden social model-X fields={forbidden}")


def _build_social_slot_index(
    ordered_slots: pd.DataFrame,
    availability: np.ndarray,
    frames: pd.DataFrame,
) -> pd.DataFrame:
    frame = ordered_slots[
        [
            "cache_row",
            "window_id",
            "slot_index",
            "frame_uid",
            "object_track_key",
            "frame_index",
            "source_type",
            "dataset_id",
            "lineage_scope",
            "human_review_complete",
        ]
    ].copy()
    partner = frames[["frame_uid", "nearest_pig_id", "nearest_track_id"]]
    frame = frame.merge(partner, on="frame_uid", how="left", validate="many_to_one")
    frame["social_partner_key"] = _partner_keys(frame)
    frame["social_relation_available"] = availability.reshape(-1)
    frame["social_window_slot_uid"] = (
        frame["window_id"].astype(str)
        + "::slot="
        + frame["slot_index"].astype(str)
    )
    frame = frame.drop(columns=["nearest_pig_id", "nearest_track_id"])
    _validate_social_slot_index(frame)
    return frame


def _validate_social_slot_index(frame: pd.DataFrame) -> None:
    available = _strict_bool(
        frame["social_relation_available"],
        name="social relation availability",
    )
    partner = frame["social_partner_key"].fillna("").astype(str)
    if not partner.ne("").equals(available.reset_index(drop=True)):
        raise ValueError("social availability/partner key drift")
    if frame["social_window_slot_uid"].astype(str).duplicated().any():
        raise ValueError("duplicate social window-slot identities")


def _validate_social_indexes(
    windows: pd.DataFrame,
    slots: pd.DataFrame,
) -> None:
    _validate_order_frames(windows, slots)
    required = {
        "social_partner_key",
        "social_relation_available",
        "social_window_slot_uid",
    }
    missing = sorted(required - set(slots.columns))
    if missing:
        raise ValueError(f"social slot index missing columns={missing}")
    _validate_social_slot_index(slots)


def _partner_keys(frame: pd.DataFrame) -> pd.Series:
    pig = frame["nearest_pig_id"].fillna("").astype(str).str.strip()
    track = frame["nearest_track_id"].fillna("").astype(str).str.strip()
    return pig.where(pig.ne(""), track)


def _cache_content_audit(
    social: np.ndarray,
    availability: np.ndarray,
    windows: pd.DataFrame,
    slots: pd.DataFrame,
    exported: SpatialSequenceExport,
    source_probe: dict[str, Any],
) -> dict[str, Any]:
    patterns, counts = np.unique(
        availability.astype(np.uint8),
        axis=0,
        return_counts=True,
    )
    summaries = {
        name: {
            "min": float(social[..., index].min()),
            "max": float(social[..., index].max()),
            "mean": float(social[..., index].mean()),
            "std": float(social[..., index].std()),
        }
        for index, name in enumerate(SOCIAL_RELATION_FEATURE_NAMES)
    }
    return {
        "model_window_rows": int(len(windows)),
        "model_slot_rows": int(len(slots)),
        "role_window_counts": {
            str(name): int(value)
            for name, value in windows["l5_role"].value_counts().items()
        },
        "social_relation_shape": list(social.shape),
        "availability_shape": list(availability.shape),
        "social_relation_dtype": str(social.dtype),
        "availability_dtype": str(availability.dtype),
        "available_slots": int(availability.sum()),
        "unavailable_slots": int((~availability).sum()),
        "availability_pattern_count": int(len(patterns)),
        "availability_patterns": [
            {"pattern": row.astype(int).tolist(), "windows": int(count)}
            for row, count in zip(patterns, counts, strict=True)
        ],
        "feature_summaries": summaries,
        "ordered_window_id_sha256": _ordered_sha256(windows["window_id"]),
        "window_index_content_sha256": _dataframe_sha256(windows),
        "slot_index_content_sha256": _dataframe_sha256(slots),
        "spatial_export_audit": {
            "social_rebased_windows": int(
                exported.audit["social_rebased_windows"]
            ),
            "social_valid_pair_count": int(
                exported.audit["social_valid_pair_count"]
            ),
            "social_reset_row_count": int(
                exported.audit["social_reset_row_count"]
            ),
            "missing_frame_slots": 0,
        },
        "source_probe": source_probe,
        "numeric_social_only": True,
        "partner_identity_values_used": False,
        "top_k_partner_features_used": False,
        "unit_aggregate_features_used": False,
        "geometry_values_in_model_x": False,
        "motion_values_in_model_x": False,
        "roi_values_in_model_x": False,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "errors": [],
        "valid": True,
    }


def _validate_parent_decision(
    config: LegacyL6SocialRelationCacheConfig,
) -> None:
    decision = _read_json(config.bound_path("parents", "l6_roi_full_decision"))
    value = _object(decision.get("decision"), "ROI full decision")
    expected = "DO_NOT_EXPAND_ROI_RELATION_FROM_CURRENT_SHORT_EVIDENCE"
    if value.get("decision") != expected:
        raise ValueError("social cache parent ROI decision drift")
    if value.get("full_confirmation_complete") is not True:
        raise ValueError("social cache parent ROI confirmation incomplete")
    if value.get("next_action") != (
        "continue_l6_from_parameter_matched_zero_without_roi_values"
    ):
        raise ValueError("social cache parent ROI next action drift")


def _source_bindings(
    config: LegacyL6SocialRelationCacheConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in ("parents", "inputs"):
        for name, value in _object(config.payload[section], section).items():
            spec = _object(value, f"{section}.{name}")
            result[f"{section}.{name}"] = {
                "path": str(spec["path"]),
                "sha256": str(spec["sha256"]),
            }
    return result


def _validate_written_manifest(
    config: LegacyL6SocialRelationCacheConfig,
    manifest: dict[str, Any],
) -> None:
    expected = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_SOCIAL_RELATION_CACHE",
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": CANONICAL_SOURCE_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_sha256": config.sha256,
        "implementation_hashes": {
            name: str(spec["sha256"])
            for name, spec in config.payload["implementation"].items()
        },
        "errors": [],
        "valid": True,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"social manifest {field} drift")


def _validate_cache_config_payload(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "source_identity",
        "parents",
        "inputs",
        "order_authority",
        "features",
        "implementation",
        "execution_guard",
        "output",
    }
    _require_exact_keys(payload, required, "social cache config")
    expected = {
        "schema_version": CACHE_CONFIG_SCHEMA,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise ValueError(f"social cache {field}={payload[field]!r}")
    source = _object(payload["source_identity"], "source_identity")
    source_expected = {
        "canonical_short_name": CANONICAL_SOURCE_NAME,
        "expected_rows": EXPECTED_RAW_ROWS,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "merged_data": False,
    }
    for field, value in source_expected.items():
        if source.get(field) != value:
            raise ValueError(f"social cache source {field} drift")
    _require_sha(str(source["raw_sha256"]), "raw_sha256")
    if set(_object(payload["parents"], "parents")) != {
        "temporal_ladder_config",
        "l5_decision",
        "l6_roi_full_decision",
    }:
        raise ValueError("social cache parent set drift")
    if set(_object(payload["inputs"], "inputs")) != {"harmonized_frames"}:
        raise ValueError("social cache input set drift")
    for section in ("parents", "inputs"):
        for name, value in _object(payload[section], section).items():
            _validate_bound_spec(value, f"{section}.{name}")
    order = _object(payload["order_authority"], "order_authority")
    if set(order) != {
        "config",
        "manifest",
        "root_relative_path",
        "geometry_values_used",
    }:
        raise ValueError("social cache order authority keys drift")
    _validate_bound_spec(order["config"], "order.config")
    _validate_bound_spec(order["manifest"], "order.manifest")
    if order["geometry_values_used"] is not False:
        raise ValueError("social cache cannot consume geometry values")
    features = _object(payload["features"], "features")
    expected_features = {
        "view_id": VIEW_ID,
        "temporal_view_name": TEMPORAL_VIEW_NAME,
        "sequence_length": SEQUENCE_LENGTH,
        "model_window_rows": EXPECTED_MODEL_WINDOWS,
        "model_slot_rows": EXPECTED_MODEL_SLOTS,
        "feature_names": list(SOCIAL_RELATION_FEATURE_NAMES),
        "feature_dim": SOCIAL_RELATION_DIM,
        "feature_dtype": "float32",
        "availability_dtype": "bool",
        "window_local_rebase": True,
        "numeric_social_only": True,
        "partner_identity_values_allowed": False,
        "top_k_partner_features_allowed": False,
        "unit_aggregate_features_allowed": False,
        "source_media_fallback_allowed": False,
    }
    if features != expected_features:
        raise ValueError("social cache feature contract drift")
    implementation = _object(payload["implementation"], "implementation")
    expected_implementation = {
        "cache_builder",
        "spatial_exporter",
        "geometry_cache_helpers",
        "motion_cache_helpers",
    }
    if set(implementation) != expected_implementation:
        raise ValueError("social cache implementation set drift")
    for name, value in implementation.items():
        _validate_bound_spec(value, f"implementation.{name}")
    guard = _object(payload["execution_guard"], "execution_guard")
    if set(guard) != {"allowed_dirty_paths", "required_tracked_paths"}:
        raise ValueError("social cache execution guard keys drift")
    output = _object(payload["output"], "output")
    if set(output) != {"cache_root_relative_path"}:
        raise ValueError("social cache output keys drift")


def _validate_repeat_config(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "primary",
        "repeat",
        "output_path",
    }
    _require_exact_keys(payload, required, "social cache repeat config")
    expected = {
        "schema_version": REPEAT_CONFIG_SCHEMA,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise ValueError(f"social repeat {field} drift")
    fields = {
        "config_path",
        "config_sha256",
        "manifest_path",
        "manifest_sha256",
    }
    for name in ("primary", "repeat"):
        spec = _object(payload[name], name)
        if set(spec) != fields:
            raise ValueError(f"social repeat {name} keys drift")
        _require_sha(str(spec["config_sha256"]), f"{name} config sha")
        _require_sha(str(spec["manifest_sha256"]), f"{name} manifest sha")


def _repeat_packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": str(packet["config"].path),
        "config_sha256": packet["config"].sha256,
        "manifest_path": str(packet["manifest_path"]),
        "manifest_sha256": packet["audit"]["manifest_sha256"],
        "verified_artifacts": packet["audit"]["verified_artifacts"],
        "social_relation_shape": packet["audit"]["social_relation_shape"],
        "availability_shape": packet["audit"]["availability_shape"],
        "available_slots": packet["audit"]["available_slots"],
        "errors": [],
        "valid": True,
    }


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    path = Path(str(spec["path"]))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name}.path is not repository relative")
    _require_sha(str(spec["sha256"]), f"{name}.sha256")


def _validated_rows(values: np.ndarray, maximum: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("social cache rows must be a nonempty vector")
    if rows.min() < 0 or rows.max() >= maximum:
        raise ValueError("social cache rows are out of bounds")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("social cache rows contain duplicates")
    return rows
