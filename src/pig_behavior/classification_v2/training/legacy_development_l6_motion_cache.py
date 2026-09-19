"""Window-local T6 motion cache for legacy-only L6 development."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.spatial_sequence_export import (
    LEGACY_SPATIAL_FRAME_FEATURES,
    SpatialSequenceExport,
    export_legacy_development_spatial_sequences,
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
    load_geometry_cache,
    load_geometry_cache_config,
    single_source_probe_audit,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

EXPECTED_TEMPORAL_SLOT_ROWS = 109_296
MOTION_FEATURE_NAMES = tuple(LEGACY_SPATIAL_FRAME_FEATURES["motion_delta"])
MOTION_DIM = len(MOTION_FEATURE_NAMES)
MOTION_DTYPE = np.dtype(np.float32)
AVAILABILITY_DTYPE = np.dtype(np.bool_)

MOTION_POSITION_FIELDS = (
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
)
MOTION_QUALITY_FIELDS = (
    "bbox_valid",
    "actor_bbox_valid",
    "geometry_feature_valid",
    "spatiotemporal_feature_valid",
)
MOTION_FRAME_COLUMNS = (
    "source_type",
    "dataset_id",
    "frame_uid",
    "object_track_key",
    "frame_index",
    "timestamp_sec",
    "lineage_scope",
    "human_review_complete",
    *MOTION_POSITION_FIELDS,
    *MOTION_FEATURE_NAMES,
    *MOTION_QUALITY_FIELDS,
)

CACHE_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_cache_config.v1"
)
CACHE_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_cache_manifest.v1"
)
CACHE_AUDIT_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_cache_audit.v1"
)

CACHE_FILES = {
    "motion": "motion_raw_f32.npy",
    "availability": "motion_available_bool.npy",
    "window_index": "motion_window_index.csv",
    "slot_index": "motion_slot_index.csv",
    "manifest": "motion_cache_manifest.json",
}


@dataclass(frozen=True, slots=True)
class LegacyL6MotionCacheConfig:
    """Hash-bound motion cache build specification."""

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

    def bound_path(self, section: str, name: str | None = None) -> Path:
        value: Any = self.payload[section]
        if name is not None:
            value = value[name]
        spec = _object(value, f"{section}.{name}" if name else section)
        return _resolve_inside(self.repo_root, str(spec["path"]))


@dataclass(frozen=True, slots=True)
class LegacyL6MotionCache:
    """Audited motion arrays aligned to the frozen L5 T6 view."""

    root: Path
    motion_path: Path
    availability_path: Path
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def load_motion(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Copy cache rows and close the mmap before returning."""

        mapping = np.load(self.motion_path, mmap_mode="r")
        try:
            if rows is None:
                values = np.asarray(mapping, dtype=MOTION_DTYPE).copy()
            else:
                indices = _validated_rows(rows, len(self.window_index))
                values = np.asarray(
                    mapping[indices],
                    dtype=MOTION_DTYPE,
                ).copy()
        finally:
            _close_memmap(mapping)
        return values

    def load_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        """Copy cache availability rows and close the mmap."""

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


def load_motion_cache_config(path: Path) -> LegacyL6MotionCacheConfig:
    """Load one motion-cache config and verify every immutable parent."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_cache_config_payload(payload)
    config = LegacyL6MotionCacheConfig(
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
    for section in ("parents", "inputs", "implementation"):
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
    _validate_parent_decision(config)
    _load_order_authority(config)
    return config


def preflight_motion_cache(
    config: LegacyL6MotionCacheConfig,
) -> dict[str, Any]:
    """Run the CPU-only lineage, order, Git, and output gate."""

    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    windows = 0
    slots = 0
    roles: dict[str, int] = {}
    try:
        order = _load_order_authority(config)
        windows = len(order.window_index)
        slots = len(order.slot_index)
        roles = {
            str(key): int(value)
            for key, value in order.window_index["l5_role"]
            .value_counts()
            .items()
        }
        if windows != EXPECTED_MODEL_WINDOWS:
            errors.append(f"model_window_rows={windows}")
        if slots != EXPECTED_MODEL_SLOTS:
            errors.append(f"model_slot_rows={slots}")
        if roles != {"train": 14_608, "validation": 980}:
            errors.append(f"model_window_roles={roles}")
        if config.output_root.exists():
            errors.append(f"cache_output_exists={config.output_root}")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = motion_cache_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("motion cache preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "motion_cache_preflight.v1"
        ),
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_MOTION_CACHE_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_MOTION_CACHE_PREFLIGHT"
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
        "model_window_roles": roles,
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "git_guard": git_guard,
        "build_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def build_motion_cache(
    config: LegacyL6MotionCacheConfig,
) -> tuple[Path, dict[str, Any]]:
    """Build one immutable cache from canonical window-local motion export."""

    preflight = preflight_motion_cache(config)
    if not preflight["build_authorized"]:
        raise RuntimeError(f"motion cache preflight failed={preflight['errors']}")
    order = _load_order_authority(config)
    frames = _load_harmonized_frames(config)
    result = materialize_motion_cache(order, frames)
    root = config.output_root
    temporary = root.with_name(f"{root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"motion cache temporary output exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    paths = {name: temporary / filename for name, filename in CACHE_FILES.items()}
    np.save(paths["motion"], result["motion"], allow_pickle=False)
    np.save(
        paths["availability"],
        result["availability"],
        allow_pickle=False,
    )
    result["window_index"].to_csv(
        paths["window_index"],
        index=False,
        lineterminator="\n",
    )
    result["slot_index"].to_csv(
        paths["slot_index"],
        index=False,
        lineterminator="\n",
    )
    artifacts = {
        name: {
            "filename": CACHE_FILES[name],
            "sha256": file_sha256(paths[name]),
            "size_bytes": int(paths[name].stat().st_size),
        }
        for name in ("motion", "availability", "window_index", "slot_index")
    }
    implementation = {
        name: {
            "path": str(spec["path"]),
            "sha256": str(spec["sha256"]),
        }
        for name, value in config.payload["implementation"].items()
        for spec in [_object(value, f"implementation.{name}")]
    }
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_MOTION_CACHE",
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
        "implementation": implementation,
        "git_guard": preflight["git_guard"],
        "parent_view": {
            "view_id": VIEW_ID,
            "temporal_view_name": TEMPORAL_VIEW_NAME,
            "sequence_length": SEQUENCE_LENGTH,
            "model_window_rows": EXPECTED_MODEL_WINDOWS,
            "model_slot_rows": EXPECTED_MODEL_SLOTS,
            "ordered_window_id_sha256": _ordered_sha256(
                result["window_index"]["window_id"]
            ),
            "geometry_values_consumed": False,
            "order_authority_manifest_sha256": str(
                config.payload["order_authority"]["manifest"]["sha256"]
            ),
        },
        "source_bindings": _source_bindings(config),
        "feature_contract": {
            "feature_names": list(MOTION_FEATURE_NAMES),
            "feature_dim": MOTION_DIM,
            "feature_dtype": str(MOTION_DTYPE),
            "availability_dtype": str(AVAILABILITY_DTYPE),
            "motion_source": "canonical_spatial_sequence_export_v1",
            "window_local_rebase_required": True,
            "first_slot_motion_zero_required": True,
            "first_slot_availability_false_required": True,
            "normalization": "none_raw_cache_fold_train_pairs_only_at_consumer",
            "availability_is_behavior_evidence": False,
            "labels_ids_paths_folds_review_or_unit_aggregates_in_model_x": False,
            "source_media_fallback_allowed": False,
        },
        "cache_scope": {
            "roles": ["train", "validation"],
            "outer_holdout_slots_materialized": 0,
            "source_media_reads": 0,
            "video_decodes": 0,
        },
        "content_audit": result["content_audit"],
        "artifacts": artifacts,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(paths["manifest"], manifest)
    temporary.replace(root)
    audit = audit_motion_cache(config, cache_root=root)
    if not audit["valid"]:
        raise RuntimeError(f"written motion cache failed audit={audit['errors']}")
    return root / CACHE_FILES["manifest"], audit


def materialize_motion_cache(
    order: LegacyL6GeometryCache,
    frames: pd.DataFrame,
) -> dict[str, Any]:
    """Create motion arrays from an order authority and harmonized frames."""

    windows, ordered_slots = _windows_from_order(order)
    _validate_harmonized_frames(frames)
    export_frames = frames[
        [
            "object_track_key",
            "frame_index",
            "timestamp_sec",
            *MOTION_POSITION_FIELDS,
            *MOTION_FEATURE_NAMES,
            *MOTION_QUALITY_FIELDS,
        ]
    ].copy()
    exported = export_legacy_development_spatial_sequences(
        windows,
        export_frames,
        max_window_length=SEQUENCE_LENGTH,
        feature_schema={
            "motion_delta": list(
                LEGACY_SPATIAL_FRAME_FEATURES["motion_delta"]
            ),
            "quality_mask": list(
                LEGACY_SPATIAL_FRAME_FEATURES["quality_mask"][:4]
            ),
        },
    )
    _validate_spatial_export(exported)
    motion = np.asarray(
        exported.arrays["motion_delta"],
        dtype=MOTION_DTYPE,
    ).copy()
    availability = _motion_pair_availability(exported)
    if np.any(motion[~availability] != 0.0):
        raise ValueError("unavailable motion slots contain nonzero values")
    window_index = order.window_index.copy().reset_index(drop=True)
    slot_index = _build_motion_slot_index(
        ordered_slots,
        availability,
    )
    reset_audit = _window_start_reset_audit(ordered_slots, frames, motion)
    source_probe = single_source_probe_audit(slot_index)
    content = _cache_content_audit(
        motion,
        availability,
        window_index,
        slot_index,
        exported,
        reset_audit=reset_audit,
        source_probe=source_probe,
    )
    return {
        "motion": motion,
        "availability": availability,
        "window_index": window_index,
        "slot_index": slot_index,
        "content_audit": content,
    }


def load_motion_cache(
    config: LegacyL6MotionCacheConfig,
    *,
    cache_root: Path | None = None,
) -> LegacyL6MotionCache:
    """Load and audit one motion cache without retaining mmap handles."""

    root = (cache_root or config.output_root).resolve()
    audit = audit_motion_cache(config, cache_root=root)
    if not audit["valid"]:
        raise ValueError(f"motion cache audit failed={audit['errors']}")
    return LegacyL6MotionCache(
        root=root,
        motion_path=root / CACHE_FILES["motion"],
        availability_path=root / CACHE_FILES["availability"],
        window_index=pd.read_csv(root / CACHE_FILES["window_index"]),
        slot_index=pd.read_csv(root / CACHE_FILES["slot_index"]),
        manifest=_read_json(root / CACHE_FILES["manifest"]),
        audit=audit,
    )


def audit_motion_cache(
    config: LegacyL6MotionCacheConfig,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Re-hash and structurally verify a written motion cache."""

    root = (cache_root or config.output_root).resolve()
    errors: list[str] = []
    verified = 0
    motion_shape: list[int] = []
    availability_shape: list[int] = []
    available_slots = 0
    try:
        _require_inside(config.repo_root, root)
        manifest = _read_json(root / CACHE_FILES["manifest"])
        _validate_written_manifest(config, manifest)
        artifacts = _object(manifest["artifacts"], "artifacts")
        for name in ("motion", "availability", "window_index", "slot_index"):
            spec = _object(artifacts[name], f"artifacts.{name}")
            path = root / str(spec["filename"])
            if file_sha256(path) != str(spec["sha256"]):
                errors.append(f"artifact_hash_mismatch={name}")
            elif int(path.stat().st_size) != int(spec["size_bytes"]):
                errors.append(f"artifact_size_mismatch={name}")
            else:
                verified += 1
        motion = np.load(root / CACHE_FILES["motion"], mmap_mode="r")
        availability = np.load(
            root / CACHE_FILES["availability"],
            mmap_mode="r",
        )
        try:
            motion_shape = list(motion.shape)
            availability_shape = list(availability.shape)
            expected_motion = (
                EXPECTED_MODEL_WINDOWS,
                SEQUENCE_LENGTH,
                MOTION_DIM,
            )
            expected_available = (EXPECTED_MODEL_WINDOWS, SEQUENCE_LENGTH)
            if motion.shape != expected_motion:
                errors.append(f"motion_shape={motion_shape}")
            if availability.shape != expected_available:
                errors.append(f"availability_shape={availability_shape}")
            if motion.dtype != MOTION_DTYPE:
                errors.append(f"motion_dtype={motion.dtype}")
            if availability.dtype != AVAILABILITY_DTYPE:
                errors.append(f"availability_dtype={availability.dtype}")
            if not np.isfinite(motion).all():
                errors.append("motion_contains_nonfinite")
            if availability.shape == expected_available:
                available_slots = int(availability.sum())
                if availability[:, 0].any():
                    errors.append("first_slot_availability_not_false")
                if np.any(motion[~availability] != 0.0):
                    errors.append("unavailable_motion_is_nonzero")
        finally:
            _close_memmap(motion)
            _close_memmap(availability)
        window_index = pd.read_csv(root / CACHE_FILES["window_index"])
        slot_index = pd.read_csv(root / CACHE_FILES["slot_index"])
        _validate_motion_indexes(window_index, slot_index)
        observed = _ordered_sha256(window_index["window_id"])
        expected = str(manifest["parent_view"]["ordered_window_id_sha256"])
        if observed != expected:
            errors.append("ordered_window_id_sha256_mismatch")
        if int(manifest["content_audit"]["available_pair_slots"]) != (
            available_slots
        ):
            errors.append("available_pair_slot_count_mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    valid = not errors
    return {
        "schema_version": CACHE_AUDIT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_MOTION_CACHE_AUDIT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_MOTION_CACHE_AUDIT"
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
        "motion_shape": motion_shape,
        "availability_shape": availability_shape,
        "available_pair_slots": available_slots,
        "first_slot_unavailable_rows": EXPECTED_MODEL_WINDOWS,
        "outer_holdout_slots_materialized": 0,
        "source_media_reads": 0,
        "errors": errors,
        "valid": valid,
    }


def _load_order_authority(
    config: LegacyL6MotionCacheConfig,
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
    expected_manifest = config.bound_path("order_authority", "manifest")
    if expected_manifest != root / "geometry_cache_manifest.json":
        raise ValueError("motion order-authority manifest path drift")
    if cache.audit.get("manifest_sha256") != file_sha256(expected_manifest):
        raise ValueError("motion order-authority manifest audit hash drift")
    return cache


def _load_harmonized_frames(
    config: LegacyL6MotionCacheConfig,
) -> pd.DataFrame:
    path = config.bound_path("inputs", "harmonized_frames")
    try:
        frame = pd.read_csv(path, usecols=list(MOTION_FRAME_COLUMNS))
    except ValueError as error:
        raise ValueError(f"harmonized motion columns unavailable: {error}") from error
    _validate_harmonized_frames(frame)
    return frame


def _validate_harmonized_frames(frame: pd.DataFrame) -> None:
    missing = sorted(set(MOTION_FRAME_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"harmonized motion frames missing columns={missing}")
    if len(frame) != EXPECTED_RAW_ROWS:
        raise ValueError(f"harmonized motion frame rows={len(frame)}")
    if set(frame["source_type"].astype(str)) != {SOURCE_TYPE}:
        raise ValueError("harmonized motion source_type drift")
    if set(frame["dataset_id"].astype(str)) != {DATASET_ID}:
        raise ValueError("harmonized motion dataset_id drift")
    _validate_claim_columns(frame, "harmonized motion frames")
    if frame["frame_uid"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("harmonized motion frames contain blank frame_uid")
    if frame["frame_uid"].astype(str).duplicated().any():
        raise ValueError("harmonized motion frames contain duplicate frame_uid")
    keys = frame[["object_track_key", "frame_index"]].copy()
    keys["object_track_key"] = (
        keys["object_track_key"].fillna("").astype(str).str.strip()
    )
    keys["frame_index"] = pd.to_numeric(keys["frame_index"], errors="coerce")
    invalid = (
        keys["object_track_key"].eq("")
        | keys["frame_index"].isna()
        | keys["frame_index"].mod(1).ne(0)
    )
    if invalid.any() or keys.duplicated().any():
        raise ValueError(
            "harmonized motion frame key contract failed "
            f"invalid={int(invalid.sum())} duplicate={int(keys.duplicated().sum())}"
        )


def _windows_from_order(
    order: LegacyL6GeometryCache,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = order.window_index.copy().reset_index(drop=True)
    slots = order.slot_index.copy()
    _validate_order_frames(windows, slots)
    slots["cache_row"] = pd.to_numeric(slots["cache_row"]).astype(np.int64)
    slots["slot_index"] = pd.to_numeric(slots["slot_index"]).astype(np.int64)
    slots["frame_index"] = pd.to_numeric(slots["frame_index"]).astype(np.int64)
    slots = slots.sort_values(
        ["cache_row", "slot_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    object_keys = slots["object_track_key"].astype(str).to_numpy().reshape(
        EXPECTED_MODEL_WINDOWS,
        SEQUENCE_LENGTH,
    )
    frame_indices = slots["frame_index"].to_numpy().reshape(
        EXPECTED_MODEL_WINDOWS,
        SEQUENCE_LENGTH,
    )
    if np.any(object_keys != object_keys[:, :1]):
        raise ValueError("motion order window crosses object tracks")
    if not np.all(np.diff(frame_indices, axis=1) == 1):
        raise ValueError("motion order window frame indices are not contiguous")
    export_windows = pd.DataFrame(
        {
            "window_id": windows["window_id"].astype(str),
            "object_track_key": object_keys[:, 0],
            "window_start_frame": frame_indices[:, 0],
            "window_end_frame": frame_indices[:, -1],
            "window_length_frames": SEQUENCE_LENGTH,
        }
    )
    return export_windows, slots


def _validate_order_frames(
    windows: pd.DataFrame,
    slots: pd.DataFrame,
) -> None:
    required_windows = {
        "cache_row",
        "window_id",
        "temporal_unit_key",
        "l5_role",
        "source_type",
        "dataset_id",
        "lineage_scope",
        "human_review_complete",
        "sequence_length",
    }
    required_slots = {
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
    }
    missing_windows = sorted(required_windows - set(windows.columns))
    missing_slots = sorted(required_slots - set(slots.columns))
    if missing_windows or missing_slots:
        raise ValueError(
            "motion order columns missing "
            f"windows={missing_windows} slots={missing_slots}"
        )
    if len(windows) != EXPECTED_MODEL_WINDOWS:
        raise ValueError(f"motion order window rows={len(windows)}")
    if len(slots) != EXPECTED_MODEL_SLOTS:
        raise ValueError(f"motion order slot rows={len(slots)}")
    expected_rows = np.arange(EXPECTED_MODEL_WINDOWS, dtype=np.int64)
    observed_rows = pd.to_numeric(windows["cache_row"]).to_numpy(np.int64)
    if not np.array_equal(observed_rows, expected_rows):
        raise ValueError("motion order window cache_row drift")
    ordered = slots.copy()
    ordered["cache_row"] = pd.to_numeric(ordered["cache_row"]).astype(np.int64)
    ordered["slot_index"] = pd.to_numeric(ordered["slot_index"]).astype(np.int64)
    ordered = ordered.sort_values(
        ["cache_row", "slot_index"],
        kind="mergesort",
    )
    expected_cache_rows = np.repeat(expected_rows, SEQUENCE_LENGTH)
    expected_slot_rows = np.tile(
        np.arange(SEQUENCE_LENGTH, dtype=np.int64),
        EXPECTED_MODEL_WINDOWS,
    )
    if not np.array_equal(ordered["cache_row"], expected_cache_rows):
        raise ValueError("motion order slot cache_row drift")
    if not np.array_equal(ordered["slot_index"], expected_slot_rows):
        raise ValueError("motion order slot_index drift")
    expected_ids = np.repeat(
        windows["window_id"].astype(str).to_numpy(),
        SEQUENCE_LENGTH,
    )
    if not np.array_equal(ordered["window_id"].astype(str), expected_ids):
        raise ValueError("motion order slot window_id drift")
    _validate_claim_columns(windows, "motion order window index")
    _validate_claim_columns(slots, "motion order slot index")


def _validate_spatial_export(exported: SpatialSequenceExport) -> None:
    audit = exported.audit
    if audit.get("errors"):
        raise ValueError(f"motion spatial export errors={audit['errors']}")
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
        "motion_rebased_windows": EXPECTED_MODEL_WINDOWS,
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ValueError(
                f"motion spatial export {field}={audit.get(field)!r}!={value!r}"
            )
    observed_names = tuple(exported.feature_names.get("motion_delta", []))
    if observed_names != MOTION_FEATURE_NAMES:
        raise ValueError(f"motion feature order drift={observed_names}")
    selected_model_x = [*observed_names, "motion_available"]
    forbidden = [
        name
        for name in selected_model_x
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
                "fold",
            )
        )
    ]
    if forbidden:
        raise ValueError(f"forbidden motion model-X fields={forbidden}")


def _motion_pair_availability(
    exported: SpatialSequenceExport,
) -> np.ndarray:
    observed = np.asarray(exported.arrays["observed_mask"], dtype=bool)
    quality = np.asarray(exported.arrays["quality_mask"], dtype=np.float32)
    names = list(exported.feature_names["quality_mask"])
    missing = sorted(set(MOTION_QUALITY_FIELDS) - set(names))
    if missing:
        raise ValueError(f"motion quality fields missing={missing}")
    row_valid = observed.copy()
    for field in MOTION_QUALITY_FIELDS:
        row_valid &= quality[..., names.index(field)] > 0.5
    adjacent = np.asarray(
        exported.arrays["adjacent_motion_pair_mask"],
        dtype=bool,
    )
    availability = (
        adjacent & observed & row_valid
    ).astype(AVAILABILITY_DTYPE)
    expected_pairs = int(exported.audit["motion_adjacent_pair_count"])
    if int(availability.sum()) != expected_pairs:
        raise ValueError(
            "motion availability/export pair count drift "
            f"{int(availability.sum())}!={expected_pairs}"
        )
    return availability


def _build_motion_slot_index(
    ordered_slots: pd.DataFrame,
    availability: np.ndarray,
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
    available = availability.reshape(-1)
    frame["motion_available"] = available
    previous = frame.groupby("cache_row", sort=False)["frame_uid"].shift(1)
    frame["previous_frame_uid"] = previous.fillna("").astype(str)
    pair = frame["previous_frame_uid"] + "->" + frame["frame_uid"].astype(str)
    frame["motion_pair_uid"] = pair.where(available, "")
    columns = [
        "cache_row",
        "window_id",
        "slot_index",
        "frame_uid",
        "previous_frame_uid",
        "motion_pair_uid",
        "object_track_key",
        "frame_index",
        "source_type",
        "dataset_id",
        "motion_available",
        "lineage_scope",
        "human_review_complete",
    ]
    frame = frame[columns]
    _validate_motion_slot_pairs(frame)
    return frame


def _validate_motion_slot_pairs(frame: pd.DataFrame) -> None:
    available = _strict_bool(
        frame["motion_available"],
        name="motion slot availability",
    )
    first = frame["slot_index"].astype(int).eq(0)
    previous_frame_uid = (
        frame["previous_frame_uid"].fillna("").astype(str)
    )
    motion_pair_uid = frame["motion_pair_uid"].fillna("").astype(str)
    if available[first].any():
        raise ValueError("motion first slots are available")
    if previous_frame_uid[first].ne("").any():
        raise ValueError("motion first slots expose previous frame IDs")
    if motion_pair_uid[first].ne("").any():
        raise ValueError("motion first slots expose pair IDs")
    if motion_pair_uid[available].eq("").any():
        raise ValueError("available motion slots lack pair IDs")
    conflicts = (
        frame.loc[available]
        .assign(motion_pair_uid=motion_pair_uid[available])
        .groupby("motion_pair_uid", sort=False)["frame_uid"]
        .nunique(dropna=False)
    )
    if conflicts.gt(1).any():
        raise ValueError("motion pair IDs map to conflicting current frames")


def _window_start_reset_audit(
    ordered_slots: pd.DataFrame,
    frames: pd.DataFrame,
    motion: np.ndarray,
) -> dict[str, Any]:
    first = ordered_slots.loc[ordered_slots["slot_index"].astype(int).eq(0)]
    raw = first[["frame_uid", "frame_index"]].merge(
        frames[["frame_uid", *MOTION_FEATURE_NAMES]],
        on="frame_uid",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing_frames = int(raw["_merge"].ne("both").sum())
    if missing_frames:
        raise ValueError(f"motion reset audit missing frame joins={missing_frames}")
    numeric = raw[list(MOTION_FEATURE_NAMES)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    raw_values = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_values = raw_values.to_numpy(dtype=np.float64)
    raw_nonzero = np.any(np.abs(raw_values) > 1e-12, axis=1)
    starts_after_zero = pd.to_numeric(raw["frame_index"]).gt(0).to_numpy()
    cached_first = motion[:, 0, :]
    cached_nonzero = np.any(cached_first != 0.0, axis=1)
    if cached_nonzero.any():
        raise ValueError("window-local motion reset left nonzero first slots")
    return {
        "window_rows": int(len(raw)),
        "windows_starting_after_frame_zero": int(starts_after_zero.sum()),
        "raw_nonzero_motion_at_window_start_rows": int(raw_nonzero.sum()),
        "raw_nonzero_after_frame_zero_start_rows": int(
            (raw_nonzero & starts_after_zero).sum()
        ),
        "cached_nonzero_first_slot_rows": int(cached_nonzero.sum()),
        "outside_window_motion_context_consumed": False,
        "valid": True,
    }


def _cache_content_audit(
    motion: np.ndarray,
    availability: np.ndarray,
    window_index: pd.DataFrame,
    slot_index: pd.DataFrame,
    exported: SpatialSequenceExport,
    *,
    reset_audit: dict[str, Any],
    source_probe: dict[str, Any],
) -> dict[str, Any]:
    roles = {
        str(key): int(value)
        for key, value in window_index["l5_role"].value_counts().items()
    }
    available = int(availability.sum())
    unavailable = int(availability.size - available)
    flattened = motion.reshape(-1, MOTION_DIM).astype(np.float64)
    statistics = {
        name: {
            "minimum": float(flattened[:, index].min()),
            "maximum": float(flattened[:, index].max()),
            "mean": float(flattened[:, index].mean()),
            "population_std": float(flattened[:, index].std(ddof=0)),
        }
        for index, name in enumerate(MOTION_FEATURE_NAMES)
    }
    availability_patterns = np.unique(availability, axis=0)
    if availability_patterns.shape[0] != 1:
        raise ValueError(
            "motion availability pattern varies across model windows "
            f"patterns={availability_patterns.shape[0]}"
        )
    return {
        "model_window_rows": int(len(window_index)),
        "model_slot_rows": int(len(slot_index)),
        "role_window_counts": roles,
        "available_pair_slots": available,
        "unavailable_slots": unavailable,
        "unavailable_first_slots": int((~availability[:, 0]).sum()),
        "unavailable_nonfirst_slots": int((~availability[:, 1:]).sum()),
        "motion_shape": list(motion.shape),
        "availability_shape": list(availability.shape),
        "motion_dtype": str(motion.dtype),
        "availability_dtype": str(availability.dtype),
        "motion_statistics": statistics,
        "availability_pattern_count": int(availability_patterns.shape[0]),
        "availability_pattern": availability_patterns[0].astype(int).tolist(),
        "ordered_window_id_sha256": _ordered_sha256(
            window_index["window_id"]
        ),
        "window_index_content_sha256": _dataframe_sha256(window_index),
        "slot_index_content_sha256": _dataframe_sha256(slot_index),
        "spatial_export_audit": exported.audit,
        "window_start_reset_audit": reset_audit,
        "source_probe": source_probe,
        "availability_only_is_diagnostic_not_behavior_evidence": True,
        "unit_aggregate_features_selected": [],
        "geometry_values_consumed": False,
        "outer_holdout_slots_materialized": 0,
        "source_media_reads": 0,
        "errors": [],
        "valid": True,
    }


def _validate_motion_indexes(
    window_index: pd.DataFrame,
    slot_index: pd.DataFrame,
) -> None:
    if len(window_index) != EXPECTED_MODEL_WINDOWS:
        raise ValueError(f"motion window index rows={len(window_index)}")
    if len(slot_index) != EXPECTED_MODEL_SLOTS:
        raise ValueError(f"motion slot index rows={len(slot_index)}")
    expected = np.arange(EXPECTED_MODEL_WINDOWS, dtype=np.int64)
    if not np.array_equal(
        pd.to_numeric(window_index["cache_row"]).to_numpy(np.int64),
        expected,
    ):
        raise ValueError("motion window index cache_row drift")
    if window_index["window_id"].astype(str).duplicated().any():
        raise ValueError("motion window index IDs are duplicated")
    if slot_index[["window_id", "slot_index"]].duplicated().any():
        raise ValueError("motion slot index keys are duplicated")
    _validate_claim_columns(window_index, "motion window index")
    _validate_claim_columns(slot_index, "motion slot index")
    _validate_motion_slot_pairs(slot_index)


def motion_cache_git_guard(
    config: LegacyL6MotionCacheConfig,
) -> dict[str, Any]:
    """Require committed cache sources/config and preserve known user dirt."""

    guard = _object(config.payload["execution_guard"], "execution_guard")
    status = _git(
        config.repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    entries = [line for line in status.splitlines() if line.strip()]
    observed = sorted(_status_path(line) for line in entries)
    allowed = sorted(
        str(path).replace("\\", "/") for path in guard["allowed_dirty_paths"]
    )
    unexpected = sorted(set(observed).difference(allowed))
    required = [
        str(path).replace("\\", "/")
        for path in guard["required_tracked_paths"]
    ]
    untracked: list[str] = []
    for path in required:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(config.repo_root),
                "ls-files",
                "--error-unmatch",
                "--",
                path,
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            untracked.append(path)
    errors: list[str] = []
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if untracked:
        errors.append(f"required_paths_untracked={untracked}")
    return {
        "code_sha": _git(config.repo_root, "rev-parse", "HEAD").strip(),
        "dirty_entries": entries,
        "allowed_dirty_paths": allowed,
        "observed_dirty_paths": observed,
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": required,
        "untracked_required_paths": untracked,
        "errors": errors,
        "valid": not errors,
    }


def _validate_parent_decision(config: LegacyL6MotionCacheConfig) -> None:
    decision = _read_json(config.bound_path("parents", "l5_decision"))
    if decision.get("lineage_scope") != LINEAGE_SCOPE:
        raise ValueError("L5 decision lineage scope drift")
    expected_status = "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_DECISION"
    if decision.get("status") != expected_status:
        raise ValueError("L5 temporal decision status is not PASS")
    selected = _object(decision.get("decision"), "L5 decision")
    if selected.get("selected_working_view") != VIEW_ID:
        raise ValueError("L5 temporal decision did not select t6_sliding")


def _source_bindings(config: LegacyL6MotionCacheConfig) -> dict[str, Any]:
    source = _object(config.payload["source_identity"], "source_identity")
    return {
        "raw_authority": {
            "path": str(source["raw_authority_path"]),
            "sha256": str(source["raw_sha256"]),
            "rows": EXPECTED_RAW_ROWS,
        },
        "parents": config.payload["parents"],
        "inputs": config.payload["inputs"],
        "order_authority": config.payload["order_authority"],
    }


def _validate_written_manifest(
    config: LegacyL6MotionCacheConfig,
    manifest: dict[str, Any],
) -> None:
    expected = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_MOTION_CACHE",
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
        "valid": True,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"motion cache manifest {field}="
                f"{manifest.get(field)!r}!={value!r}"
            )
    content = _object(manifest.get("content_audit"), "content_audit")
    if not content.get("valid") or content.get("errors"):
        raise ValueError("motion cache manifest content audit is invalid")
    if content.get("geometry_values_consumed") is not False:
        raise ValueError("motion cache consumed geometry values")
    if content.get("unit_aggregate_features_selected") != []:
        raise ValueError("motion cache selected unit aggregates")
    source_probe = _object(content.get("source_probe"), "source_probe")
    if source_probe.get("status") != "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE":
        raise ValueError("motion cache source probe status drift")


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
    _require_exact_keys(payload, required, "motion cache config")
    identity = {
        "schema_version": CACHE_CONFIG_SCHEMA,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
    }
    for field, value in identity.items():
        if payload[field] != value:
            raise ValueError(f"motion cache config {field}={payload[field]!r}")
    source = _object(payload["source_identity"], "source_identity")
    source_expected = {
        "canonical_short_name": CANONICAL_SOURCE_NAME,
        "raw_authority_path": (
            "data/raw/legacy_full_multigt_masked_nodup_16f/"
            "legacy_dense_tracklet_map.csv"
        ),
        "raw_sha256": (
            "ff73c158ef879eb8177b0c18783fc751945fe1d6af97a4b8235cd71681fabbcb"
        ),
        "expected_rows": EXPECTED_RAW_ROWS,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "merged_data": False,
    }
    if source != source_expected:
        raise ValueError(f"motion cache source identity drift={source}")
    parents = _object(payload["parents"], "parents")
    _require_exact_keys(
        parents,
        {"temporal_ladder_config", "l5_decision"},
        "parents",
    )
    inputs = _object(payload["inputs"], "inputs")
    _require_exact_keys(
        inputs,
        {"harmonized_frames", "temporal_slot_manifest"},
        "inputs",
    )
    expected_input_rows = {
        "harmonized_frames": EXPECTED_RAW_ROWS,
        "temporal_slot_manifest": EXPECTED_TEMPORAL_SLOT_ROWS,
    }
    for section_name, values in (
        ("parents", parents),
        ("inputs", inputs),
    ):
        for name, value in values.items():
            spec = _object(value, f"{section_name}.{name}")
            expected_keys = {"path", "sha256"}
            if section_name == "inputs":
                expected_keys.add("expected_rows")
            _require_exact_keys(spec, expected_keys, f"{section_name}.{name}")
            _require_sha(str(spec["sha256"]), f"{section_name}.{name}.sha256")
            if section_name == "inputs" and int(spec["expected_rows"]) != (
                expected_input_rows[name]
            ):
                raise ValueError(f"inputs.{name}.expected_rows drift")
    order = _object(payload["order_authority"], "order_authority")
    _require_exact_keys(
        order,
        {"config", "manifest", "root_relative_path", "geometry_values_used"},
        "order_authority",
    )
    for name in ("config", "manifest"):
        spec = _object(order[name], f"order_authority.{name}")
        _require_exact_keys(spec, {"path", "sha256"}, f"order_authority.{name}")
        _require_sha(str(spec["sha256"]), f"order_authority.{name}.sha256")
    if order["geometry_values_used"] is not False:
        raise ValueError("motion cache must not use geometry values")
    features = _object(payload["features"], "features")
    feature_expected = {
        "view_id": VIEW_ID,
        "temporal_view_name": TEMPORAL_VIEW_NAME,
        "sequence_length": SEQUENCE_LENGTH,
        "model_window_rows": EXPECTED_MODEL_WINDOWS,
        "model_slot_rows": EXPECTED_MODEL_SLOTS,
        "feature_names": list(MOTION_FEATURE_NAMES),
        "feature_dim": MOTION_DIM,
        "feature_dtype": str(MOTION_DTYPE),
        "availability_dtype": str(AVAILABILITY_DTYPE),
        "window_local_rebase": True,
        "first_slot_zero_and_unavailable": True,
        "unit_aggregate_features_allowed": False,
        "normalization": "none_raw_cache_train_pairs_only_at_consumer",
        "source_media_fallback_allowed": False,
    }
    if features != feature_expected:
        raise ValueError(f"motion cache feature contract drift={features}")
    implementation = _object(payload["implementation"], "implementation")
    _require_exact_keys(
        implementation,
        {"cache_builder", "spatial_exporter"},
        "implementation",
    )
    for name, value in implementation.items():
        spec = _object(value, f"implementation.{name}")
        _require_exact_keys(spec, {"path", "sha256"}, f"implementation.{name}")
        _require_sha(str(spec["sha256"]), f"implementation.{name}.sha256")
    guard = _object(payload["execution_guard"], "execution_guard")
    _require_exact_keys(
        guard,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )
    output = _object(payload["output"], "output")
    _require_exact_keys(output, {"cache_root_relative_path"}, "output")


def _validate_claim_columns(frame: pd.DataFrame, name: str) -> None:
    missing = sorted(
        {"lineage_scope", "human_review_complete"} - set(frame.columns)
    )
    if missing:
        raise ValueError(f"{name} missing claim columns={missing}")
    if set(frame["lineage_scope"].astype(str)) != {LINEAGE_SCOPE}:
        raise ValueError(f"{name} lineage_scope drift")
    if _strict_bool(
        frame["human_review_complete"],
        name=f"{name}.human_review_complete",
    ).any():
        raise ValueError(f"{name} claims completed human review")


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{name} contains missing booleans")
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"{name} contains invalid numeric booleans")
        return values.eq(1)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "t": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "f": False,
    }
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(f"{name} contains unknown booleans={unknown}")
    return normalized.map(mapping).astype(bool)


def _validated_rows(values: np.ndarray, maximum: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("motion cache row indices must be one-dimensional")
    if len(rows) and (rows.min() < 0 or rows.max() >= maximum):
        raise IndexError("motion cache row indices are out of bounds")
    return rows


def _ordered_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_bound_file(path: Path, expected_sha: str, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    observed = file_sha256(path)
    if observed != expected_sha:
        raise ValueError(
            f"{name} hash mismatch: expected={expected_sha}, observed={observed}"
        )


def _resolve_inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    _require_inside(root, path)
    return path


def _require_inside(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes repository: {path}") from error


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    name: str,
) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - observed)},"
            f"extra={sorted(observed - expected)}"
        )


def _require_sha(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} is not a lowercase SHA256")


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"git command failed: {' '.join(arguments)}")
    return completed.stdout


def _status_path(line: str) -> str:
    value = line[3:].strip().replace("\\", "/")
    if " -> " in value:
        value = value.split(" -> ", 1)[1]
    return value.strip('"')


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _close_memmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()
