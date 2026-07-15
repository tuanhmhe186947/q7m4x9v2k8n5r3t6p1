"""Immutable T6 geometry cache for legacy-only L6 development controls."""

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

from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    load_temporal_ladder_config,
    load_temporal_ladder_view,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

LINEAGE_SCOPE = "legacy-only-unreviewed-development"
CANONICAL_SOURCE_NAME = "legacy_16f"
SOURCE_TYPE = "legacy_recovered"
DATASET_ID = "legacy_recovered_16f"
VIEW_ID = "t6_sliding"
TEMPORAL_VIEW_NAME = "legacy_t6_all_sliding_observed_time"
SEQUENCE_LENGTH = 6
EXPECTED_RAW_ROWS = 72_864
EXPECTED_MODEL_WINDOWS = 15_588
EXPECTED_MODEL_SLOTS = 93_528
EXPECTED_ALL_TIER_SLOTS = 109_296

GEOMETRY_FEATURE_NAMES = (
    "cx_n",
    "cy_n",
    "bw_n",
    "bh_n",
    "area_n",
    "aspect_ratio",
    "box_diag_n",
    "box_compactness",
)
GEOMETRY_DIM = len(GEOMETRY_FEATURE_NAMES)
GEOMETRY_DTYPE = np.dtype(np.float32)
AVAILABILITY_DTYPE = np.dtype(np.bool_)

CACHE_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_cache_config.v1"
)
CACHE_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_cache_manifest.v1"
)
CACHE_AUDIT_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_cache_audit.v1"
)

CACHE_FILES = {
    "geometry": "geometry_raw_f32.npy",
    "availability": "geometry_available_bool.npy",
    "window_index": "geometry_window_index.csv",
    "slot_index": "geometry_slot_index.csv",
    "manifest": "geometry_cache_manifest.json",
}


@dataclass(frozen=True, slots=True)
class LegacyL6GeometryCacheConfig:
    """Hash-bound cache build specification."""

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
class LegacyL6GeometryCache:
    """Audited geometry arrays aligned to the frozen L5 T6 view."""

    root: Path
    geometry_path: Path
    availability_path: Path
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def load_geometry(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Copy bounded cache rows and close the mmap before returning."""

        mapping = np.load(self.geometry_path, mmap_mode="r")
        try:
            if rows is None:
                values = np.asarray(mapping, dtype=GEOMETRY_DTYPE).copy()
            else:
                indices = _validated_rows(rows, len(self.window_index))
                values = np.asarray(mapping[indices], dtype=GEOMETRY_DTYPE).copy()
        finally:
            _close_memmap(mapping)
        return values

    def load_availability(self, rows: np.ndarray | None = None) -> np.ndarray:
        """Copy bounded availability rows and close the mmap."""

        mapping = np.load(self.availability_path, mmap_mode="r")
        try:
            if rows is None:
                values = np.asarray(mapping, dtype=AVAILABILITY_DTYPE).copy()
            else:
                indices = _validated_rows(rows, len(self.window_index))
                values = np.asarray(
                    mapping[indices],
                    dtype=AVAILABILITY_DTYPE,
                ).copy()
        finally:
            _close_memmap(mapping)
        return values


def load_geometry_cache_config(path: Path) -> LegacyL6GeometryCacheConfig:
    """Load the cache config and verify every immutable parent."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_cache_config_payload(payload)
    config = LegacyL6GeometryCacheConfig(
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
    for name, spec_value in _object(payload["parents"], "parents").items():
        spec = _object(spec_value, f"parents.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"parent {name}",
        )
    for name, spec_value in _object(payload["inputs"], "inputs").items():
        spec = _object(spec_value, f"inputs.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"input {name}",
        )
    implementation = _object(payload["implementation"], "implementation")
    _validate_bound_file(
        _resolve_inside(config.repo_root, str(implementation["path"])),
        str(implementation["sha256"]),
        "cache implementation",
    )
    _validate_parent_decision(config)
    return config


def preflight_geometry_cache(
    config: LegacyL6GeometryCacheConfig,
) -> dict[str, Any]:
    """Run the CPU-only parent, routing, Git, and output-availability gate."""

    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    windows = 0
    roles: dict[str, int] = {}
    try:
        ladder_path = config.bound_path("parents", "temporal_ladder_config")
        ladder = load_temporal_ladder_config(ladder_path)
        _, view, _ = load_temporal_ladder_view(ladder, VIEW_ID)
        windows = len(view.windows)
        roles = {
            str(key): int(value)
            for key, value in view.windows["l5_role"].value_counts().items()
        }
        if windows != EXPECTED_MODEL_WINDOWS:
            errors.append(f"model_window_rows={windows}")
        if roles != {"train": 14_608, "validation": 980}:
            errors.append(f"model_window_roles={roles}")
        if config.output_root.exists():
            errors.append(f"cache_output_exists={config.output_root}")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = geometry_cache_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("geometry cache preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "geometry_cache_preflight.v1"
        ),
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_CACHE_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_GEOMETRY_CACHE_PREFLIGHT"
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


def build_geometry_cache(
    config: LegacyL6GeometryCacheConfig,
) -> tuple[Path, dict[str, Any]]:
    """Build the immutable model-visible cache after the committed preflight."""

    preflight = preflight_geometry_cache(config)
    if not preflight["build_authorized"]:
        raise RuntimeError(f"geometry cache preflight failed={preflight['errors']}")
    ladder_path = config.bound_path("parents", "temporal_ladder_config")
    ladder = load_temporal_ladder_config(ladder_path)
    _, view, parent = load_temporal_ladder_view(ladder, VIEW_ID)
    slots = _load_model_slots(config, view.windows)
    context = _load_image_context(config)
    joined = slots.merge(
        context,
        left_on=["object_track_key_audit", "frame_index_expected_audit"],
        right_on=["object_track_key", "frame_index"],
        how="left",
        validate="many_to_one",
    )
    _validate_joined_slots(joined)
    geometry = compute_geometry_features(joined)
    reference_audit = _validate_geometry_reference(config, joined, geometry)
    availability = geometry_availability(joined, geometry)
    ordered, geometry_array, availability_array = _reshape_cache_arrays(
        view.windows,
        joined,
        geometry,
        availability,
    )
    window_index = _build_window_index(view.windows, ordered)
    slot_index = _build_slot_index(ordered, geometry, availability)
    source_probe = single_source_probe_audit(slot_index)
    cache_audit = _cache_content_audit(
        geometry_array,
        availability_array,
        window_index,
        slot_index,
        reference_audit=reference_audit,
        source_probe=source_probe,
    )
    root = config.output_root
    temporary = root.with_name(f"{root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"geometry cache temporary output exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    geometry_path = temporary / CACHE_FILES["geometry"]
    availability_path = temporary / CACHE_FILES["availability"]
    window_index_path = temporary / CACHE_FILES["window_index"]
    slot_index_path = temporary / CACHE_FILES["slot_index"]
    np.save(geometry_path, geometry_array, allow_pickle=False)
    np.save(availability_path, availability_array, allow_pickle=False)
    window_index.to_csv(window_index_path, index=False, lineterminator="\n")
    slot_index.to_csv(slot_index_path, index=False, lineterminator="\n")
    artifacts = {
        name: {
            "filename": CACHE_FILES[name],
            "sha256": file_sha256(temporary / CACHE_FILES[name]),
            "size_bytes": int((temporary / CACHE_FILES[name]).stat().st_size),
        }
        for name in ("geometry", "availability", "window_index", "slot_index")
    }
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_CACHE",
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
        "implementation_sha256": str(
            config.payload["implementation"]["sha256"]
        ),
        "git_guard": preflight["git_guard"],
        "parent_view": {
            "view_id": VIEW_ID,
            "temporal_view_name": TEMPORAL_VIEW_NAME,
            "sequence_length": SEQUENCE_LENGTH,
            "model_window_rows": EXPECTED_MODEL_WINDOWS,
            "model_slot_rows": EXPECTED_MODEL_SLOTS,
            "consumer_parent": parent,
            "ordered_window_id_sha256": _ordered_sha256(
                window_index["window_id"]
            ),
        },
        "source_bindings": _source_bindings(config),
        "feature_contract": {
            "feature_names": list(GEOMETRY_FEATURE_NAMES),
            "feature_dim": GEOMETRY_DIM,
            "feature_dtype": str(GEOMETRY_DTYPE),
            "availability_dtype": str(AVAILABILITY_DTYPE),
            "geometry_source": (
                "recomputed_from_image_context_bbox_and_image_size_v1"
            ),
            "reference_validation": "allclose_against_frame_geometry_v1",
            "normalization": "none_raw_cache_fold_train_only_at_consumer",
            "availability_definition": (
                "bbox_context_valid_and_finite_declared_geometry_v1"
            ),
            "availability_is_behavior_evidence": False,
            "labels_ids_paths_or_folds_in_model_x": False,
        },
        "cache_scope": {
            "roles": ["train", "validation"],
            "outer_holdout_slots_materialized": 0,
            "source_media_reads": 0,
            "video_decodes": 0,
        },
        "content_audit": cache_audit,
        "artifacts": artifacts,
        "errors": [],
        "valid": True,
    }
    manifest_path = temporary / CACHE_FILES["manifest"]
    _write_json_exclusive(manifest_path, manifest)
    temporary.replace(root)
    audit = audit_geometry_cache(config, cache_root=root)
    if not audit["valid"]:
        raise RuntimeError(f"written geometry cache failed audit={audit['errors']}")
    return root / CACHE_FILES["manifest"], audit


def load_geometry_cache(
    config: LegacyL6GeometryCacheConfig,
    *,
    cache_root: Path | None = None,
) -> LegacyL6GeometryCache:
    """Load and audit the cache without retaining mmap handles."""

    root = (cache_root or config.output_root).resolve()
    audit = audit_geometry_cache(config, cache_root=root)
    if not audit["valid"]:
        raise ValueError(f"geometry cache audit failed={audit['errors']}")
    manifest = _read_json(root / CACHE_FILES["manifest"])
    window_index = pd.read_csv(root / CACHE_FILES["window_index"])
    slot_index = pd.read_csv(root / CACHE_FILES["slot_index"])
    return LegacyL6GeometryCache(
        root=root,
        geometry_path=root / CACHE_FILES["geometry"],
        availability_path=root / CACHE_FILES["availability"],
        window_index=window_index,
        slot_index=slot_index,
        manifest=manifest,
        audit=audit,
    )


def audit_geometry_cache(
    config: LegacyL6GeometryCacheConfig,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Re-hash and structurally verify a written geometry cache."""

    root = (cache_root or config.output_root).resolve()
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    verified_artifacts = 0
    geometry_shape: list[int] = []
    availability_shape: list[int] = []
    try:
        _require_inside(config.repo_root, root)
        manifest_path = root / CACHE_FILES["manifest"]
        manifest = _read_json(manifest_path)
        _validate_written_manifest(config, manifest)
        artifacts = _object(manifest["artifacts"], "cache artifacts")
        for name in ("geometry", "availability", "window_index", "slot_index"):
            spec = _object(artifacts[name], f"artifacts.{name}")
            path = root / str(spec["filename"])
            if file_sha256(path) != str(spec["sha256"]):
                errors.append(f"artifact_hash_mismatch={name}")
            elif int(path.stat().st_size) != int(spec["size_bytes"]):
                errors.append(f"artifact_size_mismatch={name}")
            else:
                verified_artifacts += 1
        geometry = np.load(root / CACHE_FILES["geometry"], mmap_mode="r")
        availability = np.load(
            root / CACHE_FILES["availability"],
            mmap_mode="r",
        )
        try:
            geometry_shape = list(geometry.shape)
            availability_shape = list(availability.shape)
            if geometry.shape != (
                EXPECTED_MODEL_WINDOWS,
                SEQUENCE_LENGTH,
                GEOMETRY_DIM,
            ):
                errors.append(f"geometry_shape={geometry_shape}")
            if availability.shape != (EXPECTED_MODEL_WINDOWS, SEQUENCE_LENGTH):
                errors.append(f"availability_shape={availability_shape}")
            if geometry.dtype != GEOMETRY_DTYPE:
                errors.append(f"geometry_dtype={geometry.dtype}")
            if availability.dtype != AVAILABILITY_DTYPE:
                errors.append(f"availability_dtype={availability.dtype}")
            if not np.isfinite(geometry).all():
                errors.append("geometry_contains_nonfinite")
        finally:
            _close_memmap(geometry)
            _close_memmap(availability)
        window_index = pd.read_csv(root / CACHE_FILES["window_index"])
        slot_index = pd.read_csv(root / CACHE_FILES["slot_index"])
        _validate_index_frames(window_index, slot_index)
        observed_window_hash = _ordered_sha256(window_index["window_id"])
        expected_window_hash = str(
            manifest["parent_view"]["ordered_window_id_sha256"]
        )
        if observed_window_hash != expected_window_hash:
            errors.append("ordered_window_id_sha256_mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    valid = not errors
    return {
        "schema_version": CACHE_AUDIT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_CACHE_AUDIT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_GEOMETRY_CACHE_AUDIT"
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
        "verified_artifacts": verified_artifacts,
        "geometry_shape": geometry_shape,
        "availability_shape": availability_shape,
        "outer_holdout_slots_materialized": 0,
        "source_media_reads": 0,
        "errors": errors,
        "valid": valid,
    }


def compute_geometry_features(frame: pd.DataFrame) -> np.ndarray:
    """Recompute the eight explicit geometry fields from bbox and image size."""

    required = {"image_width", "image_height", "x1", "y1", "x2", "y2"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"geometry input missing columns={missing}")
    numeric = frame[list(sorted(required))].apply(pd.to_numeric, errors="coerce")
    image_width = numeric["image_width"].to_numpy(dtype=np.float64)
    image_height = numeric["image_height"].to_numpy(dtype=np.float64)
    x1 = numeric["x1"].to_numpy(dtype=np.float64)
    y1 = numeric["y1"].to_numpy(dtype=np.float64)
    x2 = numeric["x2"].to_numpy(dtype=np.float64)
    y2 = numeric["y2"].to_numpy(dtype=np.float64)
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    image_area = image_width * image_height
    bbox_diag = np.sqrt(bbox_width**2 + bbox_height**2)
    image_diag = np.sqrt(image_width**2 + image_height**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        area_n = bbox_area / image_area
        box_diag_n = bbox_diag / image_diag
        values = np.column_stack(
            [
                center_x / image_width,
                center_y / image_height,
                bbox_width / image_width,
                bbox_height / image_height,
                area_n,
                bbox_width / bbox_height,
                box_diag_n,
                area_n / (box_diag_n**2),
            ]
        )
    if values.shape != (len(frame), GEOMETRY_DIM):
        raise ValueError(f"geometry computation shape drift={values.shape}")
    return values


def geometry_availability(
    frame: pd.DataFrame,
    geometry: np.ndarray,
) -> np.ndarray:
    """Return explicit availability without using behavior or review fields."""

    if "bbox_context_valid" not in frame:
        raise ValueError("geometry availability missing bbox_context_valid")
    values = np.asarray(geometry, dtype=np.float64)
    if values.shape != (len(frame), GEOMETRY_DIM):
        raise ValueError("geometry availability shape mismatch")
    bbox_valid = _strict_bool(
        frame["bbox_context_valid"],
        name="bbox_context_valid",
    ).to_numpy(dtype=np.bool_)
    return bbox_valid & np.isfinite(values).all(axis=1)


def single_source_probe_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Fail closed to a non-estimable status for the single legacy source."""

    required = {"source_type", "dataset_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"source probe input missing columns={missing}")
    sources = sorted(frame["source_type"].fillna("").astype(str).unique())
    datasets = sorted(frame["dataset_id"].fillna("").astype(str).unique())
    if sources != [SOURCE_TYPE] or datasets != [DATASET_ID]:
        raise ValueError(
            "legacy geometry source identity drift: "
            f"sources={sources}, datasets={datasets}"
        )
    return {
        "schema_version": (
            "classification_v2.legacy_development_l6.source_probe.v1"
        ),
        "status": "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE",
        "source_type_values": sources,
        "dataset_id_values": datasets,
        "source_type_cardinality": 1,
        "dataset_id_cardinality": 1,
        "probe_fit_performed": False,
        "two_source_result_reported": False,
        "estimable": False,
        "reason": "all_model_visible_rows_have_one_legacy_source_identity",
        "errors": [],
        "valid": True,
    }


def _load_model_slots(
    config: LegacyL6GeometryCacheConfig,
    model_windows: pd.DataFrame,
) -> pd.DataFrame:
    path = config.bound_path("inputs", "temporal_slot_manifest")
    columns = [
        "temporal_view_name",
        "parent_window_id",
        "temporal_unit_key",
        "item_order",
        "slot_index",
        "object_track_key_audit",
        "frame_index_expected_audit",
        "length_mask",
        "observed_mask",
        "padding_mask",
        "lineage_scope",
        "human_review_complete",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if len(frame) != EXPECTED_ALL_TIER_SLOTS:
        raise ValueError(f"all T6 slot rows={len(frame)}")
    wanted = set(model_windows["window_id"].astype(str))
    frame = frame.loc[
        frame["parent_window_id"].astype(str).isin(wanted)
    ].copy()
    if len(frame) != EXPECTED_MODEL_SLOTS:
        raise ValueError(f"model-visible T6 slot rows={len(frame)}")
    if frame[["parent_window_id", "slot_index"]].duplicated().any():
        raise ValueError("model-visible T6 slots contain duplicate keys")
    if set(frame["temporal_view_name"].astype(str)) != {TEMPORAL_VIEW_NAME}:
        raise ValueError("model-visible T6 temporal view drift")
    for name in ("length_mask", "observed_mask"):
        if not _strict_bool(frame[name], name=name).all():
            raise ValueError(f"model-visible T6 {name} contains false slots")
    if _strict_bool(frame["padding_mask"], name="padding_mask").any():
        raise ValueError("model-visible T6 slots contain padding")
    _validate_claim_columns(frame, "temporal slots")
    slot_sets = frame.groupby("parent_window_id", sort=False)["slot_index"].apply(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if not slot_sets.map(lambda value: value == tuple(range(SEQUENCE_LENGTH))).all():
        raise ValueError("model-visible T6 slot sets are incomplete")
    return frame


def _load_image_context(config: LegacyL6GeometryCacheConfig) -> pd.DataFrame:
    path = config.bound_path("inputs", "image_context_manifest")
    columns = [
        "frame_uid",
        "object_track_key",
        "frame_index",
        "source_type",
        "dataset_id",
        "image_width",
        "image_height",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_context_valid",
        "lineage_scope",
        "human_review_complete",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if len(frame) != EXPECTED_RAW_ROWS:
        raise ValueError(f"image context rows={len(frame)}")
    if frame[["object_track_key", "frame_index"]].duplicated().any():
        raise ValueError("image context object/frame keys are duplicated")
    if frame["frame_uid"].astype(str).duplicated().any():
        raise ValueError("image context frame_uid is duplicated")
    if set(frame["source_type"].astype(str)) != {SOURCE_TYPE}:
        raise ValueError("image context source_type drift")
    if set(frame["dataset_id"].astype(str)) != {DATASET_ID}:
        raise ValueError("image context dataset_id drift")
    _validate_claim_columns(frame, "image context")
    return frame


def _validate_joined_slots(joined: pd.DataFrame) -> None:
    if len(joined) != EXPECTED_MODEL_SLOTS:
        raise ValueError(f"joined geometry slot rows={len(joined)}")
    missing = int(joined["frame_uid"].isna().sum())
    if missing:
        raise ValueError(f"joined geometry slots missing context={missing}")
    if joined[["parent_window_id", "slot_index"]].duplicated().any():
        raise ValueError("joined geometry slots contain duplicate keys")
    if set(joined["source_type"].astype(str)) != {SOURCE_TYPE}:
        raise ValueError("joined geometry source_type drift")
    if set(joined["dataset_id"].astype(str)) != {DATASET_ID}:
        raise ValueError("joined geometry dataset_id drift")
    _validate_claim_columns(joined, "joined geometry slots")


def _validate_geometry_reference(
    config: LegacyL6GeometryCacheConfig,
    joined: pd.DataFrame,
    geometry: np.ndarray,
) -> dict[str, Any]:
    path = config.bound_path("inputs", "frame_geometry")
    columns = [
        "frame_uid",
        "source_type",
        "dataset_id",
        *GEOMETRY_FEATURE_NAMES,
        "geometry_feature_valid",
        "lineage_scope",
        "human_review_complete",
    ]
    reference = pd.read_csv(path, usecols=columns, low_memory=False)
    if len(reference) != EXPECTED_RAW_ROWS:
        raise ValueError(f"frame geometry reference rows={len(reference)}")
    if reference["frame_uid"].astype(str).duplicated().any():
        raise ValueError("frame geometry reference frame_uid is duplicated")
    _validate_claim_columns(reference, "frame geometry reference")
    candidate = joined[["frame_uid"]].copy()
    for index, name in enumerate(GEOMETRY_FEATURE_NAMES):
        candidate[f"candidate_{name}"] = geometry[:, index]
    compared = candidate.merge(
        reference,
        on="frame_uid",
        how="left",
        validate="many_to_one",
    )
    if len(compared) != EXPECTED_MODEL_SLOTS:
        raise ValueError("geometry reference comparison lost slots")
    if compared["source_type"].isna().any():
        raise ValueError("geometry reference comparison has missing frame_uid")
    if not _strict_bool(
        compared["geometry_feature_valid"],
        name="geometry_feature_valid",
    ).all():
        raise ValueError("model-visible geometry reference contains invalid rows")
    maximum_errors: dict[str, float] = {}
    for name in GEOMETRY_FEATURE_NAMES:
        expected = pd.to_numeric(compared[name], errors="coerce").to_numpy(
            dtype=np.float64
        )
        observed = compared[f"candidate_{name}"].to_numpy(dtype=np.float64)
        if not np.isfinite(expected).all():
            raise ValueError(f"geometry reference {name} contains nonfinite values")
        delta = np.abs(expected - observed)
        maximum_errors[name] = float(delta.max(initial=0.0))
        if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"geometry reference mismatch={name}:max={maximum_errors[name]}"
            )
    return {
        "comparison_rows": int(len(compared)),
        "feature_names": list(GEOMETRY_FEATURE_NAMES),
        "absolute_tolerance": 1e-12,
        "relative_tolerance": 0.0,
        "maximum_absolute_error": maximum_errors,
        "reference_match": True,
        "errors": [],
        "valid": True,
    }


def _reshape_cache_arrays(
    windows: pd.DataFrame,
    joined: pd.DataFrame,
    geometry: np.ndarray,
    availability: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = joined.copy()
    for index, name in enumerate(GEOMETRY_FEATURE_NAMES):
        work[name] = geometry[:, index]
    work["geometry_available"] = np.asarray(
        availability,
        dtype=np.bool_,
    )
    positions = windows[["window_id"]].reset_index().rename(
        columns={"index": "window_position", "window_id": "parent_window_id"}
    )
    work = work.merge(
        positions,
        on="parent_window_id",
        how="left",
        validate="many_to_one",
    )
    if work["window_position"].isna().any():
        raise ValueError("geometry cache slots contain unknown model windows")
    work = work.sort_values(
        ["window_position", "slot_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    expected_positions = np.repeat(
        np.arange(EXPECTED_MODEL_WINDOWS, dtype=np.int64),
        SEQUENCE_LENGTH,
    )
    if not np.array_equal(
        work["window_position"].to_numpy(dtype=np.int64),
        expected_positions,
    ):
        raise ValueError("geometry cache window order drift")
    expected_slots = np.tile(
        np.arange(SEQUENCE_LENGTH, dtype=np.int64),
        EXPECTED_MODEL_WINDOWS,
    )
    if not np.array_equal(
        work["slot_index"].to_numpy(dtype=np.int64),
        expected_slots,
    ):
        raise ValueError("geometry cache slot order drift")
    raw = work[list(GEOMETRY_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    available = work["geometry_available"].to_numpy(dtype=np.bool_)
    clean = np.where(available[:, None], raw, 0.0)
    if not np.isfinite(clean).all():
        raise ValueError("geometry cache contains nonfinite values after masking")
    geometry_array = clean.astype(GEOMETRY_DTYPE).reshape(
        EXPECTED_MODEL_WINDOWS,
        SEQUENCE_LENGTH,
        GEOMETRY_DIM,
    )
    availability_array = available.reshape(
        EXPECTED_MODEL_WINDOWS,
        SEQUENCE_LENGTH,
    )
    return work, geometry_array, availability_array


def _build_window_index(
    windows: pd.DataFrame,
    ordered_slots: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "window_id",
        "temporal_unit_key",
        "l5_role",
        "source_type",
        "dataset_id",
        "lineage_scope",
        "human_review_complete",
    ]
    missing = sorted(set(required).difference(windows.columns))
    if missing:
        raise ValueError(f"geometry window index missing parent fields={missing}")
    index = windows[required].copy().reset_index(drop=True)
    index.insert(0, "cache_row", np.arange(len(index), dtype=np.int64))
    identity_hashes = (
        ordered_slots.groupby("window_position", sort=True)["frame_uid"]
        .apply(_ordered_sha256)
        .reset_index(name="ordered_frame_uid_sha256")
    )
    index = index.merge(
        identity_hashes,
        left_on="cache_row",
        right_on="window_position",
        how="left",
        validate="one_to_one",
    ).drop(columns="window_position")
    index["sequence_length"] = SEQUENCE_LENGTH
    _validate_claim_columns(index, "geometry window index")
    return index


def _build_slot_index(
    ordered: pd.DataFrame,
    geometry: np.ndarray,
    availability: np.ndarray,
) -> pd.DataFrame:
    del geometry, availability
    columns = [
        "window_position",
        "parent_window_id",
        "slot_index",
        "frame_uid",
        "object_track_key",
        "frame_index",
        "source_type",
        "dataset_id",
        "geometry_available",
        "lineage_scope",
        "human_review_complete",
    ]
    frame = ordered[columns].copy().rename(
        columns={
            "window_position": "cache_row",
            "parent_window_id": "window_id",
        }
    )
    frame["frame_index"] = frame["frame_index"].astype(np.int64)
    frame["slot_index"] = frame["slot_index"].astype(np.int64)
    _validate_claim_columns(frame, "geometry slot index")
    return frame


def _cache_content_audit(
    geometry: np.ndarray,
    availability: np.ndarray,
    window_index: pd.DataFrame,
    slot_index: pd.DataFrame,
    *,
    reference_audit: dict[str, Any],
    source_probe: dict[str, Any],
) -> dict[str, Any]:
    available_slots = int(np.asarray(availability, dtype=np.bool_).sum())
    if available_slots != EXPECTED_MODEL_SLOTS:
        raise ValueError(f"geometry available slots={available_slots}")
    roles = {
        str(key): int(value)
        for key, value in window_index["l5_role"].value_counts().items()
    }
    if roles != {"train": 14_608, "validation": 980}:
        raise ValueError(f"geometry cache role counts={roles}")
    flattened = geometry.reshape(-1, GEOMETRY_DIM).astype(np.float64)
    statistics = {
        name: {
            "minimum": float(flattened[:, index].min()),
            "maximum": float(flattened[:, index].max()),
            "mean": float(flattened[:, index].mean()),
            "population_std": float(flattened[:, index].std(ddof=0)),
        }
        for index, name in enumerate(GEOMETRY_FEATURE_NAMES)
    }
    return {
        "model_window_rows": int(len(window_index)),
        "model_slot_rows": int(len(slot_index)),
        "role_window_counts": roles,
        "available_slots": available_slots,
        "unavailable_slots": EXPECTED_MODEL_SLOTS - available_slots,
        "geometry_shape": list(geometry.shape),
        "availability_shape": list(availability.shape),
        "geometry_dtype": str(geometry.dtype),
        "availability_dtype": str(availability.dtype),
        "geometry_statistics": statistics,
        "ordered_window_id_sha256": _ordered_sha256(window_index["window_id"]),
        "window_index_content_sha256": _dataframe_sha256(window_index),
        "slot_index_content_sha256": _dataframe_sha256(slot_index),
        "reference_audit": reference_audit,
        "source_probe": source_probe,
        "availability_is_constant_one_in_legacy_16f": True,
        "availability_only_is_diagnostic_not_behavior_evidence": True,
        "outer_holdout_slots_materialized": 0,
        "source_media_reads": 0,
        "errors": [],
        "valid": True,
    }


def _validate_index_frames(
    window_index: pd.DataFrame,
    slot_index: pd.DataFrame,
) -> None:
    if len(window_index) != EXPECTED_MODEL_WINDOWS:
        raise ValueError(f"geometry window index rows={len(window_index)}")
    if len(slot_index) != EXPECTED_MODEL_SLOTS:
        raise ValueError(f"geometry slot index rows={len(slot_index)}")
    expected_rows = np.arange(EXPECTED_MODEL_WINDOWS, dtype=np.int64)
    if not np.array_equal(
        window_index["cache_row"].to_numpy(dtype=np.int64),
        expected_rows,
    ):
        raise ValueError("geometry window index cache_row drift")
    if window_index["window_id"].astype(str).duplicated().any():
        raise ValueError("geometry window index IDs are duplicated")
    if slot_index[["window_id", "slot_index"]].duplicated().any():
        raise ValueError("geometry slot index keys are duplicated")
    if set(window_index["source_type"].astype(str)) != {SOURCE_TYPE}:
        raise ValueError("geometry window index source_type drift")
    if set(window_index["dataset_id"].astype(str)) != {DATASET_ID}:
        raise ValueError("geometry window index dataset_id drift")
    _validate_claim_columns(window_index, "geometry window index")
    _validate_claim_columns(slot_index, "geometry slot index")


def geometry_cache_git_guard(
    config: LegacyL6GeometryCacheConfig,
) -> dict[str, Any]:
    """Require committed cache sources/config and preserve only known user dirt."""

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
        str(path).replace("\\", "/") for path in guard["required_tracked_paths"]
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


def _validate_parent_decision(config: LegacyL6GeometryCacheConfig) -> None:
    decision_path = config.bound_path("parents", "l5_decision")
    decision = _read_json(decision_path)
    if decision.get("lineage_scope") != LINEAGE_SCOPE:
        raise ValueError("L5 decision lineage scope drift")
    if decision.get("status") != (
        "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_DECISION"
    ):
        raise ValueError("L5 temporal decision status is not PASS")
    selected = _object(decision.get("decision"), "L5 decision")
    if selected.get("selected_working_view") != VIEW_ID:
        raise ValueError("L5 temporal decision did not select t6_sliding")
    boundary = _object(
        decision.get("interpretation_boundary"),
        "L5 interpretation boundary",
    )
    expected = {
        "decision_scope": LINEAGE_SCOPE,
        "legacy_dataset_is_legacy_16f_not_merged": True,
        "legacy_rare_support_generalizes_to_merged_data": False,
        "merged_data_has_materially_more_rare_behaviors": True,
        "merged_reviewed_reassessment_required": True,
        "local_vram_is_architecture_limit": False,
        "rented_gpu_allowed_after_target_environment_gate": True,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
    }
    if boundary != expected:
        raise ValueError(f"L5 interpretation boundary drift={boundary}")


def _source_bindings(
    config: LegacyL6GeometryCacheConfig,
) -> dict[str, Any]:
    source = _object(config.payload["source_identity"], "source_identity")
    parents = _object(config.payload["parents"], "parents")
    inputs = _object(config.payload["inputs"], "inputs")
    return {
        "raw_authority": {
            "path": str(source["raw_authority_path"]),
            "sha256": str(source["raw_sha256"]),
            "rows": EXPECTED_RAW_ROWS,
        },
        "parents": {
            name: {
                "path": str(spec["path"]),
                "sha256": str(spec["sha256"]),
            }
            for name, spec in parents.items()
        },
        "inputs": {
            name: {
                "path": str(spec["path"]),
                "sha256": str(spec["sha256"]),
                "expected_rows": int(spec["expected_rows"]),
            }
            for name, spec in inputs.items()
        },
    }


def _validate_written_manifest(
    config: LegacyL6GeometryCacheConfig,
    manifest: dict[str, Any],
) -> None:
    expected = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_CACHE",
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
        "implementation_sha256": str(config.payload["implementation"]["sha256"]),
        "valid": True,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(
                f"geometry cache manifest {name}="
                f"{manifest.get(name)!r}!={value!r}"
            )
    content = _object(manifest.get("content_audit"), "content_audit")
    if not content.get("valid") or content.get("errors"):
        raise ValueError("geometry cache manifest content audit is invalid")
    source_probe = _object(content.get("source_probe"), "source_probe")
    if source_probe.get("status") != "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE":
        raise ValueError("geometry cache source probe status drift")
    if source_probe.get("probe_fit_performed") is not False:
        raise ValueError("geometry cache fitted an invalid single-source probe")


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
        "features",
        "implementation",
        "execution_guard",
        "output",
    }
    _require_exact_keys(payload, required, "geometry cache config")
    expected_claims = {
        "schema_version": CACHE_CONFIG_SCHEMA,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
    }
    for name, value in expected_claims.items():
        if payload[name] != value:
            raise ValueError(f"geometry cache config {name}={payload[name]!r}")
    source = _object(payload["source_identity"], "source_identity")
    _require_exact_keys(
        source,
        {
            "canonical_short_name",
            "raw_authority_path",
            "raw_sha256",
            "expected_rows",
            "source_type",
            "dataset_id",
            "merged_data",
        },
        "source_identity",
    )
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
        raise ValueError(f"geometry cache source identity drift={source}")
    parents = _object(payload["parents"], "parents")
    _require_exact_keys(
        parents,
        {"temporal_ladder_config", "l5_decision"},
        "parents",
    )
    for name, spec_value in parents.items():
        spec = _object(spec_value, f"parents.{name}")
        _require_exact_keys(spec, {"path", "sha256"}, f"parents.{name}")
        _require_sha(str(spec["sha256"]), f"parents.{name}.sha256")
    inputs = _object(payload["inputs"], "inputs")
    expected_rows = {
        "image_context_manifest": EXPECTED_RAW_ROWS,
        "temporal_slot_manifest": EXPECTED_ALL_TIER_SLOTS,
        "frame_geometry": EXPECTED_RAW_ROWS,
    }
    _require_exact_keys(inputs, set(expected_rows), "inputs")
    for name, rows in expected_rows.items():
        spec = _object(inputs[name], f"inputs.{name}")
        _require_exact_keys(
            spec,
            {"path", "sha256", "expected_rows"},
            f"inputs.{name}",
        )
        _require_sha(str(spec["sha256"]), f"inputs.{name}.sha256")
        if int(spec["expected_rows"]) != rows:
            raise ValueError(f"inputs.{name}.expected_rows={spec['expected_rows']}")
    features = _object(payload["features"], "features")
    feature_expected = {
        "view_id": VIEW_ID,
        "temporal_view_name": TEMPORAL_VIEW_NAME,
        "sequence_length": SEQUENCE_LENGTH,
        "model_window_rows": EXPECTED_MODEL_WINDOWS,
        "model_slot_rows": EXPECTED_MODEL_SLOTS,
        "feature_names": list(GEOMETRY_FEATURE_NAMES),
        "feature_dim": GEOMETRY_DIM,
        "feature_dtype": str(GEOMETRY_DTYPE),
        "availability_dtype": str(AVAILABILITY_DTYPE),
        "normalization": "none_raw_cache_train_positions_only_at_consumer",
        "source_media_fallback_allowed": False,
    }
    if features != feature_expected:
        raise ValueError(f"geometry cache feature contract drift={features}")
    implementation = _object(payload["implementation"], "implementation")
    _require_exact_keys(implementation, {"path", "sha256"}, "implementation")
    _require_sha(str(implementation["sha256"]), "implementation.sha256")
    guard = _object(payload["execution_guard"], "execution_guard")
    _require_exact_keys(
        guard,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )
    output = _object(payload["output"], "output")
    _require_exact_keys(output, {"cache_root_relative_path"}, "output")


def _validate_claim_columns(frame: pd.DataFrame, name: str) -> None:
    required = {"lineage_scope", "human_review_complete"}
    missing = sorted(required.difference(frame.columns))
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
        raise ValueError("geometry cache row indices must be one-dimensional")
    if len(rows) and (rows.min() < 0 or rows.max() >= maximum):
        raise IndexError("geometry cache row indices are out of bounds")
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
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
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
