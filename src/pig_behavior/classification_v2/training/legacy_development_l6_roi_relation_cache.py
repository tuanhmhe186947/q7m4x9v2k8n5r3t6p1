"""Immutable all-class ROI-relation cache for legacy L6 development."""

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
    _close_memmap,
    _object,
    _ordered_sha256,
    _read_json,
    _require_exact_keys,
    _require_inside,
    _require_sha,
    _resolve_inside,
    _strict_bool,
    _validate_bound_file,
    _validated_rows,
    _write_json_exclusive,
    geometry_cache_git_guard,
    load_geometry_cache,
    load_geometry_cache_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

ROI_RELATION_FEATURE_NAMES = tuple(
    SPATIAL_FRAME_FEATURES["roi_class_relation"]
)
ROI_RELATION_DIM = len(ROI_RELATION_FEATURE_NAMES)
ROI_RELATION_DTYPE = np.dtype(np.float32)
AVAILABILITY_DTYPE = np.dtype(np.bool_)
ROI_AVAILABILITY_FIELDS = (
    "roi_feeder_available",
    "roi_drinker_available",
    "roi_toy_available",
)
BOOLEAN_RELATION_FIELDS = tuple(
    name
    for name in ROI_RELATION_FEATURE_NAMES
    if name.endswith(("_center_inside", "_near", "_contact"))
)

CACHE_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_cache_config.v1"
)
CACHE_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_cache_manifest.v1"
)
CACHE_AUDIT_SCHEMA = (
    "classification_v2.legacy_development_l6.roi_relation_cache_audit.v1"
)

CACHE_FILES = {
    "roi_relation": "roi_relation_raw_f32.npy",
    "availability": "roi_relation_available_bool.npy",
    "window_index": "roi_relation_window_index.csv",
    "slot_index": "roi_relation_slot_index.csv",
    "manifest": "roi_relation_cache_manifest.json",
}


@dataclass(frozen=True, slots=True)
class LegacyL6ROIRelationCacheConfig:
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

    def bound_path(self, section: str, name: str) -> Path:
        value = _object(self.payload[section], section)[name]
        spec = _object(value, f"{section}.{name}")
        return _resolve_inside(self.repo_root, str(spec["path"]))


@dataclass(frozen=True, slots=True)
class LegacyL6ROIRelationCache:
    """Audited all-class ROI arrays aligned to the frozen L5 T6 view."""

    root: Path
    roi_relation_path: Path
    availability_path: Path
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def load_roi_relation(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        mapping = np.load(self.roi_relation_path, mmap_mode="r")
        try:
            if rows is None:
                values = np.asarray(mapping, dtype=ROI_RELATION_DTYPE).copy()
            else:
                indices = _validated_rows(rows, len(self.window_index))
                values = np.asarray(
                    mapping[indices],
                    dtype=ROI_RELATION_DTYPE,
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


def load_roi_relation_cache_config(
    path: Path,
) -> LegacyL6ROIRelationCacheConfig:
    """Load one config and verify every immutable parent."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_cache_config_payload(payload)
    config = LegacyL6ROIRelationCacheConfig(
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
    _validate_bound_file(
        _resolve_inside(config.repo_root, str(implementation["path"])),
        str(implementation["sha256"]),
        "cache implementation",
    )
    _validate_motion_parent(config)
    _load_order_authority(config)
    return config


def preflight_roi_relation_cache(
    config: LegacyL6ROIRelationCacheConfig,
) -> dict[str, Any]:
    """Run the CPU-only parent, schema, Git, and output gate."""

    cuda_before = torch.cuda.is_initialized()
    errors: list[str] = []
    windows = 0
    slots = 0
    frame_rows = 0
    availability_counts: dict[str, int] = {}
    try:
        order = _load_order_authority(config)
        windows = len(order.window_index)
        slots = len(order.slot_index)
        frame = _load_roi_frames(config)
        frame_rows = len(frame)
        availability_counts = {
            name: int(_strict_bool(frame[name], name=name).sum())
            for name in ROI_AVAILABILITY_FIELDS
        }
        if config.output_root.exists():
            errors.append(f"cache_output_exists={config.output_root}")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    git_guard = geometry_cache_git_guard(config)
    errors.extend(str(value) for value in git_guard["errors"])
    cuda_after = torch.cuda.is_initialized()
    if cuda_before or cuda_after:
        errors.append("ROI relation cache preflight initialized CUDA")
    valid = not errors
    return {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "roi_relation_cache_preflight.v1"
        ),
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_CACHE_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_ROI_RELATION_CACHE_PREFLIGHT"
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
        "roi_frame_rows": frame_rows,
        "availability_counts": availability_counts,
        "cuda_runtime_initialized_before": cuda_before,
        "cuda_runtime_initialized_after": cuda_after,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "git_guard": git_guard,
        "build_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def build_roi_relation_cache(
    config: LegacyL6ROIRelationCacheConfig,
) -> tuple[Path, dict[str, Any]]:
    """Build one immutable cache after the committed preflight passes."""

    preflight = preflight_roi_relation_cache(config)
    if not preflight["build_authorized"]:
        raise RuntimeError(
            f"ROI relation cache preflight failed={preflight['errors']}"
        )
    order = _load_order_authority(config)
    frame = _load_roi_frames(config)
    roi, availability, window_index, slot_index = materialize_roi_relation_cache(
        order.window_index,
        order.slot_index,
        frame,
    )
    content = _cache_content_audit(
        roi,
        availability,
        window_index,
        slot_index,
    )
    root = config.output_root
    temporary = root.with_name(f"{root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"ROI cache temporary output exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    np.save(temporary / CACHE_FILES["roi_relation"], roi, allow_pickle=False)
    np.save(
        temporary / CACHE_FILES["availability"],
        availability,
        allow_pickle=False,
    )
    window_index.to_csv(
        temporary / CACHE_FILES["window_index"],
        index=False,
        lineterminator="\n",
    )
    slot_index.to_csv(
        temporary / CACHE_FILES["slot_index"],
        index=False,
        lineterminator="\n",
    )
    artifacts = {
        name: {
            "filename": CACHE_FILES[name],
            "sha256": file_sha256(temporary / CACHE_FILES[name]),
            "size_bytes": int(
                (temporary / CACHE_FILES[name]).stat().st_size
            ),
        }
        for name in (
            "roi_relation",
            "availability",
            "window_index",
            "slot_index",
        )
    }
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_CACHE",
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
        "parent_view": {
            "view_id": VIEW_ID,
            "temporal_view_name": TEMPORAL_VIEW_NAME,
            "sequence_length": SEQUENCE_LENGTH,
            "model_window_rows": EXPECTED_MODEL_WINDOWS,
            "model_slot_rows": EXPECTED_MODEL_SLOTS,
            "ordered_window_id_sha256": _ordered_sha256(
                window_index["window_id"]
            ),
        },
        "source_bindings": _source_bindings(config),
        "feature_contract": {
            "feature_names": list(ROI_RELATION_FEATURE_NAMES),
            "feature_dim": ROI_RELATION_DIM,
            "feature_dtype": str(ROI_RELATION_DTYPE),
            "availability_dtype": str(AVAILABILITY_DTYPE),
            "roi_classes": ["feeder", "drinker", "toy"],
            "all_classes_exposed_independently": True,
            "target_selected_roi_fields_used": False,
            "unit_aggregate_features_used": False,
            "geometry_values_used": False,
            "motion_values_used": False,
            "normalization": "none_raw_cache_fold_train_only_at_consumer",
            "availability_definition": (
                "all_three_roi_classes_available_and_features_finite_v1"
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
        "content_audit": content,
        "artifacts": artifacts,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(temporary / CACHE_FILES["manifest"], manifest)
    temporary.replace(root)
    audit = audit_roi_relation_cache(config, cache_root=root)
    if not audit["valid"]:
        raise RuntimeError(f"written ROI cache failed audit={audit['errors']}")
    return root / CACHE_FILES["manifest"], audit


def materialize_roi_relation_cache(
    window_index: pd.DataFrame,
    order_slots: pd.DataFrame,
    frames: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Join frame ROI values to the immutable window-slot order."""

    _validate_order_indexes(window_index, order_slots)
    _validate_roi_frames(frames)
    payload = frames[
        [
            "frame_uid",
            *ROI_RELATION_FEATURE_NAMES,
            *ROI_AVAILABILITY_FIELDS,
        ]
    ].copy()
    joined = order_slots.merge(
        payload,
        on="frame_uid",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if len(joined) != EXPECTED_MODEL_SLOTS:
        raise ValueError(f"ROI joined slots={len(joined)}")
    if joined["_merge"].ne("both").any():
        raise ValueError("ROI cache has unmatched frame_uid")
    joined = joined.sort_values(
        ["cache_row", "slot_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    numeric = joined[list(ROI_RELATION_FEATURE_NAMES)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = numeric.to_numpy(dtype=np.float64)
    class_available = np.column_stack(
        [
            _strict_bool(joined[name], name=name).to_numpy(dtype=np.bool_)
            for name in ROI_AVAILABILITY_FIELDS
        ]
    )
    available = class_available.all(axis=1) & np.isfinite(values).all(axis=1)
    if not available.all():
        raise ValueError(
            f"legacy ROI unavailable slots={int((~available).sum())}"
        )
    _validate_relation_bounds(values)
    roi = values.astype(ROI_RELATION_DTYPE).reshape(
        EXPECTED_MODEL_WINDOWS,
        SEQUENCE_LENGTH,
        ROI_RELATION_DIM,
    )
    availability = available.reshape(
        EXPECTED_MODEL_WINDOWS,
        SEQUENCE_LENGTH,
    )
    output_window_index = window_index.copy().reset_index(drop=True)
    output_slots = order_slots.drop(
        columns=["geometry_available"],
        errors="ignore",
    ).copy()
    output_slots["roi_relation_available"] = available
    return roi, availability, output_window_index, output_slots


def load_roi_relation_cache(
    config: LegacyL6ROIRelationCacheConfig,
    *,
    cache_root: Path | None = None,
) -> LegacyL6ROIRelationCache:
    root = (cache_root or config.output_root).resolve()
    audit = audit_roi_relation_cache(config, cache_root=root)
    if not audit["valid"]:
        raise ValueError(f"ROI relation cache audit failed={audit['errors']}")
    return LegacyL6ROIRelationCache(
        root=root,
        roi_relation_path=root / CACHE_FILES["roi_relation"],
        availability_path=root / CACHE_FILES["availability"],
        window_index=pd.read_csv(root / CACHE_FILES["window_index"]),
        slot_index=pd.read_csv(root / CACHE_FILES["slot_index"]),
        manifest=_read_json(root / CACHE_FILES["manifest"]),
        audit=audit,
    )


def audit_roi_relation_cache(
    config: LegacyL6ROIRelationCacheConfig,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    root = (cache_root or config.output_root).resolve()
    errors: list[str] = []
    verified = 0
    roi_shape: list[int] = []
    availability_shape: list[int] = []
    try:
        _require_inside(config.repo_root, root)
        manifest = _read_json(root / CACHE_FILES["manifest"])
        _validate_written_manifest(config, manifest)
        artifacts = _object(manifest["artifacts"], "artifacts")
        for name in (
            "roi_relation",
            "availability",
            "window_index",
            "slot_index",
        ):
            spec = _object(artifacts[name], f"artifacts.{name}")
            path = root / str(spec["filename"])
            if file_sha256(path) != str(spec["sha256"]):
                errors.append(f"artifact_hash_mismatch={name}")
            elif path.stat().st_size != int(spec["size_bytes"]):
                errors.append(f"artifact_size_mismatch={name}")
            else:
                verified += 1
        roi = np.load(root / CACHE_FILES["roi_relation"], mmap_mode="r")
        availability = np.load(
            root / CACHE_FILES["availability"],
            mmap_mode="r",
        )
        try:
            roi_shape = list(roi.shape)
            availability_shape = list(availability.shape)
            if roi.shape != (
                EXPECTED_MODEL_WINDOWS,
                SEQUENCE_LENGTH,
                ROI_RELATION_DIM,
            ):
                errors.append(f"roi_relation_shape={roi_shape}")
            if availability.shape != (
                EXPECTED_MODEL_WINDOWS,
                SEQUENCE_LENGTH,
            ):
                errors.append(f"availability_shape={availability_shape}")
            if roi.dtype != ROI_RELATION_DTYPE:
                errors.append(f"roi_relation_dtype={roi.dtype}")
            if availability.dtype != AVAILABILITY_DTYPE:
                errors.append(f"availability_dtype={availability.dtype}")
            if not np.isfinite(roi).all():
                errors.append("roi_relation_contains_nonfinite")
            if not availability.all():
                errors.append("legacy_roi_relation_unavailable")
        finally:
            _close_memmap(roi)
            _close_memmap(availability)
        window_index = pd.read_csv(root / CACHE_FILES["window_index"])
        slot_index = pd.read_csv(root / CACHE_FILES["slot_index"])
        _validate_order_indexes(window_index, slot_index)
        expected_hash = manifest["parent_view"]["ordered_window_id_sha256"]
        if _ordered_sha256(window_index["window_id"]) != expected_hash:
            errors.append("ordered_window_id_sha256_mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    valid = not errors
    return {
        "schema_version": CACHE_AUDIT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_CACHE_AUDIT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_ROI_RELATION_CACHE_AUDIT"
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
        "roi_relation_shape": roi_shape,
        "availability_shape": availability_shape,
        "outer_holdout_slots_materialized": 0,
        "source_media_reads": 0,
        "errors": errors,
        "valid": valid,
    }


def _load_order_authority(
    config: LegacyL6ROIRelationCacheConfig,
) -> Any:
    order = _object(config.payload["order_authority"], "order_authority")
    order_config = load_geometry_cache_config(
        config.bound_path("order_authority", "config")
    )
    order_root = _resolve_inside(
        config.repo_root,
        str(order["root_relative_path"]),
    )
    cache = load_geometry_cache(order_config, cache_root=order_root)
    expected_manifest = config.bound_path("order_authority", "manifest")
    if expected_manifest != order_root / "geometry_cache_manifest.json":
        raise ValueError("ROI order manifest path drift")
    if cache.audit["manifest_sha256"] != file_sha256(expected_manifest):
        raise ValueError("ROI order manifest hash drift")
    return cache


def _load_roi_frames(config: LegacyL6ROIRelationCacheConfig) -> pd.DataFrame:
    path = config.bound_path("inputs", "frame_roi")
    columns = [
        "frame_uid",
        "source_type",
        "dataset_id",
        "lineage_scope",
        "human_review_complete",
        *ROI_RELATION_FEATURE_NAMES,
        *ROI_AVAILABILITY_FIELDS,
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    _validate_roi_frames(frame)
    return frame


def _validate_roi_frames(frame: pd.DataFrame) -> None:
    required = {
        "frame_uid",
        "source_type",
        "dataset_id",
        "lineage_scope",
        "human_review_complete",
        *ROI_RELATION_FEATURE_NAMES,
        *ROI_AVAILABILITY_FIELDS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ROI frame columns missing={missing}")
    if len(frame) != EXPECTED_RAW_ROWS:
        raise ValueError(f"ROI frame rows={len(frame)}")
    if frame["frame_uid"].isna().any() or frame["frame_uid"].duplicated().any():
        raise ValueError("ROI frame_uid is blank or duplicated")
    _require_set(frame["source_type"], {SOURCE_TYPE}, "ROI source_type")
    _require_set(frame["dataset_id"], {DATASET_ID}, "ROI dataset_id")
    _require_set(frame["lineage_scope"], {LINEAGE_SCOPE}, "ROI lineage")
    if _strict_bool(
        frame["human_review_complete"],
        name="human_review_complete",
    ).any():
        raise ValueError("ROI frames claim human review complete")
    if any("target" in name or "behavior" in name for name in required):
        raise ValueError("target-selected ROI field entered cache contract")


def _validate_order_indexes(
    window_index: pd.DataFrame,
    slot_index: pd.DataFrame,
) -> None:
    if len(window_index) != EXPECTED_MODEL_WINDOWS:
        raise ValueError(f"ROI window index rows={len(window_index)}")
    if len(slot_index) != EXPECTED_MODEL_SLOTS:
        raise ValueError(f"ROI slot index rows={len(slot_index)}")
    expected = np.arange(EXPECTED_MODEL_WINDOWS, dtype=np.int64)
    if not np.array_equal(
        pd.to_numeric(window_index["cache_row"]).to_numpy(dtype=np.int64),
        expected,
    ):
        raise ValueError("ROI window cache_row order drift")
    slots = slot_index.sort_values(
        ["cache_row", "slot_index"],
        kind="mergesort",
    )
    observed = slots.groupby("cache_row", sort=False)["slot_index"].apply(list)
    if any(value != list(range(SEQUENCE_LENGTH)) for value in observed):
        raise ValueError("ROI slot order is not exact T6")
    if slot_index["frame_uid"].isna().any():
        raise ValueError("ROI order contains blank frame_uid")


def _validate_relation_bounds(values: np.ndarray) -> None:
    if values.shape[1] != ROI_RELATION_DIM or not np.isfinite(values).all():
        raise ValueError("ROI relation tensor is nonfinite or width-drifted")
    for index, name in enumerate(ROI_RELATION_FEATURE_NAMES):
        column = values[:, index]
        if name.endswith("_min_dist_n") and (column < 0.0).any():
            raise ValueError(f"negative ROI distance={name}")
        if name.endswith(("_max_overlap_ratio", "_max_iou")) and (
            (column < 0.0) | (column > 1.0)
        ).any():
            raise ValueError(f"ROI ratio out of bounds={name}")
        if name in BOOLEAN_RELATION_FIELDS and not np.isin(
            column,
            [0.0, 1.0],
        ).all():
            raise ValueError(f"ROI boolean relation is not binary={name}")


def _cache_content_audit(
    roi: np.ndarray,
    availability: np.ndarray,
    window_index: pd.DataFrame,
    slot_index: pd.DataFrame,
) -> dict[str, Any]:
    flattened = roi.reshape(-1, ROI_RELATION_DIM).astype(np.float64)
    return {
        "model_window_rows": len(window_index),
        "model_slot_rows": len(slot_index),
        "roi_relation_shape": list(roi.shape),
        "availability_shape": list(availability.shape),
        "available_slots": int(availability.sum()),
        "unavailable_slots": int((~availability).sum()),
        "availability_pattern": availability.all(axis=0).astype(int).tolist(),
        "feature_summaries": {
            name: {
                "minimum": float(flattened[:, index].min()),
                "maximum": float(flattened[:, index].max()),
                "mean": float(flattened[:, index].mean()),
            }
            for index, name in enumerate(ROI_RELATION_FEATURE_NAMES)
        },
        "source_probe": {
            "status": "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE",
            "source_type_values": [SOURCE_TYPE],
            "dataset_id_values": [DATASET_ID],
            "source_probe_auc": None,
            "source_probe_macro_f1": None,
        },
        "target_selected_roi_fields_used": False,
        "unit_aggregate_features_used": False,
        "geometry_values_used": False,
        "motion_values_used": False,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "valid": True,
    }


def _source_bindings(
    config: LegacyL6ROIRelationCacheConfig,
) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for section in ("parents", "inputs"):
        bindings[section] = {
            name: {
                "path": value["path"],
                "sha256": value["sha256"],
            }
            for name, value in _object(
                config.payload[section],
                section,
            ).items()
        }
    bindings["order_authority"] = {
        name: {
            "path": config.payload["order_authority"][name]["path"],
            "sha256": config.payload["order_authority"][name]["sha256"],
        }
        for name in ("config", "manifest")
    }
    return bindings


def _validate_motion_parent(config: LegacyL6ROIRelationCacheConfig) -> None:
    decision = _read_json(config.bound_path("parents", "l6_motion_decision"))
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L6_MOTION_SHORT_DECISION",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "valid": True,
    }
    for field, value in expected.items():
        if decision.get(field) != value:
            raise ValueError(f"motion parent {field} drift")
    outcome = _object(decision.get("decision"), "motion parent decision")
    if outcome.get("decision") != (
        "DO_NOT_EXPAND_MOTION_FROM_CURRENT_SHORT_EVIDENCE"
    ):
        raise ValueError("motion parent decision drift")
    if outcome.get("full_motion_expansion_authorized") is not False:
        raise ValueError("motion parent unexpectedly authorizes full expansion")


def _validate_written_manifest(
    config: LegacyL6ROIRelationCacheConfig,
    manifest: dict[str, Any],
) -> None:
    expected = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_ROI_RELATION_CACHE",
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
        "errors": [],
        "valid": True,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"ROI manifest {field} drift")
    contract = _object(manifest["feature_contract"], "feature_contract")
    if contract.get("feature_names") != list(ROI_RELATION_FEATURE_NAMES):
        raise ValueError("ROI manifest feature order drift")
    if contract.get("target_selected_roi_fields_used") is not False:
        raise ValueError("ROI manifest target-selected field drift")


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
    _require_exact_keys(payload, required, "ROI relation cache config")
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
        if payload.get(field) != value:
            raise ValueError(f"ROI config {field} drift")
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
        "expected_rows": EXPECTED_RAW_ROWS,
        "source_type": SOURCE_TYPE,
        "dataset_id": DATASET_ID,
        "merged_data": False,
    }
    for field, value in source_expected.items():
        if source.get(field) != value:
            raise ValueError(f"ROI source {field} drift")
    _require_sha(str(source["raw_sha256"]), "source raw SHA256")
    parents = _object(payload["parents"], "parents")
    _require_exact_keys(
        parents,
        {"temporal_ladder_config", "l5_decision", "l6_motion_decision"},
        "parents",
    )
    inputs = _object(payload["inputs"], "inputs")
    _require_exact_keys(inputs, {"frame_roi"}, "inputs")
    for section, values in (("parents", parents), ("inputs", inputs)):
        for name, value in values.items():
            _validate_bound_spec(value, f"{section}.{name}")
    order = _object(payload["order_authority"], "order_authority")
    _require_exact_keys(
        order,
        {"config", "manifest", "root_relative_path", "geometry_values_used"},
        "order_authority",
    )
    _validate_bound_spec(order["config"], "order_authority.config")
    _validate_bound_spec(order["manifest"], "order_authority.manifest")
    if order["geometry_values_used"] is not False:
        raise ValueError("ROI cache may not consume geometry values")
    feature_expected = {
        "view_id": VIEW_ID,
        "temporal_view_name": TEMPORAL_VIEW_NAME,
        "sequence_length": SEQUENCE_LENGTH,
        "model_window_rows": EXPECTED_MODEL_WINDOWS,
        "model_slot_rows": EXPECTED_MODEL_SLOTS,
        "feature_names": list(ROI_RELATION_FEATURE_NAMES),
        "feature_dim": ROI_RELATION_DIM,
        "feature_dtype": str(ROI_RELATION_DTYPE),
        "availability_dtype": str(AVAILABILITY_DTYPE),
        "all_classes_exposed_independently": True,
        "target_selected_roi_fields_allowed": False,
        "unit_aggregate_features_allowed": False,
        "normalization": "none_raw_cache_train_frames_only_at_consumer",
        "source_media_fallback_allowed": False,
    }
    if _object(payload["features"], "features") != feature_expected:
        raise ValueError("ROI relation feature contract drift")
    _validate_bound_spec(payload["implementation"], "implementation")
    guard = _object(payload["execution_guard"], "execution_guard")
    _require_exact_keys(
        guard,
        {"allowed_dirty_paths", "required_tracked_paths"},
        "execution_guard",
    )
    output = _object(payload["output"], "output")
    _require_exact_keys(output, {"cache_root_relative_path"}, "output")


def _validate_bound_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    _require_exact_keys(spec, {"path", "sha256"}, name)
    _require_sha(str(spec["sha256"]), f"{name}.sha256")


def _require_set(series: pd.Series, expected: set[str], name: str) -> None:
    observed = set(series.fillna("").astype(str))
    if observed != expected:
        raise ValueError(f"{name}={sorted(observed)}")
