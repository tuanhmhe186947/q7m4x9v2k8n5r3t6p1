"""Hash-bound T6 pen-context cache for legacy-only development.

This cache is deliberately separate from the frozen geometry and motion
caches.  Geometry and generic motion remain fixed controls; only the seven
continuous pen-boundary values are added by the downstream ablation.
"""

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

from pig_behavior.classification_v2.features.pen_context import (
    PEN_CONTEXT_LEGACY_MODEL_FEATURE_COLUMNS,
    REQUIRED_PEN_CONTEXT_INPUT_COLUMNS,
    audit_pen_context_features,
    build_pen_context_features,
)
from pig_behavior.classification_v2.spatial_sequence_export import (
    LEGACY_SPATIAL_FRAME_FEATURES,
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
    LegacyL6GeometryCache,
    load_geometry_cache,
    load_geometry_cache_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

PEN_FEATURE_NAMES = tuple(PEN_CONTEXT_LEGACY_MODEL_FEATURE_COLUMNS)
PEN_STATIC_FEATURE_COUNT = 3
PEN_DIM = len(PEN_FEATURE_NAMES)
PEN_DTYPE = np.dtype(np.float32)
MASK_DTYPE = np.dtype(np.bool_)

CACHE_CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_cache_config.v1"
)
CACHE_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_cache_manifest.v1"
)
CACHE_AUDIT_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_cache_audit.v1"
)
CACHE_REPEAT_SCHEMA = (
    "classification_v2.legacy_development_l6.pen_context_cache_repeat.v1"
)

CACHE_FILES = {
    "pen": "pen_context_raw_f32.npy",
    "feature_availability": "pen_feature_available_bool.npy",
    "availability": "pen_context_available_bool.npy",
    "quality": "pen_quality_valid_bool.npy",
    "motion_availability": "pen_motion_available_bool.npy",
    "window_index": "pen_context_window_index.csv",
    "slot_index": "pen_context_slot_index.csv",
    "manifest": "pen_context_cache_manifest.json",
}

_FRAME_COLUMNS = tuple(
    dict.fromkeys(
        [
            *REQUIRED_PEN_CONTEXT_INPUT_COLUMNS,
            "lineage_scope",
            "human_review_complete",
            "cx_n",
            "cy_n",
            "bw_n",
            "bh_n",
            "area_n",
            "aspect_ratio",
            "actor_bbox_valid",
            "geometry_feature_valid",
            "spatiotemporal_feature_valid",
        ]
    )
)


@dataclass(frozen=True, slots=True)
class LegacyL6PenContextCacheConfig:
    """Immutable cache specification with primary and repeat roots."""

    path: Path
    payload: dict[str, Any]
    repo_root: Path

    @property
    def sha256(self) -> str:
        return file_sha256(self.path)

    def cache_root(self, variant: str) -> Path:
        if variant not in {"primary", "repeat"}:
            raise ValueError(f"unknown pen cache variant={variant}")
        value = self.payload["output"][f"{variant}_cache_root_relative_path"]
        return _resolve_inside(self.repo_root, str(value))

    def bound_path(self, section: str, name: str | None = None) -> Path:
        value: Any = self.payload[section]
        if name is not None:
            value = _object(value, section)[name]
        spec = _object(value, f"{section}.{name}" if name else section)
        return _resolve_inside(self.repo_root, str(spec["path"]))


@dataclass(frozen=True, slots=True)
class LegacyL6PenContextCache:
    """Audited pen tensors aligned to the frozen legacy T6 order."""

    root: Path
    pen_path: Path
    feature_availability_path: Path
    availability_path: Path
    quality_path: Path
    motion_availability_path: Path
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    manifest: dict[str, Any]
    audit: dict[str, Any]

    def load_pen(self, rows: np.ndarray | None = None) -> np.ndarray:
        return _load_array(self.pen_path, PEN_DTYPE, rows, len(self.window_index))

    def load_feature_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        return _load_array(
            self.feature_availability_path,
            MASK_DTYPE,
            rows,
            len(self.window_index),
        )

    def load_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        """Load the effective branch mask: available and quality-valid."""

        return _load_array(
            self.availability_path,
            MASK_DTYPE,
            rows,
            len(self.window_index),
        )

    def load_quality(self, rows: np.ndarray | None = None) -> np.ndarray:
        return _load_array(
            self.quality_path,
            MASK_DTYPE,
            rows,
            len(self.window_index),
        )

    def load_motion_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        return _load_array(
            self.motion_availability_path,
            MASK_DTYPE,
            rows,
            len(self.window_index),
        )


def load_pen_context_cache_config(
    path: Path,
) -> LegacyL6PenContextCacheConfig:
    """Load a cache config and verify every scientific input hash."""

    resolved = path.resolve()
    payload = _read_json(resolved)
    _validate_config_payload(payload)
    config = LegacyL6PenContextCacheConfig(
        path=resolved,
        payload=payload,
        repo_root=resolved.parents[2],
    )
    source = _object(payload["source_identity"], "source_identity")
    _validate_bound_file(
        _resolve_inside(config.repo_root, str(source["raw_authority_path"])),
        str(source["raw_sha256"]),
        "legacy raw authority",
    )
    for section in ("parents", "inputs", "implementation"):
        for name, value in _object(payload[section], section).items():
            spec = _object(value, f"{section}.{name}")
            _validate_bound_file(
                _resolve_inside(config.repo_root, str(spec["path"])),
                str(spec["sha256"]),
                f"pen cache {section}.{name}",
            )
    order = _object(payload["order_authority"], "order_authority")
    for name in ("config", "manifest"):
        spec = _object(order[name], f"order_authority.{name}")
        _validate_bound_file(
            _resolve_inside(config.repo_root, str(spec["path"])),
            str(spec["sha256"]),
            f"pen cache order_authority.{name}",
        )
    mask = _object(payload["mask_contract"], "mask_contract")
    _validate_bound_file(
        _resolve_inside(config.repo_root, str(mask["path"])),
        str(mask["sha256"]),
        "pen calibration mask",
    )
    return config


def preflight_pen_context_cache(
    config: LegacyL6PenContextCacheConfig,
    *,
    variant: str,
) -> dict[str, Any]:
    """Check lineage, order, mask, and exclusive output before building."""

    errors: list[str] = []
    windows = 0
    slots = 0
    roles: dict[str, int] = {}
    root = config.cache_root(variant)
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
        if root.exists():
            errors.append(f"cache_output_exists={root}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    valid = not errors
    return {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "pen_context_cache_preflight.v1"
        ),
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_CACHE_PREFLIGHT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_CACHE_PREFLIGHT"
        ),
        "variant": variant,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "output_root": str(root),
        "model_window_rows": windows,
        "model_slot_rows": slots,
        "model_window_roles": roles,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "errors": errors,
        "valid": valid,
    }


def build_pen_context_cache(
    config: LegacyL6PenContextCacheConfig,
    *,
    variant: str,
) -> tuple[Path, dict[str, Any]]:
    """Build one exclusive cache without decoding source video."""

    preflight = preflight_pen_context_cache(config, variant=variant)
    if not preflight["valid"]:
        raise RuntimeError(f"pen cache preflight failed={preflight['errors']}")
    order = _load_order_authority(config)
    frames = _load_harmonized_frames(config)
    mask = _object(config.payload["mask_contract"], "mask_contract")
    result = materialize_pen_context_cache(
        order.window_index,
        order.slot_index,
        frames,
        mask_path=_resolve_inside(config.repo_root, str(mask["path"])),
        expected_mask_sha256=str(mask["sha256"]),
        mask_threshold=int(mask["threshold"]),
        near_boundary_clearance_ratio=float(
            mask["near_boundary_clearance_ratio"]
        ),
    )
    root = config.cache_root(variant)
    temporary = root.with_name(f"{root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"pen cache temporary output exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    paths = {name: temporary / filename for name, filename in CACHE_FILES.items()}
    for name in (
        "pen",
        "feature_availability",
        "availability",
        "quality",
        "motion_availability",
    ):
        np.save(paths[name], result[name], allow_pickle=False)
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
    artifact_names = tuple(name for name in CACHE_FILES if name != "manifest")
    artifacts = {
        name: {
            "filename": CACHE_FILES[name],
            "sha256": file_sha256(paths[name]),
            "size_bytes": int(paths[name].stat().st_size),
        }
        for name in artifact_names
    }
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_CACHE",
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
        "code_sha": _git(config.repo_root, "rev-parse", "HEAD").strip(),
        "dirty_worktree": bool(
            _git(
                config.repo_root,
                "status",
                "--porcelain",
                "--untracked-files=all",
            ).strip()
        ),
        "mask_contract": dict(mask),
        "parent_view": {
            "view_id": "t6_sliding",
            "sequence_length": SEQUENCE_LENGTH,
            "model_window_rows": EXPECTED_MODEL_WINDOWS,
            "model_slot_rows": EXPECTED_MODEL_SLOTS,
            "ordered_window_id_sha256": _ordered_sha256(
                result["window_index"]["window_id"]
            ),
        },
        "feature_contract": {
            "feature_names": list(PEN_FEATURE_NAMES),
            "feature_dim": PEN_DIM,
            "feature_dtype": str(PEN_DTYPE),
            "feature_availability_dtype": str(MASK_DTYPE),
            "static_features": list(PEN_FEATURE_NAMES[:PEN_STATIC_FEATURE_COUNT]),
            "pair_features": list(PEN_FEATURE_NAMES[PEN_STATIC_FEATURE_COUNT:]),
            "effective_branch_mask": (
                "pen_context_available_and_pen_context_quality_valid"
            ),
            "window_local_pair_rebase": True,
            "first_slot_pair_features_zero": True,
            "normalization": "fold_train_unique_frame_or_pair_at_consumer",
            "availability_is_behavior_evidence": False,
            "paths_ids_review_labels_in_model_x": False,
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
    audit = audit_pen_context_cache(config, cache_root=root)
    if not audit["valid"]:
        raise RuntimeError(f"written pen cache failed audit={audit['errors']}")
    return root / CACHE_FILES["manifest"], audit


def materialize_pen_context_cache(
    window_index: pd.DataFrame,
    slot_index: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    mask_path: Path,
    expected_mask_sha256: str,
    mask_threshold: int = 127,
    near_boundary_clearance_ratio: float = 1.0,
) -> dict[str, Any]:
    """Create aligned values and separate static, quality, and pair masks."""

    windows, slots, export_windows = _validated_order(window_index, slot_index)
    derived = build_pen_context_features(
        frames,
        mask_path=mask_path,
        mask_threshold=mask_threshold,
        near_boundary_clearance_ratio=near_boundary_clearance_ratio,
        expected_mask_sha256=expected_mask_sha256,
    )
    feature_audit = audit_pen_context_features(
        derived,
        mask_path=mask_path,
        mask_threshold=mask_threshold,
        near_boundary_clearance_ratio=near_boundary_clearance_ratio,
        input_rows=len(frames),
        expected_mask_sha256=expected_mask_sha256,
    )
    if feature_audit["errors"]:
        raise ValueError(f"pen frame feature audit={feature_audit['errors']}")
    exported = export_legacy_development_spatial_sequences(
        export_windows,
        derived,
        max_window_length=SEQUENCE_LENGTH,
        feature_schema={
            "pen_boundary_context": list(
                LEGACY_SPATIAL_FRAME_FEATURES["pen_boundary_context"]
            ),
            "quality_mask": list(
                LEGACY_SPATIAL_FRAME_FEATURES["quality_mask"][:4]
            ),
        },
    )
    names = tuple(exported.feature_names.get("pen_boundary_context", []))
    if names != PEN_FEATURE_NAMES:
        raise ValueError(f"pen exported feature order={names}")
    values = np.asarray(
        exported.arrays["pen_boundary_context"],
        dtype=PEN_DTYPE,
    ).copy()
    joined = _join_slot_quality(slots, derived)
    raw_available = joined["pen_context_available"].to_numpy(bool).reshape(
        len(windows),
        SEQUENCE_LENGTH,
    )
    quality = joined["pen_context_quality_valid"].to_numpy(bool).reshape(
        len(windows),
        SEQUENCE_LENGTH,
    )
    branch_available = raw_available & quality
    frame_index = joined["frame_index"].to_numpy(np.int64).reshape(
        len(windows),
        SEQUENCE_LENGTH,
    )
    pair_available = np.zeros_like(branch_available)
    pair_available[:, 1:] = (
        branch_available[:, :-1]
        & branch_available[:, 1:]
        & (np.diff(frame_index, axis=1) > 0)
    )
    feature_available = np.zeros(values.shape, dtype=MASK_DTYPE)
    feature_available[..., :PEN_STATIC_FEATURE_COUNT] = branch_available[
        ..., None
    ]
    feature_available[..., PEN_STATIC_FEATURE_COUNT:] = pair_available[
        ..., None
    ]
    values[~feature_available] = 0.0
    if not np.isfinite(values).all():
        raise ValueError("pen cache values contain nonfinite entries")
    if int(exported.audit["pen_valid_pair_count"]) != int(pair_available.sum()):
        raise ValueError("pen exported and cache pair counts differ")
    output_slots = _build_slot_index(
        joined,
        raw_available=raw_available,
        quality=quality,
        branch_available=branch_available,
        pair_available=pair_available,
    )
    content = _content_audit(
        windows,
        output_slots,
        values,
        feature_available,
        raw_available,
        quality,
        branch_available,
        pair_available,
        exported.audit,
        feature_audit,
    )
    return {
        "pen": values,
        "feature_availability": feature_available,
        "availability": branch_available.astype(MASK_DTYPE),
        "quality": quality.astype(MASK_DTYPE),
        "motion_availability": pair_available.astype(MASK_DTYPE),
        "window_index": windows,
        "slot_index": output_slots,
        "content_audit": content,
    }


def load_pen_context_cache(
    config: LegacyL6PenContextCacheConfig,
    *,
    cache_root: Path | None = None,
) -> LegacyL6PenContextCache:
    root = (cache_root or config.cache_root("primary")).resolve()
    audit = audit_pen_context_cache(config, cache_root=root)
    if not audit["valid"]:
        raise ValueError(f"pen cache audit failed={audit['errors']}")
    return LegacyL6PenContextCache(
        root=root,
        pen_path=root / CACHE_FILES["pen"],
        feature_availability_path=root / CACHE_FILES["feature_availability"],
        availability_path=root / CACHE_FILES["availability"],
        quality_path=root / CACHE_FILES["quality"],
        motion_availability_path=root / CACHE_FILES["motion_availability"],
        window_index=pd.read_csv(root / CACHE_FILES["window_index"]),
        slot_index=pd.read_csv(root / CACHE_FILES["slot_index"]),
        manifest=_read_json(root / CACHE_FILES["manifest"]),
        audit=audit,
    )


def audit_pen_context_cache(
    config: LegacyL6PenContextCacheConfig,
    *,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Re-hash and structurally validate a materialized cache."""

    root = (cache_root or config.cache_root("primary")).resolve()
    errors: list[str] = []
    verified = 0
    shapes: dict[str, list[int]] = {}
    try:
        _require_inside(config.repo_root, root)
        manifest = _read_json(root / CACHE_FILES["manifest"])
        expected_manifest = {
            "schema_version": CACHE_MANIFEST_SCHEMA,
            "status": "PASS_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_CACHE",
            "lineage_scope": LINEAGE_SCOPE,
            "config_sha256": config.sha256,
            "human_review_complete": False,
            "reviewed_or_final_claim_allowed": False,
            "q2_claim_allowed": False,
            "canonical_full_oof_authorized": False,
            "errors": [],
            "valid": True,
        }
        for field, expected in expected_manifest.items():
            if manifest.get(field) != expected:
                errors.append(
                    f"manifest_{field}={manifest.get(field)!r}!={expected!r}"
                )
        artifacts = _object(manifest.get("artifacts"), "artifacts")
        for name in CACHE_FILES:
            if name == "manifest":
                continue
            spec = _object(artifacts.get(name), f"artifact.{name}")
            path = root / str(spec.get("filename", ""))
            if not path.is_file():
                errors.append(f"artifact_missing={name}")
            elif file_sha256(path) != spec.get("sha256"):
                errors.append(f"artifact_hash_mismatch={name}")
            elif int(path.stat().st_size) != int(spec.get("size_bytes", -1)):
                errors.append(f"artifact_size_mismatch={name}")
            else:
                verified += 1
        arrays = {
            "pen": np.load(root / CACHE_FILES["pen"], mmap_mode="r"),
            "feature_availability": np.load(
                root / CACHE_FILES["feature_availability"],
                mmap_mode="r",
            ),
            "availability": np.load(
                root / CACHE_FILES["availability"],
                mmap_mode="r",
            ),
            "quality": np.load(root / CACHE_FILES["quality"], mmap_mode="r"),
            "motion_availability": np.load(
                root / CACHE_FILES["motion_availability"],
                mmap_mode="r",
            ),
        }
        try:
            shapes = {name: list(array.shape) for name, array in arrays.items()}
            _validate_written_arrays(arrays, errors)
        finally:
            for array in arrays.values():
                _close_memmap(array)
        windows = pd.read_csv(root / CACHE_FILES["window_index"])
        slots = pd.read_csv(root / CACHE_FILES["slot_index"])
        _, _, _ = _validated_order(windows, slots)
        ordered = _ordered_sha256(windows["window_id"])
        expected = manifest["parent_view"]["ordered_window_id_sha256"]
        if ordered != expected:
            errors.append("ordered_window_id_sha256_mismatch")
        content = _object(manifest.get("content_audit"), "content_audit")
        if content.get("errors") or content.get("valid") is not True:
            errors.append("manifest_content_audit_invalid")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    valid = not errors
    return {
        "schema_version": CACHE_AUDIT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_CACHE_AUDIT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_CACHE_AUDIT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "cache_root": str(root),
        "config_sha256": config.sha256,
        "manifest_sha256": (
            file_sha256(root / CACHE_FILES["manifest"])
            if (root / CACHE_FILES["manifest"]).is_file()
            else None
        ),
        "verified_artifacts": verified,
        "array_shapes": shapes,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "errors": errors,
        "valid": valid,
    }


def audit_pen_context_cache_repeat(
    config: LegacyL6PenContextCacheConfig,
) -> dict[str, Any]:
    """Require independent primary/repeat builds to be byte-identical."""

    primary = audit_pen_context_cache(
        config,
        cache_root=config.cache_root("primary"),
    )
    repeat = audit_pen_context_cache(
        config,
        cache_root=config.cache_root("repeat"),
    )
    errors = [*primary["errors"], *repeat["errors"]]
    artifact_equality: dict[str, bool] = {}
    for name, filename in CACHE_FILES.items():
        left = config.cache_root("primary") / filename
        right = config.cache_root("repeat") / filename
        equal = left.is_file() and right.is_file() and (
            file_sha256(left) == file_sha256(right)
        )
        artifact_equality[name] = equal
        if not equal:
            errors.append(f"repeat_artifact_differs={name}")
    valid = not errors
    return {
        "schema_version": CACHE_REPEAT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_CACHE_REPEAT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_PEN_CONTEXT_CACHE_REPEAT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "config_sha256": config.sha256,
        "primary": primary,
        "repeat": repeat,
        "artifact_sha256_equal": artifact_equality,
        "all_artifact_sha256_equal": all(artifact_equality.values()),
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "errors": errors,
        "valid": valid,
    }


def _load_order_authority(
    config: LegacyL6PenContextCacheConfig,
) -> LegacyL6GeometryCache:
    order = _object(config.payload["order_authority"], "order_authority")
    order_config = load_geometry_cache_config(
        config.bound_path("order_authority", "config")
    )
    root = _resolve_inside(config.repo_root, str(order["root_relative_path"]))
    cache = load_geometry_cache(order_config, cache_root=root)
    expected = config.bound_path("order_authority", "manifest")
    if cache.audit["manifest_sha256"] != file_sha256(expected):
        raise ValueError("pen order-authority manifest hash drift")
    return cache


def _load_harmonized_frames(
    config: LegacyL6PenContextCacheConfig,
) -> pd.DataFrame:
    path = config.bound_path("inputs", "harmonized_frames")
    try:
        frame = pd.read_csv(path, usecols=list(_FRAME_COLUMNS))
    except ValueError as error:
        raise ValueError(f"pen frame columns unavailable: {error}") from error
    if len(frame) != EXPECTED_RAW_ROWS:
        raise ValueError(f"pen frame rows={len(frame)}")
    if set(frame["source_type"].astype(str)) != {SOURCE_TYPE}:
        raise ValueError("pen frame source_type drift")
    if set(frame["dataset_id"].astype(str)) != {DATASET_ID}:
        raise ValueError("pen frame dataset_id drift")
    if set(frame["lineage_scope"].astype(str)) != {LINEAGE_SCOPE}:
        raise ValueError("pen frame lineage_scope drift")
    if _strict_bool(frame["human_review_complete"]).any():
        raise ValueError("pen frame table claims completed human review")
    if frame["frame_uid"].astype(str).duplicated().any():
        raise ValueError("pen frame table has duplicate frame_uid")
    return frame


def _validated_order(
    window_index: pd.DataFrame,
    slot_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = window_index.copy().reset_index(drop=True)
    slots = slot_index.copy()
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
            f"pen order missing windows={missing_windows} slots={missing_slots}"
        )
    if windows["window_id"].astype(str).duplicated().any():
        raise ValueError("pen order has duplicate window_id")
    windows["cache_row"] = pd.to_numeric(windows["cache_row"]).astype(int)
    expected_rows = np.arange(len(windows), dtype=np.int64)
    if not np.array_equal(windows["cache_row"], expected_rows):
        raise ValueError("pen order cache_row drift")
    slots["cache_row"] = pd.to_numeric(slots["cache_row"]).astype(int)
    slots["slot_index"] = pd.to_numeric(slots["slot_index"]).astype(int)
    slots["frame_index"] = pd.to_numeric(slots["frame_index"]).astype(int)
    slots = slots.sort_values(
        ["cache_row", "slot_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    expected_slots = len(windows) * SEQUENCE_LENGTH
    if len(slots) != expected_slots:
        raise ValueError(f"pen order slot rows={len(slots)}!={expected_slots}")
    if slots[["window_id", "slot_index"]].duplicated().any():
        raise ValueError("pen order has duplicate window-slot key")
    cache_rows = np.repeat(expected_rows, SEQUENCE_LENGTH)
    slot_rows = np.tile(np.arange(SEQUENCE_LENGTH), len(windows))
    if not np.array_equal(slots["cache_row"], cache_rows):
        raise ValueError("pen order slot cache_row drift")
    if not np.array_equal(slots["slot_index"], slot_rows):
        raise ValueError("pen order slot_index drift")
    object_keys = slots["object_track_key"].astype(str).to_numpy().reshape(
        len(windows),
        SEQUENCE_LENGTH,
    )
    frame_indices = slots["frame_index"].to_numpy().reshape(
        len(windows),
        SEQUENCE_LENGTH,
    )
    if np.any(object_keys != object_keys[:, :1]):
        raise ValueError("pen order window crosses object tracks")
    if not np.all(np.diff(frame_indices, axis=1) == 1):
        raise ValueError("pen order frame indices are not contiguous")
    export_windows = pd.DataFrame(
        {
            "window_id": windows["window_id"].astype(str),
            "object_track_key": object_keys[:, 0],
            "window_start_frame": frame_indices[:, 0],
            "window_end_frame": frame_indices[:, -1],
            "window_length_frames": SEQUENCE_LENGTH,
        }
    )
    return windows, slots, export_windows


def _join_slot_quality(
    slots: pd.DataFrame,
    derived: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "frame_uid",
        "object_track_key",
        "frame_index",
        "pen_context_available",
        "pen_context_quality_valid",
    ]
    right = derived[columns].copy()
    joined = slots.merge(
        right,
        on="frame_uid",
        how="left",
        validate="many_to_one",
        suffixes=("", "_frame"),
        sort=False,
    )
    if len(joined) != len(slots):
        raise ValueError("pen slot join changed row count")
    if joined["pen_context_available"].isna().any():
        raise ValueError("pen slot join lost frame rows")
    if not joined["object_track_key"].astype(str).equals(
        joined["object_track_key_frame"].astype(str)
    ):
        raise ValueError("pen slot join object_track_key mismatch")
    if not pd.to_numeric(joined["frame_index"]).equals(
        pd.to_numeric(joined["frame_index_frame"])
    ):
        raise ValueError("pen slot join frame_index mismatch")
    return joined.drop(
        columns=["object_track_key_frame", "frame_index_frame"]
    )


def _build_slot_index(
    joined: pd.DataFrame,
    *,
    raw_available: np.ndarray,
    quality: np.ndarray,
    branch_available: np.ndarray,
    pair_available: np.ndarray,
) -> pd.DataFrame:
    frame = joined[
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
    frame["pen_context_available_raw"] = raw_available.reshape(-1)
    frame["pen_context_quality_valid"] = quality.reshape(-1)
    frame["pen_context_available"] = branch_available.reshape(-1)
    frame["pen_motion_available"] = pair_available.reshape(-1)
    previous = frame.groupby("cache_row", sort=False)["frame_uid"].shift(1)
    frame["previous_frame_uid"] = previous.fillna("").astype(str)
    pair_uid = frame["previous_frame_uid"] + "->" + frame["frame_uid"].astype(str)
    frame["pen_pair_uid"] = pair_uid.where(frame["pen_motion_available"], "")
    frame["pen_window_slot_uid"] = (
        frame["window_id"].astype(str)
        + "::slot="
        + frame["slot_index"].astype(str)
    )
    if frame[["window_id", "slot_index"]].duplicated().any():
        raise ValueError("pen output slot index contains duplicates")
    return frame


def _content_audit(
    windows: pd.DataFrame,
    slots: pd.DataFrame,
    values: np.ndarray,
    feature_available: np.ndarray,
    raw_available: np.ndarray,
    quality: np.ndarray,
    branch_available: np.ndarray,
    pair_available: np.ndarray,
    spatial_audit: dict[str, Any],
    feature_audit: dict[str, Any],
) -> dict[str, Any]:
    feature_summaries: dict[str, Any] = {}
    for index, name in enumerate(PEN_FEATURE_NAMES):
        selected = values[..., index][feature_available[..., index]]
        feature_summaries[name] = {
            "available_values": int(selected.size),
            "min": float(selected.min()) if selected.size else None,
            "mean": float(selected.mean()) if selected.size else None,
            "max": float(selected.max()) if selected.size else None,
        }
    return {
        "window_rows": int(len(windows)),
        "slot_rows": int(len(slots)),
        "window_ids_unique": not windows["window_id"].duplicated().any(),
        "window_slot_keys_unique": not slots[
            ["window_id", "slot_index"]
        ].duplicated().any(),
        "pen_shape": list(values.shape),
        "feature_availability_shape": list(feature_available.shape),
        "raw_available_slots": int(raw_available.sum()),
        "quality_valid_slots": int(quality.sum()),
        "effective_branch_available_slots": int(branch_available.sum()),
        "motion_pair_available_slots": int(pair_available.sum()),
        "availability_pattern": branch_available.sum(axis=0).astype(int).tolist(),
        "pair_availability_pattern": pair_available.sum(axis=0).astype(int).tolist(),
        "feature_summaries": feature_summaries,
        "spatial_export": spatial_audit,
        "frame_feature_audit": feature_audit,
        "source_probe": {
            "status": "NOT_ESTIMABLE_SINGLE_LEGACY_SOURCE",
            "source_type": SOURCE_TYPE,
        },
        "selected_model_x": [*PEN_FEATURE_NAMES, "pen_context_available"],
        "binary_near_boundary_selected": False,
        "paths_ids_review_labels_selected": False,
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "errors": [],
        "valid": True,
    }


def _validate_written_arrays(
    arrays: dict[str, np.ndarray],
    errors: list[str],
) -> None:
    expected = {
        "pen": (EXPECTED_MODEL_WINDOWS, SEQUENCE_LENGTH, PEN_DIM),
        "feature_availability": (
            EXPECTED_MODEL_WINDOWS,
            SEQUENCE_LENGTH,
            PEN_DIM,
        ),
        "availability": (EXPECTED_MODEL_WINDOWS, SEQUENCE_LENGTH),
        "quality": (EXPECTED_MODEL_WINDOWS, SEQUENCE_LENGTH),
        "motion_availability": (EXPECTED_MODEL_WINDOWS, SEQUENCE_LENGTH),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            errors.append(f"{name}_shape={list(arrays[name].shape)}")
    if arrays["pen"].dtype != PEN_DTYPE:
        errors.append(f"pen_dtype={arrays['pen'].dtype}")
    for name in expected:
        if name != "pen" and arrays[name].dtype != MASK_DTYPE:
            errors.append(f"{name}_dtype={arrays[name].dtype}")
    if not np.isfinite(arrays["pen"]).all():
        errors.append("pen_contains_nonfinite")
    feature_mask = np.asarray(arrays["feature_availability"], dtype=bool)
    if np.any(arrays["pen"][~feature_mask] != 0.0):
        errors.append("unavailable_pen_values_are_nonzero")
    branch = np.asarray(arrays["availability"], dtype=bool)
    quality = np.asarray(arrays["quality"], dtype=bool)
    if np.any(branch & ~quality):
        errors.append("branch_availability_outside_quality")
    if not np.array_equal(feature_mask[..., 0], branch):
        errors.append("static_feature_mask_differs_from_branch_mask")
    motion = np.asarray(arrays["motion_availability"], dtype=bool)
    if motion[:, 0].any():
        errors.append("first_slot_pen_motion_available")
    if not np.array_equal(
        feature_mask[..., PEN_STATIC_FEATURE_COUNT],
        motion,
    ):
        errors.append("pair_feature_mask_differs_from_motion_mask")


def _validate_config_payload(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "source_identity",
        "mask_contract",
        "parents",
        "inputs",
        "order_authority",
        "features",
        "implementation",
        "output",
    }
    if set(payload) != required:
        raise ValueError(
            "pen cache config keys differ: "
            f"missing={sorted(required - set(payload))},"
            f"extra={sorted(set(payload) - required)}"
        )
    expected_identity = {
        "schema_version": CACHE_CONFIG_SCHEMA,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
    }
    for field, expected in expected_identity.items():
        if payload[field] != expected:
            raise ValueError(f"pen cache config {field}={payload[field]!r}")
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
        raise ValueError("pen cache source identity drift")
    mask = _object(payload["mask_contract"], "mask_contract")
    expected_mask_keys = {
        "path",
        "sha256",
        "width",
        "height",
        "threshold",
        "resize_interpolation",
        "near_boundary_clearance_ratio",
        "camera_scope",
    }
    if set(mask) != expected_mask_keys:
        raise ValueError("pen cache mask contract keys drift")
    if mask["resize_interpolation"] != "nearest":
        raise ValueError("pen cache mask resize must use nearest")
    if mask["camera_scope"] != "current_single_fixed_camera_only":
        raise ValueError("pen cache mask camera scope drift")
    for section in ("parents", "inputs", "implementation"):
        values = _object(payload[section], section)
        for name, value in values.items():
            _validate_hash_spec(value, f"{section}.{name}")
    order = _object(payload["order_authority"], "order_authority")
    if set(order) != {"config", "manifest", "root_relative_path"}:
        raise ValueError("pen cache order authority keys drift")
    _validate_hash_spec(order["config"], "order_authority.config")
    _validate_hash_spec(order["manifest"], "order_authority.manifest")
    features = _object(payload["features"], "features")
    expected_features = {
        "view_id": "t6_sliding",
        "sequence_length": SEQUENCE_LENGTH,
        "model_window_rows": EXPECTED_MODEL_WINDOWS,
        "model_slot_rows": EXPECTED_MODEL_SLOTS,
        "feature_names": list(PEN_FEATURE_NAMES),
        "feature_dim": PEN_DIM,
        "feature_dtype": str(PEN_DTYPE),
        "mask_dtype": str(MASK_DTYPE),
        "window_local_pair_rebase": True,
        "binary_near_boundary_allowed_in_model_x": False,
        "source_media_fallback_allowed": False,
    }
    if features != expected_features:
        raise ValueError("pen cache feature contract drift")
    output = _object(payload["output"], "output")
    if set(output) != {
        "primary_cache_root_relative_path",
        "repeat_cache_root_relative_path",
        "repeat_gate_relative_path",
    }:
        raise ValueError("pen cache output contract keys drift")
    for value in output.values():
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("pen cache output path is unsafe")


def _validate_hash_spec(value: object, name: str) -> None:
    spec = _object(value, name)
    if set(spec) != {"path", "sha256"}:
        raise ValueError(f"{name} keys drift")
    sha = str(spec["sha256"])
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError(f"{name}.sha256 is invalid")


def _load_array(
    path: Path,
    dtype: np.dtype[Any],
    rows: np.ndarray | None,
    maximum: int,
) -> np.ndarray:
    mapping = np.load(path, mmap_mode="r")
    try:
        if rows is None:
            return np.asarray(mapping, dtype=dtype).copy()
        indices = np.asarray(rows, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("pen cache row indices must be one-dimensional")
        if len(indices) and (indices.min() < 0 or indices.max() >= maximum):
            raise IndexError("pen cache row indices are out of bounds")
        return np.asarray(mapping[indices], dtype=dtype).copy()
    finally:
        _close_memmap(mapping)


def _strict_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(f"invalid boolean values={unknown}")
    return normalized.map(mapping).astype(bool)


def _ordered_sha256(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.fillna("").astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_bound_file(path: Path, expected_sha: str, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")
    observed = file_sha256(path)
    if observed != expected_sha:
        raise ValueError(
            f"{name} hash mismatch expected={expected_sha} observed={observed}"
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
    return _object(payload, str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        handle.write("\n")


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


def _close_memmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


__all__ = [
    "CACHE_FILES",
    "LegacyL6PenContextCache",
    "LegacyL6PenContextCacheConfig",
    "PEN_DIM",
    "PEN_FEATURE_NAMES",
    "PEN_STATIC_FEATURE_COUNT",
    "audit_pen_context_cache",
    "audit_pen_context_cache_repeat",
    "build_pen_context_cache",
    "load_pen_context_cache",
    "load_pen_context_cache_config",
    "materialize_pen_context_cache",
    "preflight_pen_context_cache",
]
