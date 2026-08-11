"""Hash-bound, inner-only RGB bindings for the PRE-S1 calibration.

The scientific binding is deliberately separate from a machine-specific cache
path realization.  It contains only the T6 FOLD_3 train/validation windows
selected by the frozen calibration authority, their six ordered context IDs,
and the cache rows that serve those contexts.  This keeps outer rows out of
the executor's metadata loader while preserving a raw-byte hash for the
immutable packed RGB tensor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_context_index import (
    IMAGE_CONTEXT_SEQUENCE_DELIMITER,
)
from pig_behavior.classification_v2.training.cvat_media_resolution import (
    CvatMediaResolutionError,
    attach_registered_cvat_media_paths,
)

SCIENTIFIC_RGB_BINDING_SCHEMA = "classification_v2.pre_s1_calibration_rgb_binding.v1"
DATA_BINDINGS_SCHEMA = "classification_v2.pre_s1_calibration_data_bindings.v2"
EXECUTION_PATH_REALIZATION_SCHEMA = (
    "classification_v2.pre_s1_calibration_execution_path_realization.v1"
)
SOURCE_INTEGRITY_SCHEMA = (
    "classification_v2.pre_s1_calibration_rgb_source_integrity.v1"
)
INNER_ROLES = frozenset({"train", "validation"})
SEQUENCE_LENGTH = 6

WINDOW_SOURCE_COLUMNS = [
    "window_id",
    "source_type",
    "object_track_key",
    "window_length_frames",
    "window_start_frame",
    "window_end_frame",
    "selected_frame_indices",
    "view_type",
    "window_valid_for_main_train",
    "lineage_scope",
    "human_review_complete",
    "dataset_id",
    "video_key",
    "pig_id",
    "track_id",
    "expected_frame_indices",
    "scene_frame_uid_sequence",
    "frame_uid_sequence",
    "image_context_id_sequence",
    "observed_image_context_rows",
    "loadable_image_context_rows",
    "missing_image_context_slots",
    "window_image_context_complete",
]
FRAME_SOURCE_COLUMNS = [
    "identifier_schema_version",
    "scene_frame_uid",
    "frame_uid",
    "source_type",
    "dataset_id",
    "video_key",
    "source_video_key",
    "source_video_path",
    "object_track_key",
    "pig_id",
    "track_id",
    "frame_index",
    "temporal_unit_key",
    "image_width",
    "image_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_valid",
    "lineage_scope",
    "human_review_complete",
    "image_context_id",
    "image_context_source",
    "resolved_media_path",
    "resolved_media_exists",
    "bbox_context_valid",
    "full_frame_context_available",
    "partner_context_available",
    "image_context_loadable",
    "image_context_error",
]
PACKED_INDEX_COLUMNS = [
    "image_context_id",
    "packed_row",
    "lineage_scope",
    "human_review_complete",
]


class RgbBindingError(ValueError):
    """Raised when an RGB scientific or execution binding is unsafe."""


@dataclass(frozen=True, slots=True)
class ResolvedRgbBinding:
    """Subset paths and hashes safe for a single inner-only loader."""

    frame_context_path: Path
    window_context_path: Path
    packed_index_path: Path
    packed_cache_path: Path
    hashes: Mapping[str, str]
    coverage: Mapping[str, int]
    audit: Mapping[str, Any]


def materialize_inner_rgb_binding(
    *,
    output_dir: Path,
    rgb_source_root: Path,
    requested_roles: pd.DataFrame,
    authority_sha256: str,
    provenance_hashes: Mapping[str, str],
    expected_train_windows: int,
    expected_validation_windows: int,
    input_parity_evidence: Mapping[str, object] | None = None,
    source_integrity_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create a new immutable binding without creating or rewriting media."""

    roles = _prepare_requested_roles(
        requested_roles,
        expected_train_windows=expected_train_windows,
        expected_validation_windows=expected_validation_windows,
    )
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RgbBindingError(f"binding output already exists={output_dir}")

    source_paths = _source_paths(Path(rgb_source_root).resolve())
    for label, path in source_paths.items():
        if not path.is_file():
            raise RgbBindingError(f"RGB source artifact missing={label}:{path}")

    source_hashes, source_identities = _resolve_source_integrity(
        source_paths,
        source_integrity_evidence,
    )
    _validate_existing_parity(source_hashes, input_parity_evidence)

    source_windows = _select_rows_by_key(
        source_paths["window_context"],
        key="window_id",
        requested=set(roles["window_id"]),
        columns=WINDOW_SOURCE_COLUMNS,
    )
    windows = roles.merge(
        source_windows,
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    if windows["source_type"].isna().any():
        missing = int(windows["source_type"].isna().sum())
        raise RgbBindingError(f"authorized inner window is absent from RGB context={missing}")
    windows["temporal_view"] = "T6"
    windows = windows.sort_values("window_id", kind="stable").reset_index(drop=True)

    context_ids = _context_ids_from_windows(windows)
    source_frames = _select_rows_by_key(
        source_paths["frame_context"],
        key="image_context_id",
        requested=context_ids,
        columns=FRAME_SOURCE_COLUMNS,
    )
    frames = _sanitize_frame_paths(source_frames)
    frames = frames.sort_values("image_context_id", kind="stable").reset_index(drop=True)

    packed_index = _select_rows_by_key(
        source_paths["packed_index"],
        key="image_context_id",
        requested=context_ids,
        columns=PACKED_INDEX_COLUMNS,
    )
    packed_index = packed_index.sort_values("image_context_id", kind="stable").reset_index(
        drop=True
    )

    audit = audit_inner_rgb_binding(
        windows=windows,
        frames=frames,
        packed_index=packed_index,
        requested_roles=roles,
    )
    _raise_if_invalid_audit(audit)

    packed_tensor = np.load(source_paths["packed_cache"], mmap_mode="r")
    try:
        packed_shape = list(packed_tensor.shape)
        packed_dtype = str(packed_tensor.dtype)
    finally:
        del packed_tensor
    if packed_shape[1:] != [64, 64, 3] or packed_dtype != "uint8":
        raise RgbBindingError(
            "packed RGB tensor contract drifted="
            f"shape={packed_shape},dtype={packed_dtype}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_paths = {
        "window_context": output_dir / "inner_window_context.csv",
        "frame_context": output_dir / "inner_frame_context.csv",
        "packed_index": output_dir / "inner_packed_image_cache_index.csv",
    }
    _write_csv_atomic(artifact_paths["window_context"], windows)
    _write_csv_atomic(artifact_paths["frame_context"], frames)
    _write_csv_atomic(artifact_paths["packed_index"], packed_index)

    artifacts = {
        key: {
            "relative_path": path.name,
            "sha256": _sha256_file(path),
            "rows": int(len(frame)),
        }
        for key, path, frame in (
            ("window_context", artifact_paths["window_context"], windows),
            ("frame_context", artifact_paths["frame_context"], frames),
            ("packed_index", artifact_paths["packed_index"], packed_index),
        )
    }
    scientific = {
        "schema_version": SCIENTIFIC_RGB_BINDING_SCHEMA,
        "calibration": {
            "authority_sha256": str(authority_sha256),
            "fold": "FOLD_3",
            "temporal_view": "T6",
            "roles": ["train", "validation"],
        },
        "provenance_hashes": _normalized_hash_mapping(provenance_hashes),
        "artifacts": artifacts,
        "source_media": {
            "logical_root": "reviewed_rgb_v1",
            "packed_cache_logical_path": (
                "actor_rgb_64_full/packed_rgb_64_letterbox.npy"
            ),
            "packed_cache_sha256": source_hashes["source_packed_cache_sha256"],
            "packed_cache_shape": packed_shape,
            "packed_cache_dtype": packed_dtype,
            **source_hashes,
        },
        "coverage": audit["coverage"],
    }
    scientific_path = output_dir / "scientific_rgb_binding.json"
    _write_json_atomic(scientific_path, scientific)
    scientific_sha256 = _sha256_file(scientific_path)
    data_bindings_path = write_execution_path_realization(
        output_dir=output_dir,
        scientific_binding_path=scientific_path,
        packed_cache_path=source_paths["packed_cache"],
        verified_packed_cache_sha256=source_hashes["source_packed_cache_sha256"],
        packed_cache_identity=source_identities["packed_cache"],
    )
    return {
        "scientific_binding_path": str(scientific_path),
        "scientific_binding_sha256": scientific_sha256,
        "data_bindings_path": str(data_bindings_path),
        "data_bindings_sha256": _sha256_file(data_bindings_path),
        "coverage": audit["coverage"],
        "source_hashes": source_hashes,
    }


def build_rgb_source_integrity_evidence(
    *,
    rgb_source_root: Path,
    output_path: Path,
    input_parity_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Hash immutable RGB source artifacts once and save reusable evidence."""

    source_paths = _source_paths(Path(rgb_source_root).resolve())
    for label, path in source_paths.items():
        if not path.is_file():
            raise RgbBindingError(f"RGB source artifact missing={label}:{path}")
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise RgbBindingError(f"source integrity evidence already exists={output_path}")
    source_hashes, source_identities = _hash_source_artifacts(source_paths)
    _validate_existing_parity(source_hashes, input_parity_evidence)
    payload = {
        "schema_version": SOURCE_INTEGRITY_SCHEMA,
        "logical_root": "reviewed_rgb_v1",
        "source_hashes": source_hashes,
        "source_file_identity": source_identities,
    }
    _write_json_atomic(output_path, payload)
    return {
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "source_hashes": source_hashes,
    }


def write_execution_path_realization(
    *,
    output_dir: Path,
    scientific_binding_path: Path,
    packed_cache_path: Path | str,
    filename: str = "pre_s1_calibration_data_bindings.json",
    verified_packed_cache_sha256: str | None = None,
    packed_cache_identity: Mapping[str, int] | None = None,
) -> Path:
    """Write a machine-specific cache location without changing science."""

    output_dir = Path(output_dir).resolve()
    scientific_binding_path = Path(scientific_binding_path).resolve()
    if not scientific_binding_path.is_relative_to(output_dir):
        raise RgbBindingError("scientific binding must be colocated with its realization")
    path = output_dir / filename
    if path.exists():
        raise RgbBindingError(f"execution path realization already exists={path}")
    realization: dict[str, object] = {
        "schema_version": EXECUTION_PATH_REALIZATION_SCHEMA,
        "packed_cache_path": str(packed_cache_path),
    }
    if verified_packed_cache_sha256 is not None:
        identity = (
            dict(packed_cache_identity)
            if packed_cache_identity is not None
            else _file_identity(Path(packed_cache_path))
        )
        if len(verified_packed_cache_sha256) != 64:
            raise RgbBindingError("attested packed RGB cache hash is invalid")
        if set(identity) != {"size_bytes", "mtime_ns"}:
            raise RgbBindingError("attested packed RGB cache identity is invalid")
        realization["packed_cache_identity_attestation"] = {
            "sha256": verified_packed_cache_sha256,
            **identity,
        }
    payload = {
        "schema_version": DATA_BINDINGS_SCHEMA,
        "scientific_binding": {
            "relative_path": scientific_binding_path.name,
            "sha256": _sha256_file(scientific_binding_path),
        },
        "execution_path_realization": realization,
    }
    _write_json_atomic(path, payload)
    return path


def resolve_execution_rgb_binding(
    *,
    data_bindings_path: Path,
    requested_roles: pd.DataFrame,
    authority_sha256: str,
    provenance_hashes: Mapping[str, str],
) -> ResolvedRgbBinding:
    """Validate and resolve a v2 binding before any RGB tensor is opened."""

    data_bindings_path = Path(data_bindings_path).resolve()
    payload = _read_json(data_bindings_path)
    if set(payload) != {
        "schema_version",
        "scientific_binding",
        "execution_path_realization",
    }:
        raise RgbBindingError("unexpected PRE-S1 execution binding fields")
    if payload.get("schema_version") != DATA_BINDINGS_SCHEMA:
        raise RgbBindingError("unsupported PRE-S1 RGB data-bindings schema")
    science_ref = payload.get("scientific_binding")
    if not isinstance(science_ref, dict) or set(science_ref) != {
        "relative_path",
        "sha256",
    }:
        raise RgbBindingError("scientific RGB binding reference is invalid")
    scientific_path = _safe_relative_path(
        data_bindings_path.parent,
        str(science_ref["relative_path"]),
    )
    if _sha256_file(scientific_path) != str(science_ref["sha256"]):
        raise RgbBindingError("scientific RGB binding hash mismatch")
    scientific = _read_json(scientific_path)
    _validate_scientific_binding(
        scientific,
        authority_sha256=authority_sha256,
        provenance_hashes=provenance_hashes,
    )

    realization = payload.get("execution_path_realization")
    valid_realization_fields = {
        "schema_version",
        "packed_cache_path",
        "packed_cache_identity_attestation",
    }
    if (
        not isinstance(realization, dict)
        or not {"schema_version", "packed_cache_path"}.issubset(realization)
        or not set(realization).issubset(valid_realization_fields)
    ):
        raise RgbBindingError("execution RGB path realization is invalid")
    if realization.get("schema_version") != EXECUTION_PATH_REALIZATION_SCHEMA:
        raise RgbBindingError("unsupported execution RGB path realization schema")
    packed_cache_path = Path(str(realization["packed_cache_path"])).resolve()
    source_media = scientific["source_media"]
    expected_cache_sha256 = str(source_media["packed_cache_sha256"])
    attestation = realization.get("packed_cache_identity_attestation")
    if attestation is None:
        observed_cache_sha256 = _sha256_file(packed_cache_path)
    else:
        if not isinstance(attestation, dict) or set(attestation) != {
            "sha256",
            "size_bytes",
            "mtime_ns",
        }:
            raise RgbBindingError("packed RGB cache identity attestation is invalid")
        if str(attestation["sha256"]) != expected_cache_sha256:
            raise RgbBindingError("packed RGB cache identity attestation hash drifted")
        expected_identity = {
            "size_bytes": int(attestation["size_bytes"]),
            "mtime_ns": int(attestation["mtime_ns"]),
        }
        observed_cache_sha256 = (
            expected_cache_sha256
            if _file_identity(packed_cache_path) == expected_identity
            else _sha256_file(packed_cache_path)
        )
    if observed_cache_sha256 != expected_cache_sha256:
        raise RgbBindingError("packed RGB cache hash mismatch")

    artifacts = scientific["artifacts"]
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for key in ("window_context", "frame_context", "packed_index"):
        entry = artifacts[key]
        if not isinstance(entry, dict) or set(entry) != {
            "relative_path",
            "sha256",
            "rows",
        }:
            raise RgbBindingError(f"scientific RGB artifact descriptor invalid={key}")
        path = _safe_relative_path(scientific_path.parent, str(entry["relative_path"]))
        observed = _sha256_file(path)
        if observed != str(entry["sha256"]):
            raise RgbBindingError(f"scientific RGB artifact hash mismatch={key}")
        artifact_paths[key] = path
        artifact_hashes[f"rgb_bound_{key}"] = observed

    windows = pd.read_csv(artifact_paths["window_context"], low_memory=False)
    frames = pd.read_csv(artifact_paths["frame_context"], low_memory=False)
    packed_index = pd.read_csv(artifact_paths["packed_index"], low_memory=False)
    for key, frame in (
        ("window_context", windows),
        ("frame_context", frames),
        ("packed_index", packed_index),
    ):
        if int(artifacts[key]["rows"]) != len(frame):
            raise RgbBindingError(f"scientific RGB artifact row count mismatch={key}")

    roles = _prepare_requested_roles(requested_roles)
    audit = audit_inner_rgb_binding(
        windows=windows,
        frames=frames,
        packed_index=packed_index,
        requested_roles=roles,
    )
    _raise_if_invalid_audit(audit)
    coverage = audit["coverage"]
    hashes = {
        "rgb_scientific_binding": _sha256_file(scientific_path),
        "rgb_packed_cache": expected_cache_sha256,
        **artifact_hashes,
    }
    return ResolvedRgbBinding(
        frame_context_path=artifact_paths["frame_context"],
        window_context_path=artifact_paths["window_context"],
        packed_index_path=artifact_paths["packed_index"],
        packed_cache_path=packed_cache_path,
        hashes=hashes,
        coverage={key: int(value) for key, value in coverage.items()},
        audit=audit,
    )


def audit_inner_rgb_binding(
    *,
    windows: pd.DataFrame,
    frames: pd.DataFrame,
    packed_index: pd.DataFrame,
    requested_roles: pd.DataFrame,
) -> dict[str, object]:
    """Fail closed on missing, reordered, cross-video, or outer bindings."""

    roles = _prepare_requested_roles(requested_roles)
    _require_columns(
        windows,
        {
            "window_id",
            "calibration_role",
            "temporal_view",
            "source_type",
            "video_key",
            "object_track_key",
            "pig_id",
            "track_id",
            "selected_frame_indices",
            "expected_frame_indices",
            "image_context_id_sequence",
            "window_image_context_complete",
        },
        "inner RGB windows",
    )
    _require_columns(
        frames,
        {
            "image_context_id",
            "source_type",
            "video_key",
            "object_track_key",
            "pig_id",
            "track_id",
            "frame_index",
            "resolved_media_path",
            "resolved_media_exists",
            "image_context_loadable",
        },
        "inner RGB frames",
    )
    _require_columns(packed_index, {"image_context_id", "packed_row"}, "inner RGB index")

    requested_by_window = dict(
        zip(
            roles["window_id"].astype(str),
            roles["calibration_role"].astype(str),
            strict=True,
        )
    )
    window_ids = windows["window_id"].astype(str)
    duplicate_windows = int(window_ids.duplicated().sum())
    actual_ids = set(window_ids)
    requested_ids = set(requested_by_window)
    missing_windows = len(requested_ids.difference(actual_ids))
    unexpected_windows = len(actual_ids.difference(requested_ids))
    role_violations = int(
        sum(
            requested_by_window.get(str(row.window_id)) != str(row.calibration_role)
            or str(row.calibration_role) not in INNER_ROLES
            for row in windows.itertuples(index=False)
        )
    )
    role_violations += int(
        sum(role not in INNER_ROLES for role in requested_by_window.values())
    )
    temporal_violations = int(
        windows["temporal_view"].astype(str).ne("T6").sum()
    )
    complete_violations = int(
        (~_strict_bool(windows["window_image_context_complete"])).sum()
    )

    frame_ids = frames["image_context_id"].astype(str)
    duplicate_context_ids = int(frame_ids.duplicated().sum())
    index_ids = packed_index["image_context_id"].astype(str)
    duplicate_index_ids = int(index_ids.duplicated().sum())
    sequence_rows = _audit_sequence_rows(windows)
    bad_sequence_windows = set(
        sequence_rows.loc[
            ~sequence_rows["sequence_shape_valid"]
            | ~sequence_rows["context_ids_present"],
            "window_id",
        ]
    )
    sequence_order_windows = set(
        sequence_rows.loc[
            ~sequence_rows["selected_matches_expected"],
            "window_id",
        ]
    )
    valid_sequence_rows = sequence_rows.loc[
        sequence_rows["sequence_shape_valid"]
        & sequence_rows["context_ids_present"]
    ].copy()
    expanded = valid_sequence_rows.explode(
        ["context_ids", "expected_frames"],
        ignore_index=True,
    ).rename(
        columns={
            "context_ids": "_context_id",
            "expected_frames": "_expected_frame",
        }
    )
    frame_lookup = _frame_lookup(frames, frame_ids)
    frame_id_set = set(frame_lookup["_context_id"])
    index_id_set = set(index_ids)
    missing_context_ids = set(expanded.loc[
        ~expanded["_context_id"].isin(frame_id_set),
        "_context_id",
    ])
    missing_index_ids = set(expanded.loc[
        ~expanded["_context_id"].isin(index_id_set),
        "_context_id",
    ])
    joined = expanded.merge(
        frame_lookup,
        on="_context_id",
        how="left",
        validate="many_to_one",
    )
    complete_rows = (
        joined["_frame_present"]
        .fillna(False)
        .astype(bool)
        .groupby(joined["_binding_row"], sort=False)
        .all()
    )
    joined = joined.loc[
        joined["_binding_row"].isin(complete_rows[complete_rows].index)
    ].copy()

    cross_video_windows: set[str] = set()
    actor_identity_windows: set[str] = set()
    media_reference_windows: set[str] = set()
    if not joined.empty:
        observed_frames = pd.to_numeric(joined["_frame_index"], errors="coerce")
        expected_frames = pd.to_numeric(joined["_expected_frame"], errors="coerce")
        frame_order_invalid = (
            observed_frames.isna()
            | expected_frames.isna()
            | observed_frames.mod(1).ne(0)
            | expected_frames.mod(1).ne(0)
            | observed_frames.ne(expected_frames)
        )
        sequence_order_windows.update(
            joined.loc[frame_order_invalid, "window_id"].astype(str)
        )
        cross_video_windows.update(
            _invalid_windows(
                joined,
                joined["_frame_video_key"].astype(str).ne(joined["video_key"].astype(str)),
            )
        )
        actor_identity_windows.update(
            _invalid_windows(
                joined,
                joined["_frame_source_type"].astype(str).ne(
                    joined["source_type"].astype(str)
                ),
            )
        )
        for field in ("object_track_key", "pig_id", "track_id"):
            expected = joined[field].map(_normalized_value)
            observed = joined[f"_frame_{field}"].map(_normalized_value)
            actor_identity_windows.update(
                _invalid_windows(joined, expected.ne("") & observed.ne(expected))
            )
        media_ok = (
            _strict_bool(joined["_frame_resolved_media_exists"])
            & _strict_bool(joined["_frame_image_context_loadable"])
            & joined["_frame_resolved_media_path"].astype(str).str.strip().ne("")
        )
        media_reference_windows.update(_invalid_windows(joined, ~media_ok))

    train_bound = int(windows["calibration_role"].astype(str).eq("train").sum())
    validation_bound = int(
        windows["calibration_role"].astype(str).eq("validation").sum()
    )
    coverage = {
        "train_windows_bound": train_bound,
        "validation_windows_bound": validation_bound,
        "missing_windows": int(missing_windows),
        "duplicate_windows": int(duplicate_windows),
        "bad_sequence_length": int(len(bad_sequence_windows)),
        "role_violations": int(role_violations),
        "cross_video_violations": int(len(cross_video_windows)),
        "unexpected_windows": int(unexpected_windows),
        "temporal_violations": int(temporal_violations),
        "incomplete_window_violations": int(complete_violations),
        "missing_context_ids": int(len(missing_context_ids)),
        "missing_packed_index_ids": int(len(missing_index_ids)),
        "duplicate_context_ids": int(duplicate_context_ids),
        "duplicate_packed_index_ids": int(duplicate_index_ids),
        "sequence_order_violations": int(len(sequence_order_windows)),
        "actor_identity_violations": int(len(actor_identity_windows)),
        "media_reference_violations": int(len(media_reference_windows)),
    }
    invalid_keys = (
        "missing_windows",
        "duplicate_windows",
        "bad_sequence_length",
        "role_violations",
        "cross_video_violations",
        "unexpected_windows",
        "temporal_violations",
        "incomplete_window_violations",
        "missing_context_ids",
        "missing_packed_index_ids",
        "duplicate_context_ids",
        "duplicate_packed_index_ids",
        "sequence_order_violations",
        "actor_identity_violations",
        "media_reference_violations",
    )
    errors = [key for key in invalid_keys if coverage[key] != 0]
    return {
        "schema_version": SCIENTIFIC_RGB_BINDING_SCHEMA,
        "coverage": coverage,
        "valid": not errors,
        "errors": errors,
    }


def _audit_sequence_rows(windows: pd.DataFrame) -> pd.DataFrame:
    """Parse the six-slot sequences once before bulk identity validation."""

    rows: list[dict[str, object]] = []
    for binding_row, row in enumerate(windows.itertuples(index=False)):
        context_ids = _split_sequence(row.image_context_id_sequence)
        expected_frames = _integer_sequence(row.expected_frame_indices)
        selected_frames = _integer_sequence(row.selected_frame_indices)
        rows.append(
            {
                "_binding_row": binding_row,
                "window_id": str(row.window_id),
                "video_key": row.video_key,
                "source_type": row.source_type,
                "object_track_key": row.object_track_key,
                "pig_id": row.pig_id,
                "track_id": row.track_id,
                "context_ids": context_ids,
                "expected_frames": expected_frames,
                "sequence_shape_valid": (
                    len(context_ids) == SEQUENCE_LENGTH
                    and len(expected_frames) == SEQUENCE_LENGTH
                ),
                "context_ids_present": not any(not value for value in context_ids),
                "selected_matches_expected": selected_frames == expected_frames,
            }
        )
    return pd.DataFrame(rows)


def _frame_lookup(frames: pd.DataFrame, frame_ids: pd.Series) -> pd.DataFrame:
    """Prepare a deterministic one-row lookup while duplicate IDs remain audited."""

    lookup = frames.loc[
        :,
        [
            "frame_index",
            "video_key",
            "source_type",
            "object_track_key",
            "pig_id",
            "track_id",
            "resolved_media_path",
            "resolved_media_exists",
            "image_context_loadable",
        ],
    ].copy()
    lookup.insert(0, "_context_id", frame_ids)
    lookup = lookup.drop_duplicates("_context_id", keep="first")
    return lookup.rename(
        columns={
            "frame_index": "_frame_index",
            "video_key": "_frame_video_key",
            "source_type": "_frame_source_type",
            "object_track_key": "_frame_object_track_key",
            "pig_id": "_frame_pig_id",
            "track_id": "_frame_track_id",
            "resolved_media_path": "_frame_resolved_media_path",
            "resolved_media_exists": "_frame_resolved_media_exists",
            "image_context_loadable": "_frame_image_context_loadable",
        }
    ).assign(_frame_present=True)


def _invalid_windows(joined: pd.DataFrame, invalid: pd.Series) -> set[str]:
    """Return window identities with one or more failed bound observations."""

    return set(joined.loc[invalid, "window_id"].astype(str))


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "frame_context": root / "image_context_v2" / "image_frame_context_manifest.csv",
        "window_context": root / "image_context_v2" / "image_window_context_manifest.csv",
        "packed_cache": root / "actor_rgb_64_full" / "packed_rgb_64_letterbox.npy",
        "packed_index": root / "actor_rgb_64_full" / "packed_image_cache_index.csv",
        "cache_manifest": root / "actor_rgb_64_full" / "manifest.csv",
        "cache_audit": root / "actor_rgb_64_full" / "cache_audit.json",
        "packed_cache_audit": root / "actor_rgb_64_full" / "packed_image_cache_audit.json",
    }


def _resolve_source_integrity(
    source_paths: Mapping[str, Path],
    evidence: Mapping[str, object] | None,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    if evidence is None:
        return _hash_source_artifacts(source_paths)
    required = {
        "schema_version",
        "logical_root",
        "source_hashes",
        "source_file_identity",
    }
    if set(evidence) != required:
        raise RgbBindingError("RGB source integrity evidence fields are invalid")
    if evidence.get("schema_version") != SOURCE_INTEGRITY_SCHEMA:
        raise RgbBindingError("RGB source integrity evidence schema is unsupported")
    if evidence.get("logical_root") != "reviewed_rgb_v1":
        raise RgbBindingError("RGB source integrity evidence logical root drifted")
    source_hashes = evidence.get("source_hashes")
    source_identities = evidence.get("source_file_identity")
    if not isinstance(source_hashes, Mapping) or not isinstance(
        source_identities,
        Mapping,
    ):
        raise RgbBindingError("RGB source integrity evidence is malformed")
    expected_hashes = set(_source_hash_keys().values())
    if set(source_hashes) != expected_hashes or set(source_identities) != set(
        source_paths
    ):
        raise RgbBindingError("RGB source integrity evidence is incomplete")
    normalized_hashes = _normalized_hash_mapping(source_hashes)
    normalized_identities: dict[str, dict[str, int]] = {}
    for label, path in source_paths.items():
        entry = source_identities[label]
        if not isinstance(entry, Mapping) or set(entry) != {"size_bytes", "mtime_ns"}:
            raise RgbBindingError(f"RGB source identity is invalid={label}")
        expected_identity = {
            "size_bytes": int(entry["size_bytes"]),
            "mtime_ns": int(entry["mtime_ns"]),
        }
        if _file_identity(path) != expected_identity:
            raise RgbBindingError(f"RGB source integrity evidence is stale={label}")
        normalized_identities[label] = expected_identity
    return normalized_hashes, normalized_identities


def _hash_source_artifacts(
    source_paths: Mapping[str, Path],
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    hashes: dict[str, str] = {}
    identities: dict[str, dict[str, int]] = {}
    hash_keys = _source_hash_keys()
    for label, path in source_paths.items():
        before = _file_identity(path)
        observed = _sha256_file(path)
        after = _file_identity(path)
        if before != after:
            raise RgbBindingError(f"RGB source changed while its hash was computed={label}")
        hashes[hash_keys[label]] = observed
        identities[label] = after
    return hashes, identities


def _source_hash_keys() -> dict[str, str]:
    return {
        "frame_context": "source_frame_context_sha256",
        "window_context": "source_window_context_sha256",
        "packed_index": "source_packed_index_sha256",
        "packed_cache": "source_packed_cache_sha256",
        "cache_manifest": "source_cache_manifest_sha256",
        "cache_audit": "source_cache_audit_sha256",
        "packed_cache_audit": "source_packed_cache_audit_sha256",
    }


def _prepare_requested_roles(
    requested_roles: pd.DataFrame,
    *,
    expected_train_windows: int | None = None,
    expected_validation_windows: int | None = None,
) -> pd.DataFrame:
    role_column = (
        "primary_s1_role"
        if "primary_s1_role" in requested_roles.columns
        else "calibration_role"
    )
    _require_columns(
        requested_roles,
        {"window_id", role_column},
        "authorized calibration population",
    )
    roles = requested_roles.loc[:, ["window_id", role_column]].copy()
    roles.columns = ["window_id", "calibration_role"]
    roles["window_id"] = roles["window_id"].astype(str)
    roles["calibration_role"] = roles["calibration_role"].astype(str)
    if roles["window_id"].duplicated().any():
        raise RgbBindingError("authorized calibration population has duplicate window_id")
    invalid_roles = sorted(set(roles["calibration_role"]).difference(INNER_ROLES))
    if invalid_roles:
        raise RgbBindingError(f"outer/test role in requested RGB binding={invalid_roles}")
    actual_train = int(roles["calibration_role"].eq("train").sum())
    actual_validation = int(roles["calibration_role"].eq("validation").sum())
    if expected_train_windows is not None and actual_train != expected_train_windows:
        raise RgbBindingError(
            "authorized train count mismatch for RGB binding="
            f"{actual_train}!={expected_train_windows}"
        )
    if (
        expected_validation_windows is not None
        and actual_validation != expected_validation_windows
    ):
        raise RgbBindingError(
            "authorized validation count mismatch for RGB binding="
            f"{actual_validation}!={expected_validation_windows}"
        )
    return roles.sort_values("window_id", kind="stable").reset_index(drop=True)


def _select_rows_by_key(
    path: Path,
    *,
    key: str,
    requested: set[str],
    columns: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=50_000, low_memory=False):
        _require_columns(chunk, set(columns), f"RGB source={path.name}")
        selected = chunk.loc[chunk[key].astype(str).isin(requested)].copy()
        if len(selected):
            parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=columns)
    result = pd.concat(parts, ignore_index=True)
    if result[key].astype(str).duplicated().any():
        raise RgbBindingError(f"RGB source has duplicate selected {key}")
    return result


def _context_ids_from_windows(windows: pd.DataFrame) -> set[str]:
    context_ids: set[str] = set()
    for row in windows.itertuples(index=False):
        values = _split_sequence(row.image_context_id_sequence)
        if len(values) != SEQUENCE_LENGTH or any(not value for value in values):
            raise RgbBindingError(f"invalid T6 image context sequence={row.window_id}")
        context_ids.update(values)
    return context_ids


def _sanitize_frame_paths(frames: pd.DataFrame) -> pd.DataFrame:
    sanitized = frames.copy()
    identity = sanitized["image_context_id"].astype(str).map(
        lambda value: f"reviewed_rgb_v1/{value}"
    )
    sanitized["media_logical_identity"] = identity
    sanitized["resolved_media_path"] = identity
    try:
        sanitized = attach_registered_cvat_media_paths(sanitized)
    except CvatMediaResolutionError as error:
        raise RgbBindingError(str(error)) from error
    cvat = sanitized["source_type"].astype(str).eq("cvat_tracking_xml")
    sanitized.loc[cvat, "resolved_media_path"] = sanitized.loc[
        cvat, "registered_relative_media_path"
    ]
    return sanitized


def _validate_existing_parity(
    source_hashes: Mapping[str, str],
    evidence: Mapping[str, object] | None,
) -> None:
    if evidence is None:
        return
    source = evidence.get("source_authorities", evidence)
    if not isinstance(source, Mapping):
        raise RgbBindingError("existing input parity evidence has no source authorities")
    checks = {
        "rgb_cache_index_sha256": "source_packed_index_sha256",
        "rgb_cache_manifest_sha256": "source_cache_manifest_sha256",
    }
    for evidence_key, source_key in checks.items():
        expected = source.get(evidence_key)
        if expected is not None and str(expected) != source_hashes[source_key]:
            raise RgbBindingError(f"existing RGB parity mismatch={evidence_key}")


def _validate_scientific_binding(
    scientific: Mapping[str, object],
    *,
    authority_sha256: str,
    provenance_hashes: Mapping[str, str],
) -> None:
    required = {
        "schema_version",
        "calibration",
        "provenance_hashes",
        "artifacts",
        "source_media",
        "coverage",
    }
    if set(scientific) != required:
        raise RgbBindingError("unexpected scientific RGB binding fields")
    if scientific.get("schema_version") != SCIENTIFIC_RGB_BINDING_SCHEMA:
        raise RgbBindingError("unsupported scientific RGB binding schema")
    calibration = scientific.get("calibration")
    if not isinstance(calibration, Mapping) or calibration != {
        "authority_sha256": str(authority_sha256),
        "fold": "FOLD_3",
        "temporal_view": "T6",
        "roles": ["train", "validation"],
    }:
        raise RgbBindingError("scientific RGB calibration identity drifted")
    if scientific.get("provenance_hashes") != _normalized_hash_mapping(provenance_hashes):
        raise RgbBindingError("scientific RGB provenance hash drifted")
    artifacts = scientific.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "window_context",
        "frame_context",
        "packed_index",
    }:
        raise RgbBindingError("scientific RGB artifacts are incomplete")
    source_media = scientific.get("source_media")
    if not isinstance(source_media, Mapping):
        raise RgbBindingError("scientific RGB source media is invalid")
    required_media = {
        "logical_root",
        "packed_cache_logical_path",
        "packed_cache_sha256",
        "packed_cache_shape",
        "packed_cache_dtype",
        "source_frame_context_sha256",
        "source_window_context_sha256",
        "source_packed_index_sha256",
        "source_packed_cache_sha256",
        "source_cache_manifest_sha256",
        "source_cache_audit_sha256",
        "source_packed_cache_audit_sha256",
    }
    if set(source_media) != required_media:
        raise RgbBindingError("scientific RGB source media fields are invalid")
    if source_media.get("packed_cache_sha256") != source_media.get(
        "source_packed_cache_sha256"
    ):
        raise RgbBindingError("scientific RGB packed cache provenance drifted")


def _raise_if_invalid_audit(audit: Mapping[str, object]) -> None:
    if not bool(audit.get("valid")):
        raise RgbBindingError(f"inner RGB binding failures={audit.get('errors')}")


def _safe_relative_path(base: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise RgbBindingError("scientific RGB artifact path must be relative")
    resolved_base = base.resolve()
    path = (resolved_base / relative).resolve()
    if not path.is_relative_to(resolved_base):
        raise RgbBindingError("scientific RGB artifact path escapes binding directory")
    return path


def _split_sequence(value: object) -> list[str]:
    return [
        item.strip()
        for item in str(value).split(IMAGE_CONTEXT_SEQUENCE_DELIMITER)
    ]


def _integer_sequence(value: object) -> list[int]:
    serialized = str(value).strip()
    if serialized.startswith("[") and serialized.endswith("]"):
        try:
            parsed = json.loads(serialized)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return _integer_values(parsed)
    return _integer_values(serialized.split("|"))


def _integer_values(values: object) -> list[int]:
    converted = pd.to_numeric(pd.Series(values), errors="coerce")
    if converted.isna().any() or ((converted % 1) != 0).any():
        return []
    return converted.astype(int).tolist()


def _normalized_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _strict_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise RgbBindingError(f"{label} missing columns={missing}")


def _normalized_hash_mapping(values: Mapping[str, str]) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in values.items()}
    invalid = [key for key, value in normalized.items() if len(value) != 64]
    if invalid:
        raise RgbBindingError(f"invalid provenance SHA256 fields={sorted(invalid)}")
    return dict(sorted(normalized.items()))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RgbBindingError(f"JSON object required={path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


__all__ = [
    "DATA_BINDINGS_SCHEMA",
    "EXECUTION_PATH_REALIZATION_SCHEMA",
    "RgbBindingError",
    "ResolvedRgbBinding",
    "SCIENTIFIC_RGB_BINDING_SCHEMA",
    "SOURCE_INTEGRITY_SCHEMA",
    "audit_inner_rgb_binding",
    "build_rgb_source_integrity_evidence",
    "materialize_inner_rgb_binding",
    "resolve_execution_rgb_binding",
    "write_execution_path_realization",
]
