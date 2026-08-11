"""Hash-bound, inner-only RGB bindings for Stage-1 temporal screening.

This module intentionally leaves the historical PRE-S1 T6 binding contract
unchanged.  It reuses its proven source-integrity mechanics while declaring a
separate Stage-1 scientific identity for each authorized target-duration view.
Scientific bindings contain no machine-specific media paths; those are kept in
the colocated execution-path realization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.training import pre_s1_rgb_binding as _legacy
from pig_behavior.classification_v2.training.cvat_source_registration import (
    CvatSourceRegistrationError,
    audit_cvat_source_path_enrichment,
    enrich_cvat_source_video_paths,
)

SCIENTIFIC_RGB_BINDING_SCHEMA = "classification_v2.s1_stage1_temporal_rgb_binding.v1"
DATA_BINDINGS_SCHEMA = "classification_v2.s1_stage1_temporal_data_bindings.v1"
EXECUTION_PATH_REALIZATION_SCHEMA = (
    "classification_v2.s1_stage1_temporal_execution_path_realization.v1"
)
INNER_ROLES = frozenset({"train", "validation"})


class Stage1RgbBindingError(ValueError):
    """Raised when a Stage-1 RGB scientific or execution binding is unsafe."""


@dataclass(frozen=True, slots=True)
class ResolvedStage1RgbBinding:
    """Subset paths and hashes safe for one inner-only Stage-1 loader."""

    frame_context_path: Path
    window_context_path: Path
    packed_index_path: Path
    packed_cache_path: Path
    hashes: Mapping[str, str]
    coverage: Mapping[str, int]
    audit: Mapping[str, Any]


def materialize_stage1_rgb_binding(
    *,
    output_dir: Path,
    rgb_source_root: Path,
    requested_roles: pd.DataFrame,
    authority_sha256: str,
    provenance_hashes: Mapping[str, str],
    view: str,
    sequence_length: int,
    expected_train_windows: int,
    expected_validation_windows: int,
    input_parity_evidence: Mapping[str, object] | None = None,
    source_integrity_evidence: Mapping[str, object] | None = None,
    cvat_source_registration_path: Path | None = None,
) -> dict[str, object]:
    """Create one immutable Stage-1 binding without rewriting RGB media."""

    _validate_view(view, sequence_length)
    roles = _prepare_requested_roles(
        requested_roles,
        expected_train_windows=expected_train_windows,
        expected_validation_windows=expected_validation_windows,
    )
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise Stage1RgbBindingError(f"binding output already exists={output_dir}")

    source_paths = _legacy._source_paths(Path(rgb_source_root).resolve())
    for label, path in source_paths.items():
        if not path.is_file():
            raise Stage1RgbBindingError(f"RGB source artifact missing={label}:{path}")
    try:
        source_hashes, source_identities = _legacy._resolve_source_integrity(
            source_paths,
            source_integrity_evidence,
        )
        _legacy._validate_existing_parity(source_hashes, input_parity_evidence)
        source_windows = _legacy._select_rows_by_key(
            source_paths["window_context"],
            key="window_id",
            requested=set(roles["window_id"]),
            columns=_legacy.WINDOW_SOURCE_COLUMNS,
        )
    except _legacy.RgbBindingError as exc:
        raise Stage1RgbBindingError(str(exc)) from exc

    windows = roles.merge(
        source_windows,
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    if windows["source_type"].isna().any():
        missing = int(windows["source_type"].isna().sum())
        raise Stage1RgbBindingError(
            f"authorized Stage-1 window is absent from RGB context={missing}"
        )
    windows["temporal_view"] = view
    windows = windows.sort_values("window_id", kind="stable").reset_index(drop=True)

    context_ids = _context_ids_from_windows(windows, sequence_length=sequence_length)
    try:
        source_frames = _legacy._select_rows_by_key(
            source_paths["frame_context"],
            key="image_context_id",
            requested=context_ids,
            columns=_legacy.FRAME_SOURCE_COLUMNS,
        )
        registration_sha256 = None
        registration_audit = None
        if cvat_source_registration_path is not None:
            before_registration = source_frames.copy(deep=True)
            source_frames, registration_sha256 = enrich_cvat_source_video_paths(
                source_frames,
                registration_path=cvat_source_registration_path,
            )
            registration_audit = audit_cvat_source_path_enrichment(
                before_registration,
                source_frames,
            )
        frames = _legacy._sanitize_frame_paths(source_frames)
        frames = frames.sort_values("image_context_id", kind="stable").reset_index(
            drop=True
        )
        packed_index = _legacy._select_rows_by_key(
            source_paths["packed_index"],
            key="image_context_id",
            requested=context_ids,
            columns=_legacy.PACKED_INDEX_COLUMNS,
        )
    except (_legacy.RgbBindingError, CvatSourceRegistrationError) as exc:
        raise Stage1RgbBindingError(str(exc)) from exc
    packed_index = packed_index.sort_values("image_context_id", kind="stable").reset_index(
        drop=True
    )

    audit = audit_stage1_rgb_binding(
        windows=windows,
        frames=frames,
        packed_index=packed_index,
        requested_roles=roles,
        view=view,
        sequence_length=sequence_length,
    )
    _raise_if_invalid(audit)

    packed_tensor = np.load(source_paths["packed_cache"], mmap_mode="r")
    try:
        packed_shape = list(packed_tensor.shape)
        packed_dtype = str(packed_tensor.dtype)
    finally:
        del packed_tensor
    if packed_shape[1:] != [64, 64, 3] or packed_dtype != "uint8":
        raise Stage1RgbBindingError(
            "packed RGB tensor contract drifted="
            f"shape={packed_shape},dtype={packed_dtype}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_paths = {
        "window_context": output_dir / "stage1_window_context.csv",
        "frame_context": output_dir / "stage1_frame_context.csv",
        "packed_index": output_dir / "stage1_packed_image_cache_index.csv",
    }
    for key, frame in (
        ("window_context", windows),
        ("frame_context", frames),
        ("packed_index", packed_index),
    ):
        _legacy._write_csv_atomic(artifact_paths[key], frame)

    artifacts = {
        key: {
            "relative_path": path.name,
            "sha256": _legacy._sha256_file(path),
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
        "stage1": {
            "authority_sha256": str(authority_sha256),
            "run_kind": "S1_STAGE1_TEMPORAL_SCREENING",
            "fold": "FOLD_3",
            "temporal_view": view,
            "sequence_length": sequence_length,
            "roles": ["train", "validation"],
        },
        "provenance_hashes": _normalized_hash_mapping(provenance_hashes),
        "artifacts": artifacts,
        "source_media": {
            "logical_root": "reviewed_rgb_v1",
            "packed_cache_logical_path": "actor_rgb_64_full/packed_rgb_64_letterbox.npy",
            "packed_cache_sha256": source_hashes["source_packed_cache_sha256"],
            "packed_cache_shape": packed_shape,
            "packed_cache_dtype": packed_dtype,
            **source_hashes,
        },
        "coverage": audit["coverage"],
    }
    if registration_sha256 is not None:
        scientific["source_media"]["cvat_source_registration_sha256"] = registration_sha256
        scientific["source_media"]["cvat_source_registration_audit"] = registration_audit
    scientific_path = output_dir / "scientific_stage1_rgb_binding.json"
    _legacy._write_json_atomic(scientific_path, scientific)
    data_bindings_path = write_stage1_execution_path_realization(
        output_dir=output_dir,
        scientific_binding_path=scientific_path,
        packed_cache_path=source_paths["packed_cache"],
        verified_packed_cache_sha256=source_hashes["source_packed_cache_sha256"],
        packed_cache_identity=source_identities["packed_cache"],
    )
    return {
        "scientific_binding_path": str(scientific_path),
        "scientific_binding_sha256": _legacy._sha256_file(scientific_path),
        "data_bindings_path": str(data_bindings_path),
        "data_bindings_sha256": _legacy._sha256_file(data_bindings_path),
        "coverage": audit["coverage"],
        "source_hashes": source_hashes,
    }


def write_stage1_execution_path_realization(
    *,
    output_dir: Path,
    scientific_binding_path: Path,
    packed_cache_path: Path | str,
    filename: str = "stage1_temporal_data_bindings.json",
    verified_packed_cache_sha256: str | None = None,
    packed_cache_identity: Mapping[str, int] | None = None,
) -> Path:
    """Bind one verified local cache realization without changing science."""

    output_dir = Path(output_dir).resolve()
    scientific_binding_path = Path(scientific_binding_path).resolve()
    if not scientific_binding_path.is_relative_to(output_dir):
        raise Stage1RgbBindingError("scientific binding must be colocated with realization")
    path = output_dir / filename
    if path.exists():
        raise Stage1RgbBindingError(f"execution path realization already exists={path}")
    realization: dict[str, object] = {
        "schema_version": EXECUTION_PATH_REALIZATION_SCHEMA,
        "packed_cache_path": str(packed_cache_path),
    }
    if verified_packed_cache_sha256 is not None:
        identity = (
            dict(packed_cache_identity)
            if packed_cache_identity is not None
            else _legacy._file_identity(Path(packed_cache_path))
        )
        if len(verified_packed_cache_sha256) != 64:
            raise Stage1RgbBindingError("attested packed RGB cache hash is invalid")
        if set(identity) != {"size_bytes", "mtime_ns"}:
            raise Stage1RgbBindingError("attested packed RGB cache identity is invalid")
        realization["packed_cache_identity_attestation"] = {
            "sha256": verified_packed_cache_sha256,
            **identity,
        }
    payload = {
        "schema_version": DATA_BINDINGS_SCHEMA,
        "scientific_binding": {
            "relative_path": scientific_binding_path.name,
            "sha256": _legacy._sha256_file(scientific_binding_path),
        },
        "execution_path_realization": realization,
    }
    _legacy._write_json_atomic(path, payload)
    return path


def resolve_stage1_execution_rgb_binding(
    *,
    data_bindings_path: Path,
    requested_roles: pd.DataFrame,
    authority_sha256: str,
    provenance_hashes: Mapping[str, str],
    view: str,
    sequence_length: int,
) -> ResolvedStage1RgbBinding:
    """Validate a Stage-1 binding before any RGB tensor is opened."""

    _validate_view(view, sequence_length)
    data_bindings_path = Path(data_bindings_path).resolve()
    payload = _read_json(data_bindings_path)
    if set(payload) != {
        "schema_version",
        "scientific_binding",
        "execution_path_realization",
    }:
        raise Stage1RgbBindingError("unexpected Stage-1 execution binding fields")
    if payload.get("schema_version") != DATA_BINDINGS_SCHEMA:
        raise Stage1RgbBindingError("unsupported Stage-1 data-bindings schema")
    science_ref = payload.get("scientific_binding")
    if not isinstance(science_ref, Mapping) or set(science_ref) != {
        "relative_path",
        "sha256",
    }:
        raise Stage1RgbBindingError("scientific RGB binding reference is invalid")
    scientific_path = _safe_relative_path(
        data_bindings_path.parent,
        str(science_ref["relative_path"]),
    )
    if _legacy._sha256_file(scientific_path) != str(science_ref["sha256"]):
        raise Stage1RgbBindingError("scientific RGB binding hash mismatch")
    scientific = _read_json(scientific_path)
    _validate_scientific_binding(
        scientific,
        authority_sha256=authority_sha256,
        provenance_hashes=provenance_hashes,
        view=view,
        sequence_length=sequence_length,
    )

    realization = payload.get("execution_path_realization")
    valid_realization_fields = {
        "schema_version",
        "packed_cache_path",
        "packed_cache_identity_attestation",
    }
    if (
        not isinstance(realization, Mapping)
        or not {"schema_version", "packed_cache_path"}.issubset(realization)
        or not set(realization).issubset(valid_realization_fields)
    ):
        raise Stage1RgbBindingError("execution RGB path realization is invalid")
    if realization.get("schema_version") != EXECUTION_PATH_REALIZATION_SCHEMA:
        raise Stage1RgbBindingError("unsupported Stage-1 path realization schema")
    packed_cache_path = Path(str(realization["packed_cache_path"])).resolve()
    expected_cache_sha256 = str(scientific["source_media"]["packed_cache_sha256"])
    attestation = realization.get("packed_cache_identity_attestation")
    if attestation is None:
        observed_cache_sha256 = _legacy._sha256_file(packed_cache_path)
    else:
        if not isinstance(attestation, Mapping) or set(attestation) != {
            "sha256",
            "size_bytes",
            "mtime_ns",
        }:
            raise Stage1RgbBindingError("packed RGB cache identity attestation is invalid")
        if str(attestation["sha256"]) != expected_cache_sha256:
            raise Stage1RgbBindingError("packed RGB cache attestation hash drifted")
        expected_identity = {
            "size_bytes": int(attestation["size_bytes"]),
            "mtime_ns": int(attestation["mtime_ns"]),
        }
        observed_cache_sha256 = (
            expected_cache_sha256
            if _legacy._file_identity(packed_cache_path) == expected_identity
            else _legacy._sha256_file(packed_cache_path)
        )
    if observed_cache_sha256 != expected_cache_sha256:
        raise Stage1RgbBindingError("packed RGB cache hash mismatch")

    artifacts = scientific["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise Stage1RgbBindingError("scientific RGB artifacts are invalid")
    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for key in ("window_context", "frame_context", "packed_index"):
        entry = artifacts.get(key)
        if not isinstance(entry, Mapping) or set(entry) != {
            "relative_path",
            "sha256",
            "rows",
        }:
            raise Stage1RgbBindingError(f"scientific RGB artifact descriptor invalid={key}")
        path = _safe_relative_path(scientific_path.parent, str(entry["relative_path"]))
        observed = _legacy._sha256_file(path)
        if observed != str(entry["sha256"]):
            raise Stage1RgbBindingError(f"scientific RGB artifact hash mismatch={key}")
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
            raise Stage1RgbBindingError(
                f"scientific RGB artifact row count mismatch={key}"
            )
    roles = _prepare_requested_roles(requested_roles)
    audit = audit_stage1_rgb_binding(
        windows=windows,
        frames=frames,
        packed_index=packed_index,
        requested_roles=roles,
        view=view,
        sequence_length=sequence_length,
    )
    _raise_if_invalid(audit)
    coverage = audit["coverage"]
    return ResolvedStage1RgbBinding(
        frame_context_path=artifact_paths["frame_context"],
        window_context_path=artifact_paths["window_context"],
        packed_index_path=artifact_paths["packed_index"],
        packed_cache_path=packed_cache_path,
        hashes={
            "rgb_scientific_binding": _legacy._sha256_file(scientific_path),
            "rgb_packed_cache": expected_cache_sha256,
            **artifact_hashes,
        },
        coverage={key: int(value) for key, value in coverage.items()},
        audit=audit,
    )


def audit_stage1_rgb_binding(
    *,
    windows: pd.DataFrame,
    frames: pd.DataFrame,
    packed_index: pd.DataFrame,
    requested_roles: pd.DataFrame,
    view: str,
    sequence_length: int,
) -> dict[str, object]:
    """Fail closed on outer, wrong-view, sequence, identity, or media errors."""

    _validate_view(view, sequence_length)
    roles = _prepare_requested_roles(requested_roles)
    _require_columns(
        windows,
        {
            "window_id",
            "stage1_role",
            "temporal_view",
            "source_type",
            "video_key",
            "object_track_key",
            "pig_id",
            "track_id",
            "window_length_frames",
            "selected_frame_indices",
            "expected_frame_indices",
            "image_context_id_sequence",
            "window_image_context_complete",
        },
        "Stage-1 RGB windows",
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
        "Stage-1 RGB frames",
    )
    _require_columns(packed_index, {"image_context_id", "packed_row"}, "Stage-1 RGB index")
    requested_by_window = dict(
        zip(
            roles["window_id"].astype(str),
            roles["stage1_role"].astype(str),
            strict=True,
        )
    )
    window_ids = windows["window_id"].astype(str)
    actual_ids = set(window_ids)
    requested_ids = set(requested_by_window)
    role_violations = int(
        sum(
            requested_by_window.get(str(row.window_id)) != str(row.stage1_role)
            or str(row.stage1_role) not in INNER_ROLES
            for row in windows.itertuples(index=False)
        )
    )
    role_violations += int(sum(role not in INNER_ROLES for role in requested_by_window.values()))
    temporal_violations = int(windows["temporal_view"].astype(str).ne(view).sum())
    lengths = pd.to_numeric(windows["window_length_frames"], errors="coerce")
    length_violations = int(lengths.ne(sequence_length).sum())
    complete_violations = int((~_strict_bool(windows["window_image_context_complete"])).sum())

    sequence_rows = _audit_sequence_rows(windows, sequence_length=sequence_length)
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
    valid_sequences = sequence_rows.loc[
        sequence_rows["sequence_shape_valid"] & sequence_rows["context_ids_present"]
    ].copy()
    expanded = valid_sequences.explode(
        ["context_ids", "expected_frames"],
        ignore_index=True,
    ).rename(columns={"context_ids": "_context_id", "expected_frames": "_expected_frame"})
    frame_ids = frames["image_context_id"].astype(str)
    index_ids = packed_index["image_context_id"].astype(str)
    frame_lookup = _legacy._frame_lookup(frames, frame_ids)
    missing_context_ids = set(
        expanded.loc[
            ~expanded["_context_id"].isin(set(frame_lookup["_context_id"])),
            "_context_id",
        ]
    )
    missing_index_ids = set(
        expanded.loc[~expanded["_context_id"].isin(set(index_ids)), "_context_id"]
    )
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
    joined = joined.loc[joined["_binding_row"].isin(complete_rows[complete_rows].index)].copy()
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
            _legacy._invalid_windows(
                joined,
                joined["_frame_video_key"].astype(str).ne(joined["video_key"].astype(str)),
            )
        )
        actor_identity_windows.update(
            _legacy._invalid_windows(
                joined,
                joined["_frame_source_type"].astype(str).ne(
                    joined["source_type"].astype(str)
                ),
            )
        )
        for field in ("object_track_key", "pig_id", "track_id"):
            expected = joined[field].map(_legacy._normalized_value)
            observed = joined[f"_frame_{field}"].map(_legacy._normalized_value)
            actor_identity_windows.update(
                _legacy._invalid_windows(joined, expected.ne("") & observed.ne(expected))
            )
        media_ok = (
            _strict_bool(joined["_frame_resolved_media_exists"])
            & _strict_bool(joined["_frame_image_context_loadable"])
            & joined["_frame_resolved_media_path"].astype(str).str.strip().ne("")
        )
        media_reference_windows.update(_legacy._invalid_windows(joined, ~media_ok))

    coverage = {
        "train_windows_bound": int(windows["stage1_role"].astype(str).eq("train").sum()),
        "validation_windows_bound": int(
            windows["stage1_role"].astype(str).eq("validation").sum()
        ),
        "missing_windows": int(len(requested_ids.difference(actual_ids))),
        "duplicate_windows": int(window_ids.duplicated().sum()),
        "bad_sequence_length": int(len(bad_sequence_windows)),
        "role_violations": int(role_violations),
        "cross_video_violations": int(len(cross_video_windows)),
        "unexpected_windows": int(len(actual_ids.difference(requested_ids))),
        "temporal_violations": int(temporal_violations),
        "window_length_violations": int(length_violations),
        "incomplete_window_violations": int(complete_violations),
        "missing_context_ids": int(len(missing_context_ids)),
        "missing_packed_index_ids": int(len(missing_index_ids)),
        "duplicate_context_ids": int(frame_ids.duplicated().sum()),
        "duplicate_packed_index_ids": int(index_ids.duplicated().sum()),
        "sequence_order_violations": int(len(sequence_order_windows)),
        "actor_identity_violations": int(len(actor_identity_windows)),
        "media_reference_violations": int(len(media_reference_windows)),
    }
    invalid_keys = {
        "missing_windows",
        "duplicate_windows",
        "bad_sequence_length",
        "role_violations",
        "cross_video_violations",
        "unexpected_windows",
        "temporal_violations",
        "window_length_violations",
        "incomplete_window_violations",
        "missing_context_ids",
        "missing_packed_index_ids",
        "duplicate_context_ids",
        "duplicate_packed_index_ids",
        "sequence_order_violations",
        "actor_identity_violations",
        "media_reference_violations",
    }
    errors = sorted(key for key in invalid_keys if coverage[key] != 0)
    return {
        "schema_version": SCIENTIFIC_RGB_BINDING_SCHEMA,
        "coverage": coverage,
        "valid": not errors,
        "errors": errors,
    }


def _prepare_requested_roles(
    requested_roles: pd.DataFrame,
    *,
    expected_train_windows: int | None = None,
    expected_validation_windows: int | None = None,
) -> pd.DataFrame:
    role_column = "primary_s1_role" if "primary_s1_role" in requested_roles else "stage1_role"
    _require_columns(requested_roles, {"window_id", role_column}, "Stage-1 population")
    roles = requested_roles.loc[:, ["window_id", role_column]].copy()
    roles.columns = ["window_id", "stage1_role"]
    roles["window_id"] = roles["window_id"].astype(str)
    roles["stage1_role"] = roles["stage1_role"].astype(str)
    if roles["window_id"].duplicated().any():
        raise Stage1RgbBindingError("Stage-1 population has duplicate window_id")
    invalid_roles = sorted(set(roles["stage1_role"]).difference(INNER_ROLES))
    if invalid_roles:
        raise Stage1RgbBindingError(f"outer/test role in Stage-1 RGB binding={invalid_roles}")
    actual_train = int(roles["stage1_role"].eq("train").sum())
    actual_validation = int(roles["stage1_role"].eq("validation").sum())
    if expected_train_windows is not None and actual_train != expected_train_windows:
        raise Stage1RgbBindingError(
            f"authorized train count mismatch={actual_train}!={expected_train_windows}"
        )
    if (
        expected_validation_windows is not None
        and actual_validation != expected_validation_windows
    ):
        raise Stage1RgbBindingError(
            "authorized validation count mismatch="
            f"{actual_validation}!={expected_validation_windows}"
        )
    return roles.sort_values("window_id", kind="stable").reset_index(drop=True)


def _context_ids_from_windows(windows: pd.DataFrame, *, sequence_length: int) -> set[str]:
    context_ids: set[str] = set()
    for row in windows.itertuples(index=False):
        values = _legacy._split_sequence(row.image_context_id_sequence)
        if len(values) != sequence_length or any(not value for value in values):
            raise Stage1RgbBindingError(
                f"invalid {row.temporal_view} image context sequence={row.window_id}"
            )
        context_ids.update(values)
    return context_ids


def _audit_sequence_rows(
    windows: pd.DataFrame,
    *,
    sequence_length: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for binding_row, row in enumerate(windows.itertuples(index=False)):
        context_ids = _legacy._split_sequence(row.image_context_id_sequence)
        expected_frames = _legacy._integer_sequence(row.expected_frame_indices)
        selected_frames = _legacy._integer_sequence(row.selected_frame_indices)
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
                    len(context_ids) == sequence_length
                    and len(expected_frames) == sequence_length
                ),
                "context_ids_present": not any(not value for value in context_ids),
                "selected_matches_expected": selected_frames == expected_frames,
            }
        )
    return pd.DataFrame(rows)


def _validate_scientific_binding(
    scientific: Mapping[str, object],
    *,
    authority_sha256: str,
    provenance_hashes: Mapping[str, str],
    view: str,
    sequence_length: int,
) -> None:
    required = {
        "schema_version",
        "stage1",
        "provenance_hashes",
        "artifacts",
        "source_media",
        "coverage",
    }
    if set(scientific) != required:
        raise Stage1RgbBindingError("unexpected scientific RGB binding fields")
    if scientific.get("schema_version") != SCIENTIFIC_RGB_BINDING_SCHEMA:
        raise Stage1RgbBindingError("unsupported Stage-1 scientific binding schema")
    expected_stage1 = {
        "authority_sha256": str(authority_sha256),
        "run_kind": "S1_STAGE1_TEMPORAL_SCREENING",
        "fold": "FOLD_3",
        "temporal_view": view,
        "sequence_length": sequence_length,
        "roles": ["train", "validation"],
    }
    if scientific.get("stage1") != expected_stage1:
        raise Stage1RgbBindingError("Stage-1 RGB scientific identity drifted")
    if scientific.get("provenance_hashes") != _normalized_hash_mapping(provenance_hashes):
        raise Stage1RgbBindingError("Stage-1 RGB provenance hash drifted")
    artifacts = scientific.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "window_context",
        "frame_context",
        "packed_index",
    }:
        raise Stage1RgbBindingError("Stage-1 RGB artifacts are incomplete")
    source_media = scientific.get("source_media")
    if not isinstance(source_media, Mapping):
        raise Stage1RgbBindingError("Stage-1 RGB source media is invalid")
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
    allowed_media = required_media | {
        "cvat_source_registration_sha256",
        "cvat_source_registration_audit",
    }
    if not required_media.issubset(source_media) or not set(source_media).issubset(
        allowed_media
    ):
        raise Stage1RgbBindingError("Stage-1 RGB source media fields are invalid")
    if source_media.get("packed_cache_sha256") != source_media.get(
        "source_packed_cache_sha256"
    ):
        raise Stage1RgbBindingError("Stage-1 RGB packed-cache provenance drifted")
    registration_hash = source_media.get("cvat_source_registration_sha256")
    if registration_hash is not None and (
        not isinstance(registration_hash, str) or len(registration_hash) != 64
    ):
        raise Stage1RgbBindingError("CVAT source-registration provenance is invalid")
    registration_audit = source_media.get("cvat_source_registration_audit")
    if registration_audit is not None and (
        not isinstance(registration_audit, Mapping)
        or registration_audit.get("valid") is not True
        or registration_audit.get("scientific_projection_sha256_before")
        != registration_audit.get("scientific_projection_sha256_after")
        or registration_audit.get("review_projection_sha256_before")
        != registration_audit.get("review_projection_sha256_after")
    ):
        raise Stage1RgbBindingError("CVAT source-registration audit is invalid")


def _validate_view(view: str, sequence_length: int) -> None:
    expected = {"T6": 6, "T8": 8, "T12": 12, "T16": 16}
    if expected.get(view) != sequence_length:
        raise Stage1RgbBindingError(
            f"unsupported Stage-1 temporal binding view={view}:{sequence_length}"
        )


def _raise_if_invalid(audit: Mapping[str, object]) -> None:
    if not bool(audit.get("valid")):
        raise Stage1RgbBindingError(f"Stage-1 RGB binding failures={audit.get('errors')}")


def _safe_relative_path(base: Path, value: str) -> Path:
    try:
        return _legacy._safe_relative_path(base, value)
    except _legacy.RgbBindingError as exc:
        raise Stage1RgbBindingError(str(exc)) from exc


def _strict_bool(values: pd.Series) -> pd.Series:
    return _legacy._strict_bool(values)


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise Stage1RgbBindingError(f"{label} missing columns={missing}")


def _normalized_hash_mapping(values: Mapping[str, str]) -> dict[str, str]:
    try:
        return _legacy._normalized_hash_mapping(values)
    except _legacy.RgbBindingError as exc:
        raise Stage1RgbBindingError(str(exc)) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _legacy._read_json(path)
    except _legacy.RgbBindingError as exc:
        raise Stage1RgbBindingError(str(exc)) from exc


__all__ = [
    "DATA_BINDINGS_SCHEMA",
    "EXECUTION_PATH_REALIZATION_SCHEMA",
    "ResolvedStage1RgbBinding",
    "SCIENTIFIC_RGB_BINDING_SCHEMA",
    "Stage1RgbBindingError",
    "audit_stage1_rgb_binding",
    "materialize_stage1_rgb_binding",
    "resolve_stage1_execution_rgb_binding",
    "write_stage1_execution_path_realization",
]
