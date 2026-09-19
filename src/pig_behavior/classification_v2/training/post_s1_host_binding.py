"""Materialize validated host-specific bindings for the post-S1 T6 screen.

The Stage-1 RGB scientific binding remains immutable and portable.  This module
owns only the reproducible, host-specific execution realization required by the
post-S1 resolution arms.  Missing realizations are therefore regenerated from
registered authorities; mismatched scientific identities fail closed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.training.cvat_source_registration import (
    CvatSourceRegistrationError,
    load_cvat_source_registration,
)
from pig_behavior.classification_v2.training.legacy_media_resolution import (
    LEGACY_SOURCE_RESOLUTION_VERSION,
)
from pig_behavior.classification_v2.training.remote_input_resolution import (
    RemoteInputAuthority,
)
from pig_behavior.classification_v2.training.stage1_rgb_binding import (
    ResolvedStage1RgbBinding,
    Stage1RgbBindingError,
    materialize_stage1_rgb_binding,
    resolve_stage1_execution_rgb_binding,
)

HOST_BINDING_SCHEMA = "classification_v2.post_s1_t6_host_binding.v1"
HOST_BINDING_PRODUCER = "post_s1_host_binding"
HOST_BINDING_VERSION = 1
EXECUTION_PHASE = "POST_S1_T6_PURE_RESOLUTION_SCREEN"
TEMPORAL_VIEW = "T6"
SEQUENCE_LENGTH = 6
ALLOWED_RESOLUTIONS = frozenset({64, 128, 160})


class PostS1HostBindingError(ValueError):
    """Raised when a host binding cannot be safely reused or regenerated."""


@dataclass(frozen=True, slots=True)
class ResolvedPostS1HostBinding:
    """A verified host realization plus the reusable Stage-1 RGB binding."""

    binding_path: Path
    binding_sha256: str
    rgb: ResolvedStage1RgbBinding
    payload: Mapping[str, Any]
    regenerated: bool


def ensure_post_s1_t6_host_binding(
    *,
    binding_path: Path,
    canonical_code_sha: str,
    input_authority: RemoteInputAuthority,
    runtime_input_binding: Mapping[str, Any],
    media_root: Path,
    rgb_source_root: Path,
    t6_population_authority_sha256: str,
    t6_population_provenance_hashes: Mapping[str, str],
    requested_roles: pd.DataFrame,
    input_resolution: int,
    cvat_source_registration_path: Path | None = None,
) -> ResolvedPostS1HostBinding:
    """Accept, or deterministically rematerialize, one derived host binding."""

    expected = _expected_identity(
        canonical_code_sha=canonical_code_sha,
        input_authority=input_authority,
        runtime_input_binding=runtime_input_binding,
        media_root=media_root,
        rgb_source_root=rgb_source_root,
        t6_population_authority_sha256=t6_population_authority_sha256,
        t6_population_provenance_hashes=t6_population_provenance_hashes,
        requested_roles=requested_roles,
        input_resolution=input_resolution,
        cvat_source_registration_path=cvat_source_registration_path,
    )
    binding_path = Path(binding_path).resolve()
    if binding_path.exists():
        payload = _read_json(binding_path)
        _assert_scientific_identity(payload, expected)
        try:
            return _resolve_existing(binding_path, expected, requested_roles)
        except PostS1HostBindingError:
            # The authority identity is established above.  A stale runtime or
            # code realization is derived state and must be regenerated.
            pass
    return _materialize(binding_path, expected, requested_roles)


def _expected_identity(
    *,
    canonical_code_sha: str,
    input_authority: RemoteInputAuthority,
    runtime_input_binding: Mapping[str, Any],
    media_root: Path,
    rgb_source_root: Path,
    t6_population_authority_sha256: str,
    t6_population_provenance_hashes: Mapping[str, str],
    requested_roles: pd.DataFrame,
    input_resolution: int,
    cvat_source_registration_path: Path | None,
) -> dict[str, Any]:
    if len(canonical_code_sha) != 40 or any(
        character not in "0123456789abcdef" for character in canonical_code_sha
    ):
        raise PostS1HostBindingError("canonical code SHA must be lowercase Git SHA-1")
    if input_resolution not in ALLOWED_RESOLUTIONS:
        raise PostS1HostBindingError("unregistered post-S1 input resolution")
    media_root = Path(media_root).resolve()
    rgb_source_root = Path(rgb_source_root).resolve()
    if not rgb_source_root.is_relative_to(media_root):
        raise PostS1HostBindingError("RGB source must remain below verified media root")
    roles = _roles_contract(requested_roles)
    runtime_root = Path(str(runtime_input_binding.get("effective_remote_input_root", "")))
    if runtime_root.resolve() != media_root:
        raise PostS1HostBindingError("runtime input root does not match media root")
    if runtime_input_binding.get("scientific_input_authority_id") != input_authority.authority_id:
        raise PostS1HostBindingError("runtime input authority identity drifted")
    if (
        runtime_input_binding.get("expected_file_count")
        != input_authority.expected_file_count
        or runtime_input_binding.get("expected_total_bytes")
        != input_authority.expected_total_bytes
    ):
        raise PostS1HostBindingError("runtime input parity population drifted")
    if len(t6_population_authority_sha256) != 64:
        raise PostS1HostBindingError("T6 population authority hash is invalid")
    provenance = _normalized_hashes(t6_population_provenance_hashes)
    registration_sha256 = None
    if cvat_source_registration_path is not None:
        try:
            _, registration_sha256 = load_cvat_source_registration(
                cvat_source_registration_path
            )
        except CvatSourceRegistrationError as error:
            raise PostS1HostBindingError(str(error)) from error
    return {
        "scientific_identity": {
            "scientific_input_authority_id": input_authority.authority_id,
            "t6_population_authority_sha256": t6_population_authority_sha256,
            "t6_population_provenance_hashes": provenance,
            "fold": "FOLD_3",
            "temporal_view": TEMPORAL_VIEW,
            "sequence_length": SEQUENCE_LENGTH,
            "roles": ["train", "validation"],
            "role_counts": roles["counts"],
            "role_population_sha256": roles["sha256"],
        },
        "runtime_realization": {
            "canonical_code_sha": canonical_code_sha,
            "execution_phase": EXECUTION_PHASE,
            "input_resolution": input_resolution,
            "effective_remote_input_root": str(media_root),
            "rgb_source_root": str(rgb_source_root),
            "expected_file_count": input_authority.expected_file_count,
            "expected_total_bytes": input_authority.expected_total_bytes,
            "parity_report_sha256": runtime_input_binding.get("parity_report_sha256"),
            "cvat_source_registration_path": (
                str(Path(cvat_source_registration_path).resolve())
                if cvat_source_registration_path is not None
                else None
            ),
            "cvat_source_registration_sha256": registration_sha256,
            "legacy_source_resolution": LEGACY_SOURCE_RESOLUTION_VERSION,
        },
        "requested_roles": roles["frame"],
    }


def _roles_contract(requested_roles: pd.DataFrame) -> dict[str, Any]:
    required = {"window_id", "primary_s1_role"}
    if not required.issubset(requested_roles.columns):
        raise PostS1HostBindingError("T6 roles require window_id and primary_s1_role")
    roles = requested_roles.loc[:, ["window_id", "primary_s1_role"]].copy()
    roles["window_id"] = roles["window_id"].astype(str)
    roles["primary_s1_role"] = roles["primary_s1_role"].astype(str)
    if roles["window_id"].duplicated().any():
        raise PostS1HostBindingError("T6 roles contain duplicate window IDs")
    if set(roles["primary_s1_role"]) - {"train", "validation"}:
        raise PostS1HostBindingError("outer/test role cannot enter T6 host binding")
    counts = {
        "train": int(roles["primary_s1_role"].eq("train").sum()),
        "validation": int(roles["primary_s1_role"].eq("validation").sum()),
    }
    if not all(counts.values()):
        raise PostS1HostBindingError("T6 host binding requires train and validation roles")
    roles = roles.sort_values("window_id", kind="stable").reset_index(drop=True)
    canonical = roles.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return {
        "frame": roles,
        "counts": counts,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _resolve_existing(
    binding_path: Path,
    expected: Mapping[str, Any],
    requested_roles: pd.DataFrame,
) -> ResolvedPostS1HostBinding:
    payload = _read_json(binding_path)
    _assert_complete_payload(payload, expected)
    _assert_hash_sidecar(binding_path)
    artifacts = payload["artifacts"]
    data_bindings_path = _safe_relative(binding_path.parent, artifacts["data_bindings"])
    if _sha256_file(data_bindings_path) != artifacts["data_bindings_sha256"]:
        raise PostS1HostBindingError("host binding data-bindings hash drifted")
    try:
        rgb = resolve_stage1_execution_rgb_binding(
            data_bindings_path=data_bindings_path,
            requested_roles=requested_roles,
            authority_sha256=expected["scientific_identity"]["t6_population_authority_sha256"],
            provenance_hashes=expected["scientific_identity"]["t6_population_provenance_hashes"],
            view=TEMPORAL_VIEW,
            sequence_length=SEQUENCE_LENGTH,
        )
    except Stage1RgbBindingError as error:
        raise PostS1HostBindingError(str(error)) from error
    return ResolvedPostS1HostBinding(
        binding_path=binding_path,
        binding_sha256=_sha256_file(binding_path),
        rgb=rgb,
        payload=payload,
        regenerated=False,
    )


def _materialize(
    binding_path: Path,
    expected: Mapping[str, Any],
    requested_roles: pd.DataFrame,
) -> ResolvedPostS1HostBinding:
    materialization_key = _materialization_key(expected)
    artifact_dir = binding_path.parent / "post_s1_host_binding_artifacts" / materialization_key
    if not artifact_dir.exists():
        try:
            report = materialize_stage1_rgb_binding(
                output_dir=artifact_dir,
                rgb_source_root=Path(expected["runtime_realization"]["rgb_source_root"]),
                requested_roles=requested_roles,
                authority_sha256=expected["scientific_identity"]["t6_population_authority_sha256"],
                provenance_hashes=expected["scientific_identity"][
                    "t6_population_provenance_hashes"
                ],
                view=TEMPORAL_VIEW,
                sequence_length=SEQUENCE_LENGTH,
                expected_train_windows=expected["scientific_identity"]["role_counts"]["train"],
                expected_validation_windows=expected["scientific_identity"][
                    "role_counts"
                ]["validation"],
                cvat_source_registration_path=(
                    Path(expected["runtime_realization"]["cvat_source_registration_path"])
                    if expected["runtime_realization"].get(
                        "cvat_source_registration_path"
                    )
                    else None
                ),
                preserve_legacy_physical_paths=True,
            )
        except Stage1RgbBindingError as error:
            raise PostS1HostBindingError(str(error)) from error
    else:
        report = {
            "data_bindings_path": str(artifact_dir / "stage1_temporal_data_bindings.json"),
            "scientific_binding_path": str(artifact_dir / "scientific_stage1_rgb_binding.json"),
        }
    data_bindings_path = Path(str(report["data_bindings_path"])).resolve()
    scientific_path = Path(str(report["scientific_binding_path"])).resolve()
    payload = {
        "schema_version": HOST_BINDING_SCHEMA,
        "producer": {"name": HOST_BINDING_PRODUCER, "version": HOST_BINDING_VERSION},
        "scientific_identity": expected["scientific_identity"],
        "runtime_realization": expected["runtime_realization"],
        "artifacts": {
            "data_bindings": str(data_bindings_path.relative_to(binding_path.parent)),
            "data_bindings_sha256": _sha256_file(data_bindings_path),
            "scientific_binding": str(scientific_path.relative_to(binding_path.parent)),
            "scientific_binding_sha256": _sha256_file(scientific_path),
        },
    }
    _write_json_atomic(binding_path, payload)
    _write_hash_sidecar_atomic(binding_path)
    resolved = _resolve_existing(binding_path, expected, requested_roles)
    return ResolvedPostS1HostBinding(
        binding_path=resolved.binding_path,
        binding_sha256=resolved.binding_sha256,
        rgb=resolved.rgb,
        payload=resolved.payload,
        regenerated=True,
    )


def _assert_scientific_identity(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != HOST_BINDING_SCHEMA:
        raise PostS1HostBindingError("unsupported post-S1 host binding schema")
    if payload.get("scientific_identity") != expected["scientific_identity"]:
        raise PostS1HostBindingError("post-S1 host binding scientific identity drifted")


def _assert_complete_payload(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if set(payload) != {
        "schema_version",
        "producer",
        "scientific_identity",
        "runtime_realization",
        "artifacts",
    }:
        raise PostS1HostBindingError("unexpected post-S1 host binding fields")
    _assert_scientific_identity(payload, expected)
    if payload.get("producer") != {
        "name": HOST_BINDING_PRODUCER,
        "version": HOST_BINDING_VERSION,
    }:
        raise PostS1HostBindingError("post-S1 host binding producer drifted")
    if payload.get("runtime_realization") != expected["runtime_realization"]:
        raise PostS1HostBindingError("post-S1 host binding runtime realization is stale")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "data_bindings",
        "data_bindings_sha256",
        "scientific_binding",
        "scientific_binding_sha256",
    }:
        raise PostS1HostBindingError("post-S1 host binding artifacts are invalid")


def _materialization_key(expected: Mapping[str, Any]) -> str:
    value = {
        "scientific_identity": expected["scientific_identity"],
        "runtime_realization": expected["runtime_realization"],
    }
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _safe_relative(base: Path, value: object) -> Path:
    candidate = (base / str(value)).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise PostS1HostBindingError("host binding artifact path escapes binding root")
    return candidate


def _assert_hash_sidecar(binding_path: Path) -> None:
    sidecar = binding_path.with_suffix(binding_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise PostS1HostBindingError("host binding SHA256 sidecar is missing")
    expected = sidecar.read_text(encoding="utf-8").strip()
    actual = _sha256_file(binding_path)
    if expected != actual:
        raise PostS1HostBindingError("host binding SHA256 audit failed")


def _write_hash_sidecar_atomic(binding_path: Path) -> None:
    sidecar = binding_path.with_suffix(binding_path.suffix + ".sha256")
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(_sha256_file(binding_path) + "\n", encoding="utf-8")
    temporary.replace(sidecar)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PostS1HostBindingError(f"invalid host binding JSON={path}") from error
    if not isinstance(value, dict):
        raise PostS1HostBindingError("host binding must be a JSON object")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    temporary.replace(path)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _normalized_hashes(values: Mapping[str, str]) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in values.items()}
    if not normalized or any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in normalized.values()
    ):
        raise PostS1HostBindingError("T6 provenance hashes are invalid")
    return dict(sorted(normalized.items()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "HOST_BINDING_SCHEMA",
    "PostS1HostBindingError",
    "ResolvedPostS1HostBinding",
    "ensure_post_s1_t6_host_binding",
]
