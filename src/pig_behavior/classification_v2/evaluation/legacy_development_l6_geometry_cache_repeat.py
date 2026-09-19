"""Exact repeat gate for the immutable legacy L6 geometry cache."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.legacy_development_l6_geometry_cache import (
    CANONICAL_SOURCE_NAME,
    DATASET_ID,
    LINEAGE_SCOPE,
    SOURCE_TYPE,
    audit_geometry_cache,
    load_geometry_cache_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_cache_repeat_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_cache_repeat_gate.v1"
)
ARTIFACT_NAMES = ("geometry", "availability", "window_index", "slot_index")


def evaluate_geometry_cache_repeat(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Audit two cache packets and require exact data-artifact equality."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    implementation = _object(config["implementation_source"], "implementation")
    implementation_path = _resolve_inside(root, str(implementation["path"]))
    _validate_bound_file(
        implementation_path,
        str(implementation["sha256"]),
        "repeat implementation",
    )
    packets: dict[str, dict[str, Any]] = {}
    for name in ("primary", "repeat"):
        spec = _object(config[name], name)
        cache_config_path = _resolve_inside(root, str(spec["config_path"]))
        cache_manifest_path = _resolve_inside(root, str(spec["manifest_path"]))
        _validate_bound_file(
            cache_config_path,
            str(spec["config_sha256"]),
            f"{name} cache config",
        )
        _validate_bound_file(
            cache_manifest_path,
            str(spec["manifest_sha256"]),
            f"{name} cache manifest",
        )
        cache_config = load_geometry_cache_config(cache_config_path)
        audit = audit_geometry_cache(
            cache_config,
            cache_root=cache_manifest_path.parent,
        )
        if not audit["valid"]:
            raise ValueError(f"{name} cache audit failed={audit['errors']}")
        packets[name] = {
            "config": cache_config,
            "config_payload": cache_config.payload,
            "manifest_path": cache_manifest_path,
            "manifest": _read_json(cache_manifest_path),
            "audit": audit,
        }
    semantic = _semantic_config_comparison(
        packets["primary"]["config_payload"],
        packets["repeat"]["config_payload"],
    )
    artifacts = _artifact_comparison(
        packets["primary"]["manifest"],
        packets["repeat"]["manifest"],
    )
    content = _content_comparison(
        packets["primary"]["manifest"],
        packets["repeat"]["manifest"],
    )
    git_guard = _git_guard(root, config)
    errors: list[str] = []
    if not semantic["valid"]:
        errors.extend(str(value) for value in semantic["errors"])
    if not artifacts["valid"]:
        errors.extend(str(value) for value in artifacts["errors"])
    if not content["valid"]:
        errors.extend(str(value) for value in content["errors"])
    errors.extend(str(value) for value in git_guard["errors"])
    valid = not errors
    return {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_CACHE_REPEAT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_GEOMETRY_CACHE_REPEAT"
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
        "config_path": str(resolved_config),
        "config_sha256": file_sha256(resolved_config),
        "implementation_source_path": str(implementation_path),
        "implementation_source_sha256": str(implementation["sha256"]),
        "primary": _packet_summary(packets["primary"]),
        "repeat": _packet_summary(packets["repeat"]),
        "semantic_config_comparison": semantic,
        "artifact_comparison": artifacts,
        "content_comparison": content,
        "separate_output_roots": (
            packets["primary"]["manifest_path"].parent
            != packets["repeat"]["manifest_path"].parent
        ),
        "fresh_process_claim": (
            "separate_cli_invocations_not_pid_bound_in_cache_schema_v1"
        ),
        "source_media_reads": 0,
        "outer_holdout_slots_materialized": 0,
        "git_guard": git_guard,
        "errors": errors,
        "valid": valid,
    }


def configured_output_path(config_path: Path, project_root: Path) -> Path:
    """Return the repository-contained repeat-gate output path."""

    config = _read_json(config_path.resolve())
    _validate_config(config)
    return _resolve_inside(project_root.resolve(), str(config["output_path"]))


def _semantic_config_comparison(
    primary: dict[str, Any],
    repeat: dict[str, Any],
) -> dict[str, Any]:
    compared_sections = [
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
    ]
    mismatches = [name for name in compared_sections if primary[name] != repeat[name]]
    primary_root = str(primary["output"]["cache_root_relative_path"])
    repeat_root = str(repeat["output"]["cache_root_relative_path"])
    errors: list[str] = []
    if mismatches:
        errors.append(f"semantic_config_sections_differ={mismatches}")
    if primary_root == repeat_root:
        errors.append("cache_repeat_output_roots_are_identical")
    primary_allowed = primary["execution_guard"]["allowed_dirty_paths"]
    repeat_allowed = repeat["execution_guard"]["allowed_dirty_paths"]
    if primary_allowed != repeat_allowed:
        errors.append("cache_repeat_allowed_dirty_paths_differ")
    return {
        "compared_sections": compared_sections,
        "different_sections": mismatches,
        "primary_output_root": primary_root,
        "repeat_output_root": repeat_root,
        "only_output_and_config_tracking_differ": not errors,
        "errors": errors,
        "valid": not errors,
    }


def _artifact_comparison(
    primary: dict[str, Any],
    repeat: dict[str, Any],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    errors: list[str] = []
    for name in ARTIFACT_NAMES:
        left = _object(primary["artifacts"][name], f"primary.{name}")
        right = _object(repeat["artifacts"][name], f"repeat.{name}")
        sha_equal = left["sha256"] == right["sha256"]
        size_equal = int(left["size_bytes"]) == int(right["size_bytes"])
        rows[name] = {
            "primary_sha256": left["sha256"],
            "repeat_sha256": right["sha256"],
            "primary_size_bytes": int(left["size_bytes"]),
            "repeat_size_bytes": int(right["size_bytes"]),
            "sha256_equal": sha_equal,
            "size_equal": size_equal,
        }
        if not sha_equal or not size_equal:
            errors.append(f"cache_artifact_repeat_mismatch={name}")
    return {
        "artifact_count": len(ARTIFACT_NAMES),
        "artifacts": rows,
        "all_artifact_sha256_equal": not errors,
        "errors": errors,
        "valid": not errors,
    }


def _content_comparison(
    primary: dict[str, Any],
    repeat: dict[str, Any],
) -> dict[str, Any]:
    left = _object(primary["content_audit"], "primary content audit")
    right = _object(repeat["content_audit"], "repeat content audit")
    fields = [
        "model_window_rows",
        "model_slot_rows",
        "role_window_counts",
        "available_slots",
        "unavailable_slots",
        "geometry_shape",
        "availability_shape",
        "geometry_dtype",
        "availability_dtype",
        "geometry_statistics",
        "ordered_window_id_sha256",
        "window_index_content_sha256",
        "slot_index_content_sha256",
        "reference_audit",
        "source_probe",
    ]
    mismatches = [name for name in fields if left[name] != right[name]]
    errors = [f"cache_content_repeat_mismatch={name}" for name in mismatches]
    return {
        "compared_fields": fields,
        "different_fields": mismatches,
        "source_probe_status": left["source_probe"]["status"],
        "reference_match": bool(left["reference_audit"]["reference_match"]),
        "errors": errors,
        "valid": not errors,
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    manifest = packet["manifest"]
    audit = packet["audit"]
    return {
        "config_path": str(packet["config"].path),
        "config_sha256": packet["config"].sha256,
        "manifest_path": str(packet["manifest_path"]),
        "manifest_sha256": audit["manifest_sha256"],
        "verified_artifacts": audit["verified_artifacts"],
        "geometry_shape": audit["geometry_shape"],
        "availability_shape": audit["availability_shape"],
        "git_code_sha": manifest["git_guard"]["code_sha"],
        "errors": [],
        "valid": True,
    }


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "lineage_scope",
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
        "implementation_source",
        "execution_guard",
        "primary",
        "repeat",
        "output_path",
    }
    if set(config) != required:
        raise ValueError("geometry cache repeat config keys differ")
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
    }
    for name, value in expected.items():
        if config[name] != value:
            raise ValueError(f"geometry cache repeat {name}={config[name]!r}")
    implementation = _object(config["implementation_source"], "implementation")
    if set(implementation) != {"path", "sha256"}:
        raise ValueError("geometry cache repeat implementation keys differ")
    _require_sha(str(implementation["sha256"]), "implementation sha256")
    packet_fields = {
        "config_path",
        "config_sha256",
        "manifest_path",
        "manifest_sha256",
    }
    for name in ("primary", "repeat"):
        spec = _object(config[name], name)
        if set(spec) != packet_fields:
            raise ValueError(f"geometry cache repeat {name} keys differ")
        _require_sha(str(spec["config_sha256"]), f"{name} config sha256")
        _require_sha(str(spec["manifest_sha256"]), f"{name} manifest sha256")
    guard = _object(config["execution_guard"], "execution_guard")
    if set(guard) != {"allowed_dirty_paths", "required_tracked_paths"}:
        raise ValueError("geometry cache repeat execution guard keys differ")


def _git_guard(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    guard = _object(config["execution_guard"], "execution_guard")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
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
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", path],
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
        "code_sha": _git(root, "rev-parse", "HEAD").strip(),
        "dirty_entries": entries,
        "allowed_dirty_paths": allowed,
        "observed_dirty_paths": observed,
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": required,
        "untracked_required_paths": untracked,
        "errors": errors,
        "valid": not errors,
    }


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
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes project root: {path}") from error
    return path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


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
