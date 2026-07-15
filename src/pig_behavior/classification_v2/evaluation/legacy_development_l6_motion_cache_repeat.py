"""Exact repeat gate for the immutable legacy L6 motion cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.evaluation.legacy_development_l6_geometry_cache_repeat import (
    _git_guard,
    _object,
    _read_json,
    _require_sha,
    _resolve_inside,
    _semantic_config_comparison,
    _validate_bound_file,
)
from pig_behavior.classification_v2.training.legacy_development_l6_motion_cache import (
    CANONICAL_SOURCE_NAME,
    DATASET_ID,
    LINEAGE_SCOPE,
    SOURCE_TYPE,
    audit_motion_cache,
    load_motion_cache_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_cache_repeat_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6.motion_cache_repeat_gate.v1"
)
ARTIFACT_NAMES = ("motion", "availability", "window_index", "slot_index")


def evaluate_motion_cache_repeat(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Audit two motion caches and require exact data-artifact equality."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    implementation = _object(config["implementation_source"], "implementation")
    implementation_path = _resolve_inside(root, str(implementation["path"]))
    _validate_bound_file(
        implementation_path,
        str(implementation["sha256"]),
        "motion repeat implementation",
    )
    packets: dict[str, dict[str, Any]] = {}
    for name in ("primary", "repeat"):
        spec = _object(config[name], name)
        cache_config_path = _resolve_inside(root, str(spec["config_path"]))
        cache_manifest_path = _resolve_inside(root, str(spec["manifest_path"]))
        _validate_bound_file(
            cache_config_path,
            str(spec["config_sha256"]),
            f"{name} motion cache config",
        )
        _validate_bound_file(
            cache_manifest_path,
            str(spec["manifest_sha256"]),
            f"{name} motion cache manifest",
        )
        cache_config = load_motion_cache_config(cache_config_path)
        audit = audit_motion_cache(
            cache_config,
            cache_root=cache_manifest_path.parent,
        )
        if not audit["valid"]:
            raise ValueError(f"{name} motion cache audit failed={audit['errors']}")
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
    for audit in (semantic, artifacts, content, git_guard):
        errors.extend(str(value) for value in audit["errors"])
    valid = not errors
    return {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_MOTION_CACHE_REPEAT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_MOTION_CACHE_REPEAT"
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
            errors.append(f"motion_cache_artifact_repeat_mismatch={name}")
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
        "available_pair_slots",
        "unavailable_slots",
        "unavailable_first_slots",
        "unavailable_nonfirst_slots",
        "motion_shape",
        "availability_shape",
        "motion_dtype",
        "availability_dtype",
        "motion_statistics",
        "availability_pattern_count",
        "availability_pattern",
        "ordered_window_id_sha256",
        "window_index_content_sha256",
        "slot_index_content_sha256",
        "spatial_export_audit",
        "window_start_reset_audit",
        "source_probe",
        "unit_aggregate_features_selected",
        "geometry_values_consumed",
    ]
    mismatches = [name for name in fields if left[name] != right[name]]
    errors = [
        f"motion_cache_content_repeat_mismatch={name}"
        for name in mismatches
    ]
    return {
        "compared_fields": fields,
        "different_fields": mismatches,
        "source_probe_status": left["source_probe"]["status"],
        "window_local_reset_valid": bool(
            left["window_start_reset_audit"]["valid"]
        ),
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
        "motion_shape": audit["motion_shape"],
        "availability_shape": audit["availability_shape"],
        "available_pair_slots": audit["available_pair_slots"],
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
        raise ValueError("motion cache repeat config keys differ")
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
            raise ValueError(f"motion cache repeat {name}={config[name]!r}")
    implementation = _object(config["implementation_source"], "implementation")
    if set(implementation) != {"path", "sha256"}:
        raise ValueError("motion cache repeat implementation keys differ")
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
            raise ValueError(f"motion cache repeat {name} keys differ")
        _require_sha(str(spec["config_sha256"]), f"{name} config sha256")
        _require_sha(str(spec["manifest_sha256"]), f"{name} manifest sha256")
    guard = _object(config["execution_guard"], "execution_guard")
    if set(guard) != {"allowed_dirty_paths", "required_tracked_paths"}:
        raise ValueError("motion cache repeat execution guard keys differ")
