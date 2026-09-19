"""Exact repeat gate for the immutable legacy L6 social-relation cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.evaluation.legacy_development_l6_geometry_cache_repeat import (
    _git_guard,
    _object,
    _read_json,
    _require_sha,
    _resolve_inside,
    _validate_bound_file,
)
from pig_behavior.classification_v2.training.legacy_development_l6_social_relation_cache import (
    CANONICAL_SOURCE_NAME,
    DATASET_ID,
    LINEAGE_SCOPE,
    SOURCE_TYPE,
    audit_social_relation_cache,
    load_social_relation_cache_config,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

CONFIG_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "social_relation_cache_repeat_config.v1"
)
RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6."
    "social_relation_cache_repeat_gate.v1"
)
ARTIFACT_NAMES = (
    "social_relation",
    "availability",
    "window_index",
    "slot_index",
)


def evaluate_social_relation_cache_repeat(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Audit two social caches and require exact data-artifact equality."""

    root = project_root.resolve()
    resolved_config = config_path.resolve()
    config = _read_json(resolved_config)
    _validate_config(config)
    implementation = _object(config["implementation_source"], "implementation")
    implementation_path = _resolve_inside(root, str(implementation["path"]))
    _validate_bound_file(
        implementation_path,
        str(implementation["sha256"]),
        "social repeat implementation",
    )
    packets: dict[str, dict[str, Any]] = {}
    for name in ("primary", "repeat"):
        spec = _object(config[name], name)
        cache_config_path = _resolve_inside(root, str(spec["config_path"]))
        cache_manifest_path = _resolve_inside(root, str(spec["manifest_path"]))
        _validate_bound_file(
            cache_config_path,
            str(spec["config_sha256"]),
            f"{name} social cache config",
        )
        _validate_bound_file(
            cache_manifest_path,
            str(spec["manifest_sha256"]),
            f"{name} social cache manifest",
        )
        cache_config = load_social_relation_cache_config(cache_config_path)
        audit = audit_social_relation_cache(
            cache_config,
            cache_root=cache_manifest_path.parent,
        )
        if not audit["valid"]:
            raise ValueError(f"{name} social cache invalid={audit['errors']}")
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
    compared = [
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
    ]
    mismatches = [name for name in compared if primary[name] != repeat[name]]
    primary_root = str(primary["output"]["cache_root_relative_path"])
    repeat_root = str(repeat["output"]["cache_root_relative_path"])
    errors: list[str] = []
    if mismatches:
        errors.append(f"semantic_config_sections_differ={mismatches}")
    if primary_root == repeat_root:
        errors.append("social_cache_repeat_output_roots_are_identical")
    return {
        "compared_sections": compared,
        "different_sections": mismatches,
        "primary_output_root": primary_root,
        "repeat_output_root": repeat_root,
        "only_output_and_guard_differ": not errors,
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
            errors.append(f"social_cache_artifact_repeat_mismatch={name}")
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
    mismatches = [name for name in left if left[name] != right.get(name)]
    extra = sorted(set(right) - set(left))
    errors = [
        f"social_cache_content_repeat_mismatch={name}"
        for name in [*mismatches, *extra]
    ]
    return {
        "different_fields": [*mismatches, *extra],
        "source_probe_status": left["source_probe"]["status"],
        "numeric_social_only": left["numeric_social_only"],
        "top_k_partner_features_used": left["top_k_partner_features_used"],
        "errors": errors,
        "valid": not errors,
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    audit = packet["audit"]
    return {
        "config_path": str(packet["config"].path),
        "config_sha256": packet["config"].sha256,
        "manifest_path": str(packet["manifest_path"]),
        "manifest_sha256": audit["manifest_sha256"],
        "verified_artifacts": audit["verified_artifacts"],
        "social_relation_shape": audit["social_relation_shape"],
        "availability_shape": audit["availability_shape"],
        "available_slots": audit["available_slots"],
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
        raise ValueError("social cache repeat config keys differ")
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
            raise ValueError(f"social cache repeat {name}={config[name]!r}")
    implementation = _object(config["implementation_source"], "implementation")
    if set(implementation) != {"path", "sha256"}:
        raise ValueError("social cache repeat implementation keys differ")
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
            raise ValueError(f"social cache repeat {name} keys differ")
        _require_sha(str(spec["config_sha256"]), f"{name} config sha256")
        _require_sha(str(spec["manifest_sha256"]), f"{name} manifest sha256")
    guard = _object(config["execution_guard"], "execution_guard")
    if set(guard) != {"allowed_dirty_paths", "required_tracked_paths"}:
        raise ValueError("social cache repeat execution guard keys differ")
