"""Semantic fingerprinting and fail-closed lineage contracts for Phase 4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from pig_behavior.classification_v2.contracts.runtime_dependencies import (
    assert_stage_runtime_dependencies_complete,
)
from pig_behavior.classification_v2.features.motion_schema import (
    MOTION_FEATURE_NAMES,
    MOTION_SCHEMA_HASH,
    MOTION_SCHEMA_ID,
    MOTION_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.features.native_evidence_contract import (
    NATIVE_EVIDENCE_SEMANTICS_VERSION,
    NATIVE_FEATURE_COMPUTATION_GRAIN,
    NATIVE_PAIR_SCOPE_KEY,
)
from pig_behavior.classification_v2.features.spatial_semantics import (
    AXIS_DISTANCE_METRIC_ID,
    AXIS_DISTANCE_METRIC_VERSION,
    DIAGONAL_DISTANCE_METRIC_ID,
    DIAGONAL_DISTANCE_METRIC_VERSION,
    ROI_AGGREGATION_VERSION,
    ROI_TARGET_MODEL_POLICY_VERSION,
    SOCIAL_IDENTITY_VERSION,
    SOCIAL_NEAR_THRESHOLD_ID,
    SOCIAL_NEAR_THRESHOLD_UNITS,
    SOCIAL_NEAR_THRESHOLD_VALUE,
    SOCIAL_TIE_BREAK_RULE,
    SOCIAL_TIE_BREAK_VERSION,
    TARGET_ROI_SHARED_POLICY_ID,
    target_roi_model_policy_registry,
)

CANONICALIZATION_VERSION = "classification_v2.canonical_json.v1"
SEMANTIC_REGISTRY_VERSION = "classification_v2.semantic_domains.v6"
SEMANTIC_BUNDLE_ID = "bundle.classification_v2.phase1_4"
SEMANTIC_BUNDLE_VERSION = "classification_v2.semantic_bundle.v6"
STAGE_GRAPH_VERSION = "classification_v2.stage_dependency_graph.v4"
ARTIFACT_MANIFEST_VERSION = "classification_v2.artifact_manifest.v7"
MANIFEST_BUILDER_ID = "builder.classification_v2.candidate_manifest"
MANIFEST_BUILDER_VERSION = (
    "classification_v2.candidate_manifest_builder.v2"
)
CANDIDATE_AUTHORITY_STATE = "CANDIDATE_VALIDATED"
OFFICIAL_AUTHORITY_STATE = "OFFICIAL_PROMOTED"
CANDIDATE_TRANSACTION_STATE_PENDING = "RENAMED_PENDING_REVALIDATION"
CANDIDATE_TRANSACTION_STATE_COMMITTED = "COMMITTED_VALIDATED"
RELEASE_AUTHORITY_SCHEMA_VERSION = (
    "classification_v2.release_authority_preflight.v4"
)
CHANGE_IMPACT_REGISTRY_VERSION = (
    "classification_v2.change_impact_registry.v4"
)
HIDDEN_CARRY_FORWARD_VERSION = (
    "classification_v2.hidden_decision_carry_forward.v4"
)
BEHAVIOR_CARRY_FORWARD_VERSION = (
    "classification_v2.behavior_decision_carry_forward.v4"
)

HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
GIT_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CODE_AUTHORITY_VCS_GIT = "git"
SUPPORTED_GIT_OBJECT_FORMATS = frozenset({"sha1", "sha256"})
EPHEMERAL_SEMANTIC_FIELDS = frozenset(
    {
        "canonical_hash",
        "created_at",
        "file_mtime",
        "generated_at",
        "manifest_hash",
        "mtime",
        "semantic_bundle_hash",
        "timestamp",
        "updated_at",
    }
)

STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "stage.legacy_cvat_source_merge": (),
    "stage.frame_local_primitives": (
        "stage.legacy_cvat_source_merge",
    ),
    "stage.hidden_review_design": (
        "stage.frame_local_primitives",
    ),
    "stage.hidden_decision_migration": (
        "stage.hidden_review_design",
    ),
    "stage.hidden_gui": (
        "stage.hidden_review_design",
        "stage.hidden_decision_migration",
    ),
    "stage.hidden_coverage_scientific_gate": (
        "stage.hidden_review_design",
        "stage.hidden_gui",
    ),
    "stage.hidden_apply": (
        "stage.frame_local_primitives",
        "stage.hidden_coverage_scientific_gate",
    ),
    "stage.temporal_harmonization": (
        "stage.hidden_apply",
    ),
    "stage.native_review_evidence": (
        "stage.temporal_harmonization",
    ),
    "stage.pig_strenet_evidence": (
        "stage.temporal_harmonization",
        "stage.native_review_evidence",
    ),
    "stage.behavior_review_unit_construction": (
        "stage.temporal_harmonization",
        "stage.pig_strenet_evidence",
    ),
    "stage.behavior_gui": (
        "stage.behavior_review_unit_construction",
    ),
    "stage.behavior_decision_apply": (
        "stage.temporal_harmonization",
        "stage.behavior_gui",
    ),
    "stage.train_ready_export": (
        "stage.behavior_decision_apply",
    ),
    "stage.tensor_export": (
        "stage.train_ready_export",
    ),
    "stage.model_input": (
        "stage.train_ready_export",
        "stage.tensor_export",
    ),
    "stage.model_execution": (
        "stage.model_input",
    ),
}

ARTIFACT_PRODUCERS: dict[str, str | None] = {
    "artifact.source_annotations": None,
    "artifact.merged_frame_objects": "stage.legacy_cvat_source_merge",
    "artifact.frame_local_primitives": "stage.frame_local_primitives",
    "artifact.hidden_manifest": "stage.hidden_review_design",
    "artifact.hidden_decisions": "stage.hidden_coverage_scientific_gate",
    "artifact.hidden_reviewed_frames": "stage.hidden_apply",
    "artifact.temporal_intervals": "stage.temporal_harmonization",
    "artifact.harmonized_frames": "stage.temporal_harmonization",
    "artifact.native_evidence": "stage.native_review_evidence",
    "artifact.pig_strenet": "stage.pig_strenet_evidence",
    "artifact.behavior_review_units": (
        "stage.behavior_review_unit_construction"
    ),
    "artifact.behavior_decisions": "stage.behavior_gui",
    "artifact.reviewed_frames": "stage.behavior_decision_apply",
    "artifact.train_ready": "stage.train_ready_export",
    "artifact.spatial_tensor": "stage.tensor_export",
    "artifact.model_input": "stage.model_input",
    "artifact.model_output": "stage.model_execution",
}

ARTIFACT_MANIFEST_REQUIRED_FIELDS = (
    "artifact_manifest_version",
    "manifest_builder_id",
    "manifest_builder_version",
    "manifest_builder_code_hash",
    "authority_state",
    "candidate_transaction_id",
    "candidate_transaction_state",
    "candidate_transaction_provenance_hash",
    "artifact_id",
    "artifact_class",
    "stage_id",
    "stage_version",
    "code_authority_vcs",
    "code_authority_object_format",
    "created_by_code_authority_sha",
    "stage_code_hash",
    "stage_semantics_hash",
    "stage_input_fingerprint",
    "stage_execution_fingerprint",
    "execution_parameters_hash",
    "semantic_bundle_hash",
    "contract_manifest_hash",
    "input_artifact_ids",
    "input_artifact_fingerprints",
    "input_file_sha256",
    "output_path",
    "output_file_sha256",
    "output_byte_size",
    "output_inspector_id",
    "output_inspector_version",
    "output_schema_id",
    "output_schema_version",
    "output_schema_hash",
    "row_count",
    "column_count",
    "ordered_columns",
    "stage_specific_metadata",
    "feature_computation_grain",
    "pair_scope_key",
    "distance_metric_ids",
    "distance_metric_versions",
    "social_identity_version",
    "social_tie_break_version",
    "roi_aggregation_version",
    "motion_schema_id",
    "motion_schema_version",
    "motion_schema_hash",
    "human_decision_authority",
    "review_key_schema_version",
    "status",
    "validation_errors",
    "validation_warnings",
)

RELEASE_AUTHORIZATION_FIELDS = (
    "authorizes_source_rebuild",
    "authorizes_frame_local_rebuild",
    "authorizes_hidden_review",
    "authorizes_temporal_harmonization",
    "authorizes_native_evidence",
    "authorizes_pig_strenet",
    "authorizes_behavior_gui",
    "authorizes_final_windows",
    "authorizes_train_ready_export",
    "authorizes_training",
)

INVENTORY_STATUSES = frozenset(
    {
        "VALID_CURRENT_AUTHORITY",
        "STALE_SEMANTICS",
        "STALE_CODE",
        "STALE_INPUT",
        "MISSING_MANIFEST",
        "HASH_MISMATCH",
        "FAILED_DIAGNOSTIC",
        "NON_OFFICIAL_AUDIT",
        "HUMAN_DECISION_EVIDENCE",
        "UNKNOWN_NOT_PROMOTABLE",
    }
)

INVENTORY_SCOPE_ID = "scope.classification_v2.scientific_artifacts"
INVENTORY_SCOPE_VERSION = "classification_v2.inventory_scope.v1"
INVENTORY_INCLUDED_ROOTS = (
    "outputs/classification_v2",
    "human_review_workspace/classification_v2",
)
INVENTORY_EXCLUDE_PATTERNS = (
    "**/.staging/**",
    "**/*.staging",
    "**/*.manifest.json",
    "**/pytest_tmp/**",
    "**/pytest_upstream_tmp/**",
)

DECISION_CLASSIFICATIONS = frozenset(
    {
        "EXACT_CARRY_FORWARD_CANDIDATE",
        "REQUIRES_HUMAN_REVALIDATION",
        "OLD_ONLY_AUDIT_EVIDENCE",
        "NEW_ONLY_REQUIRES_REVIEW",
        "CONFLICT",
        "INVALID_DECISION_SCHEMA",
    }
)


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    """Comparable semantic, code and immutable-input authority hashes."""

    semantic_domain_hashes: dict[str, str]
    stage_code_hashes: dict[str, str]
    input_hashes: dict[str, str]


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _path_text(value: str, repo_root: str | Path | None) -> str:
    normalized = _normalize_newlines(value).replace("\\", "/")
    if repo_root is None:
        if re.match(r"^[A-Za-z]:/", normalized):
            raise ValueError(
                "absolute path requires repo_root for canonicalization"
            )
        return normalized
    root = str(repo_root).replace("\\", "/").rstrip("/")
    if normalized.casefold() == root.casefold():
        return "."
    prefix = root + "/"
    if normalized.casefold().startswith(prefix.casefold()):
        return normalized[len(prefix) :]
    if re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("/"):
        raise ValueError(f"path escapes canonical repo root: {value}")
    return normalized


def _looks_like_path(value: str) -> bool:
    if "\n" in value:
        return False
    return (
        "\\" in value
        or value.startswith(("./", "../", "/"))
        or bool(re.match(r"^[A-Za-z]:[/\\]", value))
    )


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("NaN and Infinity are forbidden")
    if value == 0:
        return "0"
    rendered = format(value.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _sort_token(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalize_canonical(
    value: Any,
    *,
    path: tuple[str, ...],
    repo_root: str | Path | None,
    unordered_paths: frozenset[tuple[str, ...]],
    exclude_fields: frozenset[str],
) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("NaN and Infinity are forbidden")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Decimal):
        return _decimal_text(value)
    if isinstance(value, Path):
        return _path_text(str(value), repo_root)
    if isinstance(value, str):
        normalized = _normalize_newlines(value)
        if _looks_like_path(normalized):
            return _path_text(normalized, repo_root)
        return normalized
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            key_text = str(key)
            if key_text in exclude_fields:
                continue
            result[key_text] = _normalize_canonical(
                nested,
                path=(*path, key_text),
                repo_root=repo_root,
                unordered_paths=unordered_paths,
                exclude_fields=exclude_fields,
            )
        return result
    if isinstance(value, (set, frozenset)):
        normalized_items = [
            _normalize_canonical(
                item,
                path=(*path, "[]"),
                repo_root=repo_root,
                unordered_paths=unordered_paths,
                exclude_fields=exclude_fields,
            )
            for item in value
        ]
        return sorted(normalized_items, key=_sort_token)
    if isinstance(value, (list, tuple)):
        normalized_items = [
            _normalize_canonical(
                item,
                path=(*path, "[]"),
                repo_root=repo_root,
                unordered_paths=unordered_paths,
                exclude_fields=exclude_fields,
            )
            for item in value
        ]
        if path in unordered_paths:
            return sorted(normalized_items, key=_sort_token)
        return normalized_items
    raise TypeError(f"unsupported canonical value type: {type(value)!r}")


def canonical_json_bytes(
    payload: Any,
    *,
    repo_root: str | Path | None = None,
    unordered_paths: Iterable[tuple[str, ...]] = (),
    exclude_fields: Iterable[str] = (),
) -> bytes:
    """Serialize an object deterministically without ambient machine state."""

    normalized = _normalize_canonical(
        payload,
        path=(),
        repo_root=repo_root,
        unordered_paths=frozenset(unordered_paths),
        exclude_fields=frozenset(exclude_fields),
    )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(
    payload: Any,
    *,
    repo_root: str | Path | None = None,
    unordered_paths: Iterable[tuple[str, ...]] = (),
    exclude_fields: Iterable[str] = (),
) -> str:
    """Return a SHA-256 over canonical UTF-8 JSON."""

    return hashlib.sha256(
        canonical_json_bytes(
            payload,
            repo_root=repo_root,
            unordered_paths=unordered_paths,
            exclude_fields=exclude_fields,
        )
    ).hexdigest()


def candidate_transaction_provenance_hash(
    transaction_id: str,
    transaction_state: str,
) -> str:
    """Bind candidate transaction state to versioned manifest authority."""

    return canonical_sha256(
        {
            "artifact_manifest_version": ARTIFACT_MANIFEST_VERSION,
            "manifest_builder_id": MANIFEST_BUILDER_ID,
            "manifest_builder_version": MANIFEST_BUILDER_VERSION,
            "candidate_transaction_id": transaction_id,
            "candidate_transaction_state": transaction_state,
        }
    )


def semantic_sha256(
    payload: Any,
    *,
    repo_root: str | Path | None = None,
    unordered_paths: Iterable[tuple[str, ...]] = (),
) -> str:
    """Hash semantic content while excluding generated metadata fields."""

    return canonical_sha256(
        payload,
        repo_root=repo_root,
        unordered_paths=unordered_paths,
        exclude_fields=EPHEMERAL_SEMANTIC_FIELDS,
    )


def semantic_hash_from_json_text(
    payload: str,
    *,
    repo_root: str | Path | None = None,
) -> str:
    """Hash parsed JSON so source whitespace is non-semantic."""

    return semantic_sha256(json.loads(payload), repo_root=repo_root)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_code_authority(
    repo_root: Path,
    *,
    object_id: str | None = None,
) -> dict[str, str]:
    """Resolve the repository Git object authority without fabricating a hash."""

    root = repo_root.resolve()
    object_format = _git_rev_parse(root, "--show-object-format")
    resolved_object_id = object_id or _git_rev_parse(root, "HEAD")
    authority = {
        "code_authority_vcs": CODE_AUTHORITY_VCS_GIT,
        "code_authority_object_format": object_format,
        "created_by_code_authority_sha": resolved_object_id,
    }
    errors = validate_git_code_authority(authority)
    if errors:
        raise ValueError(f"invalid Git code authority: {errors}")
    return authority


def validate_git_code_authority(
    authority: Mapping[str, Any],
) -> list[str]:
    """Validate Git object metadata separately from SHA-256 fingerprints."""

    errors: list[str] = []
    vcs = str(authority.get("code_authority_vcs", "")).strip().lower()
    object_format = str(
        authority.get("code_authority_object_format", "")
    ).strip().lower()
    object_id = str(
        authority.get("created_by_code_authority_sha", "")
    ).strip()
    if not vcs:
        errors.append("BLANK_CODE_AUTHORITY_VCS")
    elif vcs != CODE_AUTHORITY_VCS_GIT:
        errors.append(f"UNSUPPORTED_CODE_AUTHORITY_VCS:{vcs}")
    if not object_format:
        errors.append("BLANK_CODE_AUTHORITY_OBJECT_FORMAT")
    elif object_format not in SUPPORTED_GIT_OBJECT_FORMATS:
        errors.append(f"UNSUPPORTED_GIT_OBJECT_FORMAT:{object_format}")
    if not object_id:
        errors.append("BLANK_GIT_OBJECT_ID")
        return errors
    if not re.fullmatch(r"[0-9a-f]+", object_id):
        errors.append("INVALID_GIT_OBJECT_ID")
        return errors
    pattern = {
        "sha1": GIT_SHA1_PATTERN,
        "sha256": GIT_SHA256_PATTERN,
    }.get(object_format)
    if pattern is not None and not pattern.fullmatch(object_id):
        errors.append(
            "GIT_OBJECT_FORMAT_MISMATCH:"
            f"{object_format}:length={len(object_id)}"
        )
    return errors


def _git_rev_parse(repo_root: Path, argument: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", argument],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"unable to resolve Git authority {argument}"
        ) from exc
    value = completed.stdout.strip().lower()
    if not value:
        raise ValueError(f"Git authority {argument} is blank")
    return value


def load_scientific_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scientific contract root must be a mapping")
    return payload


def deterministic_topological_order(
    dependencies: Mapping[str, Sequence[str]],
    declared_order: Sequence[str],
) -> list[str]:
    """Return a stable topological order or fail on cycles/references."""

    declared = list(declared_order)
    declared_set = set(declared)
    if len(declared) != len(declared_set):
        raise ValueError("duplicate declared stage IDs")
    if set(dependencies) != declared_set:
        missing = sorted(declared_set - set(dependencies))
        extra = sorted(set(dependencies) - declared_set)
        raise ValueError(
            f"dependency stage mismatch: missing={missing}, extra={extra}"
        )
    position = {stage_id: index for index, stage_id in enumerate(declared)}
    indegree = {stage_id: 0 for stage_id in declared}
    children: dict[str, list[str]] = defaultdict(list)
    downstream_dependencies: list[tuple[str, str]] = []
    for stage_id, parents in dependencies.items():
        if len(parents) != len(set(parents)):
            raise ValueError(f"duplicate dependencies for {stage_id}")
        for parent in parents:
            if parent not in declared_set:
                raise ValueError(
                    f"{stage_id} depends on unknown stage {parent}"
                )
            if position[parent] >= position[stage_id]:
                downstream_dependencies.append((stage_id, parent))
            indegree[stage_id] += 1
            children[parent].append(stage_id)
    ready = sorted(
        (stage_id for stage_id, count in indegree.items() if count == 0),
        key=position.__getitem__,
    )
    result: list[str] = []
    while ready:
        stage_id = ready.pop(0)
        result.append(stage_id)
        for child in sorted(
            children.get(stage_id, []),
            key=position.__getitem__,
        ):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=position.__getitem__)
    if len(result) != len(declared):
        raise ValueError("stage dependency graph contains a cycle")
    if downstream_dependencies:
        raise ValueError(
            "stage depends on downstream stage: "
            f"{downstream_dependencies}"
        )
    return result


def transitive_descendants(
    dependencies: Mapping[str, Sequence[str]],
    start_stages: Iterable[str],
) -> set[str]:
    children: dict[str, set[str]] = defaultdict(set)
    for child, parents in dependencies.items():
        for parent in parents:
            children[parent].add(child)
    result = set(start_stages)
    frontier = list(result)
    while frontier:
        stage_id = frontier.pop()
        for child in children.get(stage_id, set()):
            if child not in result:
                result.add(child)
                frontier.append(child)
    return result


def validate_stage_dependency_graph(
    contract: Mapping[str, Any],
    dependencies: Mapping[str, Sequence[str]] = STAGE_DEPENDENCIES,
) -> dict[str, Any]:
    """Validate exact contract stages, DAG order and artifact authorities."""

    stage_rows = list(contract["stages"])
    stage_ids = [str(row["stage_id"]) for row in stage_rows]
    artifact_ids = {
        str(row["artifact_id"]) for row in contract["artifacts"]
    }
    errors: list[str] = []
    try:
        order = deterministic_topological_order(dependencies, stage_ids)
    except ValueError as exc:
        errors.append(str(exc))
        order = []
    if set(ARTIFACT_PRODUCERS) != artifact_ids:
        errors.append(
            "artifact producer mapping mismatch: "
            f"missing={sorted(artifact_ids - set(ARTIFACT_PRODUCERS))},"
            f"extra={sorted(set(ARTIFACT_PRODUCERS) - artifact_ids)}"
        )
    stage_by_id = {str(row["stage_id"]): row for row in stage_rows}
    for artifact_id, producer in ARTIFACT_PRODUCERS.items():
        if producer is None:
            if artifact_id != "artifact.source_annotations":
                errors.append(f"unexpected external artifact={artifact_id}")
            continue
        if producer not in stage_by_id:
            errors.append(
                f"artifact={artifact_id}:unknown_producer={producer}"
            )
        elif artifact_id not in stage_by_id[producer]["output_artifacts"]:
            errors.append(
                f"artifact={artifact_id}:producer_output_mismatch={producer}"
            )
    reachable = set()
    if order:
        reachable = transitive_descendants(
            dependencies,
            {"stage.legacy_cvat_source_merge"},
        )
        if reachable != set(stage_ids):
            errors.append(
                "stages unreachable from immutable source="
                f"{sorted(set(stage_ids) - reachable)}"
            )
    return {
        "graph_version": STAGE_GRAPH_VERSION,
        "declared_stage_count": len(stage_ids),
        "implemented_stage_count": len(dependencies),
        "topological_order": order,
        "artifact_producers": dict(ARTIFACT_PRODUCERS),
        "errors": errors,
        "valid": not errors,
    }


def build_stage_dependency_graph(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_stage_dependency_graph(contract)
    if not validation["valid"]:
        raise ValueError(
            f"invalid stage dependency graph: {validation['errors']}"
        )
    stage_by_id = {
        str(stage["stage_id"]): stage for stage in contract["stages"]
    }
    nodes = []
    for stage_id in validation["topological_order"]:
        stage = stage_by_id[stage_id]
        nodes.append(
            {
                "stage_id": stage_id,
                "stage_version": stage["schema_version"],
                "depends_on": list(STAGE_DEPENDENCIES[stage_id]),
                "input_artifacts": list(stage["input_artifacts"]),
                "output_artifacts": list(stage["output_artifacts"]),
            }
        )
    return {
        **validation,
        "nodes": nodes,
    }


def _stage_schema_version(
    contract: Mapping[str, Any],
    stage_id: str,
) -> str:
    for stage in contract["stages"]:
        if stage["stage_id"] == stage_id:
            return str(stage["schema_version"])
    raise KeyError(stage_id)


def _domain_specs(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    def stage_version(value: str) -> str:
        return _stage_schema_version(contract, value)

    return [
        {
            "semantic_domain_id": "semantic.source_parsing_selection",
            "semantic_domain_version": "classification_v2.source_semantics.v6",
            "authority_files": [
                "src/pig_behavior/classification_v2/merge_sources.py",
                (
                    "src/pig_behavior/classification_v2/contracts/"
                    "identifiers.py"
                ),
                (
                    "docs/classification_v2/scientific_contract_v1/"
                    "00_pipeline_contract.yaml"
                ),
                "docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md",
            ],
            "authority_symbols": [
                "merge_frame_object_sources",
                "ensure_object_track_keys",
            ],
            "canonical_payload": {
                "stage_schema": stage_version(
                    "stage.legacy_cvat_source_merge"
                ),
                "authority_selection": (
                    "declared_source_allowlist_and_one_native_authority"
                ),
                "object_track_key_authority": dict(
                    contract["object_track_key_contract"]
                ),
                "positional_matching": False,
            },
            "directly_affected_stages": [
                "stage.legacy_cvat_source_merge",
            ],
            "human_decision_implications": [
                "source identity change requires review-key revalidation",
            ],
            "notes": "Immutable source bytes are separate input authority.",
        },
        {
            "semantic_domain_id": "semantic.frame_local_geometry",
            "semantic_domain_version": "classification_v2.geometry.v4",
            "authority_files": [
                "src/pig_behavior/classification_v2/features/frame_local.py",
                "src/pig_behavior/classification_v2/features/geometry.py",
            ],
            "authority_symbols": [
                "build_frame_local_primitives",
                "build_geometry_features",
            ],
            "canonical_payload": {
                "stage_schema": stage_version(
                    "stage.frame_local_primitives"
                ),
                "coordinate_system": "image_pixel_and_axis_normalized",
                "grain": "FRAME_LOCAL_PRIMITIVES",
                "temporal_features_forbidden": True,
            },
            "directly_affected_stages": [
                "stage.frame_local_primitives",
            ],
            "human_decision_implications": [
                "visual crop authority must be revalidated",
            ],
            "notes": "Frame-local geometry contains no temporal pairing.",
        },
        {
            "semantic_domain_id": "semantic.image_distance_metrics",
            "semantic_domain_version": "classification_v2.distance_metrics.v1",
            "authority_files": [
                "src/pig_behavior/classification_v2/features/spatial_semantics.py",
                "src/pig_behavior/classification_v2/features/social.py",
            ],
            "authority_symbols": [
                "AXIS_DISTANCE_CONTRACT",
                "DIAGONAL_DISTANCE_CONTRACT",
            ],
            "canonical_payload": {
                "axis_metric_id": AXIS_DISTANCE_METRIC_ID,
                "axis_metric_version": AXIS_DISTANCE_METRIC_VERSION,
                "axis_formula": "sqrt((dx_px/W_px)^2+(dy_px/H_px)^2)",
                "diagonal_metric_id": DIAGONAL_DISTANCE_METRIC_ID,
                "diagonal_metric_version": DIAGONAL_DISTANCE_METRIC_VERSION,
                "diagonal_formula": (
                    "sqrt(dx_px^2+dy_px^2)/sqrt(W_px^2+H_px^2)"
                ),
                "physical_distance": False,
                "social_threshold": {
                    "threshold_id": SOCIAL_NEAR_THRESHOLD_ID,
                    "value": Decimal(str(SOCIAL_NEAR_THRESHOLD_VALUE)),
                    "units": SOCIAL_NEAR_THRESHOLD_UNITS,
                    "metric_id": AXIS_DISTANCE_METRIC_ID,
                },
            },
            "directly_affected_stages": [
                "stage.frame_local_primitives",
            ],
            "human_decision_implications": [
                "distance-derived review sampling requires revalidation",
            ],
            "notes": "Neither image metric is a physical world distance.",
        },
        {
            "semantic_domain_id": "semantic.social_identity_tie_break",
            "semantic_domain_version": "classification_v2.social_semantics.v1",
            "authority_files": [
                "src/pig_behavior/classification_v2/features/spatial_semantics.py",
                "src/pig_behavior/classification_v2/features/social.py",
            ],
            "authority_symbols": [
                "canonical_social_identity",
                "SOCIAL_TIE_BREAK_RULE",
            ],
            "canonical_payload": {
                "identity_version": SOCIAL_IDENTITY_VERSION,
                "identity_hierarchy": [
                    "object_track_key",
                    "scoped_track_id",
                    "scoped_object_id",
                ],
                "pig_id_role": "descriptive_metadata_only",
                "tie_break_version": SOCIAL_TIE_BREAK_VERSION,
                "tie_break_rule": SOCIAL_TIE_BREAK_RULE,
                "self_neighbor_forbidden": True,
            },
            "directly_affected_stages": [
                "stage.frame_local_primitives",
                "stage.native_review_evidence",
            ],
            "human_decision_implications": [
                "partner-derived review evidence requires revalidation",
            ],
            "notes": "Raw row order never resolves equal-distance ties.",
        },
        {
            "semantic_domain_id": "semantic.roi_computation_aggregation",
            "semantic_domain_version": "classification_v2.roi_semantics.v5",
            "authority_files": [
                (
                    "src/pig_behavior/classification_v2/contracts/"
                    "target_roi_policy.py"
                ),
                "src/pig_behavior/classification_v2/features/roi.py",
                "src/pig_behavior/classification_v2/features/spatial_semantics.py",
                "src/pig_behavior/classification_v2/features/spatiotemporal.py",
            ],
            "authority_symbols": [
                "build_roi_features",
                "ROI_AGGREGATION_VERSION",
                "target_roi_model_policy_registry",
            ],
            "canonical_payload": {
                "aggregation_version": ROI_AGGREGATION_VERSION,
                "contact_denominator": "roi_available_frames",
                "zero_available_behavior": "unavailable_with_mask_false",
                "target_roi_policy_version": ROI_TARGET_MODEL_POLICY_VERSION,
                "target_roi_policy_id": TARGET_ROI_SHARED_POLICY_ID,
                "target_roi_policy": target_roi_model_policy_registry(),
                "target_roi_model_eligible": False,
            },
            "directly_affected_stages": [
                "stage.frame_local_primitives",
                "stage.native_review_evidence",
            ],
            "human_decision_implications": [
                "ROI-dependent review evidence requires revalidation",
            ],
            "notes": "Label-selected target ROI remains model-forbidden.",
        },
        {
            "semantic_domain_id": "semantic.hidden_selection",
            "semantic_domain_version": "classification_v2.hidden_selection.v4",
            "authority_files": [
                "src/pig_behavior/classification_v2/review/hidden_review_builder.py",
                "src/pig_behavior/classification_v2/review/hidden_review_science.py",
            ],
            "authority_symbols": [
                "build_hidden_review_manifest",
                "evaluate_hidden_scientific_gate",
            ],
            "canonical_payload": {
                "manifest_schema": stage_version(
                    "stage.hidden_review_design"
                ),
                "cohorts": [
                    "census",
                    "target_independent_high_risk",
                    "stratified_random",
                    "clean_control",
                ],
                "behavior_target_in_risk": False,
                "adjacent_frame_delta": 1,
            },
            "directly_affected_stages": [
                "stage.hidden_review_design",
                "stage.hidden_gui",
                "stage.hidden_coverage_scientific_gate",
            ],
            "human_decision_implications": [
                "exact media-bound carry-forward only",
            ],
            "notes": "Selection changes do not auto-delete human decisions.",
        },
        {
            "semantic_domain_id": "semantic.hidden_decision_application",
            "semantic_domain_version": "classification_v2.hidden_apply.v4",
            "authority_files": [
                "src/pig_behavior/classification_v2/review/hidden_review_builder.py",
                "src/pig_behavior/classification_v2/review/hidden_review_migration.py",
            ],
            "authority_symbols": [
                "apply_hidden_review_decisions",
                "carry_forward_hidden_review_decisions",
            ],
            "canonical_payload": {
                "apply_schema": stage_version("stage.hidden_apply"),
                "row_preserving": True,
                "matching": "stable_key_and_invariant_media_identity",
                "positional_matching": False,
                "carry_forward_version": HIDDEN_CARRY_FORWARD_VERSION,
            },
            "directly_affected_stages": [
                "stage.hidden_decision_migration",
                "stage.hidden_apply",
            ],
            "human_decision_implications": [
                "changed media or span requires human revalidation",
            ],
            "notes": "Human payload is not regenerated automatically.",
        },
        {
            "semantic_domain_id": "semantic.temporal_harmonization",
            "semantic_domain_version": (
                "classification_v2.temporal_harmonization.v4"
            ),
            "authority_files": [
                "src/pig_behavior/classification_v2/features/temporal_harmonization.py",
                "src/pig_behavior/classification_v2/sources/temporal_provenance.py",
            ],
            "authority_symbols": [
                "harmonize_temporal_labels",
                "apply_source_frame_clock",
            ],
            "canonical_payload": {
                "stage_schema": stage_version(
                    "stage.temporal_harmonization"
                ),
                "pair_reset_key": "temporal_unit_key",
                "legacy_frames_per_unit": 16,
                "cvat_frames_per_unit": 6,
                "source_safe_timing": True,
            },
            "directly_affected_stages": [
                "stage.temporal_harmonization",
            ],
            "human_decision_implications": [
                "behavior span identity must be revalidated",
            ],
            "notes": "No pair crosses a temporal unit.",
        },
        {
            "semantic_domain_id": "semantic.native_temporal_pairs",
            "semantic_domain_version": NATIVE_EVIDENCE_SEMANTICS_VERSION,
            "authority_files": [
                "src/pig_behavior/classification_v2/features/spatiotemporal.py",
                "src/pig_behavior/classification_v2/features/native_evidence_contract.py",
            ],
            "authority_symbols": [
                "_add_temporal_deltas",
                "check_native_review_evidence",
            ],
            "canonical_payload": {
                "feature_computation_grain": (
                    NATIVE_FEATURE_COMPUTATION_GRAIN
                ),
                "pair_scope_key": NATIVE_PAIR_SCOPE_KEY,
                "valid_pair_requires": [
                    "previous_observation",
                    "same_temporal_unit_key",
                    "same_canonical_actor",
                    "valid_previous_geometry",
                    "valid_current_geometry",
                    "finite_positive_delta_t",
                    "monotonic_timestamp",
                ],
                "invalid_pair_is_measured_zero": False,
                "aggregate_denominator": "valid_pairs_only",
            },
            "directly_affected_stages": [
                "stage.native_review_evidence",
            ],
            "human_decision_implications": [
                "native evidence and behavior review units become stale",
            ],
            "notes": "Phase 1 temporal-pair contract is frozen.",
        },
        {
            "semantic_domain_id": "semantic.motion_tensor_schema",
            "semantic_domain_version": MOTION_SCHEMA_VERSION,
            "authority_files": [
                "src/pig_behavior/classification_v2/features/motion_schema.py",
                "docs/classification_v2/scientific_contract_v1/05_tensor_schema_manifest.json",
            ],
            "authority_symbols": [
                "MOTION_FEATURE_NAMES",
                "MOTION_SCHEMA_HASH",
            ],
            "canonical_payload": {
                "schema_id": MOTION_SCHEMA_ID,
                "schema_version": MOTION_SCHEMA_VERSION,
                "schema_hash": MOTION_SCHEMA_HASH,
                "ordered_feature_names": list(MOTION_FEATURE_NAMES),
                "dimension": len(MOTION_FEATURE_NAMES),
                "acceleration_time": (
                    "centered_interval_average_velocity_midpoints"
                ),
                "zero_speed_direction_available": False,
            },
            "directly_affected_stages": [
                "stage.native_review_evidence",
                "stage.tensor_export",
            ],
            "human_decision_implications": [
                "motion-derived review evidence requires revalidation",
            ],
            "notes": "Feature order is semantic and order-sensitive.",
        },
        {
            "semantic_domain_id": "semantic.pig_strenet_evidence",
            "semantic_domain_version": "classification_v2.pig_strenet.v4",
            "authority_files": [
                "src/pig_behavior/classification_v2/features/pig_strenet_artifacts.py",
            ],
            "authority_symbols": ["build_pig_strenet_artifacts"],
            "canonical_payload": {
                "stage_schema": stage_version(
                    "stage.pig_strenet_evidence"
                ),
                "history_target_complete_required": True,
                "join_key": "temporal_unit_key",
                "pair_labels_ignored": True,
            },
            "directly_affected_stages": [
                "stage.pig_strenet_evidence",
            ],
            "human_decision_implications": [
                "downstream behavior review evidence becomes stale",
            ],
            "notes": "Pre-motion-fix evidence is not reusable.",
        },
        {
            "semantic_domain_id": "semantic.behavior_review_units",
            "semantic_domain_version": (
                "classification_v2.behavior_review_units.v4"
            ),
            "authority_files": [
                "src/pig_behavior/classification_v2/review/review_unit_builder.py",
            ],
            "authority_symbols": ["build_review_units"],
            "canonical_payload": {
                "stage_schema": stage_version(
                    "stage.behavior_review_unit_construction"
                ),
                "unit_key": "review_unit_id",
                "native_unit_key": "temporal_unit_key",
                "review_fields_model_forbidden": True,
            },
            "directly_affected_stages": [
                "stage.behavior_review_unit_construction",
                "stage.behavior_gui",
            ],
            "human_decision_implications": [
                "exact review-unit/media carry-forward only",
            ],
            "notes": "Review selection is not model input.",
        },
        {
            "semantic_domain_id": "semantic.behavior_decision_application",
            "semantic_domain_version": "classification_v2.behavior_apply.v4",
            "authority_files": [
                (
                    "scripts/classification_v2/01_review_units_gui/"
                    "classification_v2_apply_review_unit_decisions.py"
                ),
            ],
            "authority_symbols": ["main"],
            "canonical_payload": {
                "stage_schema": stage_version(
                    "stage.behavior_decision_apply"
                ),
                "row_preserving": True,
                "matching": "stable_review_unit_and_media_authority",
                "positional_matching": False,
                "carry_forward_version": BEHAVIOR_CARRY_FORWARD_VERSION,
            },
            "directly_affected_stages": [
                "stage.behavior_decision_apply",
            ],
            "human_decision_implications": [
                "changed unit or media requires human revalidation",
            ],
            "notes": "New-only units are never auto-accepted.",
        },
        {
            "semantic_domain_id": "semantic.final_view_windows",
            "semantic_domain_version": "classification_v2.final_views.v4",
            "authority_files": [
                "src/pig_behavior/classification_v2/train_ready_features.py",
                (
                    "scripts/classification_v2/00_source_feature_temporal/"
                    "classification_v2_build_sequence_windows.py"
                ),
            ],
            "authority_symbols": [
                "build_train_ready_window_tables",
                "main",
            ],
            "canonical_payload": {
                "stage_schema": stage_version(
                    "stage.train_ready_export"
                ),
                "views": ["T6", "T8", "T12", "T16", "S6@16"],
                "pair_reset_key": "window_id",
                "pair_features_recomputed_per_view": True,
            },
            "directly_affected_stages": [
                "stage.train_ready_export",
            ],
            "human_decision_implications": [
                "reviewed frames remain evidence; windows must rebuild",
            ],
            "notes": "Window order and offsets are semantic.",
        },
        {
            "semantic_domain_id": "semantic.model_input_export",
            "semantic_domain_version": "classification_v2.model_export.v5",
            "authority_files": [
                "src/pig_behavior/classification_v2/spatial_sequence_export.py",
                "src/pig_behavior/classification_v2/contracts/model_input_manifest.py",
                (
                    "src/pig_behavior/classification_v2/contracts/"
                    "target_roi_policy.py"
                ),
            ],
            "authority_symbols": [
                "export_spatial_sequences",
                "build_model_input_manifest",
                "target_roi_model_policy_registry",
            ],
            "canonical_payload": {
                "tensor_stage_schema": stage_version(
                    "stage.tensor_export"
                ),
                "model_input_stage_schema": stage_version(
                    "stage.model_input"
                ),
                "motion_schema_id": MOTION_SCHEMA_ID,
                "motion_schema_hash": MOTION_SCHEMA_HASH,
                "explicit_whitelist_only": True,
                "forbidden_columns_fail_closed": True,
                "target_roi_policy": target_roi_model_policy_registry(),
            },
            "directly_affected_stages": [
                "stage.tensor_export",
                "stage.model_input",
            ],
            "human_decision_implications": [
                "no direct invalidation of reviewed frame authority",
            ],
            "notes": "Exporter-only changes do not invalidate frame data.",
        },
        {
            "semantic_domain_id": "semantic.split_leakage",
            "semantic_domain_version": "classification_v2.split_leakage.v4",
            "authority_files": [
                "src/pig_behavior/classification_v2/contracts/model_input_manifest.py",
                (
                    "src/pig_behavior/classification_v2/contracts/"
                    "target_roi_policy.py"
                ),
                "src/pig_behavior/classification_v2/train_ready_features.py",
            ],
            "authority_symbols": [
                "build_model_input_manifest",
                "select_window_feature_columns",
                "target_roi_model_policy_registry",
            ],
            "canonical_payload": {
                "grouping": [
                    "recording_date",
                    "session",
                    "video",
                    "temporal_unit_key",
                ],
                "frame_random_split": False,
                "fold_local_fit": True,
                "review_fields_model_forbidden": True,
                "target_roi_model_forbidden": True,
            },
            "directly_affected_stages": [
                "stage.model_input",
            ],
            "human_decision_implications": [
                "no review decision invalidation",
            ],
            "notes": "Split changes invalidate model input and execution.",
        },
        {
            "semantic_domain_id": "semantic.train_ready_release",
            "semantic_domain_version": (
                "classification_v2.train_ready_release.v5"
            ),
            "authority_files": [
                "src/pig_behavior/classification_v2/contracts/semantic_lineage.py",
                "docs/CLASSIFICATION_V2_DATA_REBUILD_AND_HUMAN_REVIEW_RUNBOOK.md",
            ],
            "authority_symbols": [
                "build_release_authority_preflight",
                "validate_artifact_manifest",
            ],
            "canonical_payload": {
                "artifact_manifest_version": ARTIFACT_MANIFEST_VERSION,
                "manifest_builder": {
                    "id": MANIFEST_BUILDER_ID,
                    "version": MANIFEST_BUILDER_VERSION,
                    "candidate_state": CANDIDATE_AUTHORITY_STATE,
                    "official_state": OFFICIAL_AUTHORITY_STATE,
                    "load_bearing_caller_hashes_trusted": False,
                },
                "code_authority": {
                    "vcs": CODE_AUTHORITY_VCS_GIT,
                    "object_formats": sorted(
                        SUPPORTED_GIT_OBJECT_FORMATS
                    ),
                    "git_object_id_is_content_sha256": False,
                },
                "release_schema_version": (
                    RELEASE_AUTHORITY_SCHEMA_VERSION
                ),
                "manifest_promoted_last": True,
                "unsigned_phase4_authorizes_rebuild": False,
                "output_without_manifest_authoritative": False,
            },
            "directly_affected_stages": [
                "stage.model_input",
                "stage.model_execution",
            ],
            "human_decision_implications": [
                "human sign-off is prerequisite, never inferred",
            ],
            "notes": "Phase 4 audit evidence is non-official.",
        },
    ]


def build_semantic_domain_registry(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build 17 separately inspectable semantic domains and hashes."""

    graph = build_stage_dependency_graph(contract)
    stage_ids = set(graph["topological_order"])
    domains = []
    for spec in _domain_specs(contract):
        direct = list(spec["directly_affected_stages"])
        if not direct or not set(direct).issubset(stage_ids):
            raise ValueError(
                f"{spec['semantic_domain_id']}: invalid direct stages"
            )
        transitive = sorted(
            transitive_descendants(STAGE_DEPENDENCIES, direct),
            key=graph["topological_order"].index,
        )
        canonical_payload = spec["canonical_payload"]
        canonical_hash = semantic_sha256(canonical_payload)
        domains.append(
            {
                **spec,
                "authority_files": sorted(set(spec["authority_files"])),
                "authority_symbols": sorted(
                    set(spec["authority_symbols"])
                ),
                "directly_affected_stages": direct,
                "transitively_affected_stages": transitive,
                "canonical_payload": canonical_payload,
                "canonical_hash": canonical_hash,
            }
        )
    ids = [domain["semantic_domain_id"] for domain in domains]
    if len(domains) != 17 or len(ids) != len(set(ids)):
        raise ValueError("semantic domain registry must contain 17 unique IDs")
    return {
        "semantic_registry_version": SEMANTIC_REGISTRY_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "semantic_domain_count": len(domains),
        "semantic_domains": domains,
    }


def build_semantic_bundle(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    domain_hashes = {
        str(domain["semantic_domain_id"]): str(domain["canonical_hash"])
        for domain in registry["semantic_domains"]
    }
    payload = {
        "semantic_bundle_id": SEMANTIC_BUNDLE_ID,
        "semantic_bundle_version": SEMANTIC_BUNDLE_VERSION,
        "semantic_domain_hashes": domain_hashes,
    }
    return {
        **payload,
        "semantic_bundle_hash": semantic_sha256(payload),
    }


def load_code_contract_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stage_code_files(
    stage_id: str,
    mapping_rows: Sequence[Mapping[str, str]],
) -> list[str]:
    files = sorted(
        {
            str(row["source_file"]).replace("\\", "/")
            for row in mapping_rows
            if row.get("contract_item_type") == "stage"
            and row.get("contract_item_id") == stage_id
        }
    )
    if not files:
        raise ValueError(f"stage has no production code mapping: {stage_id}")
    forbidden = [
        path
        for path in files
        if path.startswith(("tests/", "outputs/"))
        or "__pycache__" in path
    ]
    if forbidden:
        raise ValueError(
            f"stage code mapping contains non-production files: {forbidden}"
        )
    return files


def compute_stage_code_hash(
    repo_root: Path,
    stage_id: str,
    mapping_rows: Sequence[Mapping[str, str]],
    *,
    file_overrides: Mapping[str, bytes] | None = None,
) -> str:
    """Hash only production blobs mapped to one exact contract stage."""

    assert_stage_runtime_dependencies_complete(
        repo_root,
        stage_id,
        mapping_rows,
    )
    overrides = file_overrides or {}
    blobs = []
    for relative in stage_code_files(stage_id, mapping_rows):
        content = overrides.get(relative)
        if content is None:
            source = (repo_root / PurePosixPath(relative)).resolve()
            try:
                source.relative_to(repo_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"stage code path escapes repo root: {relative}"
                ) from exc
            if not source.is_file():
                raise FileNotFoundError(source)
            content = source.read_bytes()
        blobs.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return canonical_sha256(
        {
            "stage_id": stage_id,
            "production_code_blobs": blobs,
        }
    )


def compute_all_stage_code_hashes(
    repo_root: Path,
    stage_ids: Sequence[str],
    mapping_rows: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    return {
        stage_id: compute_stage_code_hash(
            repo_root,
            stage_id,
            mapping_rows,
        )
        for stage_id in stage_ids
    }


def compute_stage_semantics_hash(
    stage_id: str,
    registry: Mapping[str, Any],
) -> str:
    domain_hashes = {
        domain["semantic_domain_id"]: domain["canonical_hash"]
        for domain in registry["semantic_domains"]
        if stage_id in domain["directly_affected_stages"]
    }
    if not domain_hashes:
        raise ValueError(f"stage has no semantic authority: {stage_id}")
    return canonical_sha256(
        {
            "stage_id": stage_id,
            "semantic_domain_hashes": domain_hashes,
        }
    )


def compute_stage_input_fingerprint(
    input_artifact_fingerprints: Mapping[str, str],
) -> str:
    for artifact_id, fingerprint in input_artifact_fingerprints.items():
        if not artifact_id or not HASH_PATTERN.fullmatch(fingerprint):
            raise ValueError(
                f"invalid input fingerprint: {artifact_id}={fingerprint}"
            )
    return canonical_sha256(
        {
            "input_artifact_fingerprints": dict(
                sorted(input_artifact_fingerprints.items())
            )
        }
    )


def compute_stage_execution_fingerprint(
    *,
    stage_id: str,
    stage_version: str,
    stage_code_hash: str,
    stage_semantics_hash: str,
    stage_input_fingerprint: str,
    schema_hashes: Mapping[str, str],
) -> str:
    for name, value in {
        "stage_code_hash": stage_code_hash,
        "stage_semantics_hash": stage_semantics_hash,
        "stage_input_fingerprint": stage_input_fingerprint,
        **schema_hashes,
    }.items():
        if not HASH_PATTERN.fullmatch(value):
            raise ValueError(f"invalid execution hash {name}={value}")
    return canonical_sha256(
        {
            "stage_id": stage_id,
            "stage_version": stage_version,
            "stage_code_hash": stage_code_hash,
            "stage_semantics_hash": stage_semantics_hash,
            "stage_input_fingerprint": stage_input_fingerprint,
            "schema_hashes": dict(sorted(schema_hashes.items())),
        }
    )


def build_stage_authority_registry(
    *,
    repo_root: Path,
    contract: Mapping[str, Any],
    mapping_rows: Sequence[Mapping[str, str]],
    semantic_registry: Mapping[str, Any],
) -> dict[str, Any]:
    graph = build_stage_dependency_graph(contract)
    code_hashes = compute_all_stage_code_hashes(
        repo_root,
        graph["topological_order"],
        mapping_rows,
    )
    stage_by_id = {
        str(stage["stage_id"]): stage for stage in contract["stages"]
    }
    artifact_fingerprints = {
        "artifact.source_annotations": canonical_sha256(
            {
                "artifact_id": "artifact.source_annotations",
                "status": "NOT_BOUND_NO_OFFICIAL_PHASE4_REBUILD",
            }
        )
    }
    stages = []
    for stage_id in graph["topological_order"]:
        stage = stage_by_id[stage_id]
        input_fingerprints = {}
        for artifact_id in stage["input_artifacts"]:
            if artifact_id not in artifact_fingerprints:
                producer = ARTIFACT_PRODUCERS.get(artifact_id)
                if producer is None:
                    raise ValueError(
                        f"missing external input fingerprint={artifact_id}"
                    )
                input_fingerprints[artifact_id] = canonical_sha256(
                    {
                        "artifact_id": artifact_id,
                        "producer_stage_id": producer,
                        "status": "NOT_OFFICIALLY_REBUILT",
                    }
                )
            else:
                input_fingerprints[artifact_id] = artifact_fingerprints[
                    artifact_id
                ]
        input_fingerprint = compute_stage_input_fingerprint(
            input_fingerprints
        )
        semantics_hash = compute_stage_semantics_hash(
            stage_id,
            semantic_registry,
        )
        schema_hashes = {"contract_stage_schema_hash": canonical_sha256(
            {
                "stage_id": stage_id,
                "schema_version": stage["schema_version"],
            }
        )}
        if stage_id in {
            "stage.native_review_evidence",
            "stage.tensor_export",
            "stage.model_input",
            "stage.model_execution",
        }:
            schema_hashes["motion_schema_hash"] = MOTION_SCHEMA_HASH
        execution = compute_stage_execution_fingerprint(
            stage_id=stage_id,
            stage_version=str(stage["schema_version"]),
            stage_code_hash=code_hashes[stage_id],
            stage_semantics_hash=semantics_hash,
            stage_input_fingerprint=input_fingerprint,
            schema_hashes=schema_hashes,
        )
        for artifact_id in stage["output_artifacts"]:
            if ARTIFACT_PRODUCERS.get(artifact_id) == stage_id:
                artifact_fingerprints[artifact_id] = canonical_sha256(
                    {
                        "artifact_id": artifact_id,
                        "producer_stage_execution_fingerprint": execution,
                        "status": "EXPECTED_AUTHORITY_NOT_PROMOTED",
                    }
                )
        stages.append(
            {
                "stage_id": stage_id,
                "stage_version": stage["schema_version"],
                "production_code_files": stage_code_files(
                    stage_id,
                    mapping_rows,
                ),
                "stage_code_hash": code_hashes[stage_id],
                "stage_semantics_hash": semantics_hash,
                "input_artifact_fingerprints": input_fingerprints,
                "stage_input_fingerprint": input_fingerprint,
                "schema_hashes": schema_hashes,
                "stage_execution_fingerprint": execution,
            }
        )
    return {
        "stage_authority_registry_version": (
            "classification_v2.stage_authorities.v4"
        ),
        "stages": stages,
    }


def build_authority_snapshot(
    *,
    semantic_registry: Mapping[str, Any],
    stage_authority_registry: Mapping[str, Any],
    input_hashes: Mapping[str, str] | None = None,
) -> AuthoritySnapshot:
    return AuthoritySnapshot(
        semantic_domain_hashes={
            domain["semantic_domain_id"]: domain["canonical_hash"]
            for domain in semantic_registry["semantic_domains"]
        },
        stage_code_hashes={
            stage["stage_id"]: stage["stage_code_hash"]
            for stage in stage_authority_registry["stages"]
        },
        input_hashes=dict(input_hashes or {}),
    )


def historical_pre_remediation_snapshot(
    current: AuthoritySnapshot,
) -> AuthoritySnapshot:
    """Represent the declared pre-Phase-1–3 semantics for comparison."""

    historical_payloads = {
        "semantic.image_distance_metrics": {
            "version": "pre_phase3_unversioned_axis_distance",
            "distance_metric_version": None,
            "physical_limit_explicit": False,
        },
        "semantic.social_identity_tie_break": {
            "version": "pre_phase3_row_order_and_pig_id",
            "identity": "pig_id",
            "tie_break": "row_order",
        },
        "semantic.roi_computation_aggregation": {
            "version": "pre_phase3_all_frame_denominator",
            "contact_denominator": "all_observed_frames",
        },
        "semantic.native_temporal_pairs": {
            "version": "pre_phase1_numeric_zero_invalid_pair",
            "invalid_pair_is_measured_zero": True,
        },
        "semantic.motion_tensor_schema": {
            "version": "pre_phase2_variable_dimension",
            "dimension": "runtime_available_columns",
        },
    }
    domain_hashes = dict(current.semantic_domain_hashes)
    for domain_id, payload in historical_payloads.items():
        if domain_id not in domain_hashes:
            raise ValueError(
                f"historical domain missing from registry: {domain_id}"
            )
        domain_hashes[domain_id] = semantic_sha256(payload)
    return AuthoritySnapshot(
        semantic_domain_hashes=domain_hashes,
        stage_code_hashes=dict(current.stage_code_hashes),
        input_hashes=dict(current.input_hashes),
    )


def change_impact_registry(
    semantic_registry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "change_impact_registry_version": (
            CHANGE_IMPACT_REGISTRY_VERSION
        ),
        "semantic_domain_to_stages": {
            domain["semantic_domain_id"]: list(
                domain["directly_affected_stages"]
            )
            for domain in semantic_registry["semantic_domains"]
        },
        "input_authority_to_stage": {
            "input.source_annotations": "stage.legacy_cvat_source_merge",
            "input.hidden_decisions": (
                "stage.hidden_coverage_scientific_gate"
            ),
            "input.behavior_decisions": "stage.behavior_decision_apply",
        },
        "non_semantic_authority_classes": [
            "documentation_only",
            "tests_only",
            "generated_audits",
            "caches",
        ],
    }


def compute_earliest_rebuild_stage(
    previous_authority: AuthoritySnapshot,
    current_authority: AuthoritySnapshot,
    artifact_inventory: Sequence[Mapping[str, Any]],
    dependency_graph: Mapping[str, Any],
    *,
    impact_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the earliest safe rebuild stage from authority differences."""

    order = list(dependency_graph["topological_order"])
    domain_to_stages = impact_registry["semantic_domain_to_stages"]
    input_to_stage = impact_registry["input_authority_to_stage"]
    changed_domains = sorted(
        domain_id
        for domain_id in (
            set(previous_authority.semantic_domain_hashes)
            | set(current_authority.semantic_domain_hashes)
        )
        if previous_authority.semantic_domain_hashes.get(domain_id)
        != current_authority.semantic_domain_hashes.get(domain_id)
    )
    changed_code = sorted(
        stage_id
        for stage_id in (
            set(previous_authority.stage_code_hashes)
            | set(current_authority.stage_code_hashes)
        )
        if previous_authority.stage_code_hashes.get(stage_id)
        != current_authority.stage_code_hashes.get(stage_id)
    )
    changed_inputs = sorted(
        input_id
        for input_id in (
            set(previous_authority.input_hashes)
            | set(current_authority.input_hashes)
        )
        if previous_authority.input_hashes.get(input_id)
        != current_authority.input_hashes.get(input_id)
    )
    direct_stages: set[str] = set(changed_code)
    reasons: list[str] = []
    for domain_id in changed_domains:
        affected = domain_to_stages.get(domain_id)
        if not affected:
            raise ValueError(
                f"changed semantic domain lacks impact mapping={domain_id}"
            )
        direct_stages.update(affected)
        reasons.append(f"SEMANTIC_DOMAIN_CHANGED:{domain_id}")
    for stage_id in changed_code:
        reasons.append(f"STAGE_CODE_CHANGED:{stage_id}")
    for input_id in changed_inputs:
        stage_id = input_to_stage.get(input_id)
        if stage_id is None:
            raise ValueError(
                f"changed input lacks impact mapping={input_id}"
            )
        direct_stages.add(stage_id)
        reasons.append(f"INPUT_HASH_CHANGED:{input_id}")
    invalidated_stages = (
        transitive_descendants(STAGE_DEPENDENCIES, direct_stages)
        if direct_stages
        else set()
    )
    earliest = (
        min(direct_stages, key=order.index) if direct_stages else None
    )
    stale_ids = sorted(
        str(record["artifact_id"])
        for record in artifact_inventory
        if record.get("stage_id") in invalidated_stages
        or record.get("classification")
        in {
            "STALE_SEMANTICS",
            "STALE_CODE",
            "STALE_INPUT",
            "MISSING_MANIFEST",
            "HASH_MISMATCH",
            "FAILED_DIAGNOSTIC",
            "UNKNOWN_NOT_PROMOTABLE",
        }
    )
    transitive_ids = sorted(
        str(record["artifact_id"])
        for record in artifact_inventory
        if record.get("stage_id") in invalidated_stages
        and record.get("stage_id") not in direct_stages
    )
    carry_candidates = sorted(
        str(record["artifact_id"])
        for record in artifact_inventory
        if record.get("classification") == "HUMAN_DECISION_EVIDENCE"
    )
    human_revalidation = any(
        domain_id
        in {
            "semantic.source_parsing_selection",
            "semantic.frame_local_geometry",
            "semantic.image_distance_metrics",
            "semantic.social_identity_tie_break",
            "semantic.roi_computation_aggregation",
            "semantic.hidden_selection",
            "semantic.hidden_decision_application",
            "semantic.temporal_harmonization",
            "semantic.native_temporal_pairs",
            "semantic.motion_tensor_schema",
            "semantic.pig_strenet_evidence",
            "semantic.behavior_review_units",
            "semantic.behavior_decision_application",
        }
        for domain_id in changed_domains
    )
    blocked_actions = (
        [
            "OFFICIAL_FRAME_LOCAL_REBUILD_REQUIRES_PHASE4_SIGNOFF",
            "HIDDEN_REVIEW_REQUIRES_RELEASE_AUTHORITY",
            "NATIVE_EVIDENCE_REQUIRES_UPSTREAM_VALID_MANIFESTS",
            "PIG_STRENET_REQUIRES_CURRENT_NATIVE_EVIDENCE",
            "TRAINING_REQUIRES_SIGNED_RELEASE_AUTHORITY",
        ]
        if direct_stages
        else []
    )
    return {
        "rebuild_required": bool(direct_stages),
        "earliest_stage_id": earliest,
        "direct_change_reasons": reasons,
        "changed_semantic_domains": changed_domains,
        "changed_stage_code_hashes": changed_code,
        "changed_input_hashes": changed_inputs,
        "directly_affected_stages": sorted(
            direct_stages,
            key=order.index,
        ),
        "invalidated_stages": sorted(
            invalidated_stages,
            key=order.index,
        ),
        "stale_artifact_ids": stale_ids,
        "transitively_invalidated_artifact_ids": transitive_ids,
        "human_decision_revalidation_required": human_revalidation,
        "carry_forward_candidates": carry_candidates,
        "blocked_actions": blocked_actions,
    }


def artifact_manifest_json_schema() -> dict[str, Any]:
    """Return the authoritative fail-closed artifact-manifest schema."""

    hash_fields = {
        name: {"type": "string", "pattern": HASH_PATTERN.pattern}
        for name in (
            "manifest_builder_code_hash",
            "candidate_transaction_id",
            "candidate_transaction_provenance_hash",
            "stage_code_hash",
            "stage_semantics_hash",
            "stage_input_fingerprint",
            "stage_execution_fingerprint",
            "execution_parameters_hash",
            "semantic_bundle_hash",
            "contract_manifest_hash",
            "input_file_sha256",
            "output_file_sha256",
            "output_schema_hash",
            "motion_schema_hash",
        )
    }
    properties: dict[str, Any] = {
        name: {"type": "string", "minLength": 1}
        for name in ARTIFACT_MANIFEST_REQUIRED_FIELDS
    }
    properties.update(hash_fields)
    properties.update(
        {
            "artifact_manifest_version": {
                "const": ARTIFACT_MANIFEST_VERSION,
            },
            "manifest_builder_id": {"const": MANIFEST_BUILDER_ID},
            "manifest_builder_version": {
                "const": MANIFEST_BUILDER_VERSION,
            },
            "authority_state": {
                "enum": [
                    CANDIDATE_AUTHORITY_STATE,
                    OFFICIAL_AUTHORITY_STATE,
                ]
            },
            "candidate_transaction_state": {
                "const": CANDIDATE_TRANSACTION_STATE_COMMITTED,
            },
            "code_authority_vcs": {
                "const": CODE_AUTHORITY_VCS_GIT,
            },
            "code_authority_object_format": {
                "enum": sorted(SUPPORTED_GIT_OBJECT_FORMATS),
            },
            "created_by_code_authority_sha": {
                "type": "string",
                "pattern": "^[0-9a-f]+$",
            },
            "stage_id": {"enum": list(STAGE_DEPENDENCIES)},
            "input_artifact_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "input_artifact_fingerprints": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "pattern": HASH_PATTERN.pattern,
                },
            },
            "distance_metric_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "distance_metric_versions": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "row_count": {"type": "integer", "minimum": 0},
            "column_count": {"type": "integer", "minimum": 0},
            "output_byte_size": {"type": "integer", "minimum": 0},
            "ordered_columns": {
                "type": "array",
                "items": {"type": "string"},
            },
            "stage_specific_metadata": {"type": "object"},
            "validation_errors": {
                "type": "array",
                "items": {"type": "string"},
            },
            "validation_warnings": {
                "type": "array",
                "items": {"type": "string"},
            },
            "status": {"const": "VALIDATED"},
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "schema.classification_v2.artifact_manifest.v7",
        "title": "Classification V2 scientific artifact manifest",
        "type": "object",
        "required": list(ARTIFACT_MANIFEST_REQUIRED_FIELDS),
        "properties": properties,
        "allOf": [
            {
                "if": {
                    "properties": {
                        "code_authority_object_format": {"const": "sha1"},
                    }
                },
                "then": {
                    "properties": {
                        "created_by_code_authority_sha": {
                            "pattern": GIT_SHA1_PATTERN.pattern,
                        }
                    }
                },
            },
            {
                "if": {
                    "properties": {
                        "code_authority_object_format": {"const": "sha256"},
                    }
                },
                "then": {
                    "properties": {
                        "created_by_code_authority_sha": {
                            "pattern": GIT_SHA256_PATTERN.pattern,
                        }
                    }
                },
            },
        ],
        "additionalProperties": False,
    }


def release_authority_json_schema() -> dict[str, Any]:
    """Return the Phase 4 release-authority schema."""

    properties: dict[str, Any] = {
        "release_authority_schema_version": {
            "const": RELEASE_AUTHORITY_SCHEMA_VERSION,
        },
        "phase4_human_signoff": {"type": "boolean"},
        "prerequisite_errors": {
            "type": "array",
            "items": {"type": "string"},
        },
        "blocked_reasons": {
            "type": "array",
            "items": {"type": "string"},
        },
    }
    properties.update(
        {name: {"type": "boolean"} for name in RELEASE_AUTHORIZATION_FIELDS}
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "schema.classification_v2.release_authority.v4",
        "title": "Classification V2 release-authority preflight",
        "type": "object",
        "required": [
            "release_authority_schema_version",
            "phase4_human_signoff",
            "prerequisite_errors",
            "blocked_reasons",
            *RELEASE_AUTHORIZATION_FIELDS,
        ],
        "properties": properties,
        "additionalProperties": False,
    }


def _blank(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def _required_manifest_hash_fields() -> tuple[str, ...]:
    return (
        "manifest_builder_code_hash",
        "candidate_transaction_id",
        "candidate_transaction_provenance_hash",
        "stage_code_hash",
        "stage_semantics_hash",
        "stage_input_fingerprint",
        "stage_execution_fingerprint",
        "execution_parameters_hash",
        "semantic_bundle_hash",
        "contract_manifest_hash",
        "input_file_sha256",
        "output_file_sha256",
        "output_schema_hash",
        "motion_schema_hash",
    )


def validate_artifact_manifest(
    manifest: Mapping[str, Any],
    *,
    output_path: Path | None = None,
    upstream_manifests: Mapping[str, Mapping[str, Any]] | None = None,
    expected_stage_execution_fingerprint: str | None = None,
    expected_schema: tuple[str, str, str] | None = None,
    require_committed_transaction: bool = True,
) -> dict[str, Any]:
    """Validate one artifact manifest without trusting producer assertions."""

    errors: list[str] = []
    missing = [
        name
        for name in ARTIFACT_MANIFEST_REQUIRED_FIELDS
        if name not in manifest
    ]
    if missing:
        errors.append(f"MISSING_REQUIRED_FIELDS:{','.join(missing)}")
    if manifest.get("artifact_manifest_version") != ARTIFACT_MANIFEST_VERSION:
        errors.append("WRONG_ARTIFACT_MANIFEST_VERSION")
    if manifest.get("manifest_builder_id") != MANIFEST_BUILDER_ID:
        errors.append("WRONG_MANIFEST_BUILDER_ID")
    if manifest.get("manifest_builder_version") != MANIFEST_BUILDER_VERSION:
        errors.append("WRONG_MANIFEST_BUILDER_VERSION")
    if manifest.get("authority_state") not in {
        CANDIDATE_AUTHORITY_STATE,
        OFFICIAL_AUTHORITY_STATE,
    }:
        errors.append("INVALID_AUTHORITY_STATE")
    transaction_state = str(
        manifest.get("candidate_transaction_state", "")
    )
    if require_committed_transaction:
        if transaction_state != CANDIDATE_TRANSACTION_STATE_COMMITTED:
            errors.append("CANDIDATE_TRANSACTION_NOT_COMMITTED")
    elif transaction_state not in {
        CANDIDATE_TRANSACTION_STATE_PENDING,
        CANDIDATE_TRANSACTION_STATE_COMMITTED,
    }:
        errors.append("INVALID_CANDIDATE_TRANSACTION_STATE")
    transaction_id = str(manifest.get("candidate_transaction_id", ""))
    transaction_provenance = str(
        manifest.get("candidate_transaction_provenance_hash", "")
    )
    if (
        HASH_PATTERN.fullmatch(transaction_id)
        and transaction_provenance
        != candidate_transaction_provenance_hash(
            transaction_id,
            transaction_state,
        )
    ):
        errors.append("CANDIDATE_TRANSACTION_PROVENANCE_MISMATCH")
    stage_id = manifest.get("stage_id")
    if stage_id not in STAGE_DEPENDENCIES:
        errors.append("UNKNOWN_STAGE_ID")
    for name in _required_manifest_hash_fields():
        value = manifest.get(name)
        if _blank(value):
            errors.append(f"BLANK_REQUIRED_HASH:{name}")
        elif not HASH_PATTERN.fullmatch(str(value)):
            errors.append(f"INVALID_HASH:{name}")
    errors.extend(validate_git_code_authority(manifest))
    artifact_id = str(manifest.get("artifact_id", ""))
    input_ids = manifest.get("input_artifact_ids")
    fingerprints = manifest.get("input_artifact_fingerprints")
    if not isinstance(input_ids, list):
        errors.append("INPUT_ARTIFACT_IDS_NOT_LIST")
        input_ids = []
    elif len(input_ids) != len(set(input_ids)):
        errors.append("DUPLICATE_INPUT_ARTIFACT_ID")
    if artifact_id and artifact_id in input_ids:
        errors.append("SELF_DEPENDENCY")
    if not isinstance(fingerprints, Mapping):
        errors.append("INPUT_FINGERPRINTS_NOT_MAPPING")
        fingerprints = {}
    elif set(input_ids) != set(fingerprints):
        errors.append("INPUT_FINGERPRINT_KEY_MISMATCH")
    for input_id, fingerprint in fingerprints.items():
        if not HASH_PATTERN.fullmatch(str(fingerprint)):
            errors.append(f"INVALID_INPUT_FINGERPRINT:{input_id}")
    if manifest.get("status") != "VALIDATED":
        errors.append("MANIFEST_STATUS_NOT_VALIDATED")
    validation_errors = manifest.get("validation_errors")
    if validation_errors not in ([], ()):
        errors.append("MANIFEST_DECLARES_VALIDATION_ERRORS")
    for count_field in ("row_count", "column_count"):
        value = manifest.get(count_field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"INVALID_COUNT:{count_field}")
    output_byte_size = manifest.get("output_byte_size")
    if (
        not isinstance(output_byte_size, int)
        or isinstance(output_byte_size, bool)
        or output_byte_size < 0
    ):
        errors.append("INVALID_OUTPUT_BYTE_SIZE")
    ordered_columns = manifest.get("ordered_columns")
    if not isinstance(ordered_columns, list):
        errors.append("ORDERED_COLUMNS_NOT_LIST")
    elif manifest.get("column_count") != len(ordered_columns):
        errors.append("COLUMN_COUNT_ORDERED_COLUMNS_MISMATCH")
    if not isinstance(manifest.get("stage_specific_metadata"), Mapping):
        errors.append("STAGE_SPECIFIC_METADATA_NOT_MAPPING")
    if output_path is not None:
        if not output_path.is_file():
            errors.append("OUTPUT_MISSING")
        else:
            if manifest.get("output_file_sha256") != file_sha256(output_path):
                errors.append("OUTPUT_HASH_MISMATCH")
            if manifest.get("output_byte_size") != output_path.stat().st_size:
                errors.append("OUTPUT_BYTE_SIZE_MISMATCH")
    if (
        expected_stage_execution_fingerprint is not None
        and manifest.get("stage_execution_fingerprint")
        != expected_stage_execution_fingerprint
    ):
        errors.append("STAGE_EXECUTION_FINGERPRINT_MISMATCH")
    if expected_schema is not None:
        actual_schema = (
            manifest.get("output_schema_id"),
            manifest.get("output_schema_version"),
            manifest.get("output_schema_hash"),
        )
        if actual_schema != expected_schema:
            errors.append("OUTPUT_SCHEMA_MISMATCH")
    if upstream_manifests is not None:
        for input_id in input_ids:
            upstream = upstream_manifests.get(input_id)
            if upstream is None:
                errors.append(f"UPSTREAM_MANIFEST_MISSING:{input_id}")
                continue
            result = validate_artifact_manifest(upstream)
            if not result["valid"]:
                errors.append(f"UPSTREAM_MANIFEST_INVALID:{input_id}")
                continue
            if (
                fingerprints.get(input_id)
                != upstream.get("stage_execution_fingerprint")
            ):
                errors.append(f"UPSTREAM_FINGERPRINT_MISMATCH:{input_id}")
    return {"valid": not errors, "errors": errors}


def validate_artifact_manifest_set(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject duplicate artifact IDs across a candidate inventory."""

    errors: list[str] = []
    seen: set[str] = set()
    for manifest in manifests:
        artifact_id = str(manifest.get("artifact_id", ""))
        if not artifact_id:
            errors.append("BLANK_ARTIFACT_ID")
        elif artifact_id in seen:
            errors.append(f"DUPLICATE_ARTIFACT_ID:{artifact_id}")
        seen.add(artifact_id)
        result = validate_artifact_manifest(manifest)
        errors.extend(
            f"{artifact_id}:{error}" for error in result["errors"]
        )
    return {"valid": not errors, "errors": errors}


def validate_artifact_manifest_pair(
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Require both artifact and valid matching manifest."""

    errors: list[str] = []
    if not output_path.is_file():
        errors.append("OUTPUT_MISSING")
    if not manifest_path.is_file():
        errors.append("MANIFEST_MISSING")
    if errors:
        return {"valid": False, "errors": errors}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "errors": [f"MANIFEST_UNREADABLE:{type(exc).__name__}"],
        }
    return validate_artifact_manifest(manifest, output_path=output_path)


def promote_artifact_transactionally(
    *,
    staging_output: Path,
    final_output: Path,
    final_manifest: Path,
    candidate_manifest: Mapping[str, Any],
    validate_output: Callable[[Path], Sequence[str]] | None = None,
    upstream_manifests: Mapping[str, Mapping[str, Any]] | None = None,
    before_manifest_promotion: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Promote a validated artifact and move its manifest last."""

    if final_output.exists() or final_manifest.exists():
        raise FileExistsError("refusing to overwrite promoted authority")
    if not staging_output.is_file():
        raise FileNotFoundError(staging_output)
    output_errors = list(validate_output(staging_output)) if validate_output else []
    if output_errors:
        raise ValueError(f"output validation failed: {output_errors}")
    manifest = dict(candidate_manifest)
    if manifest.get("authority_state") != CANDIDATE_AUTHORITY_STATE:
        raise ValueError("promotion requires CANDIDATE_VALIDATED authority")
    manifest_result = validate_artifact_manifest(
        manifest,
        output_path=staging_output,
        upstream_manifests=upstream_manifests,
    )
    if not manifest_result["valid"]:
        raise ValueError(
            f"candidate manifest invalid: {manifest_result['errors']}"
        )
    official_manifest = {
        **manifest,
        "authority_state": OFFICIAL_AUTHORITY_STATE,
    }
    official_result = validate_artifact_manifest(
        official_manifest,
        output_path=staging_output,
        upstream_manifests=upstream_manifests,
    )
    if not official_result["valid"]:
        raise ValueError(
            f"official manifest invalid: {official_result['errors']}"
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_manifest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_manifest.name}.",
        suffix=".staging",
        dir=final_manifest.parent,
    )
    os.close(descriptor)
    temporary_manifest = Path(temporary_name)
    artifact_promoted = False
    try:
        temporary_manifest.write_bytes(
            canonical_json_bytes(official_manifest) + b"\n"
        )
        os.replace(staging_output, final_output)
        artifact_promoted = True
        if before_manifest_promotion is not None:
            before_manifest_promotion()
        os.replace(temporary_manifest, final_manifest)
        return {
            "promoted": True,
            "output_path": str(final_output),
            "manifest_path": str(final_manifest),
            "output_sha256": manifest["output_file_sha256"],
        }
    except BaseException:
        if artifact_promoted and final_output.exists():
            os.replace(final_output, staging_output)
        if temporary_manifest.exists():
            temporary_manifest.unlink()
        raise


def _is_agent_audit(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    return "agent_audits" in parts


def _preserved_lineage_reason_codes(path: Path) -> list[str]:
    text = path.as_posix().casefold()
    reasons: list[str] = []
    if "v3" in text or "spatiotemporal_semantic_patch_20260722" in text:
        reasons.extend(
            [
                "STOPPED_AFTER_TEMPORAL_HARMONIZATION",
                "SPATIOTEMPORAL_SCIENTIFIC_AUDIT_FAILED",
                "NOT_RESUMABLE_AFTER_SEMANTIC_CHANGE",
                "NOT_BEHAVIOR_REVIEW_AUTHORITY",
                "NOT_TRAIN_READY",
            ]
        )
    if "pig_strenet" in text and "phase4_human_review" not in text:
        reasons.extend(
            [
                "FAILED_DIAGNOSTIC_PRE_MOTION_FIX",
                "NOT_REUSABLE",
                "NOT_REVIEW_EVIDENCE",
            ]
        )
    return reasons


def classify_existing_artifact(
    path: Path,
    *,
    current_stage_semantics: Mapping[str, str],
    current_stage_code: Mapping[str, str],
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Classify an existing artifact without using modification times."""

    reasons = _preserved_lineage_reason_codes(path)
    if _is_agent_audit(path):
        return {
            "classification": "NON_OFFICIAL_AUDIT",
            "reason_codes": [*reasons, "AUDIT_EVIDENCE_ONLY"],
            "promotable": False,
        }
    name = path.name.casefold()
    if "decision" in name:
        return {
            "classification": "HUMAN_DECISION_EVIDENCE",
            "reason_codes": [*reasons, "SEPARATE_CARRY_FORWARD_CONTRACT"],
            "promotable": False,
        }
    failed_diagnostic_reasons = {
        "SPATIOTEMPORAL_SCIENTIFIC_AUDIT_FAILED",
        "FAILED_DIAGNOSTIC_PRE_MOTION_FIX",
    }
    if failed_diagnostic_reasons.intersection(reasons):
        return {
            "classification": "FAILED_DIAGNOSTIC",
            "reason_codes": reasons,
            "promotable": False,
        }
    candidate_manifest = manifest_path or path.with_suffix(
        path.suffix + ".manifest.json"
    )
    if not candidate_manifest.is_file():
        return {
            "classification": "MISSING_MANIFEST",
            "reason_codes": [*reasons, "NO_VALID_ARTIFACT_MANIFEST"],
            "promotable": False,
        }
    try:
        manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "classification": "HASH_MISMATCH",
            "reason_codes": [*reasons, "MANIFEST_UNREADABLE"],
            "promotable": False,
        }
    result = validate_artifact_manifest(manifest, output_path=path)
    if not result["valid"]:
        return {
            "classification": "HASH_MISMATCH",
            "reason_codes": [*reasons, *result["errors"]],
            "promotable": False,
        }
    stage_id = str(manifest["stage_id"])
    if manifest["stage_semantics_hash"] != current_stage_semantics.get(
        stage_id
    ):
        classification = "STALE_SEMANTICS"
    elif manifest["stage_code_hash"] != current_stage_code.get(stage_id):
        classification = "STALE_CODE"
    elif any(
        not HASH_PATTERN.fullmatch(str(value))
        for value in manifest["input_artifact_fingerprints"].values()
    ):
        classification = "STALE_INPUT"
    else:
        classification = "VALID_CURRENT_AUTHORITY"
    return {
        "classification": classification,
        "reason_codes": reasons,
        "promotable": classification == "VALID_CURRENT_AUTHORITY",
        "stage_id": stage_id,
        "artifact_id": manifest["artifact_id"],
    }


def classification_v2_inventory_scope() -> dict[str, Any]:
    """Return the fixed Classification V2 scientific inventory boundary."""

    return {
        "inventory_scope_id": INVENTORY_SCOPE_ID,
        "inventory_scope_version": INVENTORY_SCOPE_VERSION,
        "included_roots": list(INVENTORY_INCLUDED_ROOTS),
        "excluded_roots": [
            "outputs/classification_v2/agent_audits/**/pytest_tmp",
            "outputs/classification_v2/agent_audits/**/pytest_upstream_tmp",
            "staging roots inside the included roots",
        ],
        "include_patterns": ["**/*"],
        "exclude_patterns": list(INVENTORY_EXCLUDE_PATTERNS),
        "path_normalization_rule": "repository-relative POSIX '/'",
    }


def _inventory_repository_root(
    root_or_roots: Path | Sequence[Path],
) -> Path:
    if isinstance(root_or_roots, Path):
        return root_or_roots.resolve()
    roots = [path.resolve() for path in root_or_roots]
    if not roots:
        raise ValueError("authoritative inventory roots are required")
    candidates: set[Path] = set()
    for root in roots:
        normalized = root.as_posix().casefold()
        for included in INVENTORY_INCLUDED_ROOTS:
            suffix = "/" + included.casefold()
            if normalized.endswith(suffix):
                candidates.add(root.parents[1])
                break
        else:
            raise ValueError(f"inventory root outside authority scope: {root}")
    if len(candidates) != 1:
        raise ValueError("inventory roots do not share one repository root")
    repository_root = next(iter(candidates))
    expected = {
        (repository_root / PurePosixPath(relative)).resolve()
        for relative in INVENTORY_INCLUDED_ROOTS
    }
    if set(roots) != expected:
        raise ValueError(
            "inventory roots must exactly match authoritative included roots"
        )
    return repository_root


def _inventory_path_excluded(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    lowered = tuple(part.casefold() for part in parts)
    if any(
        part in {
            ".staging",
            "pytest_tmp",
            "pytest_upstream_tmp",
        }
        for part in lowered
    ):
        return True
    name = lowered[-1] if lowered else ""
    return name.endswith((".staging", ".manifest.json"))


def inventory_existing_artifacts(
    root_or_roots: Path | Sequence[Path],
    *,
    current_stage_semantics: Mapping[str, str],
    current_stage_code: Mapping[str, str],
    max_files: int = 10_000,
) -> list[dict[str, Any]]:
    """Build the fixed-scope deterministic read-only artifact inventory."""

    repository_root = _inventory_repository_root(root_or_roots)
    paths: list[Path] = []
    for relative_root in INVENTORY_INCLUDED_ROOTS:
        root = repository_root / PurePosixPath(relative_root)
        if root.is_dir():
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and not _inventory_path_excluded(
                    path.relative_to(repository_root).as_posix()
                )
            )
    normalized_paths = [
        path.relative_to(repository_root).as_posix() for path in paths
    ]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("duplicate paths in authoritative inventory")
    paths = sorted(paths, key=lambda value: value.as_posix().casefold())
    if len(paths) > max_files:
        raise ValueError(
            f"bounded inventory exceeded max_files={max_files}: {len(paths)}"
        )
    records: list[dict[str, Any]] = []
    for path in paths:
        classification = classify_existing_artifact(
            path,
            current_stage_semantics=current_stage_semantics,
            current_stage_code=current_stage_code,
        )
        records.append(
            {
                "artifact_id": classification.get(
                    "artifact_id",
                    f"unmanifested:{path.as_posix()}",
                ),
                "path": path.as_posix(),
                "normalized_path": path.relative_to(
                    repository_root
                ).as_posix(),
                "inventory_scope_id": INVENTORY_SCOPE_ID,
                "inventory_scope_version": INVENTORY_SCOPE_VERSION,
                "stage_id": classification.get("stage_id"),
                **classification,
            }
        )
    return records


def audit_inventory_partition(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify one and only one top-level class per inventory path."""

    normalized_paths = [
        str(record.get("normalized_path", "")) for record in records
    ]
    duplicate_count = len(normalized_paths) - len(set(normalized_paths))
    unclassified = sum(
        str(record.get("classification", "")) not in INVENTORY_STATUSES
        for record in records
    )
    overlap = sum(
        isinstance(record.get("classification"), (list, tuple, set))
        and len(record["classification"]) != 1
        for record in records
    )
    counts = Counter(
        str(record.get("classification", "")) for record in records
    )
    return {
        "inventory_scope_id": INVENTORY_SCOPE_ID,
        "inventory_scope_version": INVENTORY_SCOPE_VERSION,
        "total": len(records),
        "classification_counts": dict(sorted(counts.items())),
        "unclassified": unclassified,
        "overlap": overlap,
        "duplicate_paths": duplicate_count,
        "valid": not (unclassified or overlap or duplicate_count),
    }


def _group_by_stable_key(
    records: Sequence[Mapping[str, Any]],
    key_field: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = str(record.get(key_field, "")).strip()
        grouped[key].append(record)
    return grouped


def _decision_conflict(
    records: Sequence[Mapping[str, Any]],
    decision_field: str,
) -> bool:
    values = {
        canonical_sha256(record.get(decision_field))
        for record in records
        if decision_field in record
    }
    return len(values) > 1


def evaluate_decision_carry_forward(
    old_decisions: Sequence[Mapping[str, Any]],
    new_review_units: Sequence[Mapping[str, Any]],
    *,
    key_field: str,
    identity_fields: Sequence[str],
    authority_fields: Sequence[str],
    review_schema_fields: Sequence[str],
    decision_schema_field: str,
    decision_field: str,
) -> list[dict[str, Any]]:
    """Classify exact-key human-decision carry-forward eligibility."""

    old_grouped = _group_by_stable_key(old_decisions, key_field)
    new_grouped = _group_by_stable_key(new_review_units, key_field)
    results: list[dict[str, Any]] = []
    for key in sorted(set(old_grouped) | set(new_grouped)):
        old = old_grouped.get(key, [])
        new = new_grouped.get(key, [])
        if not key:
            classification = "INVALID_DECISION_SCHEMA"
            reasons = ["BLANK_STABLE_KEY"]
        elif len(old) > 1 and _decision_conflict(old, decision_field):
            classification = "CONFLICT"
            reasons = ["DUPLICATE_CONFLICTING_DECISION_KEY"]
        elif len(new) > 1:
            classification = "CONFLICT"
            reasons = ["DUPLICATE_NEW_REVIEW_KEY"]
        elif not old:
            classification = "NEW_ONLY_REQUIRES_REVIEW"
            reasons = ["NO_OLD_DECISION"]
        elif not new:
            classification = "OLD_ONLY_AUDIT_EVIDENCE"
            reasons = ["NO_CURRENT_REVIEW_UNIT"]
        else:
            old_record = old[0]
            new_record = new[0]
            schema_fields = (
                *review_schema_fields,
                decision_schema_field,
            )
            missing_schema = [
                field
                for field in schema_fields
                if _blank(old_record.get(field))
                or _blank(new_record.get(field))
            ]
            if missing_schema:
                classification = "INVALID_DECISION_SCHEMA"
                reasons = [
                    f"MISSING_SCHEMA_AUTHORITY:{field}"
                    for field in missing_schema
                ]
            elif old_record.get(decision_schema_field) != new_record.get(
                decision_schema_field
            ):
                classification = "INVALID_DECISION_SCHEMA"
                reasons = [
                    "INCOMPATIBLE_DECISION_SCHEMA:"
                    f"{old_record.get(decision_schema_field)}->"
                    f"{new_record.get(decision_schema_field)}"
                ]
            else:
                changed_identity = [
                    field
                    for field in identity_fields
                    if old_record.get(field) != new_record.get(field)
                ]
                changed_authority = [
                    field
                    for field in (*authority_fields, *review_schema_fields)
                    if old_record.get(field) != new_record.get(field)
                ]
                if changed_identity or changed_authority:
                    classification = "REQUIRES_HUMAN_REVALIDATION"
                    reasons = [
                        *(f"IDENTITY_CHANGED:{field}" for field in changed_identity),
                        *(
                            f"AUTHORITY_CHANGED:{field}"
                            for field in changed_authority
                        ),
                    ]
                else:
                    classification = "EXACT_CARRY_FORWARD_CANDIDATE"
                    reasons = ["ALL_REQUIRED_AUTHORITIES_IDENTICAL"]
        results.append(
            {
                key_field: key,
                "classification": classification,
                "reason_codes": reasons,
                "positional_matching_used": False,
                "automatic_acceptance": False,
            }
        )
    return results


def evaluate_hidden_decision_carry_forward(
    old_decisions: Sequence[Mapping[str, Any]],
    new_review_units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the Hidden-decision exact authority contract."""

    return evaluate_decision_carry_forward(
        old_decisions,
        new_review_units,
        key_field="review_key",
        identity_fields=(
            "source_key",
            "dataset_key",
            "video_key",
            "object_track_key",
            "frame_span_key",
        ),
        authority_fields=(
            "visual_media_sha256",
            "crop_authority_sha256",
            "full_frame_authority_sha256",
        ),
        review_schema_fields=("review_schema_version",),
        decision_schema_field="decision_schema_version",
        decision_field="decision",
    )


def evaluate_behavior_decision_carry_forward(
    old_decisions: Sequence[Mapping[str, Any]],
    new_review_units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the Behavior-decision exact authority contract."""

    return evaluate_decision_carry_forward(
        old_decisions,
        new_review_units,
        key_field="review_unit_key",
        identity_fields=(
            "canonical_actor_key",
            "temporal_unit_key",
            "frame_span_key",
        ),
        authority_fields=(
            "original_label_authority_sha256",
            "visual_media_sha256",
            "review_task_semantics_hash",
        ),
        review_schema_fields=("review_schema_version",),
        decision_schema_field="decision_schema_version",
        decision_field="decision",
    )


def decision_carry_forward_contracts() -> dict[str, Any]:
    """Return explicit, non-positional human-decision rules."""

    return {
        "hidden": {
            "version": HIDDEN_CARRY_FORWARD_VERSION,
            "stable_key": "review_key",
            "requires": [
                "source/dataset/video/object identity",
                "frame/span identity",
                "visual-media hash",
                "crop/full-frame authority",
                "review schema compatibility",
                "decision schema compatibility",
            ],
        },
        "behavior": {
            "version": BEHAVIOR_CARRY_FORWARD_VERSION,
            "stable_key": "review_unit_key",
            "requires": [
                "actor identity",
                "temporal-unit/span identity",
                "original label authority",
                "visual-media authority",
                "review task semantics",
                "decision schema compatibility",
            ],
        },
        "classifications": sorted(DECISION_CLASSIFICATIONS),
        "forbidden_matching": [
            "position",
            "row_number",
            "nearest_time",
            "pig_id_only",
        ],
        "new_only_auto_accepted": False,
    }


def build_release_authority_preflight(
    *,
    artifact_gate_results: Mapping[str, bool],
    phase4_human_signoff: bool,
    manual_authorizations: Mapping[str, bool] | None = None,
    stopped_lineage: bool = False,
    failed_diagnostic: bool = False,
    non_official_audit_only: bool = False,
) -> dict[str, Any]:
    """Compute a fail-closed release preflight; defaults authorize nothing."""

    requested = dict(manual_authorizations or {})
    errors = sorted(
        f"PREREQUISITE_INVALID:{name}"
        for name, valid in artifact_gate_results.items()
        if valid is not True
    )
    if stopped_lineage:
        errors.append("STOPPED_LINEAGE_NOT_RESUMABLE")
    if failed_diagnostic:
        errors.append("FAILED_DIAGNOSTIC_NOT_AUTHORITY")
    if non_official_audit_only:
        errors.append("NON_OFFICIAL_AUDIT_NOT_AUTHORITY")
    if not phase4_human_signoff:
        errors.append("PHASE4_HUMAN_SIGNOFF_MISSING")
    authorizations: dict[str, bool] = {}
    prerequisites_valid = not errors
    for field in RELEASE_AUTHORIZATION_FIELDS:
        authorizations[field] = bool(
            prerequisites_valid and requested.get(field) is True
        )
    blocked = [
        name
        for name, authorized in authorizations.items()
        if not authorized
    ]
    return {
        "release_authority_schema_version": (
            RELEASE_AUTHORITY_SCHEMA_VERSION
        ),
        "phase4_human_signoff": phase4_human_signoff,
        "prerequisite_errors": errors,
        "blocked_reasons": blocked,
        **authorizations,
    }


def validate_release_authority_preflight(
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject malformed or internally inconsistent release authority."""

    errors: list[str] = []
    if (
        preflight.get("release_authority_schema_version")
        != RELEASE_AUTHORITY_SCHEMA_VERSION
    ):
        errors.append("WRONG_RELEASE_AUTHORITY_SCHEMA_VERSION")
    for field in RELEASE_AUTHORIZATION_FIELDS:
        if not isinstance(preflight.get(field), bool):
            errors.append(f"AUTHORIZATION_NOT_BOOLEAN:{field}")
    prerequisite_errors = preflight.get("prerequisite_errors")
    if not isinstance(prerequisite_errors, list):
        errors.append("PREREQUISITE_ERRORS_NOT_LIST")
        prerequisite_errors = ["INVALID"]
    if prerequisite_errors and any(
        preflight.get(field) is True
        for field in RELEASE_AUTHORIZATION_FIELDS
    ):
        errors.append("AUTHORIZATION_TRUE_WITH_INVALID_PREREQUISITE")
    if preflight.get("phase4_human_signoff") is not True and any(
        preflight.get(field) is True
        for field in RELEASE_AUTHORIZATION_FIELDS
    ):
        errors.append("AUTHORIZATION_TRUE_WITHOUT_PHASE4_SIGNOFF")
    return {"valid": not errors, "errors": errors}
