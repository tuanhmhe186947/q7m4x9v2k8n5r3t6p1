"""Explicit, run-bound artifact paths for reviewed classification_v2 data.

The template declares artifact semantics while a run-specific artifact map
declares every concrete path.  This separation prevents historical canonical
output directories from being reused implicitly by a new reviewed lineage.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)

ARTIFACT_MAP_SCHEMA_VERSION = "classification_v2.artifact_map.v3"
LEGACY_ARTIFACT_MAP_SCHEMA_VERSION = "classification_v2.artifact_map.v2"
DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION = (
    "classification_v2.data_contract_template.v1"
)
GENERATED_CONTRACT_SCHEMA_VERSION = (
    "classification_v2.versioned_data_contract.v2"
)
LEGACY_GENERATED_CONTRACT_SCHEMA_VERSION = (
    "classification_v2.versioned_data_contract.v1"
)
PATH_POLICY_SCHEMA_VERSION = "classification_v2.artifact_path_policy.v2"
LEGACY_PATH_POLICY_SCHEMA_VERSION = (
    "classification_v2.artifact_path_policy.v1"
)
BUILD_AUDIT_SCHEMA_VERSION = "classification_v2.data_contract_build_audit.v1"

ALLOWED_PROFILES = {
    "legacy-only-unreviewed-development",
    "mixed-reviewed",
}
ALLOWED_ARTIFACT_SCOPES = {
    "agent_derived",
    "human_review",
    "project_static",
}
ALLOWED_ARTIFACT_TYPES = {"binary", "csv", "json", "npz"}
PROJECT_STATIC_PREFIXES = {"configs", "scripts"}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
PLACEHOLDER_TOKENS = {
    "replace_with",
    "reviewer_vn",
    "yyyy",
    "yyyymmdd",
}


class VersionedDataContractError(ValueError):
    """Expose deterministic validation errors to CLI callers and tests."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__(f"invalid versioned data contract: {errors}")


@dataclass(frozen=True, slots=True)
class VersionedDataContractBuild:
    """Validated generated contract, audit payload, and resolved destination."""

    contract: dict[str, Any]
    audit: dict[str, Any]
    output_path: Path


def build_versioned_data_contract(
    template_path: Path,
    artifact_map_path: Path,
    *,
    output_path: Path,
    project_root: Path,
) -> VersionedDataContractBuild:
    """Build one explicit contract without reading or changing dataset rows."""

    root = project_root.resolve()
    template_file, template_rel = _existing_project_file(
        template_path,
        root,
        label="template",
    )
    artifact_map_file, artifact_map_rel = _existing_project_file(
        artifact_map_path,
        root,
        label="artifact_map",
    )
    destination, destination_rel = _project_path(
        output_path,
        root,
        label="output_json",
    )
    template = _read_json_object(template_file, "template")
    artifact_map = _read_json_object(artifact_map_file, "artifact_map")

    errors = _validate_root_contracts(template, artifact_map)
    lineage_ids, lineage_id_errors = _validated_lineage_ids(
        artifact_map,
        profile=artifact_map.get("profile"),
    )
    errors.extend(lineage_id_errors)
    roots, root_errors = _validated_lineage_roots(
        artifact_map.get("lineage_roots"),
        lineage_ids,
        profile=artifact_map.get("profile"),
        schema_version=artifact_map.get("schema_version"),
    )
    errors.extend(root_errors)
    errors.extend(
        _validate_map_location(
            artifact_map_rel,
            roots,
            lineage_ids,
        )
    )
    errors.extend(
        _validate_output_location(
            destination_rel,
            roots,
            lineage_ids,
        )
    )

    template_artifacts = template.get("artifacts", {})
    mapped_artifacts = artifact_map.get("artifacts", {})
    artifact_errors, resolved_artifacts = _validate_artifacts(
        template_artifacts,
        mapped_artifacts,
        roots=roots,
        lineage_ids=lineage_ids,
        project_root=root,
    )
    errors.extend(artifact_errors)
    for field in ("train_ready_root", "snapshot_output_dir"):
        errors.extend(
            _validate_lineage_path_field(
                field,
                artifact_map.get(field),
                roots=roots,
                lineage_ids=lineage_ids,
            )
        )

    if errors:
        raise VersionedDataContractError(errors)

    template_sha256 = _sha256_file(template_file)
    artifact_map_sha256 = _sha256_file(artifact_map_file)
    contract = _generated_contract(
        template,
        artifact_map,
        template_path=template_rel,
        template_sha256=template_sha256,
        artifact_map_path=artifact_map_rel,
        artifact_map_sha256=artifact_map_sha256,
        resolved_artifacts=resolved_artifacts,
        lineage_roots=roots,
        lineage_ids=lineage_ids,
    )
    audit = _build_audit(
        contract,
        output_path=destination_rel,
        output_exists=destination.exists(),
    )
    return VersionedDataContractBuild(
        contract=contract,
        audit=audit,
        output_path=destination,
    )


def write_versioned_data_contract(
    build: VersionedDataContractBuild,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Persist the validated packet, or report it without writes in dry-run."""

    existed_before_write = build.output_path.exists()
    audit = {
        **build.audit,
        "dry_run": bool(dry_run),
        "artifact_written": False,
        "overwrite": bool(overwrite),
        "output_existed_before_write": existed_before_write,
    }
    if dry_run:
        return audit
    require_output_paths_available(
        [build.output_path],
        overwrite=overwrite,
    )
    build.output_path.parent.mkdir(parents=True, exist_ok=True)
    build.output_path.write_text(
        _stable_json(build.contract),
        encoding="utf-8",
    )
    return {
        **audit,
        "artifact_written": True,
        "output_sha256": _sha256_file(build.output_path),
    }


def validate_generated_data_contract(
    contract_path: Path,
    *,
    project_root: Path,
) -> list[str]:
    """Check source hashes and path policy before a snapshot consumes a contract."""

    root = project_root.resolve()
    contract_file, _ = _existing_project_file(
        contract_path,
        root,
        label="generated_contract",
    )
    contract = _read_json_object(contract_file, "generated_contract")
    errors: list[str] = []
    generated_schema = contract.get("generated_contract_schema_version")
    supported_generated_schemas = {
        GENERATED_CONTRACT_SCHEMA_VERSION,
        LEGACY_GENERATED_CONTRACT_SCHEMA_VERSION,
    }
    if generated_schema not in supported_generated_schemas:
        errors.append("generated_contract_schema_version_mismatch")
    policy = contract.get("path_policy")
    if not isinstance(policy, dict):
        errors.append("path_policy_must_be_object")
    else:
        expected_policy = _path_policy(generated_schema)
        if policy != expected_policy:
            errors.append("path_policy_mismatch")
    source_paths: dict[str, Path] = {}
    for prefix in ("template", "artifact_map"):
        path_value = contract.get(f"{prefix}_path")
        expected_hash = contract.get(f"{prefix}_sha256")
        try:
            source_file, _ = _existing_project_file(
                Path(str(path_value or "")),
                root,
                label=prefix,
            )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{prefix}_source_invalid:{exc}")
            continue
        current_hash = _sha256_file(source_file)
        if not expected_hash or current_hash != expected_hash:
            errors.append(f"{prefix}_sha256_mismatch")
        source_paths[prefix] = source_file
    if set(source_paths) == {"template", "artifact_map"}:
        try:
            rebuilt = build_versioned_data_contract(
                source_paths["template"],
                source_paths["artifact_map"],
                output_path=contract_file,
                project_root=root,
            )
        except VersionedDataContractError as exc:
            errors.extend(
                f"generated_contract_rebuild_error:{error}"
                for error in exc.errors
            )
        else:
            if rebuilt.contract != contract:
                errors.append("generated_contract_payload_drift")
    return errors


def _validate_root_contracts(
    template: dict[str, Any],
    artifact_map: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    required_map_fields = {
        "schema_version",
        "run_id",
        "profile",
        "lineage_roots",
        "train_ready_root",
        "snapshot_output_dir",
        "artifacts",
    }
    schema_version = artifact_map.get("schema_version")
    if schema_version == ARTIFACT_MAP_SCHEMA_VERSION:
        required_map_fields.add("lineage_ids")
    unknown_map_fields = sorted(set(artifact_map).difference(required_map_fields))
    missing_map_fields = sorted(required_map_fields.difference(artifact_map))
    if missing_map_fields:
        errors.append(f"artifact_map_missing_fields={missing_map_fields}")
    if unknown_map_fields:
        errors.append(f"artifact_map_unknown_fields={unknown_map_fields}")
    if schema_version not in {
        ARTIFACT_MAP_SCHEMA_VERSION,
        LEGACY_ARTIFACT_MAP_SCHEMA_VERSION,
    }:
        errors.append("artifact_map_schema_version_mismatch")
    if (
        template.get("template_schema_version")
        != DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION
    ):
        errors.append("data_contract_template_schema_version_mismatch")
    raw_run_id = artifact_map.get("run_id")
    run_id = raw_run_id.strip() if isinstance(raw_run_id, str) else ""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        errors.append("run_id_not_path_safe")
    run_id_lower = run_id.lower()
    if any(token in run_id_lower for token in PLACEHOLDER_TOKENS):
        errors.append("run_id_contains_placeholder")
    if artifact_map.get("profile") not in ALLOWED_PROFILES:
        errors.append("unsupported_profile")
    allowed_profiles = template.get("allowed_profiles")
    if not isinstance(allowed_profiles, list) or not allowed_profiles:
        errors.append("template_allowed_profiles_must_be_nonempty_list")
    elif not all(isinstance(item, str) for item in allowed_profiles):
        errors.append("template_allowed_profiles_must_contain_strings")
    else:
        unknown_profiles = sorted(set(allowed_profiles).difference(ALLOWED_PROFILES))
        if unknown_profiles:
            errors.append(
                f"template_allowed_profiles_unknown={unknown_profiles}"
            )
        if artifact_map.get("profile") not in allowed_profiles:
            errors.append("artifact_map_profile_not_allowed_by_template")
    if not isinstance(artifact_map.get("artifacts"), dict):
        errors.append("artifact_map_artifacts_must_be_object")
    if not isinstance(template.get("artifacts"), dict):
        errors.append("template_artifacts_must_be_object")
    for field in ("contract_version", "snapshot_name"):
        value = template.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"template_missing_{field}")
    forbidden_template_fields = sorted(
        set(template).intersection(
            {"root", "train_ready_root", "snapshot_output_dir"}
        )
    )
    if forbidden_template_fields:
        errors.append(
            "template_contains_runtime_path_fields="
            f"{forbidden_template_fields}"
        )
    return errors


def _validated_lineage_ids(
    artifact_map: dict[str, Any],
    *,
    profile: Any,
) -> tuple[dict[str, str], list[str]]:
    """Resolve role-specific IDs while preserving artifact-map v2 replay."""

    run_value = artifact_map.get("run_id")
    run_id = run_value.strip() if isinstance(run_value, str) else ""
    schema_version = artifact_map.get("schema_version")
    required_roles = {"agent_derived"}
    if profile == "mixed-reviewed":
        required_roles.add("human_review")
    if schema_version == LEGACY_ARTIFACT_MAP_SCHEMA_VERSION:
        return {role: run_id for role in required_roles}, []

    values = artifact_map.get("lineage_ids")
    if not isinstance(values, dict):
        return {}, ["lineage_ids_must_be_object"]
    errors: list[str] = []
    missing = sorted(required_roles.difference(values))
    unknown = sorted(set(values).difference(required_roles))
    if missing:
        errors.append(f"lineage_id_roles_missing={missing}")
    if unknown:
        errors.append(f"lineage_id_roles_unknown={unknown}")
    lineage_ids: dict[str, str] = {}
    for role in sorted(required_roles.intersection(values)):
        value = values[role]
        lineage_id = value.strip() if isinstance(value, str) else ""
        if not RUN_ID_PATTERN.fullmatch(lineage_id):
            errors.append(f"lineage_id_not_path_safe:{role}")
        if any(token in lineage_id.lower() for token in PLACEHOLDER_TOKENS):
            errors.append(f"lineage_id_contains_placeholder:{role}")
        lineage_ids[role] = lineage_id
    if lineage_ids.get("agent_derived") != run_id:
        errors.append("run_id_must_equal_agent_derived_lineage_id")
    if (
        profile == "mixed-reviewed"
        and lineage_ids.get("human_review")
        == lineage_ids.get("agent_derived")
    ):
        errors.append("mixed_reviewed_lineage_ids_must_be_distinct")
    return lineage_ids, errors


def _validated_lineage_roots(
    values: Any,
    lineage_ids: dict[str, str],
    *,
    profile: Any,
    schema_version: Any,
) -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    if not isinstance(values, dict) or not values:
        return {}, ["lineage_roots_must_be_nonempty_object"]
    required_roles = {"agent_derived"}
    if profile == "mixed-reviewed":
        required_roles.add("human_review")
    unknown_roles = sorted(set(values).difference(required_roles))
    missing_roles = sorted(required_roles.difference(values))
    if missing_roles:
        errors.append(f"lineage_root_roles_missing={missing_roles}")
    if unknown_roles:
        errors.append(f"lineage_root_roles_unknown={unknown_roles}")
    roots: dict[str, str] = {}
    for role in sorted(set(values).intersection(required_roles)):
        value = values[role]
        try:
            normalized = _normalize_relative_path(value)
        except ValueError as exc:
            errors.append(f"invalid_lineage_root:{role}:{exc}")
            continue
        lineage_id = lineage_ids.get(role)
        if not lineage_id or lineage_id not in PurePosixPath(normalized).parts:
            errors.append(
                f"lineage_root_missing_exact_lineage_id:{role}:{normalized}"
            )
        if _is_project_static_path(normalized):
            errors.append(
                f"lineage_root_uses_static_namespace:{role}:{normalized}"
            )
        parts = PurePosixPath(normalized).parts
        if role == "human_review" and parts and parts[0] == "outputs":
            errors.append("human_review_root_must_be_outside_outputs")
        if (
            role == "agent_derived"
            and parts
            and parts[0] == "human_review_workspace"
        ):
            errors.append("agent_derived_root_must_be_outside_human_workspace")
        if schema_version == ARTIFACT_MAP_SCHEMA_VERSION:
            errors.extend(
                _reviewed_q2_namespace_errors(
                    role,
                    normalized,
                    lineage_id=lineage_id,
                    profile=profile,
                )
            )
        roots[role] = normalized
    if len(set(roots.values())) != len(roots):
        errors.append("duplicate_lineage_roots")
    root_items = list(roots.items())
    for index, (left_role, left) in enumerate(root_items):
        left_parts = PurePosixPath(left).parts
        for right_role, right in root_items[index + 1 :]:
            right_parts = PurePosixPath(right).parts
            shorter = min(len(left_parts), len(right_parts))
            if left_parts[:shorter] == right_parts[:shorter]:
                errors.append(
                    "overlapping_lineage_roots:"
                    f"{left_role}:{left}:{right_role}:{right}"
                )
    return roots, errors


def _reviewed_q2_namespace_errors(
    role: str,
    path: str,
    *,
    lineage_id: str | None,
    profile: Any,
) -> list[str]:
    """Lock mixed-reviewed v3 roots to separate operator/agent namespaces."""

    if profile != "mixed-reviewed" or not lineage_id:
        return []
    expected = {
        "human_review": (
            "human_review_workspace",
            "classification_v2",
            lineage_id,
        ),
        "agent_derived": (
            "outputs",
            "classification_v2",
            "agent_audits",
            lineage_id,
        ),
    }.get(role)
    if expected is None:
        return []
    if PurePosixPath(path).parts != expected:
        return [f"reviewed_q2_lineage_root_not_exact:{role}:{path}"]
    return []


def _validate_artifacts(
    template_artifacts: Any,
    mapped_artifacts: Any,
    *,
    roots: dict[str, str],
    lineage_ids: dict[str, str],
    project_root: Path,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not isinstance(template_artifacts, dict) or not isinstance(
        mapped_artifacts,
        dict,
    ):
        return ["artifact_objects_invalid"], {}
    errors: list[str] = []
    missing = sorted(set(template_artifacts).difference(mapped_artifacts))
    unknown = sorted(set(mapped_artifacts).difference(template_artifacts))
    if missing:
        errors.append(f"artifact_map_missing_artifacts={missing}")
    if unknown:
        errors.append(f"artifact_map_unknown_artifacts={unknown}")
    resolved: dict[str, dict[str, Any]] = {}
    observed_paths: dict[str, str] = {}
    for name in sorted(set(template_artifacts).intersection(mapped_artifacts)):
        spec = template_artifacts[name]
        entry = mapped_artifacts[name]
        if not isinstance(spec, dict) or not isinstance(entry, dict):
            errors.append(f"artifact_entry_must_be_object:{name}")
            continue
        if "path" in spec:
            errors.append(f"template_artifact_contains_path:{name}")
        scope = spec.get("scope")
        artifact_type = spec.get("type")
        if scope not in ALLOWED_ARTIFACT_SCOPES:
            errors.append(f"template_artifact_scope_invalid:{name}")
        if artifact_type not in ALLOWED_ARTIFACT_TYPES:
            errors.append(f"template_artifact_type_invalid:{name}")
        if not isinstance(spec.get("required"), bool):
            errors.append(f"template_artifact_required_not_boolean:{name}")
        unknown_entry_fields = sorted(set(entry).difference({"path", "scope"}))
        missing_entry_fields = sorted({"path", "scope"}.difference(entry))
        if unknown_entry_fields or missing_entry_fields:
            errors.append(
                f"artifact_map_entry_fields_invalid:{name}:"
                f"missing={missing_entry_fields}:unknown={unknown_entry_fields}"
            )
            continue
        if entry.get("scope") != scope:
            errors.append(f"artifact_scope_mismatch:{name}")
        try:
            path = _normalize_relative_path(entry.get("path"))
        except ValueError as exc:
            errors.append(f"artifact_path_invalid:{name}:{exc}")
            continue
        if path in observed_paths:
            errors.append(
                f"duplicate_artifact_path:{observed_paths[path]}:{name}:{path}"
            )
        observed_paths[path] = name
        if scope in {"agent_derived", "human_review"}:
            expected_root = roots.get(scope)
            if not expected_root or not _inside_root(path, expected_root):
                errors.append(f"{scope}_artifact_outside_root:{name}:{path}")
            lineage_id = lineage_ids.get(scope)
            if not lineage_id or lineage_id not in PurePosixPath(path).parts:
                errors.append(
                    f"lineage_artifact_missing_exact_lineage_id:{name}"
                )
        elif scope == "project_static":
            if not _is_project_static_path(path):
                errors.append(f"project_static_artifact_outside_policy:{name}")
            elif not (project_root / Path(path)).is_file():
                errors.append(f"project_static_artifact_missing:{name}:{path}")
        errors.extend(_validate_artifact_extension(name, path, artifact_type))
        resolved[name] = {
            **{key: value for key, value in spec.items() if key != "path"},
            "path": path,
            "scope": scope,
        }
    return errors, resolved


def _validate_lineage_path_field(
    field: str,
    value: Any,
    *,
    roots: dict[str, str],
    lineage_ids: dict[str, str],
) -> list[str]:
    try:
        path = _normalize_relative_path(value)
    except ValueError as exc:
        return [f"{field}_invalid:{exc}"]
    errors: list[str] = []
    agent_root = roots.get("agent_derived")
    if not agent_root or not _inside_root(path, agent_root):
        errors.append(f"{field}_outside_agent_derived_root")
    agent_lineage_id = lineage_ids.get("agent_derived")
    if not agent_lineage_id or agent_lineage_id not in PurePosixPath(path).parts:
        errors.append(f"{field}_missing_exact_agent_lineage_id")
    return errors


def _validate_map_location(
    path: str,
    roots: dict[str, str],
    lineage_ids: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    agent_root = roots.get("agent_derived")
    if not agent_root or not _inside_root(path, agent_root):
        errors.append("artifact_map_file_outside_agent_derived_root")
    agent_lineage_id = lineage_ids.get("agent_derived")
    if not agent_lineage_id or agent_lineage_id not in PurePosixPath(path).parts:
        errors.append("artifact_map_file_missing_exact_agent_lineage_id")
    return errors


def _validate_output_location(
    path: str,
    roots: dict[str, str],
    lineage_ids: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    if PurePosixPath(path).suffix.lower() != ".json":
        errors.append("output_json_requires_json_suffix")
    agent_root = roots.get("agent_derived")
    if not agent_root or not _inside_root(path, agent_root):
        errors.append("output_json_outside_agent_derived_root")
    agent_lineage_id = lineage_ids.get("agent_derived")
    if not agent_lineage_id or agent_lineage_id not in PurePosixPath(path).parts:
        errors.append("output_json_missing_exact_agent_lineage_id")
    return errors


def _validate_artifact_extension(
    name: str,
    path: str,
    artifact_type: Any,
) -> list[str]:
    expected = {"csv": ".csv", "json": ".json", "npz": ".npz"}.get(
        artifact_type
    )
    if expected and PurePosixPath(path).suffix.lower() != expected:
        return [f"artifact_extension_mismatch:{name}:expected={expected}"]
    return []


def _generated_contract(
    template: dict[str, Any],
    artifact_map: dict[str, Any],
    *,
    template_path: str,
    template_sha256: str,
    artifact_map_path: str,
    artifact_map_sha256: str,
    resolved_artifacts: dict[str, dict[str, Any]],
    lineage_roots: dict[str, str],
    lineage_ids: dict[str, str],
) -> dict[str, Any]:
    dynamic_fields = {
        "artifacts",
        "root",
        "snapshot_output_dir",
        "train_ready_root",
    }
    static_template = {
        key: value for key, value in template.items() if key not in dynamic_fields
    }
    is_legacy = (
        artifact_map.get("schema_version")
        == LEGACY_ARTIFACT_MAP_SCHEMA_VERSION
    )
    generated_schema = (
        LEGACY_GENERATED_CONTRACT_SCHEMA_VERSION
        if is_legacy
        else GENERATED_CONTRACT_SCHEMA_VERSION
    )
    payload = {
        **static_template,
        "generated_contract_schema_version": generated_schema,
        "root": ".",
        "run_id": artifact_map["run_id"],
        "profile": artifact_map["profile"],
        "lineage_roots": lineage_roots,
        "train_ready_root": _normalize_relative_path(
            artifact_map["train_ready_root"]
        ),
        "snapshot_output_dir": _normalize_relative_path(
            artifact_map["snapshot_output_dir"]
        ),
        "template_path": template_path,
        "template_sha256": template_sha256,
        "artifact_map_path": artifact_map_path,
        "artifact_map_sha256": artifact_map_sha256,
        "path_policy": _path_policy(generated_schema),
        "artifacts": resolved_artifacts,
    }
    if not is_legacy:
        payload["lineage_ids"] = lineage_ids
    return payload


def _path_policy(generated_schema: Any) -> dict[str, Any]:
    policy = {
        "schema_version": (
            LEGACY_PATH_POLICY_SCHEMA_VERSION
            if generated_schema == LEGACY_GENERATED_CONTRACT_SCHEMA_VERSION
            else PATH_POLICY_SCHEMA_VERSION
        ),
        "all_artifact_paths_explicit": True,
        "canonical_fallback_allowed": False,
        "project_relative_paths_required": True,
        "exact_run_id_in_lineage_paths_required": True,
        "owner_separated_lineage_roots_required": True,
        "agent_writes_human_review_root_allowed": False,
    }
    if generated_schema == LEGACY_GENERATED_CONTRACT_SCHEMA_VERSION:
        return policy
    policy["exact_run_id_in_lineage_paths_required"] = False
    policy["exact_role_lineage_id_in_paths_required"] = True
    policy["mixed_reviewed_lineage_ids_must_be_distinct"] = True
    return policy


def _build_audit(
    contract: dict[str, Any],
    *,
    output_path: str,
    output_exists: bool,
) -> dict[str, Any]:
    scopes = [spec["scope"] for spec in contract["artifacts"].values()]
    return {
        "schema_version": BUILD_AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "valid": True,
        "errors": [],
        "run_id": contract["run_id"],
        "lineage_ids": contract.get("lineage_ids"),
        "profile": contract["profile"],
        "template_path": contract["template_path"],
        "template_sha256": contract["template_sha256"],
        "artifact_map_path": contract["artifact_map_path"],
        "artifact_map_sha256": contract["artifact_map_sha256"],
        "output_json": output_path,
        "output_exists": bool(output_exists),
        "artifact_count": len(scopes),
        "agent_derived_artifact_count": scopes.count("agent_derived"),
        "human_review_artifact_count": scopes.count("human_review"),
        "project_static_artifact_count": scopes.count("project_static"),
        "all_artifact_paths_explicit": True,
        "canonical_fallback_allowed": False,
        "dataset_rows_read": 0,
        "dataset_rows_written": 0,
    }


def _existing_project_file(
    path: Path,
    root: Path,
    *,
    label: str,
) -> tuple[Path, str]:
    resolved, relative = _project_path(path, root, label=label)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {relative}")
    return resolved, relative


def _project_path(
    path: Path,
    root: Path,
    *,
    label: str,
) -> tuple[Path, str]:
    if not str(path).strip():
        raise ValueError(f"{label} path must not be blank")
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path is outside project root") from exc
    return resolved, relative.as_posix()


def _normalize_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a nonempty string")
    raw = value.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or Path(value).is_absolute():
        raise ValueError("absolute paths are forbidden")
    if ".." in path.parts:
        raise ValueError("parent traversal is forbidden")
    if any(token in raw for token in ("%", "<", ">")):
        raise ValueError("path contains unresolved placeholder syntax")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise ValueError("path must identify a project child")
    return normalized


def _inside_root(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return path_parts[: len(root_parts)] == root_parts


def _is_project_static_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts and parts[0] in PROJECT_STATIC_PREFIXES)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
