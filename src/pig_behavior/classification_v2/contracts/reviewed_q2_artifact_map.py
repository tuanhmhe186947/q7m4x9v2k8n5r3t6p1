"""Generate one explicit reviewed-Q2 artifact map from two isolated run IDs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    ARTIFACT_MAP_SCHEMA_VERSION,
    DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION,
    PLACEHOLDER_TOKENS,
    RUN_ID_PATTERN,
)

LAYOUT_SCHEMA_VERSION = (
    "classification_v2.reviewed_q2_artifact_layout.v1"
)
HUMAN_ROOT_PREFIX = PurePosixPath(
    "human_review_workspace/classification_v2"
)
AGENT_ROOT_PREFIX = PurePosixPath(
    "outputs/classification_v2/agent_audits"
)


class ReviewedQ2ArtifactMapError(ValueError):
    """Expose deterministic layout and namespace failures to CLI callers."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__(f"invalid reviewed Q2 artifact map: {errors}")


@dataclass(frozen=True, slots=True)
class ReviewedQ2ArtifactMapBuild:
    """Validated map payload, audit, and agent-owned output destination."""

    artifact_map: dict[str, Any]
    audit: dict[str, Any]
    output_path: Path


def build_reviewed_q2_artifact_map(
    template_path: Path,
    layout_path: Path,
    *,
    human_review_run_id: str,
    agent_audit_run_id: str,
    output_path: Path,
    project_root: Path,
) -> ReviewedQ2ArtifactMapBuild:
    """Resolve all paths without reading rows or creating the human root."""

    root = project_root.resolve()
    template_file = _project_file(template_path, root, "template")
    layout_file = _project_file(layout_path, root, "layout")
    destination, destination_relative = _project_path(
        output_path,
        root,
        "output_json",
    )
    template = _read_json(template_file, "template")
    layout = _read_json(layout_file, "layout")
    errors = _identifier_errors(
        human_review_run_id,
        label="human_review_run_id",
    )
    errors.extend(
        _identifier_errors(
            agent_audit_run_id,
            label="agent_audit_run_id",
        )
    )
    if human_review_run_id == agent_audit_run_id:
        errors.append("human_and_agent_run_ids_must_be_distinct")

    human_root = (HUMAN_ROOT_PREFIX / human_review_run_id).as_posix()
    agent_root = (AGENT_ROOT_PREFIX / agent_audit_run_id).as_posix()
    expected_output = (
        PurePosixPath(agent_root)
        / "contracts"
        / "reviewed_q2_artifact_map.json"
    ).as_posix()
    if destination_relative != expected_output:
        errors.append(
            "output_json_must_equal_agent_contract_path:"
            f"expected={expected_output}:actual={destination_relative}"
        )
    errors.extend(_root_payload_errors(template, layout))

    template_artifacts = template.get("artifacts")
    layout_artifacts = layout.get("artifacts")
    resolved: dict[str, dict[str, str]] = {}
    if isinstance(template_artifacts, dict) and isinstance(
        layout_artifacts,
        dict,
    ):
        missing = sorted(set(template_artifacts).difference(layout_artifacts))
        unknown = sorted(set(layout_artifacts).difference(template_artifacts))
        if missing:
            errors.append(f"layout_missing_artifacts={missing}")
        if unknown:
            errors.append(f"layout_unknown_artifacts={unknown}")
        for name in sorted(set(template_artifacts).intersection(layout_artifacts)):
            spec = template_artifacts[name]
            relative = layout_artifacts[name]
            entry, entry_errors = _resolve_artifact(
                name,
                spec,
                relative,
                human_root=human_root,
                agent_root=agent_root,
                project_root=root,
            )
            errors.extend(entry_errors)
            if entry is not None:
                resolved[name] = entry

    paths = [entry["path"] for entry in resolved.values()]
    if len(paths) != len(set(paths)):
        errors.append("resolved_artifact_paths_not_unique")
    errors = sorted(set(errors))
    if errors:
        raise ReviewedQ2ArtifactMapError(errors)

    artifact_map = {
        "schema_version": ARTIFACT_MAP_SCHEMA_VERSION,
        "run_id": agent_audit_run_id,
        "profile": "mixed-reviewed",
        "lineage_ids": {
            "human_review": human_review_run_id,
            "agent_derived": agent_audit_run_id,
        },
        "lineage_roots": {
            "human_review": human_root,
            "agent_derived": agent_root,
        },
        "train_ready_root": (
            PurePosixPath(agent_root)
            / str(layout["train_ready_relative_root"])
        ).as_posix(),
        "snapshot_output_dir": (
            PurePosixPath(agent_root)
            / str(layout["snapshot_relative_root"])
        ).as_posix(),
        "artifacts": resolved,
    }
    audit = {
        "schema_version": (
            "classification_v2.reviewed_q2_artifact_map_build_audit.v1"
        ),
        "status": "PASS",
        "valid": True,
        "errors": [],
        "human_review_run_id": human_review_run_id,
        "agent_audit_run_id": agent_audit_run_id,
        "human_review_root": human_root,
        "agent_derived_root": agent_root,
        "output_json": destination_relative,
        "artifact_count": len(resolved),
        "human_root_created": False,
        "canonical_fallback_used": False,
        "dataset_rows_read": 0,
        "dataset_rows_written": 0,
    }
    return ReviewedQ2ArtifactMapBuild(
        artifact_map=artifact_map,
        audit=audit,
        output_path=destination,
    )


def write_reviewed_q2_artifact_map(
    build: ReviewedQ2ArtifactMapBuild,
    *,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Write only the agent-owned map; never create or modify human paths."""

    audit = {
        **build.audit,
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "artifact_written": False,
    }
    if dry_run:
        return audit
    require_output_paths_available(
        [build.output_path],
        overwrite=overwrite,
    )
    build.output_path.parent.mkdir(parents=True, exist_ok=True)
    build.output_path.write_text(
        json.dumps(
            build.artifact_map,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {**audit, "artifact_written": True}


def _root_payload_errors(
    template: dict[str, Any],
    layout: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if (
        template.get("template_schema_version")
        != DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION
    ):
        errors.append("template_schema_version_mismatch")
    if layout.get("schema_version") != LAYOUT_SCHEMA_VERSION:
        errors.append("layout_schema_version_mismatch")
    if (
        layout.get("template_schema_version")
        != DATA_CONTRACT_TEMPLATE_SCHEMA_VERSION
    ):
        errors.append("layout_template_schema_version_mismatch")
    if layout.get("profile") != "mixed-reviewed":
        errors.append("layout_profile_must_be_mixed_reviewed")
    if template.get("allowed_profiles") != ["mixed-reviewed"]:
        errors.append("template_must_allow_only_mixed_reviewed")
    if not isinstance(template.get("artifacts"), dict):
        errors.append("template_artifacts_must_be_object")
    if not isinstance(layout.get("artifacts"), dict):
        errors.append("layout_artifacts_must_be_object")
    for field in ("train_ready_relative_root", "snapshot_relative_root"):
        try:
            _relative_path(layout.get(field), label=field)
        except ValueError as exc:
            errors.append(f"invalid_{field}:{exc}")
    return errors


def _resolve_artifact(
    name: str,
    spec: Any,
    relative: Any,
    *,
    human_root: str,
    agent_root: str,
    project_root: Path,
) -> tuple[dict[str, str] | None, list[str]]:
    if not isinstance(spec, dict):
        return None, [f"template_artifact_invalid:{name}"]
    scope = spec.get("scope")
    try:
        relative_path = _relative_path(relative, label=name)
    except ValueError as exc:
        return None, [f"layout_artifact_path_invalid:{name}:{exc}"]
    errors: list[str] = []
    if scope == "project_static":
        path = relative_path
        if PurePosixPath(path).parts[0] not in {"configs", "scripts"}:
            errors.append(f"static_artifact_outside_static_prefix:{name}")
        if not (project_root / Path(path)).is_file():
            errors.append(f"static_artifact_missing:{name}:{path}")
    elif scope == "human_review":
        path = (PurePosixPath(human_root) / relative_path).as_posix()
    elif scope == "agent_derived":
        path = (PurePosixPath(agent_root) / relative_path).as_posix()
    else:
        return None, [f"unsupported_artifact_scope:{name}:{scope}"]
    expected_suffix = {
        "binary": ".npy",
        "csv": ".csv",
        "json": ".json",
        "npz": ".npz",
    }.get(spec.get("type"))
    if expected_suffix and PurePosixPath(path).suffix.lower() != expected_suffix:
        errors.append(
            f"artifact_extension_mismatch:{name}:expected={expected_suffix}"
        )
    return {"path": path, "scope": str(scope)}, errors


def _identifier_errors(value: Any, *, label: str) -> list[str]:
    identifier = value.strip() if isinstance(value, str) else ""
    errors: list[str] = []
    if not RUN_ID_PATTERN.fullmatch(identifier):
        errors.append(f"{label}_not_path_safe")
    if any(token in identifier.lower() for token in PLACEHOLDER_TOKENS):
        errors.append(f"{label}_contains_placeholder")
    return errors


def _relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path")
    path = PurePosixPath(value.strip().replace("\\", "/"))
    if path.is_absolute() or Path(value).is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be project-relative without traversal")
    if any(token in value for token in ("%", "<", ">")):
        raise ValueError(f"{label} contains unresolved placeholder syntax")
    if path.as_posix() in {"", "."}:
        raise ValueError(f"{label} must identify a child path")
    return path.as_posix()


def _project_file(path: Path, root: Path, label: str) -> Path:
    resolved, relative = _project_path(path, root, label)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {relative}")
    return resolved


def _project_path(
    path: Path,
    root: Path,
    label: str,
) -> tuple[Path, str]:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path is outside project root") from exc
    return resolved, relative.as_posix()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


__all__ = [
    "LAYOUT_SCHEMA_VERSION",
    "ReviewedQ2ArtifactMapBuild",
    "ReviewedQ2ArtifactMapError",
    "build_reviewed_q2_artifact_map",
    "write_reviewed_q2_artifact_map",
]
