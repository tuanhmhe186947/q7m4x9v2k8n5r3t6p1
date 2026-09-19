"""Independent P0 gate for the reviewed all-source classification lineage.

This gate consumes only a generated data contract and its declared artifacts.
It never infers canonical paths, writes the human-review root, trains a model,
or authorizes full OOF. A valid result can authorize only the next bounded
model smoke after the separate short-run gate is satisfied.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.model_input_manifest import (
    build_model_input_manifest,
)
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.training_snapshot import (
    check_training_snapshot,
)
from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    GENERATED_CONTRACT_SCHEMA_VERSION,
    validate_generated_data_contract,
)
from pig_behavior.classification_v2.evaluation.loader_input_audit import (
    audit_loader_input_contract,
)

P0_SCHEMA_VERSION = "classification_v2.reviewed_q2_p0_preflight.v1"
CSV_CHUNK_ROWS = 100_000
FILE_CHUNK_BYTES = 1024 * 1024

HIDDEN_REVIEW_ARTIFACTS = (
    "hidden_review_unit_manifest",
    "hidden_review_decisions",
    "hidden_review_decision_coverage_audit",
    "hidden_review_scientific_gate",
    "hidden_reviewed_frame_features",
    "hidden_apply_audit",
)
BEHAVIOR_REVIEW_ARTIFACTS = (
    "full_review_unit_manifest",
    "roi_behavior_decisions",
    "motion_behavior_decisions",
    "posture_behavior_decisions",
    "interaction_behavior_decisions",
    "behavior_decision_coverage_audit",
    "behavior_apply_audit",
    "reviewed_frame_features",
)
FRAME_PARITY_ARTIFACTS = (
    "enhanced_frame_features",
    "hidden_reviewed_frame_features",
    "reviewed_frame_features",
)
AUDIT_ARTIFACTS = (
    "temporal_unit_audit",
    "q2_grouped_fold_audit",
    "leakage_audit",
    "domain_controls_audit",
    "source_matched_view_audit",
    "source_matched_view_check_audit",
    "identifier_lineage_audit",
    "source_to_window_lineage_audit",
    "spatial_sequence_audit",
    "temporal_view_audit",
)
KEYED_ARTIFACTS = (
    ("native_temporal_unit_manifest", "temporal_unit_key"),
    ("split_manifest", "window_id"),
    ("q2_outer_fold_assignments", "temporal_unit_key"),
)
PERSISTENCE_FIELDS = {
    "artifact_written",
    "dry_run",
    "output_existed_before_write",
    "output_json",
    "overwrite",
}
FORBIDDEN_CANONICAL_PREFIXES = (
    "outputs/classification_v2/train_ready_windows",
    "outputs/classification_v2/sequence_features_reviewed",
    "outputs/classification_v2/model_design",
    "outputs/classification_v2/full_multimodal_oof",
)


def build_reviewed_q2_p0_preflight(
    data_contract_json: Path,
    snapshot_json: Path,
    *,
    project_root: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Run the reviewed-Q2 P0 checks without changing any project artifact."""

    root = project_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    contract_path = _safe_existing_path(
        data_contract_json,
        root,
        errors,
        label="data_contract",
    )
    snapshot_path = _safe_existing_path(
        snapshot_json,
        root,
        errors,
        label="snapshot",
    )
    contract = _safe_json(contract_path, errors, "data_contract")
    output_path = _safe_output_path(output_json, root, contract, errors)

    contract_errors = _validate_contract(contract_path, contract, root)
    errors.extend(contract_errors)
    profile = contract.get("profile")
    if profile != "mixed-reviewed":
        errors.append(f"p0_requires_mixed_reviewed_profile={profile}")
    if (
        contract.get("generated_contract_schema_version")
        != GENERATED_CONTRACT_SCHEMA_VERSION
    ):
        errors.append("p0_requires_generated_contract_v2")

    namespace = _audit_namespace(
        contract_path,
        output_path,
        contract,
        root,
    )
    errors.extend(namespace["errors"])
    warnings.extend(namespace["warnings"])

    artifacts = contract.get("artifacts", {})
    artifact_paths = _resolve_artifact_paths(
        contract,
        root,
        errors,
    )
    errors.extend(
        _audit_declared_artifacts(
            contract,
            artifact_paths,
            root,
        )
    )

    manifest_check = _audit_model_input_manifest(
        contract_path,
        artifact_paths.get("model_input_contract"),
        root,
    )
    errors.extend(manifest_check["errors"])

    loader_check = _audit_loader_input(
        artifact_paths.get("model_input_contract"),
        artifact_paths.get("loader_input_audit"),
        root,
    )
    errors.extend(loader_check["errors"])

    snapshot_check = _audit_snapshot(
        snapshot_path,
        contract_path,
        root,
    )
    errors.extend(snapshot_check["errors"])

    hidden_check = _audit_review_layer(
        "hidden",
        HIDDEN_REVIEW_ARTIFACTS,
        contract,
        artifact_paths,
    )
    behavior_check = _audit_review_layer(
        "behavior",
        BEHAVIOR_REVIEW_ARTIFACTS,
        contract,
        artifact_paths,
    )
    errors.extend(hidden_check["errors"])
    errors.extend(behavior_check["errors"])

    parity_check = _audit_frame_parity(artifact_paths)
    errors.extend(parity_check["errors"])
    audit_check = _audit_scientific_artifacts(
        contract,
        artifact_paths,
    )
    errors.extend(audit_check["errors"])
    keyed_check = _audit_keyed_artifacts(artifact_paths)
    errors.extend(keyed_check["errors"])

    anchor_check = _required_values_check(
        "cvat_anchor_1020_audit",
        contract,
        artifact_paths,
    )
    resolver_check = _required_values_check(
        "source_image_loader_audit",
        contract,
        artifact_paths,
    )
    errors.extend(anchor_check["errors"])
    errors.extend(resolver_check["errors"])

    errors = sorted(set(errors))
    valid = not errors
    return {
        "schema_version": P0_SCHEMA_VERSION,
        "profile": profile,
        "run_id": contract.get("run_id"),
        "lineage_ids": contract.get("lineage_ids"),
        "data_contract_json": _relative_path(contract_path, root),
        "data_contract_sha256": _sha256_file(contract_path)
        if contract_path is not None
        else None,
        "snapshot_json": _relative_path(snapshot_path, root)
        if snapshot_path is not None
        else None,
        "snapshot_sha256": _optional_sha256(snapshot_path),
        "output_json": _relative_path(output_path, root)
        if output_path is not None
        else None,
        "canonical_fallback_used": False,
        "human_review_root_write_attempted": False,
        "checks": {
            "namespace": namespace,
            "generated_contract": {
                "valid": not contract_errors,
                "errors": contract_errors,
            },
            "declared_artifacts": {
                "artifact_count": len(artifacts)
                if isinstance(artifacts, dict)
                else 0,
                "resolved_count": len(artifact_paths),
            },
            "model_input_manifest": manifest_check,
            "loader_input_audit": loader_check,
            "training_snapshot": snapshot_check,
            "hidden_review": hidden_check,
            "behavior_review": behavior_check,
            "frame_parity": parity_check,
            "scientific_audits": audit_check,
            "keyed_artifacts": keyed_check,
            "cvat_anchor_1020": anchor_check,
            "gui_video_resolver": resolver_check,
        },
        "model_smoke_authorized": valid,
        "full_oof_authorized": False,
        "full_oof_authorization_required": True,
        "next_allowed_action": (
            "model_smoke_after_short_gate"
            if valid
            else "resolve_p0_blockers_and_complete_human_review"
        ),
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "valid": valid,
    }


def write_reviewed_q2_p0_preflight(
    result: dict[str, Any],
    *,
    data_contract_json: Path | None = None,
    output_json: Path,
    project_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Persist a P0 result only inside its declared agent audit root."""

    root = project_root.resolve()
    contract_value = data_contract_json or Path(
        str(result.get("data_contract_json") or "")
    )
    contract = _safe_json(
        _safe_existing_path(
            contract_value,
            root,
            [],
            label="data_contract",
        ),
        [],
        "data_contract",
    )
    output_errors: list[str] = []
    destination = _safe_output_path(
        output_json,
        root,
        contract,
        output_errors,
    )
    if output_errors:
        raise ValueError(f"P0 output path is not agent-owned: {output_errors}")
    if destination is None:
        raise ValueError("P0 output path is required and must be agent-owned")
    require_output_paths_available([destination], overwrite=overwrite)
    payload = {
        **result,
        "output_json": _relative_path(destination, root),
        "artifact_written": True,
        "overwrite": bool(overwrite),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_stable_json(payload), encoding="utf-8")
    return payload


def _validate_contract(
    contract_path: Path | None,
    contract: dict[str, Any],
    root: Path,
) -> list[str]:
    if contract_path is None:
        return ["missing_data_contract"]
    try:
        return validate_generated_data_contract(
            contract_path,
            project_root=root,
        )
    except Exception as exc:
        return [f"generated_contract_validation_failed={exc}"]


def _audit_model_input_manifest(
    contract_path: Path | None,
    manifest_path: Path | None,
    root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    persisted = _safe_json(manifest_path, errors, "model_input_manifest")
    rebuilt: dict[str, Any] = {}
    if contract_path is not None and manifest_path is not None:
        try:
            build = build_model_input_manifest(
                contract_path,
                output_path=manifest_path,
                project_root=root,
            )
            rebuilt = build.manifest
        except Exception as exc:
            errors.append(f"model_input_manifest_rebuild_failed={exc}")
    if rebuilt and persisted and rebuilt != persisted:
        errors.append("model_input_manifest_payload_drift")
    if persisted.get("errors"):
        errors.append(
            f"model_input_manifest_errors={persisted.get('errors')}"
        )
    return {
        "valid": not errors,
        "rebuilt": bool(rebuilt),
        "persisted_exists": bool(persisted),
        "errors": errors,
    }


def _audit_loader_input(
    manifest_path: Path | None,
    persisted_path: Path | None,
    root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    persisted = _safe_json(persisted_path, errors, "loader_input_audit")
    fresh: dict[str, Any] = {}
    if manifest_path is not None:
        try:
            fresh = audit_loader_input_contract(
                model_input_contract_json=manifest_path,
                project_root=root,
            )
        except Exception as exc:
            errors.append(f"loader_input_audit_rebuild_failed={exc}")
    if fresh.get("valid") is not True:
        errors.append(f"fresh_loader_input_audit_invalid={fresh.get('errors')}")
    if _without_persistence_fields(fresh) != _without_persistence_fields(
        persisted
    ):
        errors.append("loader_input_audit_payload_drift")
    return {
        "valid": not errors,
        "fresh_valid": fresh.get("valid"),
        "persisted_exists": bool(persisted),
        "errors": errors,
    }


def _audit_snapshot(
    snapshot_path: Path | None,
    contract_path: Path | None,
    root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    result: dict[str, Any] = {}
    if snapshot_path is None or contract_path is None:
        errors.append("snapshot_or_contract_missing")
    else:
        try:
            result = check_training_snapshot(
                snapshot_path,
                contract_path=contract_path,
                project_root=root,
            )
        except TypeError:
            try:
                result = check_training_snapshot(
                    snapshot_path,
                    contract_path=contract_path,
                )
            except Exception as exc:
                errors.append(f"training_snapshot_check_failed={exc}")
        except Exception as exc:
            errors.append(f"training_snapshot_check_failed={exc}")
    if result.get("valid") is not True:
        errors.append(f"training_snapshot_invalid={result.get('errors')}")
    return {
        "valid": not errors,
        "snapshot_valid": result.get("valid"),
        "snapshot_id": result.get("expected_snapshot_id"),
        "errors": errors,
    }


def _audit_review_layer(
    label: str,
    names: tuple[str, ...],
    contract: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    for name in names:
        path = artifact_paths.get(name)
        spec = _artifact_spec(contract, name)
        check = _audit_artifact_values(name, path, spec)
        details[name] = check
        errors.extend(check["errors"])
    return {
        "valid": not errors,
        "artifacts": details,
        "errors": errors,
    }


def _audit_scientific_artifacts(
    contract: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    for name in AUDIT_ARTIFACTS:
        check = _audit_artifact_values(
            name,
            artifact_paths.get(name),
            _artifact_spec(contract, name),
            require_empty_errors=True,
        )
        details[name] = check
        errors.extend(check["errors"])
    return {
        "valid": not errors,
        "artifacts": details,
        "errors": errors,
    }


def _required_values_check(
    name: str,
    contract: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    spec = _artifact_spec(contract, name)
    return _audit_artifact_values(
        name,
        artifact_paths.get(name),
        spec,
    )


def _audit_artifact_values(
    name: str,
    path: Path | None,
    spec: dict[str, Any] | None,
    *,
    require_empty_errors: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    if path is None or not path.is_file():
        errors.append(f"missing_required_artifact:{name}")
        return {"exists": False, "errors": errors}
    spec = spec or {}
    artifact_type = spec.get("type")
    if artifact_type == "json":
        payload = _safe_json(path, errors, name)
        for field, expected in spec.get("required_json_values", {}).items():
            observed = _nested_value(payload, field)
            if observed != expected:
                errors.append(
                    f"required_json_value_mismatch:{name}.{field}"
                )
        if require_empty_errors and payload.get("errors") != []:
            errors.append(f"nonempty_audit_errors:{name}")
        if name.endswith("_gate") and payload.get("valid") is not True:
            errors.append(f"invalid_scientific_gate:{name}")
    elif artifact_type == "csv":
        columns, row_count = _csv_header_and_count(path, errors)
        missing = sorted(
            set(spec.get("required_columns", [])).difference(columns)
        )
        if missing:
            errors.append(f"missing_required_columns:{name}={missing}")
        if row_count == 0:
            errors.append(f"zero_rows:{name}")
    return {
        "exists": True,
        "type": artifact_type,
        "errors": errors,
    }


def _audit_frame_parity(
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    errors: list[str] = []
    profiles = {
        name: _csv_key_profile(artifact_paths.get(name), errors, name)
        for name in FRAME_PARITY_ARTIFACTS
    }
    reference = profiles[FRAME_PARITY_ARTIFACTS[0]]
    for name in FRAME_PARITY_ARTIFACTS[1:]:
        observed = profiles[name]
        if observed.get("key_column") != reference.get("key_column"):
            errors.append(f"frame_key_column_mismatch:{name}")
        if observed.get("row_count") != reference.get("row_count"):
            errors.append(f"frame_row_count_mismatch:{name}")
        if observed.get("ordered_key_sha256") != reference.get(
            "ordered_key_sha256"
        ):
            errors.append(f"frame_ordered_key_mismatch:{name}")
        if observed.get("duplicate_key_count", 0):
            errors.append(f"frame_duplicate_keys:{name}")
    if not reference.get("key_column"):
        errors.append("frame_parity_key_missing")
    return {
        "valid": not errors,
        "profiles": profiles,
        "errors": errors,
    }


def _audit_keyed_artifacts(
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {}
    for name, key_column in KEYED_ARTIFACTS:
        profile = _csv_key_profile(
            artifact_paths.get(name),
            errors,
            name,
            preferred_key=key_column,
        )
        details[name] = profile
        if profile.get("key_column") != key_column:
            errors.append(f"key_column_mismatch:{name}")
        if profile.get("duplicate_key_count", 0):
            errors.append(f"duplicate_key:{name}")
    return {
        "valid": not errors,
        "artifacts": details,
        "errors": errors,
    }


def _resolve_artifact_paths(
    contract: dict[str, Any],
    root: Path,
    errors: list[str],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("contract_artifacts_must_be_object")
        return paths
    for name, spec in artifacts.items():
        if not isinstance(spec, dict):
            errors.append(f"artifact_spec_invalid:{name}")
            continue
        value = spec.get("path")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"artifact_path_missing:{name}")
            continue
        try:
            path = _project_path(value, root, f"artifact:{name}")
        except ValueError as exc:
            errors.append(f"artifact_path_invalid:{name}:{exc}")
            continue
        if _is_canonical_path(path, root):
            errors.append(f"canonical_fallback_path:{name}")
        paths[name] = path
    return paths


def _audit_declared_artifacts(
    contract: dict[str, Any],
    paths: dict[str, Path],
    root: Path,
) -> list[str]:
    errors: list[str] = []
    artifacts = contract.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return ["contract_artifacts_must_be_object"]
    for name, spec in artifacts.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("required") is True and not paths.get(name, Path()).is_file():
            errors.append(f"missing_required_artifact:{name}")
        scope = spec.get("scope")
        path = paths.get(name)
        if path is None:
            continue
        if scope == "human_review" and _is_under(path, root / "outputs"):
            errors.append(f"human_artifact_in_outputs:{name}")
        if scope == "agent_derived" and "agent_audits" not in path.parts:
            errors.append(f"agent_artifact_outside_agent_audits:{name}")
    return errors


def _audit_namespace(
    contract_path: Path | None,
    output_path: Path | None,
    contract: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    roots = contract.get("lineage_roots")
    agent_root_value = roots.get("agent_derived") if isinstance(roots, dict) else None
    human_root_value = roots.get("human_review") if isinstance(roots, dict) else None
    if not isinstance(agent_root_value, str):
        errors.append("missing_agent_derived_root")
        return {"valid": False, "errors": errors, "warnings": warnings}
    agent_root = root / Path(agent_root_value)
    for label, path in (
        ("contract", contract_path),
        ("output", output_path),
    ):
        if path is not None and not _is_under(path, agent_root):
            errors.append(f"{label}_outside_agent_derived_root")
    if isinstance(human_root_value, str):
        human_root = root / Path(human_root_value)
        if output_path is not None and _is_under(output_path, human_root):
            errors.append("p0_output_inside_human_review_root")
    else:
        errors.append("missing_human_review_root")
    if contract.get("path_policy", {}).get("canonical_fallback_allowed") is not False:
        errors.append("canonical_fallback_must_be_false")
    return {
        "valid": not errors,
        "agent_derived_root": agent_root_value,
        "human_review_root": human_root_value,
        "errors": errors,
        "warnings": warnings,
    }


def _artifact_spec(
    contract: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    artifacts = contract.get("artifacts")
    spec = artifacts.get(name) if isinstance(artifacts, dict) else None
    return spec if isinstance(spec, dict) else None


def _safe_existing_path(
    value: Path,
    root: Path,
    errors: list[str],
    *,
    label: str,
) -> Path | None:
    try:
        path = _project_path(value, root, label)
    except ValueError as exc:
        errors.append(f"{label}_path_invalid={exc}")
        return None
    if not path.is_file():
        errors.append(f"missing_{label}={path}")
        return None
    return path


def _safe_output_path(
    value: Path | None,
    root: Path,
    contract: dict[str, Any],
    errors: list[str],
) -> Path | None:
    if value is None:
        return None
    try:
        path = _project_path(value, root, "output")
    except ValueError as exc:
        errors.append(f"output_path_invalid={exc}")
        return None
    roots = contract.get("lineage_roots")
    agent = roots.get("agent_derived") if isinstance(roots, dict) else None
    if not isinstance(agent, str) or not _is_under(path, root / Path(agent)):
        errors.append("output_outside_agent_derived_root")
    return path


def _safe_json(
    path: Path | None,
    errors: list[str],
    label: str,
) -> dict[str, Any]:
    if path is None or not path.is_file():
        errors.append(f"missing_json:{label}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_json:{label}:{exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"json_must_be_object:{label}")
        return {}
    return payload


def _csv_header_and_count(
    path: Path,
    errors: list[str],
) -> tuple[list[str], int]:
    if not path.is_file():
        errors.append(f"missing_csv:{path}")
        return [], 0
    try:
        header = list(pd.read_csv(path, nrows=0).columns)
        count = 0
        for chunk in pd.read_csv(path, chunksize=CSV_CHUNK_ROWS, low_memory=False):
            count += len(chunk)
        return header, count
    except Exception as exc:
        errors.append(f"invalid_csv:{path}:{exc}")
        return [], 0


def _csv_key_profile(
    path: Path | None,
    errors: list[str],
    name: str,
    *,
    preferred_key: str | None = None,
) -> dict[str, Any]:
    if path is None or not path.is_file():
        errors.append(f"missing_keyed_csv:{name}")
        return {"exists": False, "row_count": 0}
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
        candidates = (
            (preferred_key,) if preferred_key else ()
        ) + (
            "scene_frame_uid",
            "frame_uid",
            "review_row_index",
            "image_key",
        )
        key = next((value for value in candidates if value in columns), None)
        if key is None:
            errors.append(f"missing_key_column:{name}")
            return {"exists": True, "row_count": 0}
        digest = hashlib.sha256()
        seen: set[str] = set()
        duplicate_count = 0
        blank_count = 0
        row_count = 0
        for chunk in pd.read_csv(
            path,
            usecols=[key],
            chunksize=CSV_CHUNK_ROWS,
            low_memory=False,
        ):
            for raw in chunk[key].astype("string").fillna("").tolist():
                value = str(raw)
                if not value:
                    blank_count += 1
                if value in seen:
                    duplicate_count += 1
                seen.add(value)
                if row_count:
                    digest.update(b"\n")
                digest.update(value.encode("utf-8"))
                row_count += 1
        return {
            "exists": True,
            "key_column": key,
            "row_count": row_count,
            "ordered_key_sha256": digest.hexdigest(),
            "duplicate_key_count": duplicate_count,
            "blank_key_count": blank_count,
        }
    except Exception as exc:
        errors.append(f"invalid_keyed_csv:{name}:{exc}")
        return {"exists": False, "row_count": 0}


def _nested_value(payload: Any, dotted_path: str) -> Any:
    current = payload
    for token in dotted_path.split("."):
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _without_persistence_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in PERSISTENCE_FIELDS
    }


def _project_path(value: Any, root: Path, label: str) -> Path:
    is_path_value = isinstance(value, Path)
    if is_path_value:
        candidate = value
    elif isinstance(value, str) and value.strip():
        candidate = Path(value.strip().replace("\\", "/"))
    else:
        raise ValueError(f"{label} must be a nonempty path")
    if (not is_path_value and candidate.is_absolute()) or ".." in candidate.parts:
        raise ValueError(f"{label} must be project-relative")
    path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    if not _is_under(path, root):
        raise ValueError(f"{label} is outside project root")
    return path


def _relative_path(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _is_canonical_path(path: Path, root: Path) -> bool:
    relative = _relative_path(path, root)
    if relative is None:
        return True
    normalized = relative.lower()
    return any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in FORBIDDEN_CANONICAL_PREFIXES
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return _sha256_file(path)


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


__all__ = [
    "P0_SCHEMA_VERSION",
    "build_reviewed_q2_p0_preflight",
    "write_reviewed_q2_p0_preflight",
]
