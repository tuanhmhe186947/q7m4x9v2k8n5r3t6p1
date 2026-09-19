"""Loader and sampler input audit for classification_v2.

The training loader is allowed to use source-domain controls as masks, weights,
or sampling manifests. It must not use source, path, review, manual, identity,
or label columns as model inputs. This audit checks the file-level contract
before smoke training so leakage is caught outside the trainer.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.model_input_manifest import (
    MODEL_INPUT_MANIFEST_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)

LOADER_INPUT_AUDIT_SCHEMA_VERSION = (
    "classification_v2.loader_input_audit.v2"
)
FILE_CHUNK_BYTES = 1024 * 1024


def audit_loader_input_contract(
    *,
    model_input_contract_json: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Validate hash-bound X and controls without canonical path fallback."""

    errors: list[str] = []
    warnings: list[str] = []
    root = project_root.resolve()
    model_contract = _read_json(
        model_input_contract_json,
        errors,
        "model_input_contract",
    )
    _check_model_contract(model_contract, errors)
    tabular_x_csv = _bound_artifact_path(
        model_contract,
        group="predictive",
        name="tabular_X",
        root=root,
        errors=errors,
    )
    whitelist_json = _bound_artifact_path(
        model_contract,
        group="feature_contract",
        name="feature_whitelist",
        root=root,
        errors=errors,
    )
    blacklist_json = _bound_artifact_path(
        model_contract,
        group="feature_contract",
        name="feature_blacklist",
        root=root,
        errors=errors,
    )
    source_manifest_csv = _bound_artifact_path(
        model_contract,
        group="mask_and_control",
        name="source_matched_view_manifest",
        root=root,
        errors=errors,
    )
    source_audit_json = _bound_artifact_path(
        model_contract,
        group="data_audits",
        name="source_matched_view_audit",
        root=root,
        errors=errors,
    )
    source_check_json = _bound_artifact_path(
        model_contract,
        group="data_audits",
        name="source_matched_view_check_audit",
        root=root,
        errors=errors,
    )
    domain_controls_json = _bound_artifact_path(
        model_contract,
        group="data_audits",
        name="domain_controls_audit",
        root=root,
        errors=errors,
    )

    whitelist_payload = _read_json(
        whitelist_json,
        errors,
        "feature_whitelist",
    )
    blacklist_payload = _read_json(
        blacklist_json,
        errors,
        "feature_blacklist",
    )
    whitelist = _string_list(
        whitelist_payload.get("features"),
        errors,
        "feature_whitelist.features",
    )
    forbidden_patterns = _string_list(
        blacklist_payload.get("forbidden_patterns"),
        errors,
        "feature_blacklist.forbidden_patterns",
    )
    manifest_forbidden = _string_list(
        model_contract.get("forbidden_model_inputs"),
        errors,
        "model_input_contract.forbidden_model_inputs",
    )
    if forbidden_patterns != manifest_forbidden:
        errors.append("feature_blacklist_does_not_match_model_manifest")
    x_columns = _read_csv_columns(tabular_x_csv, errors, "tabular_x_csv")
    forbidden_x_columns = _match_forbidden_columns(x_columns, forbidden_patterns)
    whitelist_missing_in_x = sorted(set(whitelist).difference(x_columns))
    extra_x_columns = sorted(set(x_columns).difference(whitelist))

    if not whitelist:
        errors.append("empty_tabular_feature_whitelist")
    if not x_columns:
        errors.append(f"empty_or_missing_tabular_x_columns={tabular_x_csv}")
    if forbidden_x_columns:
        errors.append(f"forbidden_x_columns={forbidden_x_columns}")
    if whitelist_missing_in_x:
        errors.append(f"whitelist_missing_in_tabular_x={whitelist_missing_in_x}")
    if extra_x_columns:
        errors.append(f"tabular_x_columns_not_in_whitelist={extra_x_columns}")
    if x_columns and whitelist and x_columns != whitelist:
        errors.append("tabular_x_column_order_does_not_match_whitelist")

    source_manifest, source_columns = _read_source_control_manifest(
        source_manifest_csv,
        errors,
    )
    source_audit = _read_json(
        source_audit_json,
        errors,
        "source_matched_view_audit",
    )
    source_check = _read_json(
        source_check_json,
        errors,
        "source_matched_view_check_audit",
    )
    domain_controls = _read_json(
        domain_controls_json,
        errors,
        "domain_controls_audit",
    )
    _check_source_controls(
        source_manifest,
        source_audit,
        source_check,
        errors,
        warnings,
    )
    _check_domain_controls(domain_controls, errors, warnings)

    return {
        "schema_version": LOADER_INPUT_AUDIT_SCHEMA_VERSION,
        "run_id": model_contract.get("run_id"),
        "agent_derived_root": (
            model_contract.get("lineage_roots", {}).get("agent_derived")
            if isinstance(model_contract.get("lineage_roots"), dict)
            else None
        ),
        "model_input_contract_json": str(model_input_contract_json),
        "canonical_fallback_used": False,
        "tabular_x_csv": str(tabular_x_csv),
        "feature_whitelist_json": str(whitelist_json),
        "feature_blacklist_json": str(blacklist_json),
        "source_matched_view_manifest_csv": str(source_manifest_csv),
        "source_matched_view_audit_json": str(source_audit_json),
        "source_matched_view_check_audit_json": str(source_check_json),
        "domain_controls_audit_json": str(domain_controls_json),
        "tabular_x_column_count": len(x_columns),
        "tabular_feature_whitelist_count": len(whitelist),
        "forbidden_x_columns": forbidden_x_columns,
        "whitelist_missing_in_tabular_x": whitelist_missing_in_x,
        "tabular_x_columns_not_in_whitelist": extra_x_columns,
        "source_selection_columns": source_columns,
        "source_control_rows": len(source_manifest),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def write_loader_input_audit(
    audit: dict[str, Any],
    *,
    output_path: Path,
    project_root: Path,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Write one audit only inside the manifest-declared agent root."""

    root = project_root.resolve()
    destination = (
        output_path.resolve()
        if output_path.is_absolute()
        else (root / output_path).resolve()
    )
    try:
        destination_relative = destination.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("loader audit output is outside project root") from exc
    agent_root_value = audit.get("agent_derived_root")
    if not isinstance(agent_root_value, str) or not agent_root_value.strip():
        raise ValueError("loader audit is missing agent_derived_root")
    agent_parts = PurePosixPath(agent_root_value)
    destination_parts = PurePosixPath(destination_relative)
    if destination_parts.parts[: len(agent_parts.parts)] != agent_parts.parts:
        raise ValueError("loader audit output is outside agent_derived_root")
    result = {
        **audit,
        "output_json": destination_relative,
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "artifact_written": False,
    }
    if dry_run:
        return result
    require_output_paths_available([destination], overwrite=overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    persisted = {**result, "artifact_written": True}
    destination.write_text(
        json.dumps(
            persisted,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return persisted


def _read_json(path: Path, errors: list[str], name: str) -> dict[str, Any]:
    """Read a JSON artifact and record missing/invalid payloads as audit errors."""

    if not path.is_file():
        errors.append(f"missing_{name}={path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_json_{name}={path}:{exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"invalid_json_object_{name}={path}")
        return {}
    return payload


def _read_csv_columns(path: Path, errors: list[str], name: str) -> list[str]:
    """Read only CSV headers so the audit is cheap on large training artifacts."""

    if not path.is_file():
        errors.append(f"missing_{name}={path}")
        return []
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        errors.append(f"invalid_csv_{name}={path}:{exc}")
        return []


def _match_forbidden_columns(columns: list[str], patterns: list[str]) -> list[str]:
    """Return columns that match forbidden leakage patterns."""

    out: list[str] = []
    for column in columns:
        if any(fnmatch.fnmatchcase(column, pattern) for pattern in patterns):
            out.append(column)
    return sorted(set(out))


def _bound_artifact_path(
    model_contract: dict[str, Any],
    *,
    group: str,
    name: str,
    root: Path,
    errors: list[str],
) -> Path:
    """Resolve and hash-check one manifest binding inside the project root."""

    groups = model_contract.get("artifact_groups")
    binding = groups.get(group, {}).get(name) if isinstance(groups, dict) else None
    invalid = root / f"__missing_{group}_{name}__"
    if not isinstance(binding, dict):
        errors.append(f"model_manifest_binding_missing:{group}:{name}")
        return invalid
    path_value = binding.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        errors.append(f"model_manifest_binding_path_missing:{group}:{name}")
        return invalid
    path = (root / Path(path_value)).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        errors.append(f"model_manifest_binding_outside_project:{group}:{name}")
        return invalid
    if binding.get("scope") != "agent_derived":
        errors.append(f"model_manifest_binding_not_agent_owned:{group}:{name}")
    if not path.is_file():
        errors.append(f"model_manifest_bound_artifact_missing:{group}:{name}")
        return path
    expected_size = binding.get("size_bytes")
    if expected_size != path.stat().st_size:
        errors.append(f"model_manifest_bound_size_mismatch:{group}:{name}")
    expected_hash = binding.get("sha256")
    if expected_hash != _sha256_file(path):
        errors.append(f"model_manifest_bound_hash_mismatch:{group}:{name}")
    return path


def _string_list(
    value: Any,
    errors: list[str],
    label: str,
) -> list[str]:
    """Require a nonempty, ordered, duplicate-free list of strings."""

    if not isinstance(value, list) or not value:
        errors.append(f"{label}_must_be_nonempty_list")
        return []
    if not all(isinstance(item, str) and item.strip() for item in value):
        errors.append(f"{label}_must_contain_nonempty_strings")
        return []
    values = [item.strip() for item in value]
    if len(values) != len(set(values)):
        errors.append(f"{label}_contains_duplicates")
    return values


def _read_source_control_manifest(
    path: Path,
    errors: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Load only source-control keys and masks; these never become model X."""

    required = [
        "window_id",
        "source_type",
        "view_matched_6frame",
        "source_class_balance_keep",
    ]
    if not path.is_file():
        errors.append(f"missing_source_matched_view_manifest={path}")
        return pd.DataFrame(columns=required), []
    try:
        frame = pd.read_csv(path, usecols=required, low_memory=False)
    except (OSError, ValueError) as exc:
        errors.append(f"invalid_source_matched_view_manifest={path}:{exc}")
        return pd.DataFrame(columns=required), []
    if frame["window_id"].isna().any() or frame["window_id"].astype(str).str.strip().eq("").any():
        errors.append("source_matched_view_blank_window_id")
    duplicate_count = int(frame["window_id"].duplicated().sum())
    if duplicate_count:
        errors.append(f"source_matched_view_duplicate_window_id={duplicate_count}")
    return frame, list(frame.columns)


def _check_source_controls(
    source_manifest: pd.DataFrame,
    source_audit: dict[str, Any],
    source_check: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Require non-destructive source controls and matching row evidence."""

    if source_audit.get("valid") is not True:
        errors.append("source_matched_view_audit_invalid")
    if source_audit.get("errors"):
        errors.append(f"source_matched_view_audit_errors={source_audit.get('errors')}")
    if source_check.get("valid") is not True:
        errors.append("source_matched_view_check_invalid")
    if source_check.get("errors"):
        errors.append(f"source_matched_view_check_errors={source_check.get('errors')}")
    rows = len(source_manifest)
    for label, audit in (
        ("source_matched_view_audit", source_audit),
        ("source_matched_view_check", source_check),
    ):
        reported = audit.get("rows")
        if reported is not None and reported != rows:
            errors.append(f"{label}_row_count_mismatch={reported}!={rows}")
    if rows == 0:
        errors.append("source_matched_view_rows_zero")
    if source_audit.get("warnings"):
        warnings.extend(
            f"source_matched_view_warning={warning}"
            for warning in source_audit.get("warnings", [])
        )


def _check_domain_controls(
    audit: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Require the aggregate shortcut-control audit to pass."""

    if audit.get("valid") is not True:
        errors.append("domain_controls_audit_invalid")
    if audit.get("errors"):
        errors.append(f"domain_controls_audit_errors={audit.get('errors')}")
    if audit.get("warnings"):
        warnings.extend(
            f"domain_controls_warning={warning}"
            for warning in audit.get("warnings", [])
        )


def _check_model_contract(
    model_contract: dict[str, Any],
    errors: list[str],
) -> None:
    """Check model input contract records the same non-leakage boundary."""

    if model_contract.get("schema_version") != MODEL_INPUT_MANIFEST_SCHEMA_VERSION:
        errors.append("model_input_contract_schema_version_mismatch")
    forbidden_model_inputs = model_contract.get("forbidden_model_inputs", [])
    if not forbidden_model_inputs:
        errors.append("model_input_contract_missing_forbidden_model_inputs")
    missing_artifacts = model_contract.get("missing_artifacts", [])
    if missing_artifacts:
        errors.append(f"model_input_contract_missing_artifacts={missing_artifacts}")
    branches = model_contract.get("model_input_branches", {})
    if not branches:
        errors.append("model_input_contract_missing_branches")
    path_policy = model_contract.get("path_policy")
    if not isinstance(path_policy, dict):
        errors.append("model_input_contract_missing_path_policy")
    elif path_policy.get("canonical_fallback_allowed") is not False:
        errors.append("model_input_contract_allows_canonical_fallback")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
