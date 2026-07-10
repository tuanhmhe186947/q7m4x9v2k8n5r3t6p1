"""Loader and sampler input audit for classification_v2.

The training loader is allowed to use source-domain controls as masks, weights,
or sampling manifests. It must not use source, path, review, manual, identity,
or label columns as model inputs. This audit checks the file-level contract
before smoke training so leakage is caught outside the trainer.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

import pandas as pd


def audit_loader_input_contract(
    *,
    trainer_contract_json: Path,
    model_input_contract_json: Path,
    source_domain_audit_json: Path,
    source_domain_manifest_csv: Path | None = None,
) -> dict[str, Any]:
    """Validate X whitelist and source-domain mask artifacts for training."""

    errors: list[str] = []
    warnings: list[str] = []
    trainer_contract = _read_json(trainer_contract_json, errors, "trainer_contract")
    model_contract = _read_json(model_input_contract_json, errors, "model_input_contract")
    source_audit = _read_json(source_domain_audit_json, errors, "source_domain_audit")

    whitelist = list(trainer_contract.get("tabular_feature_whitelist", []))
    forbidden_patterns = list(trainer_contract.get("forbidden_x_patterns", []))
    tabular_x_csv = Path(str(trainer_contract.get("tabular_x_csv", "")))
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

    source_selection_path = source_domain_manifest_csv or Path(str(source_audit.get("selection_manifest", "")))
    source_selection_columns = _read_csv_columns(source_selection_path, errors, "source_domain_selection_manifest")
    _check_source_domain_audit(source_audit, source_selection_columns, errors, warnings)
    _check_model_contract(model_contract, errors, warnings)

    return {
        "schema_version": "classification_v2_loader_input_audit_v1",
        "trainer_contract_json": str(trainer_contract_json),
        "model_input_contract_json": str(model_input_contract_json),
        "source_domain_audit_json": str(source_domain_audit_json),
        "source_domain_manifest_csv": str(source_selection_path),
        "tabular_x_csv": str(tabular_x_csv),
        "tabular_x_column_count": len(x_columns),
        "tabular_feature_whitelist_count": len(whitelist),
        "forbidden_x_columns": forbidden_x_columns,
        "whitelist_missing_in_tabular_x": whitelist_missing_in_x,
        "tabular_x_columns_not_in_whitelist": extra_x_columns,
        "source_selection_columns": source_selection_columns,
        "source_domain_rows": source_audit.get("rows"),
        "source_domain_kept_rows": source_audit.get("kept_rows"),
        "source_domain_balanced_strata_after_count": source_audit.get("balanced_strata_after_count"),
        "source_domain_imbalanced_strata_after_count": source_audit.get("imbalanced_strata_after_count"),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _read_json(path: Path, errors: list[str], name: str) -> dict[str, Any]:
    """Read a JSON artifact and record missing/invalid payloads as audit errors."""

    if not path.exists():
        errors.append(f"missing_{name}={path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid_json_{name}={path}:{exc}")
        return {}


def _read_csv_columns(path: Path, errors: list[str], name: str) -> list[str]:
    """Read only CSV headers so the audit is cheap on large training artifacts."""

    if not path.exists():
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


def _check_source_domain_audit(
    source_audit: dict[str, Any],
    selection_columns: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Ensure source-domain controls exist as mask metadata, not model X."""

    if source_audit.get("valid") is False:
        errors.append("source_domain_audit_invalid")
    if source_audit.get("errors"):
        errors.append(f"source_domain_audit_errors={source_audit.get('errors')}")
    required_manifest_cols = {
        "window_id",
        "source_type",
        "domain_control_keep",
        "domain_control_eligible",
        "domain_control_reason",
        "domain_control_stratum_key",
    }
    missing = sorted(required_manifest_cols.difference(selection_columns))
    if missing:
        errors.append(f"source_domain_selection_missing_columns={missing}")
    if source_audit.get("imbalanced_strata_after_count", 0):
        errors.append(f"source_domain_imbalanced_strata_after_count={source_audit.get('imbalanced_strata_after_count')}")
    if source_audit.get("forbidden_x_columns"):
        errors.append(f"source_domain_forbidden_x_columns={source_audit.get('forbidden_x_columns')}")
    if source_audit.get("kept_rows", 0) == 0:
        errors.append("source_domain_kept_rows_zero")
    if source_audit.get("warnings"):
        warnings.extend(f"source_domain_warning={warning}" for warning in source_audit.get("warnings", []))


def _check_model_contract(model_contract: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    """Check model input contract records the same non-leakage boundary."""

    forbidden_model_inputs = model_contract.get("forbidden_model_inputs", [])
    if not forbidden_model_inputs:
        errors.append("model_input_contract_missing_forbidden_model_inputs")
    missing_artifacts = model_contract.get("missing_artifacts", [])
    if missing_artifacts:
        warnings.append(f"model_input_contract_missing_artifacts={missing_artifacts}")
    branches = model_contract.get("model_input_branches", {})
    if not branches:
        errors.append("model_input_contract_missing_branches")
