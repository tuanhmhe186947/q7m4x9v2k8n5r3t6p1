"""Trainer input contract checks for classification_v2.

This module makes the trainer boundary explicit: tabular X must match an
approved whitelist, y/masks/weights/splits must come from known artifacts, and
metadata/review/path columns are rejected even if they are numeric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.contracts.model_io import forbidden_x_columns, read_csv_schema


def check_trainer_contract(contract_path: Path) -> dict[str, Any]:
    """Validate the trainer contract against current train-ready artifacts."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    x_path = Path(contract["tabular_x_csv"])
    x_columns = read_csv_schema(x_path)
    whitelist = list(contract["tabular_feature_whitelist"])
    forbidden = forbidden_x_columns(x_columns, contract.get("forbidden_x_patterns"))
    missing_from_x = sorted(set(whitelist).difference(x_columns))
    extra_in_x = sorted(set(x_columns).difference(whitelist))
    duplicate_whitelist = _duplicates(whitelist)

    errors: list[str] = []
    if missing_from_x:
        errors.append(f"whitelist_columns_missing_from_x={missing_from_x}")
    if extra_in_x:
        errors.append(f"x_columns_not_in_whitelist={extra_in_x}")
    if forbidden:
        errors.append(f"forbidden_x_columns={forbidden}")
    if duplicate_whitelist:
        errors.append(f"duplicate_whitelist_columns={duplicate_whitelist}")
    for artifact_name in _all_declared_artifacts(contract):
        path = Path(contract["train_ready_root"]) / artifact_name
        if not path.exists():
            errors.append(f"missing_declared_artifact={path}")

    return {
        "contract_path": str(contract_path),
        "tabular_x_csv": str(x_path),
        "feature_count": int(len(x_columns)),
        "whitelist_count": int(len(whitelist)),
        "x_matches_whitelist_order": x_columns == whitelist,
        "missing_from_x": missing_from_x,
        "extra_in_x": extra_in_x,
        "forbidden_x_columns": forbidden,
        "allowed_x_artifacts": contract.get("allowed_x_artifacts", []),
        "allowed_y_artifacts": contract.get("allowed_y_artifacts", []),
        "allowed_mask_weight_split_artifacts": contract.get("allowed_mask_weight_split_artifacts", []),
        "rules": contract.get("rules", []),
        "errors": errors,
        "valid": not errors,
    }


def _all_declared_artifacts(contract: dict[str, Any]) -> list[str]:
    artifacts: list[str] = []
    for key in ["allowed_x_artifacts", "allowed_y_artifacts", "allowed_mask_weight_split_artifacts"]:
        artifacts.extend(str(value) for value in contract.get(key, []))
    return artifacts


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for value in values:
        if value in seen:
            duplicated.add(value)
        seen.add(value)
    return sorted(duplicated)
