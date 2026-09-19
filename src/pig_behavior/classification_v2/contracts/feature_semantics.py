"""Feature semantics audit for classification_v2 model inputs.

The audit assigns every train-ready feature to a scientific signal family
(geometry, motion, ROI, social, quality, mask/index) and records whether that
family is model-usable or needs deployment-time caution. This makes the paper
claim and trainer whitelist reviewable instead of relying on implicit column
names.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from pig_behavior.classification_v2.contracts.model_io import (
    forbidden_x_columns,
    read_csv_schema,
)


def load_feature_semantics(path: Path) -> dict[str, Any]:
    """Load the semantic family contract used by audit and trainer config."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trainer_contract(contract: dict[str, Any], root: Path) -> dict[str, Any]:
    """Load the trainer contract so semantics and trainer whitelist cannot drift."""

    trainer_path = contract.get("trainer_contract_json")
    if not trainer_path:
        return {}
    return json.loads((root / str(trainer_path)).read_text(encoding="utf-8"))


def _load_tabular_trainer_contract(
    contract: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Load the exact tabular whitelist used to build model X."""

    trainer_path = contract.get("tabular_trainer_contract_json")
    if not trainer_path:
        return {}
    return json.loads((root / str(trainer_path)).read_text(encoding="utf-8"))


def _spatial_model_input_whitelist(
    trainer_contract: dict[str, Any],
    contract: dict[str, Any],
) -> list[str]:
    """Return the authoritative model-input spatial arrays for the trainer."""

    whitelist = trainer_contract.get("spatial_sequence_feature_whitelist")
    if whitelist is None:
        whitelist = contract.get("model_input_spatial_arrays", [])
    return [str(name) for name in whitelist]


def audit_feature_semantics(
    contract_path: Path,
    *,
    tabular_x_csv: Path | None = None,
    spatial_npz: Path | None = None,
) -> dict[str, Any]:
    """Validate selected tabular/spatial artifacts against one contract."""
    contract = load_feature_semantics(contract_path)
    root = Path(".").resolve()
    tabular_path = _resolve_artifact_path(
        tabular_x_csv or Path(contract["tabular_x_csv"]),
        root,
    )
    spatial_path = _resolve_artifact_path(
        spatial_npz or Path(contract["spatial_npz"]),
        root,
    )
    trainer_contract = _load_trainer_contract(contract, root)
    tabular_trainer_contract = _load_tabular_trainer_contract(contract, root)
    spatial_model_inputs = _spatial_model_input_whitelist(trainer_contract, contract)

    tabular_columns = read_csv_schema(tabular_path)
    tabular_assignments = _assign_tabular_families(tabular_columns, contract["tabular_families"])
    spatial_assignments = _assign_spatial_arrays(spatial_path, contract["spatial_arrays"])
    forbidden = forbidden_x_columns(tabular_columns, contract.get("forbidden_x_patterns"))
    expected_tabular = [
        str(column)
        for column in tabular_trainer_contract.get("tabular_feature_whitelist", [])
    ]
    tabular_missing = sorted(set(expected_tabular).difference(tabular_columns))
    tabular_unexpected = sorted(set(tabular_columns).difference(expected_tabular))
    tabular_contract_match = tabular_columns == expected_tabular

    errors: list[str] = []
    warnings: list[str] = []
    unassigned = sorted(col for col, family in tabular_assignments.items() if family is None)
    if unassigned:
        errors.append(f"unassigned_tabular_features={unassigned}")
    if forbidden:
        errors.append(f"forbidden_tabular_x_features={forbidden}")
    if not expected_tabular:
        errors.append("tabular_trainer_whitelist_empty")
    if tabular_missing:
        errors.append(f"tabular_features_missing_from_x={tabular_missing}")
    if tabular_unexpected:
        errors.append(f"unexpected_tabular_x_features={tabular_unexpected}")
    if expected_tabular and not tabular_missing and not tabular_unexpected:
        if not tabular_contract_match:
            errors.append("tabular_x_feature_order_mismatch")
    missing_spatial = sorted(set(contract["spatial_arrays"]).difference(spatial_assignments))
    if missing_spatial:
        errors.append(f"missing_spatial_arrays={missing_spatial}")
    undeclared_spatial = sorted(
        name
        for name, assignment in spatial_assignments.items()
        if not assignment.get("declared")
    )
    if undeclared_spatial:
        errors.append(f"undeclared_spatial_arrays={undeclared_spatial}")
    spatial_role_errors = _spatial_model_input_role_errors(
        spatial_assignments,
        spatial_model_inputs,
    )
    errors.extend(spatial_role_errors)
    tabular_family_counts = _family_counts(tabular_assignments)
    roi_context = _roi_context_status(tabular_family_counts, spatial_assignments)
    if not roi_context["available"]:
        warnings.append("roi_context_absent_from_model_inputs")

    return {
        "contract_path": str(contract_path),
        "tabular_x_csv": _display_path(tabular_path, root),
        "spatial_npz": _display_path(spatial_path, root),
        "trainer_contract_json": _trainer_contract_display_path(contract, root),
        "tabular_trainer_contract_json": _tabular_trainer_contract_display_path(
            contract,
            root,
        ),
        "tabular_feature_count": int(len(tabular_columns)),
        "tabular_expected_feature_count": int(len(expected_tabular)),
        "tabular_contract_match": tabular_contract_match,
        "tabular_features_missing_from_x": tabular_missing,
        "unexpected_tabular_x_features": tabular_unexpected,
        "tabular_family_counts": tabular_family_counts,
        "tabular_assignments": tabular_assignments,
        "spatial_assignments": spatial_assignments,
        "spatial_model_input_whitelist": spatial_model_inputs,
        "spatial_model_input_array_count": int(len(spatial_model_inputs)),
        "spatial_non_model_arrays": _spatial_non_model_arrays(spatial_assignments),
        "spatial_model_input_role_errors": spatial_role_errors,
        "declared_spatial_array_count": int(
            len(spatial_assignments) - len(undeclared_spatial)
        ),
        "undeclared_spatial_arrays": undeclared_spatial,
        "roi_context": roi_context,
        "forbidden_tabular_features": forbidden,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _assign_tabular_families(columns: list[str], families: dict[str, Any]) -> dict[str, str | None]:
    """Assign columns by deterministic prefix rules declared in the semantics JSON."""
    assignments: dict[str, str | None] = {}
    for column in columns:
        matched = None
        for family_name, spec in families.items():
            if any(column.startswith(prefix) for prefix in spec.get("prefixes", [])):
                matched = family_name
                break
        assignments[column] = matched
    return assignments


def _assign_spatial_arrays(
    spatial_path: Path,
    declared_arrays: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Read NPZ array names/shapes and attach declared semantic meanings."""
    arrays = np.load(spatial_path, mmap_mode="r")
    assignments: dict[str, dict[str, Any]] = {}
    for name in arrays.files:
        spec = declared_arrays.get(name, {})
        arr = arrays[name]
        assignments[name] = {
            "family": spec.get("family"),
            "model_input_role": spec.get("model_input_role"),
            "model_input_allowed": bool(spec.get("model_input_allowed", False)),
            "description": _semantic_text_lines(spec.get("description", "")),
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "declared": bool(spec),
        }
    return assignments


def _spatial_model_input_role_errors(
    spatial_assignments: dict[str, dict[str, Any]],
    spatial_model_inputs: list[str],
) -> list[str]:
    """Fail if model feature groups include mask/index/audit-only arrays."""

    errors: list[str] = []
    missing = sorted(set(spatial_model_inputs).difference(spatial_assignments))
    if missing:
        errors.append(f"missing_model_input_spatial_arrays={missing}")
    invalid: list[dict[str, Any]] = []
    for name in spatial_model_inputs:
        assignment = spatial_assignments.get(name)
        if not assignment:
            continue
        if (
            assignment.get("model_input_role") != "model_input"
            or assignment.get("model_input_allowed") is not True
        ):
            invalid.append(
                {
                    "array": name,
                    "model_input_role": assignment.get("model_input_role"),
                    "model_input_allowed": assignment.get("model_input_allowed"),
                }
            )
    if invalid:
        errors.append(f"spatial_model_input_role_errors={invalid}")
    return errors


def _spatial_non_model_arrays(
    spatial_assignments: dict[str, dict[str, Any]],
) -> list[str]:
    """List arrays intentionally reserved for masks, indexing, or audit."""

    return sorted(
        name
        for name, assignment in spatial_assignments.items()
        if assignment.get("model_input_role") != "model_input"
    )


def _family_counts(assignments: dict[str, str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for family in assignments.values():
        key = family or "unassigned"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _roi_context_status(
    tabular_family_counts: dict[str, int],
    spatial_assignments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Report ROI signal availability without requiring label-derived tabular fields."""

    spatial_roi = spatial_assignments.get("roi_class_relation") or {}
    has_spatial_roi = spatial_roi.get("family") == "roi_context_sequence"
    tabular_count = int(tabular_family_counts.get("roi_context", 0))
    return {
        "available": bool(tabular_count > 0 or has_spatial_roi),
        "tabular_feature_count": tabular_count,
        "spatial_sequence_array": "roi_class_relation" if has_spatial_roi else None,
        "spatial_sequence_shape": spatial_roi.get("shape") if has_spatial_roi else None,
        "leakage_policy": _wrap_text(
            "Use label-independent ROI relation tensors; do not add "
            "target_roi_* or label-selected ROI audit fields to model X."
        ),
    }


def _display_path(path: Path, root: Path) -> str:
    """Prefer stable relative paths in generated audits."""

    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def _resolve_artifact_path(path: Path, root: Path) -> Path:
    """Resolve CLI smoke overrides and contract-relative project paths alike."""

    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _trainer_contract_display_path(contract: dict[str, Any], root: Path) -> str | None:
    """Return a stable trainer contract path when this audit is trainer-bound."""

    trainer_path = contract.get("trainer_contract_json")
    if not trainer_path:
        return None
    return _display_path(root / str(trainer_path), root)


def _tabular_trainer_contract_display_path(
    contract: dict[str, Any],
    root: Path,
) -> str | None:
    """Return the exact tabular whitelist contract path."""

    trainer_path = contract.get("tabular_trainer_contract_json")
    if not trainer_path:
        return None
    return _display_path(root / str(trainer_path), root)


def _wrap_text(value: str, *, width: int = 82) -> list[str]:
    """Store long audit prose as short JSON lines for readable diffs."""

    return textwrap.wrap(value, width=width) if value else []


def _semantic_text_lines(value: Any, *, width: int = 82) -> list[str]:
    """Accept string or pre-wrapped JSON text and emit compact audit lines."""

    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return _wrap_text(str(value), width=width)
