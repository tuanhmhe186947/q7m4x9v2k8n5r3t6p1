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


def audit_feature_semantics(contract_path: Path) -> dict[str, Any]:
    """Validate tabular/spatial model inputs against the declared feature semantics."""
    contract = load_feature_semantics(contract_path)
    root = Path(".").resolve()
    tabular_path = root / contract["tabular_x_csv"]
    spatial_path = root / contract["spatial_npz"]

    tabular_columns = read_csv_schema(tabular_path)
    tabular_assignments = _assign_tabular_families(tabular_columns, contract["tabular_families"])
    spatial_assignments = _assign_spatial_arrays(spatial_path, contract["spatial_arrays"])
    forbidden = forbidden_x_columns(tabular_columns, contract.get("forbidden_x_patterns"))

    errors: list[str] = []
    warnings: list[str] = []
    unassigned = sorted(col for col, family in tabular_assignments.items() if family is None)
    if unassigned:
        errors.append(f"unassigned_tabular_features={unassigned}")
    if forbidden:
        errors.append(f"forbidden_tabular_x_features={forbidden}")
    missing_spatial = sorted(set(contract["spatial_arrays"]).difference(spatial_assignments))
    if missing_spatial:
        errors.append(f"missing_spatial_arrays={missing_spatial}")
    tabular_family_counts = _family_counts(tabular_assignments)
    roi_context = _roi_context_status(tabular_family_counts, spatial_assignments)
    if not roi_context["available"]:
        warnings.append("roi_context_absent_from_model_inputs")

    return {
        "contract_path": str(contract_path),
        "tabular_x_csv": _display_path(tabular_path, root),
        "spatial_npz": _display_path(spatial_path, root),
        "tabular_feature_count": int(len(tabular_columns)),
        "tabular_family_counts": tabular_family_counts,
        "tabular_assignments": tabular_assignments,
        "spatial_assignments": spatial_assignments,
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
            "description": _wrap_text(str(spec.get("description", ""))),
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "declared": bool(spec),
        }
    return assignments


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


def _wrap_text(value: str, *, width: int = 82) -> list[str]:
    """Store long audit prose as short JSON lines for readable diffs."""

    return textwrap.wrap(value, width=width) if value else []
