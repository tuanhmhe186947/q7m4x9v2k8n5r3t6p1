"""Export run-bound, leakage-safe train-ready tables and feature contracts."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.versioned_data_contract import (
    GENERATED_CONTRACT_SCHEMA_VERSION,
    validate_generated_data_contract,
)
from pig_behavior.classification_v2.contracts.window_alignment import (
    require_ordered_window_ids,
)
from pig_behavior.classification_v2.train_ready_features import (
    build_train_ready_window_tables,
)

FEATURE_SPEC_SCHEMA_VERSION = (
    "classification_v2.reviewed_q2_tabular_feature_spec.v1"
)
FEATURE_WHITELIST_SCHEMA_VERSION = (
    "classification_v2.feature_whitelist.v1"
)
FEATURE_BLACKLIST_SCHEMA_VERSION = (
    "classification_v2.feature_blacklist.v1"
)
FEATURE_AUDIT_SCHEMA_VERSION = (
    "classification_v2.feature_whitelist_audit.v1"
)
TRAIN_READY_AUDIT_SCHEMA_VERSION = (
    "classification_v2.train_ready_export_audit.v2"
)
EXPECTED_TABULAR_FEATURE_COUNT = 102


def parse_args() -> argparse.Namespace:
    """Require a generated contract instead of canonical path defaults."""

    parser = argparse.ArgumentParser(
        description=(
            "Export leakage-safe X/y/mask/sample_weight tables from reviewed "
            "sequence windows."
        )
    )
    parser.add_argument(
        "--data-contract-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--label-col", default="behavior_window_label")
    parser.add_argument("--mask-col", default="window_valid_for_main_train")
    parser.add_argument("--sample-weight-col", default="window_sample_weight")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived export files in the selected output dir.",
    )
    return parser.parse_args()


def main() -> None:
    """Export only to paths declared by one generated reviewed-Q2 contract."""

    args = parse_args()
    audit = export_train_ready_from_contract(
        args.data_contract_json,
        project_root=args.project_root,
        label_col=args.label_col,
        mask_col=args.mask_col,
        sample_weight_col=args.sample_weight_col,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=True))


def export_train_ready_from_contract(
    data_contract_json: Path,
    *,
    project_root: Path,
    label_col: str,
    mask_col: str,
    sample_weight_col: str,
    max_rows: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Build X/y/masks and run-bound feature files without path inference."""

    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be > 0")
    root = project_root.resolve()
    contract_path = _project_file(data_contract_json, root)
    contract_errors = validate_generated_data_contract(
        contract_path,
        project_root=root,
    )
    contract = _read_json(contract_path)
    if contract_errors:
        raise ValueError(f"Generated data contract is invalid: {contract_errors}")
    if (
        contract.get("generated_contract_schema_version")
        != GENERATED_CONTRACT_SCHEMA_VERSION
    ):
        raise ValueError("Reviewed-Q2 exporter requires generated contract v2")
    if contract.get("profile") != "mixed-reviewed":
        raise ValueError("Reviewed-Q2 exporter requires profile=mixed-reviewed")

    feature_spec_path = _artifact_path(
        contract,
        "tabular_feature_spec",
        root=root,
        expected_scope="project_static",
        must_exist=True,
    )
    input_path = _artifact_path(
        contract,
        "sequence_window_features",
        root=root,
        expected_scope="agent_derived",
        must_exist=True,
    )
    outputs = {
        name: _artifact_path(
            contract,
            name,
            root=root,
            expected_scope="agent_derived",
        )
        for name in (
            "tabular_X",
            "y_behavior",
            "train_mask",
            "sample_weight",
            "feature_whitelist",
            "feature_blacklist",
            "feature_whitelist_audit",
            "train_ready_audit",
        )
    }
    train_ready_root = _project_path(
        contract.get("train_ready_root"),
        root,
        label="train_ready_root",
    )
    outside_train_ready = sorted(
        name
        for name, path in outputs.items()
        if not _is_within(path, train_ready_root)
    )
    if outside_train_ready:
        raise ValueError(
            "Train-ready outputs are outside declared train_ready_root: "
            f"{outside_train_ready}"
        )
    require_output_paths_available(
        list(outputs.values()),
        overwrite=overwrite,
    )

    feature_spec = _read_json(feature_spec_path)
    features = _validate_feature_spec(feature_spec)
    forbidden_patterns = _string_list(
        contract.get("forbidden_x_patterns"),
        "forbidden_x_patterns",
    )
    leaked_features = sorted(
        feature
        for feature in features
        if any(
            fnmatch.fnmatchcase(feature, pattern)
            for pattern in forbidden_patterns
        )
    )
    if leaked_features:
        raise ValueError(
            f"Feature spec matches forbidden X patterns: {leaked_features}"
        )

    source = pd.read_csv(input_path, low_memory=False)
    source_rows = len(source)
    selected = (
        source.head(max_rows).copy()
        if max_rows is not None
        else source.copy()
    )
    if "window_id" not in selected.columns:
        raise ValueError("Input sequence windows are missing window_id")
    window_alignment = require_ordered_window_ids(
        "train_ready_input",
        selected["window_id"],
    )

    tables = build_train_ready_window_tables(
        selected,
        label_col=label_col,
        mask_col=mask_col,
        sample_weight_col=sample_weight_col,
        feature_whitelist=features,
    )
    if tables.audit["errors"]:
        raise ValueError(
            f"Train-ready feature audit failed: {tables.audit['errors']}"
        )
    if list(tables.x.columns) != features:
        raise ValueError("Exported X columns do not match feature spec order")

    contract_sha256 = _sha256_file(contract_path)
    feature_spec_sha256 = _sha256_file(feature_spec_path)
    whitelist_payload = {
        "schema_version": FEATURE_WHITELIST_SCHEMA_VERSION,
        "profile": "mixed-reviewed",
        "run_id": contract.get("run_id"),
        "data_contract": _relative_path(contract_path, root),
        "data_contract_sha256": contract_sha256,
        "feature_spec": _relative_path(feature_spec_path, root),
        "feature_spec_sha256": feature_spec_sha256,
        "feature_count": len(features),
        "features": features,
    }
    blacklist_payload = {
        "schema_version": FEATURE_BLACKLIST_SCHEMA_VERSION,
        "profile": "mixed-reviewed",
        "run_id": contract.get("run_id"),
        "data_contract": _relative_path(contract_path, root),
        "data_contract_sha256": contract_sha256,
        "forbidden_pattern_count": len(forbidden_patterns),
        "forbidden_patterns": forbidden_patterns,
    }
    feature_audit = {
        "schema_version": FEATURE_AUDIT_SCHEMA_VERSION,
        "profile": "mixed-reviewed",
        "run_id": contract.get("run_id"),
        "valid": True,
        "errors": [],
        "never_use_all_numeric_columns": True,
        "fail_closed_on_unknown_columns": True,
        "feature_spec_sha256": feature_spec_sha256,
        "data_contract_sha256": contract_sha256,
        "feature_count": len(features),
        "x_columns_match_whitelist": True,
        "forbidden_features": [],
        "feature_selection": tables.audit,
    }

    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    tables.x.to_csv(outputs["tabular_X"], index=False)
    tables.y.rename(label_col).to_frame().to_csv(
        outputs["y_behavior"],
        index=False,
    )
    tables.mask.rename(mask_col).to_frame().to_csv(
        outputs["train_mask"],
        index=False,
    )
    tables.sample_weight.rename(sample_weight_col).to_frame().to_csv(
        outputs["sample_weight"],
        index=False,
    )
    _write_json(outputs["feature_whitelist"], whitelist_payload)
    _write_json(outputs["feature_blacklist"], blacklist_payload)
    _write_json(outputs["feature_whitelist_audit"], feature_audit)

    row_counts = {
        "source_input": int(source_rows),
        "selected_input": int(len(selected)),
        "X": int(len(tables.x)),
        "y": int(len(tables.y)),
        "mask": int(len(tables.mask)),
        "sample_weight": int(len(tables.sample_weight)),
        "mask_true": int(tables.mask.sum()),
        "mask_false": int((~tables.mask).sum()),
    }
    selected_count = row_counts["selected_input"]
    aligned_counts = {
        row_counts[name]
        for name in ("X", "y", "mask", "sample_weight")
    }
    errors = []
    if aligned_counts != {selected_count}:
        errors.append("train_ready_row_count_mismatch")
    audit = {
        "schema_version": TRAIN_READY_AUDIT_SCHEMA_VERSION,
        "profile": "mixed-reviewed",
        "run_id": contract.get("run_id"),
        "valid": not errors,
        "errors": errors,
        "complete_export": max_rows is None,
        "max_rows": max_rows,
        "canonical_fallback_used": False,
        "data_contract": _relative_path(contract_path, root),
        "data_contract_sha256": contract_sha256,
        "input_csv": _relative_path(input_path, root),
        "input_csv_sha256": _sha256_file(input_path),
        "rows": {
            **row_counts,
            "row_count_preserved": aligned_counts == {selected_count},
        },
        "window_alignment": window_alignment,
        "feature_selection": tables.audit,
        "outputs": {
            name: {
                "path": _relative_path(path, root),
                "sha256": _sha256_file(path),
            }
            for name, path in outputs.items()
            if name != "train_ready_audit"
        },
    }
    if errors:
        raise ValueError(f"Train-ready export failed: {errors}")
    _write_json(outputs["train_ready_audit"], audit)
    return audit


def _project_file(path: Path, root: Path) -> Path:
    """Resolve one existing file while forbidding project-root escape."""

    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not _is_within(resolved, root):
        raise ValueError(f"Path is outside project root: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _artifact_path(
    contract: dict[str, Any],
    name: str,
    *,
    root: Path,
    expected_scope: str,
    must_exist: bool = False,
) -> Path:
    """Resolve one explicitly declared artifact and validate its owner."""

    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Generated contract artifacts must be an object")
    entry = artifacts.get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"Generated contract has no artifact: {name}")
    if entry.get("scope") != expected_scope:
        raise ValueError(
            f"Artifact {name} must have scope={expected_scope}"
        )
    resolved = _project_path(
        entry.get("path"),
        root,
        label=f"artifact:{name}",
    )
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _project_path(value: Any, root: Path, *, label: str) -> Path:
    """Resolve a nonempty project-relative path without traversal."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path")
    path = Path(value.strip())
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be project-relative without traversal")
    resolved = (root / path).resolve()
    if not _is_within(resolved, root):
        raise ValueError(f"{label} is outside project root")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject array/scalar payloads."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return payload


def _string_list(value: Any, label: str) -> list[str]:
    """Return unique, nonblank strings while preserving declared order."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    items = [item.strip() if isinstance(item, str) else "" for item in value]
    if any(not item for item in items):
        raise ValueError(f"{label} must contain only nonblank strings")
    if len(items) != len(set(items)):
        raise ValueError(f"{label} contains duplicate values")
    return items


def _validate_feature_spec(payload: dict[str, Any]) -> list[str]:
    """Validate the path-free, ordered reviewed-Q2 tabular feature spec."""

    expected_fields = {
        "schema_version",
        "profile",
        "selection_policy",
        "features",
    }
    if set(payload) != expected_fields:
        raise ValueError(
            "Feature spec fields mismatch: "
            f"expected={sorted(expected_fields)}, actual={sorted(payload)}"
        )
    if payload.get("schema_version") != FEATURE_SPEC_SCHEMA_VERSION:
        raise ValueError("Feature spec schema_version mismatch")
    if payload.get("profile") != "mixed-reviewed":
        raise ValueError("Feature spec requires profile=mixed-reviewed")
    expected_policy = {
        "explicit_ordered_whitelist": True,
        "all_numeric_selection_allowed": False,
        "unknown_feature_fails_closed": True,
        "inference_available_only": True,
    }
    if payload.get("selection_policy") != expected_policy:
        raise ValueError("Feature spec selection_policy mismatch")
    features = _string_list(payload.get("features"), "features")
    if len(features) != EXPECTED_TABULAR_FEATURE_COUNT:
        raise ValueError(
            "Feature spec count mismatch: "
            f"expected={EXPECTED_TABULAR_FEATURE_COUNT}, actual={len(features)}"
        )
    return features


def _sha256_file(path: Path) -> str:
    """Hash one artifact without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    """Return one stable POSIX path after enforcing project containment."""

    resolved = path.resolve()
    if not _is_within(resolved, root):
        raise ValueError(f"Path is outside project root: {path}")
    return resolved.relative_to(root).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON for stable content hashing."""

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is equal to or below a root."""

    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
