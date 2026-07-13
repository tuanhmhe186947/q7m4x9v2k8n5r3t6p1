"""Immutable training snapshot contract for classification_v2.

The snapshot records file hashes, schemas, row counts, ordered window-id
digests, NPZ array contracts, and leakage guards. It is intentionally separate
from feature generation so it can verify that training uses exactly the audited
artifacts without rewriting source data or silently dropping rows.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.model_io import validate_model_input_columns

CSV_CHUNK_ROWS = 100_000
FILE_CHUNK_BYTES = 1024 * 1024
SNAPSHOT_SCHEMA_VERSION = "classification_v2.training_snapshot.v2"
ORDERED_KEY_HASH_VERSION = "newline_join_v1"


@dataclass(frozen=True)
class SnapshotPaths:
    """Resolved paths used by freeze/check scripts."""

    root: Path
    contract_json: Path
    output_dir: Path


def load_contract(path: Path) -> dict[str, Any]:
    """Load the JSON data contract that declares expected train-ready artifacts."""
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_training_snapshot(
    contract_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Create an immutable snapshot manifest for the current artifact state."""
    contract = load_contract(contract_path)
    paths = _resolve_paths(contract_path, contract, output_path=output_path)
    snapshot = _build_snapshot(paths, contract)
    if snapshot["errors"]:
        raise ValueError(
            "Cannot freeze an invalid training snapshot: "
            f"{snapshot['errors']}"
        )
    snapshot_id = _snapshot_id(snapshot)
    snapshot["snapshot_id"] = snapshot_id
    destination = output_path or (paths.output_dir / f"{snapshot_id}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)

    encoded = _stable_json(snapshot)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if _snapshot_identity_payload(existing) != _snapshot_identity_payload(snapshot):
            raise FileExistsError(
                "Snapshot already exists with different artifact content: "
                f"{destination}"
            )
        # Git provenance is recorded at first freeze, while the content ID is
        # intentionally stable across later checker/code-only commits.
        return {**existing, "snapshot_path": str(destination)}
    destination.write_text(encoded, encoding="utf-8")
    return {**snapshot, "snapshot_path": str(destination)}


def check_training_snapshot(
    snapshot_path: Path,
    *,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Compare current artifacts against a frozen snapshot and report deterministic drift."""
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    contract_file = contract_path or Path(expected["contract_path"])
    contract = load_contract(contract_file)
    paths = _resolve_paths(contract_file, contract)
    current = _build_snapshot(paths, contract)
    current["snapshot_id"] = _snapshot_id(current)

    errors: list[str] = []
    warnings: list[str] = []
    if expected.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        errors.append(
            "snapshot_schema_version_mismatch="
            f"expected:{SNAPSHOT_SCHEMA_VERSION},"
            f"actual:{expected.get('snapshot_schema_version')}"
        )
    if expected.get("errors"):
        errors.append(f"frozen_snapshot_has_contract_errors={expected['errors']}")
    if current.get("errors"):
        errors.append(f"current_snapshot_has_contract_errors={current['errors']}")
    _compare_artifacts(expected.get("artifacts", {}), current.get("artifacts", {}), errors)
    if expected.get("row_alignment") != current.get("row_alignment"):
        errors.append("row_alignment_drift")
    if expected.get("key_alignment") != current.get("key_alignment"):
        errors.append("key_alignment_drift")
    if expected.get("key_coverage") != current.get("key_coverage"):
        errors.append("key_coverage_drift")
    if expected.get("model_input_audit") != current.get("model_input_audit"):
        errors.append("model_input_audit_drift")
    if expected.get("contract_digest") != current.get("contract_digest"):
        errors.append("contract_json_digest_changed")
    if expected.get("git_commit") != current.get("git_commit"):
        warnings.append("git_commit_changed")
    if expected.get("snapshot_id") != current.get("snapshot_id"):
        errors.append("snapshot_id_drift")

    return {
        "snapshot_path": str(snapshot_path),
        "expected_snapshot_id": expected.get("snapshot_id"),
        "current_snapshot_id": current["snapshot_id"],
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "current": current,
    }


def _resolve_paths(
    contract_path: Path,
    contract: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> SnapshotPaths:
    # Contract roots are project-relative by design, so scripts run from the
    # repository root produce stable paths independent of the config directory.
    root = Path(contract.get("root", ".")).resolve()
    output_dir = (root / contract["snapshot_output_dir"]).resolve()
    if output_path is not None:
        output_dir = output_path.resolve().parent
    return SnapshotPaths(root=root, contract_json=contract_path.resolve(), output_dir=output_dir)


def _build_snapshot(paths: SnapshotPaths, contract: dict[str, Any]) -> dict[str, Any]:
    artifacts = {
        name: _profile_artifact(paths.root / spec["path"], spec)
        for name, spec in contract.get("artifacts", {}).items()
    }
    errors = _validate_contract_profiles(contract, artifacts)
    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "contract_version": contract.get("contract_version"),
        "snapshot_name": contract.get("snapshot_name"),
        "contract_path": str(paths.contract_json),
        "contract_digest": _sha256_file(paths.contract_json),
        "git_commit": _git_commit(paths.root),
        "primary_key": contract.get("primary_key", "window_id"),
        "lineage_audit_artifact": contract.get("lineage_audit_artifact"),
        "required_ordered_window_artifacts": contract.get(
            "required_ordered_window_artifacts",
            [],
        ),
        "artifacts": artifacts,
        "row_alignment": _row_alignment(contract, artifacts),
        "key_alignment": _key_alignment(contract, artifacts),
        "key_coverage": _key_coverage(contract, artifacts),
        "model_input_audit": _model_input_audit(contract, artifacts),
        "errors": errors,
    }


def _profile_artifact(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "path": str(path),
        "type": spec.get("type"),
        "required": bool(spec.get("required", False)),
        "exists": path.exists(),
    }
    if not path.exists():
        return profile
    profile.update({"size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)})
    if spec.get("type") == "csv":
        profile.update(_profile_csv(path, spec))
    elif spec.get("type") == "npz":
        profile.update(_profile_npz(path, spec))
    elif spec.get("type") == "json":
        profile["json_top_level_type"] = type(json.loads(path.read_text(encoding="utf-8"))).__name__
    return profile


def _profile_csv(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    row_count = 0
    dtypes: dict[str, str] = {}
    for chunk in pd.read_csv(path, chunksize=CSV_CHUNK_ROWS, low_memory=False):
        row_count += len(chunk)
        if not dtypes:
            dtypes = {col: str(dtype) for col, dtype in chunk.dtypes.items()}
    profile: dict[str, Any] = {
        "row_count": int(row_count),
        "columns": columns,
        "dtypes": dtypes,
        "missing_required_columns": sorted(
            set(spec.get("required_columns", [])).difference(columns)
        ),
    }
    key_column = spec.get("key_column")
    if key_column and key_column in columns:
        profile.update(_ordered_key_digest(path, key_column))
    return profile


def _profile_npz(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    arrays = np.load(path, mmap_mode="r")
    array_profiles: dict[str, Any] = {}
    first_axis_lengths: dict[str, int] = {}
    for name in arrays.files:
        arr = arrays[name]
        finite_count, nonfinite_count = _finite_counts(arr)
        array_profiles[name] = {
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "finite_count": int(finite_count),
            "nonfinite_count": int(nonfinite_count),
        }
        if arr.ndim:
            first_axis_lengths[name] = int(arr.shape[0])
    return {
        "arrays": array_profiles,
        "array_names": sorted(arrays.files),
        "missing_required_arrays": sorted(
            set(spec.get("required_arrays", [])).difference(arrays.files)
        ),
        "first_axis_lengths": first_axis_lengths,
        "row_count": next(iter(first_axis_lengths.values()), 0) if first_axis_lengths else 0,
    }


def _ordered_key_digest(path: Path, key_column: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    seen: set[str] = set()
    duplicates = 0
    null_count = 0
    first_value = True
    for chunk in pd.read_csv(
        path,
        usecols=[key_column],
        chunksize=CSV_CHUNK_ROWS,
        low_memory=False,
    ):
        for value in chunk[key_column].astype("string").fillna("").tolist():
            if value == "":
                null_count += 1
            if value in seen:
                duplicates += 1
            seen.add(value)
            if not first_value:
                digest.update(b"\n")
            digest.update(value.encode("utf-8"))
            first_value = False
    return {
        "key_column": key_column,
        "ordered_key_hash_version": ORDERED_KEY_HASH_VERSION,
        "ordered_key_sha256": digest.hexdigest(),
        "key_set_sha256": _key_set_digest(seen),
        "duplicate_key_count": int(duplicates),
        "null_key_count": int(null_count),
        "unique_key_count": int(len(seen)),
    }


def _key_set_digest(keys: set[str]) -> str:
    """Hash sorted unique keys when artifact order is not part of the contract."""
    digest = hashlib.sha256()
    for value in sorted(keys):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _finite_counts(arr: np.ndarray) -> tuple[int, int]:
    if not np.issubdtype(arr.dtype, np.number):
        return int(arr.size), 0
    finite = 0
    total = int(arr.size)
    if arr.ndim == 0:
        finite = int(np.isfinite(arr).item())
    else:
        for start in range(0, arr.shape[0], 1024):
            finite += int(np.isfinite(arr[start : start + 1024]).sum())
    return finite, total - finite


def _row_alignment(contract: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    names = contract.get("row_count_alignment_group", [])
    row_counts = {name: artifacts.get(name, {}).get("row_count") for name in names}
    non_null = {k: v for k, v in row_counts.items() if v is not None}
    return {
        "group": names,
        "row_counts": row_counts,
        "aligned": len(set(non_null.values())) <= 1,
        "expected_rows": next(iter(non_null.values()), None) if non_null else None,
    }


def _key_alignment(contract: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    """Compare ordered primary-key digests for artifacts that must be row-aligned."""
    source_name = contract.get("window_id_source_artifact")
    names = contract.get("key_alignment_group", [])
    source_digest = artifacts.get(source_name, {}).get("ordered_key_sha256")
    digests = {name: artifacts.get(name, {}).get("ordered_key_sha256") for name in names}
    mismatched = sorted(
        name
        for name, digest in digests.items()
        if not source_digest or not digest or digest != source_digest
    )
    return {
        "source_artifact": source_name,
        "group": names,
        "ordered_key_hash_version": ORDERED_KEY_HASH_VERSION,
        "ordered_key_sha256": digests,
        "aligned": not mismatched,
        "mismatched": mismatched,
    }


def _key_coverage(contract: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    """Check that joinable context artifacts contain the same key set as the source."""
    configured_groups = contract.get("key_coverage_groups")
    if configured_groups:
        groups = [_key_coverage_group(group, artifacts) for group in configured_groups]
        mismatched = [
            f"{group['source_artifact']}->{name}"
            for group in groups
            for name in group["mismatched"]
        ]
        return {"groups": groups, "covered": not mismatched, "mismatched": mismatched}
    source_name = contract.get("window_id_source_artifact")
    names = contract.get("key_coverage_group", [])
    return _key_coverage_group({"source_artifact": source_name, "artifacts": names}, artifacts)


def _key_coverage_group(group: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    """Compare one declared source key set with all dependent artifacts."""

    source_name = group.get("source_artifact")
    names = group.get("artifacts", [])
    source_digest = artifacts.get(source_name, {}).get("key_set_sha256")
    digests = {name: artifacts.get(name, {}).get("key_set_sha256") for name in names}
    mismatched = sorted(
        name
        for name, digest in digests.items()
        if not source_digest or digest != source_digest
    )
    return {
        "source_artifact": source_name,
        "group": names,
        "key_set_sha256": digests,
        "covered": not mismatched,
        "mismatched": mismatched,
    }


def _model_input_audit(contract: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    audits = {}
    for name, spec in contract.get("artifacts", {}).items():
        if spec.get("model_input") and artifacts.get(name, {}).get("type") == "csv":
            audits[name] = validate_model_input_columns(
                artifacts[name].get("columns", []),
                forbidden_patterns=contract.get("forbidden_x_patterns"),
            )
    return audits


def _validate_contract_profiles(contract: dict[str, Any], artifacts: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name, spec in contract.get("artifacts", {}).items():
        profile = artifacts.get(name, {})
        if spec.get("required") and not profile.get("exists"):
            errors.append(f"missing_required_artifact:{name}")
        for column in profile.get("missing_required_columns", []):
            errors.append(f"missing_required_column:{name}.{column}")
        for array in profile.get("missing_required_arrays", []):
            errors.append(f"missing_required_array:{name}.{array}")
        if profile.get("duplicate_key_count", 0):
            errors.append(f"duplicate_key:{name}={profile['duplicate_key_count']}")
        if profile.get("null_key_count", 0):
            errors.append(f"blank_key:{name}={profile['null_key_count']}")
    alignment = _row_alignment(contract, artifacts)
    if not alignment["aligned"]:
        errors.append(f"row_count_alignment_mismatch={alignment['row_counts']}")
    key_alignment = _key_alignment(contract, artifacts)
    if not key_alignment["aligned"]:
        errors.append(f"key_alignment_mismatch={key_alignment['mismatched']}")
    required_ordered = set(
        contract.get("required_ordered_window_artifacts", [])
    )
    declared_ordered = set(contract.get("key_alignment_group", []))
    undeclared_ordered = sorted(required_ordered.difference(declared_ordered))
    if undeclared_ordered:
        errors.append(
            "required_ordered_artifacts_not_aligned="
            f"{undeclared_ordered}"
        )
    key_coverage = _key_coverage(contract, artifacts)
    if not key_coverage["covered"]:
        errors.append(f"key_coverage_mismatch={key_coverage['mismatched']}")
    for name, audit in _model_input_audit(contract, artifacts).items():
        if audit.get("forbidden_columns"):
            errors.append(f"forbidden_x_columns:{name}={audit['forbidden_columns']}")
    return errors


def _compare_artifacts(
    expected: dict[str, Any],
    current: dict[str, Any],
    errors: list[str],
) -> None:
    for name, expected_profile in expected.items():
        current_profile = current.get(name)
        if current_profile is None:
            errors.append(f"artifact_missing_from_current:{name}")
            continue
        for field in (
            "exists",
            "size_bytes",
            "sha256",
            "row_count",
            "columns",
            "array_names",
            "arrays",
            "ordered_key_hash_version",
            "key_set_sha256",
            "duplicate_key_count",
            "null_key_count",
            "unique_key_count",
        ):
            if expected_profile.get(field) != current_profile.get(field):
                errors.append(f"artifact_{field}_drift:{name}")
        if expected_profile.get("ordered_key_sha256") != current_profile.get("ordered_key_sha256"):
            errors.append(f"artifact_ordered_key_drift:{name}")


def _snapshot_id(snapshot: dict[str, Any]) -> str:
    # The snapshot ID identifies artifact/contract content. The git commit is
    # recorded for audit, but later checker commits must not rename the data.
    return "c2v2_" + hashlib.sha256(
        _stable_json(_snapshot_identity_payload(snapshot)).encode("utf-8")
    ).hexdigest()[:16]


def _snapshot_identity_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return fields that define artifact identity, excluding code provenance."""

    excluded = {"snapshot_id", "snapshot_path", "git_commit"}
    return {key: value for key, value in snapshot.items() if key not in excluded}


def _stable_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
