"""Fail-closed validation for Pig-STRENet review artifact runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_ARTIFACT_FILES = (
    "pair_manifest.csv",
    "history_features.csv",
    "stabilized_difference_summary.csv",
    "pig_strenet_artifact_audit.json",
    "artifact_manifest.json",
    "run_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pig_strenet_artifact_run(
    artifact_dir: Path,
    *,
    input_csv: Path,
    expected_run_scope: str,
    require_difference: bool = True,
    require_roi_visual: bool = True,
) -> dict[str, Any]:
    """Validate an artifact directory before review evidence is consumed."""

    root = Path(artifact_dir)
    errors: list[str] = []
    warnings: list[str] = []
    paths = {name: root / name for name in REQUIRED_ARTIFACT_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        errors.extend(f"missing_artifact_file:{name}" for name in missing)
        return _result(root, expected_run_scope, errors, warnings)

    try:
        artifact_audit = _read_json(paths["pig_strenet_artifact_audit.json"])
        artifact_manifest = _read_json(paths["artifact_manifest.json"])
        run_manifest = _read_json(paths["run_manifest.json"])
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid_artifact_json:{exc}")
        return _result(root, expected_run_scope, errors, warnings)

    if not artifact_audit.get("valid", False):
        errors.append("pig_strenet_artifact_audit_valid_false")
    if not artifact_audit.get("media_contract_valid", False):
        errors.append("pig_strenet_media_contract_valid_false")
    if not run_manifest.get("valid", False):
        errors.append("pig_strenet_run_manifest_valid_false")
    if run_manifest.get("run_scope") != expected_run_scope:
        errors.append(
            "pig_strenet_run_scope_mismatch:"
            f"{run_manifest.get('run_scope')}!={expected_run_scope}"
        )

    expected_input_sha = sha256_file(input_csv)
    audit_input_sha = artifact_audit.get("input_sha256")
    run_input = run_manifest.get("input", {})
    if audit_input_sha != expected_input_sha:
        errors.append("pig_strenet_input_sha256_mismatch")
    if run_input.get("sha256") != expected_input_sha:
        errors.append("pig_strenet_run_input_sha256_mismatch")

    manifest_ref = run_manifest.get("artifact_manifest", {})
    manifest_path = paths["artifact_manifest.json"]
    if manifest_ref.get("sha256") != sha256_file(manifest_path):
        errors.append("pig_strenet_artifact_manifest_sha256_mismatch")
    manifest_entries = artifact_manifest.get("files", [])
    manifest_names = {str(entry.get("name", "")) for entry in manifest_entries}
    for name in REQUIRED_ARTIFACT_FILES[:4]:
        if name not in manifest_names:
            errors.append(f"artifact_manifest_missing_required_file:{name}")
    for entry in manifest_entries:
        name = str(entry.get("name", ""))
        path = root / name
        if not path.is_file():
            errors.append(f"artifact_manifest_missing_file:{name}")
            continue
        if int(entry.get("size", -1)) != path.stat().st_size:
            errors.append(f"artifact_manifest_size_mismatch:{name}")
        if entry.get("sha256") != sha256_file(path):
            errors.append(f"artifact_manifest_sha256_mismatch:{name}")

    if require_difference:
        difference = artifact_audit.get("difference", {})
        if difference.get("status") not in {
            "PASS",
            "PASS_WITH_NATURAL_SKIPS",
        }:
            errors.append(
                f"difference_status_not_valid:{difference.get('status')}"
            )
        if int(difference.get("missing_available_frame_slots", 0)) != 0:
            errors.append("difference_missing_available_frame_slots")

    if require_roi_visual:
        roi = artifact_audit.get("packed_tensors", {}).get(
            "roi_visual_pixels", {}
        )
        if roi.get("status") != "PASS":
            errors.append(f"roi_visual_status_not_pass:{roi.get('status')}")
        if int(roi.get("missing_expected_rows", 0)) != 0:
            errors.append("roi_visual_missing_expected_rows")
    else:
        warnings.append("roi_visual_requirement_disabled")

    return _result(root, expected_run_scope, errors, warnings)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _result(
    root: Path,
    expected_run_scope: str,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "classification_v2.pig_strenet_artifact_gate.v1",
        "artifact_dir": str(root),
        "expected_run_scope": expected_run_scope,
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL",
    }


__all__ = ["audit_pig_strenet_artifact_run", "sha256_file"]
