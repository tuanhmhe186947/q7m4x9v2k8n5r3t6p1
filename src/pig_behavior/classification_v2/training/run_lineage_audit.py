"""Read-only integrity audit for one classification_v2 run packet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
    is_sha256,
    payload_sha256,
)
from pig_behavior.classification_v2.training.run_lineage import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    PREDICTION_MANIFEST_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    TERMINAL_STATUSES,
)
from pig_behavior.classification_v2.training.run_registry import (
    registry_registration_errors,
    validate_registry_entry,
)


def audit_run_lineage(
    run_dir: Path,
    *,
    deep_input_hashes: bool = False,
    registry_csv: Path | None = None,
) -> dict[str, Any]:
    """Verify one packet; large frozen inputs are rehashed only on request."""

    required = {
        "run": "run_manifest.json",
        "environment": "environment.json",
        "artifacts": "artifact_manifest.json",
        "checkpoints": "checkpoint_manifest.json",
        "predictions": "prediction_manifest.json",
        "config": "resolved_config.json",
    }
    errors: list[str] = []
    paths = {name: run_dir / filename for name, filename in required.items()}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {
            "run_dir": str(run_dir),
            "status": "unknown",
            "errors": [f"missing_run_packet_files={missing}"],
            "input_hash_mode": (
                "deep" if deep_input_hashes else "size_only"
            ),
            "packet_integrity_valid": False,
            "registry_registered": False,
            "integrity_valid": False,
            "run_succeeded": False,
        }
    payloads = {
        name: _read_json(path)
        for name, path in paths.items()
        if name != "config"
    }
    schemas = {
        "run": RUN_MANIFEST_SCHEMA_VERSION,
        "environment": ENVIRONMENT_SCHEMA_VERSION,
        "artifacts": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "checkpoints": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "predictions": PREDICTION_MANIFEST_SCHEMA_VERSION,
    }
    for name, expected in schemas.items():
        observed = payloads[name].get("schema_version")
        if observed != expected:
            errors.append(f"schema_mismatch={name}:{observed}")
    environment = payloads["environment"]
    if not isinstance(environment.get("initial"), dict):
        errors.append("environment_initial_missing")
    if not isinstance(environment.get("resume_events"), list):
        errors.append("environment_resume_events_invalid")
    run = payloads["run"]
    identity = run.get("identity") or {}
    identity_hash = payload_sha256(identity)
    if identity_hash != run.get("identity_sha256"):
        errors.append("run_identity_hash_mismatch")
    resolved_config = _read_json(paths["config"])
    if payload_sha256(resolved_config) != identity.get("config_sha256"):
        errors.append("resolved_config_sha256_mismatch")
    for name in ["artifacts", "checkpoints", "predictions"]:
        if payloads[name].get("run_id") != run.get("run_id"):
            errors.append(f"run_id_mismatch={name}")
        if payloads[name].get("identity_sha256") != identity_hash:
            errors.append(f"identity_hash_mismatch={name}")
    for item in payloads["artifacts"].get("inputs") or []:
        _audit_recorded_file(
            item,
            errors,
            label="input",
            verify_hash=deep_input_hashes,
        )
    checkpoint_records = payloads["checkpoints"].get("checkpoints") or []
    checkpoint_hashes = {
        str(Path(str(item.get("path", ""))).resolve()): item.get("sha256")
        for item in checkpoint_records
    }
    for item in checkpoint_records:
        _audit_recorded_file(item, errors, label="checkpoint")
        _audit_recorded_file(
            {
                "path": item.get("audit_path"),
                "sha256": item.get("audit_sha256"),
                "size_bytes": item.get("audit_size_bytes", -1),
            },
            errors,
            label="checkpoint_audit",
        )
    prediction_records = payloads["predictions"].get("predictions") or []
    for item in prediction_records:
        _audit_recorded_file(item, errors, label="prediction")
        linked_path = str(
            Path(str(item.get("checkpoint_path", ""))).resolve()
        )
        if checkpoint_hashes.get(linked_path) != item.get("checkpoint_sha256"):
            errors.append(
                f"prediction_checkpoint_link_mismatch={item.get('path')}"
            )
    for item in payloads["artifacts"].get("outputs") or []:
        if item.get("path"):
            _audit_recorded_file(item, errors, label="output")
    status = str(run.get("status", "unknown"))
    for name in ["artifacts", "checkpoints", "predictions"]:
        if payloads[name].get("status") != status:
            errors.append(f"terminal_status_mismatch={name}")
    registry_entry: dict[str, Any] | None = None
    if status in TERMINAL_STATUSES:
        registry_entry = _audit_terminal_registry(
            run_dir,
            run,
            identity,
            status,
            errors,
        )
    packet_integrity_valid = not errors
    registry_errors: list[str] = []
    if registry_entry is not None:
        declared_registry = registry_csv or Path(
            str(run.get("registry_csv_path", ""))
        )
        registry_errors = registry_registration_errors(
            declared_registry,
            registry_entry,
        )
        errors.extend(registry_errors)
    return {
        "run_dir": str(run_dir),
        "run_id": run.get("run_id"),
        "status": status,
        "checkpoint_count": len(checkpoint_records),
        "prediction_count": len(prediction_records),
        "failure_reason": run.get("failure_reason", ""),
        "input_hash_mode": "deep" if deep_input_hashes else "size_only",
        "packet_integrity_valid": packet_integrity_valid,
        "registry_registered": (
            registry_entry is not None and not registry_errors
        ),
        "errors": errors,
        "integrity_valid": not errors,
        "run_succeeded": status == "completed" and not errors,
    }


def _audit_terminal_registry(
    run_dir: Path,
    run: dict[str, Any],
    identity: dict[str, Any],
    status: str,
    errors: list[str],
) -> dict[str, Any] | None:
    entry_path = run_dir / "registry_entry.json"
    if not entry_path.is_file():
        errors.append("terminal_run_missing_registry_entry")
        return None
    expected_hash = str(run.get("registry_entry_sha256", ""))
    if not is_sha256(expected_hash):
        errors.append("registry_entry_hash_missing")
    elif file_sha256(entry_path) != expected_hash:
        errors.append("registry_entry_sha256_mismatch")
    try:
        entry = _read_json(entry_path)
        validate_registry_entry(entry)
    except ValueError as exc:
        errors.append(f"registry_entry_invalid={exc}")
        return None
    if entry.get("run_id") != run.get("run_id"):
        errors.append("registry_entry_run_id_mismatch")
    if entry.get("status") != status:
        errors.append("registry_entry_status_mismatch")
    for key, expected in identity.items():
        if key not in entry:
            continue
        if key == "modalities":
            expected = "|".join(expected)
        if entry.get(key) != expected:
            errors.append(f"registry_entry_identity_mismatch={key}")
    return entry


def _audit_recorded_file(
    item: dict[str, Any],
    errors: list[str],
    *,
    label: str,
    verify_hash: bool = True,
) -> None:
    path = Path(str(item.get("path", "")))
    if not path.is_file():
        errors.append(f"{label}_missing={path}")
        return
    expected_size = int(item.get("size_bytes", -1))
    if path.stat().st_size != expected_size:
        errors.append(f"{label}_size_drift={path}")
    expected_hash = str(item.get("sha256", ""))
    if not is_sha256(expected_hash):
        errors.append(f"{label}_invalid_sha256={path}")
    elif verify_hash and file_sha256(path) != expected_hash:
        errors.append(f"{label}_sha256_drift={path}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


__all__ = ["audit_run_lineage"]
