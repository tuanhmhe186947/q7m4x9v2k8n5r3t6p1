"""Fail-closed identity checks for active Lightning scientific executions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

RESOURCE_NAMING_CONTRACT_SCHEMA = (
    "classification_v2.lightning_resource_naming_contract.v1"
)
RESOURCE_PREFLIGHT_SCHEMA = "classification_v2.lightning_resource_preflight.v1"
EXPECTED_RESOURCE = {
    "teamspace": "pig-project",
    "studio": "pig-gpu-l4",
    "ssh_alias": "lightning-pig-gcp",
    "gpu_count": 1,
    "gpu_model": "NVIDIA L4",
}


class LightningResourceIdentityError(ValueError):
    """Raised when an active Lightning resource is not the registered resource."""


def load_lightning_resource_contract(path: Path) -> dict[str, object]:
    """Read and validate the exact active resource-naming contract."""

    payload = _read_json(path)
    if payload.get("schema_version") != RESOURCE_NAMING_CONTRACT_SCHEMA:
        raise LightningResourceIdentityError("unsupported Lightning resource contract")
    if payload.get("contract_version") != "20260810-v2":
        raise LightningResourceIdentityError("Lightning resource contract version drifted")
    if payload.get("status") != "ACTIVE":
        raise LightningResourceIdentityError("Lightning resource contract is not active")
    expected = payload.get("expected_resource")
    if expected != EXPECTED_RESOURCE:
        raise LightningResourceIdentityError("Lightning expected resource identity drifted")
    rules = payload.get("rules")
    required_rules = {
        "teamspace_and_studio_must_not_be_inferred_from_each_other": True,
        "resource_type_must_be_explicit": True,
        "exact_match_only": True,
        "deprecated_active_studio_names": ["pig_project"],
    }
    if rules != required_rules:
        raise LightningResourceIdentityError("Lightning resource identity rules drifted")
    return payload


def build_lightning_resource_preflight(
    *,
    contract_path: Path,
    observed_resource: Mapping[str, object],
) -> dict[str, object]:
    """Validate a control-plane observation and return a portable PASS record."""

    load_lightning_resource_contract(contract_path)
    resource = _normalize_resource(observed_resource)
    for key, expected in EXPECTED_RESOURCE.items():
        if resource[key] != expected:
            raise LightningResourceIdentityError(
                f"Lightning resource identity mismatch={key}"
            )
    return {
        "schema_version": RESOURCE_PREFLIGHT_SCHEMA,
        "status": "PASS",
        "resource_contract_sha256": _sha256_file(contract_path),
        "resource": resource,
    }


def write_lightning_resource_preflight(
    *,
    contract_path: Path,
    observed_resource_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Create one immutable resource preflight from a control-plane observation."""

    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise LightningResourceIdentityError(
            f"Lightning resource preflight already exists={output_path}"
        )
    observed = _read_json(observed_resource_path)
    report = build_lightning_resource_preflight(
        contract_path=Path(contract_path).resolve(),
        observed_resource=observed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new_json(output_path, report)
    return report


def validate_passing_lightning_resource_preflight(
    *,
    contract_path: Path,
    preflight_path: Path,
) -> dict[str, object]:
    """Require a PASS preflight before a permit or CUDA executor proceeds."""

    load_lightning_resource_contract(contract_path)
    report = _read_json(preflight_path)
    expected_fields = {
        "schema_version",
        "status",
        "resource_contract_sha256",
        "resource",
    }
    if set(report) != expected_fields:
        raise LightningResourceIdentityError("Lightning resource preflight fields drifted")
    if report.get("schema_version") != RESOURCE_PREFLIGHT_SCHEMA:
        raise LightningResourceIdentityError("unsupported Lightning resource preflight")
    if report.get("status") != "PASS":
        raise LightningResourceIdentityError("Lightning resource preflight is not PASS")
    if report.get("resource_contract_sha256") != _sha256_file(contract_path):
        raise LightningResourceIdentityError("Lightning resource preflight contract drifted")
    build_lightning_resource_preflight(
        contract_path=contract_path,
        observed_resource=_mapping(report.get("resource"), "resource"),
    )
    return report


def _normalize_resource(resource: Mapping[str, object]) -> dict[str, object]:
    value = _mapping(resource, "resource")
    if set(value) != set(EXPECTED_RESOURCE):
        raise LightningResourceIdentityError("Lightning resource observation fields drifted")
    gpu_count = value.get("gpu_count")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int):
        raise LightningResourceIdentityError("Lightning GPU count is invalid")
    normalized = {
        "teamspace": str(value.get("teamspace", "")),
        "studio": str(value.get("studio", "")),
        "ssh_alias": str(value.get("ssh_alias", "")),
        "gpu_count": gpu_count,
        "gpu_model": str(value.get("gpu_model", "")),
    }
    if not all(normalized[key] for key in ("teamspace", "studio", "ssh_alias", "gpu_model")):
        raise LightningResourceIdentityError("Lightning resource observation is incomplete")
    return normalized


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LightningResourceIdentityError(f"Lightning {label} must be an object")
    return value


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LightningResourceIdentityError(
            f"Lightning JSON is unreadable={Path(path)}"
        ) from error
    if not isinstance(value, dict):
        raise LightningResourceIdentityError(f"Lightning JSON object required={Path(path)}")
    return value


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    try:
        temporary.replace(path)
    except FileExistsError as error:
        temporary.unlink(missing_ok=True)
        raise LightningResourceIdentityError(
            f"Lightning resource preflight already exists={path}"
        ) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
