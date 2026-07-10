"""Experiment-record gate for classification_v2.

Engineering smoke records only need basic provenance and artifact hashes. A
paper-facing record has a stricter contract: it must point to the frozen
training snapshot, paper-grade protocol, source-domain control, native OOF
folds, and trainer input contract. This prevents later training results from
being described as Q2 evidence unless the data/review/leakage gates are
traceable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PAPER_FACING_REQUIRED_PROVENANCE = (
    "dataset_snapshot_json",
    "paper_protocol_json",
    "paper_protocol_audit_json",
    "source_domain_audit_json",
    "native_oof_audit_json",
    "trainer_contract_json",
)


def check_experiment_record(record_path: Path) -> dict[str, Any]:
    """Validate a registry record and enforce Q2 provenance when required."""

    record = json.loads(record_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    required = ["schema_version", "name", "created_at_utc", "git_commit", "git_dirty", "artifacts", "record_path"]
    missing = [key for key in required if key not in record]
    if missing:
        errors.append(f"missing_record_keys={missing}")
    for artifact in record.get("artifacts", []):
        _check_artifact_record(artifact, errors)

    paper_facing = bool(record.get("paper_facing", False))
    stage = str(record.get("experiment_stage", "engineering_smoke"))
    if paper_facing or stage == "paper_facing_candidate":
        _check_paper_facing_provenance(record.get("provenance", {}), errors, warnings)
    else:
        warnings.append("engineering_smoke_record_not_valid_for_paper_claims")

    return {
        "record_json": str(record_path),
        "name": record.get("name"),
        "schema_version": record.get("schema_version"),
        "experiment_stage": stage,
        "paper_facing": paper_facing,
        "artifact_count": len(record.get("artifacts", [])),
        "git_commit": record.get("git_commit"),
        "git_dirty": record.get("git_dirty"),
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _check_artifact_record(artifact: dict[str, Any], errors: list[str]) -> None:
    if not artifact.get("exists"):
        errors.append(f"missing_artifact={artifact.get('path')}")
    if artifact.get("hash_status") == "ok" and not artifact.get("sha256"):
        errors.append(f"missing_sha256={artifact.get('path')}")


def _check_paper_facing_provenance(
    provenance: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    missing = [name for name in PAPER_FACING_REQUIRED_PROVENANCE if not provenance.get(name)]
    if missing:
        errors.append(f"missing_paper_facing_provenance={missing}")
        return
    for name in PAPER_FACING_REQUIRED_PROVENANCE:
        artifact = provenance.get(name) or {}
        _check_artifact_record(artifact, errors)
        if artifact.get("hash_status") == "skipped_large_file":
            warnings.append(f"provenance_hash_skipped={name}")
    _check_json_payload(provenance["paper_protocol_audit_json"], "paper_protocol_audit", errors)
    _check_json_payload(provenance["source_domain_audit_json"], "source_domain_audit", errors)
    _check_json_payload(provenance["native_oof_audit_json"], "native_oof_audit", errors)


def _check_json_payload(artifact: dict[str, Any], name: str, errors: list[str]) -> None:
    path = Path(str(artifact.get("path", "")))
    if not path.exists():
        errors.append(f"missing_json_payload={name}:{path}")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("errors"):
        errors.append(f"{name}_errors={payload.get('errors')}")
    if payload.get("valid") is False:
        errors.append(f"{name}_invalid")
