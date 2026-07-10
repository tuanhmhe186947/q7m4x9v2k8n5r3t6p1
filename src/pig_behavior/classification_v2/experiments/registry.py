"""Lightweight experiment records for classification_v2.

The registry is intentionally file-based so smoke and baseline runs can record
provenance without depending on an external tracking service. Paper-facing
records must also carry the audited data snapshot, protocol, source-domain
control, and native OOF references so a result cannot be promoted from a loose
smoke run by accident.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.evaluation.native_temporal_metrics_gate import default_evaluation_contract


@dataclass(frozen=True, slots=True)
class ExperimentRecordConfig:
    """Inputs used to write a reproducible experiment record."""

    name: str
    output_dir: Path = Path("outputs/classification_v2/experiment_registry")
    metrics_json: Path | None = None
    artifacts: tuple[Path, ...] = field(default_factory=tuple)
    notes: str = ""
    experiment_stage: str = "engineering_smoke"
    paper_facing: bool = False
    dataset_snapshot_json: Path | None = None
    paper_protocol_json: Path | None = None
    paper_protocol_audit_json: Path | None = None
    source_domain_audit_json: Path | None = None
    native_oof_audit_json: Path | None = None
    trainer_contract_json: Path | None = None
    result_kind: str = "protocol_gate"
    primary_metric_unit: str = "native_temporal_unit"
    split_policy: str = "recording_group_oof"
    external_generalization_claim: bool = False
    max_hash_bytes: int = 100_000_000


def write_experiment_record(config: ExperimentRecordConfig) -> dict[str, Any]:
    """Write one experiment record and append it to the JSONL ledger."""

    if not config.name.strip():
        raise ValueError("experiment name must not be empty")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_json(config.metrics_json) if config.metrics_json else None
    artifacts = [_artifact_record(path, max_hash_bytes=config.max_hash_bytes) for path in config.artifacts]
    record: dict[str, Any] = {
        "schema_version": "classification_v2_experiment_record_v1",
        "name": config.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "metrics_json": str(config.metrics_json) if config.metrics_json else None,
        "metrics": metrics,
        "experiment_stage": config.experiment_stage,
        "paper_facing": bool(config.paper_facing),
        "provenance": _provenance_record(config),
        "evaluation_contract": _evaluation_contract(config),
        "artifacts": artifacts,
        "notes": config.notes,
    }
    record_path = config.output_dir / f"{_safe_name(config.name)}_record.json"
    ledger_path = config.output_dir / "experiment_ledger.jsonl"
    record["record_path"] = str(record_path)
    record["ledger_path"] = str(ledger_path)
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _provenance_record(config: ExperimentRecordConfig) -> dict[str, Any]:
    """Return optional Q2 gate references for an experiment record."""

    paths = {
        "dataset_snapshot_json": config.dataset_snapshot_json,
        "paper_protocol_json": config.paper_protocol_json,
        "paper_protocol_audit_json": config.paper_protocol_audit_json,
        "source_domain_audit_json": config.source_domain_audit_json,
        "native_oof_audit_json": config.native_oof_audit_json,
        "trainer_contract_json": config.trainer_contract_json,
    }
    return {
        name: _artifact_record(path, max_hash_bytes=config.max_hash_bytes) if path is not None else None
        for name, path in paths.items()
    }


def _evaluation_contract(config: ExperimentRecordConfig) -> dict[str, Any]:
    """Return the native-temporal evaluation contract stored with each record."""

    contract = default_evaluation_contract()
    contract.update(
        {
            "result_kind": config.result_kind,
            "primary_metric_unit": config.primary_metric_unit,
            "split_policy": config.split_policy,
            "external_generalization_claim": bool(config.external_generalization_claim),
        }
    )
    return contract


def _artifact_record(path: Path, *, max_hash_bytes: int) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {"path": str(path), "exists": exists}
    if not exists:
        return record
    stat = path.stat()
    record.update(
        {
            "size_bytes": int(stat.st_size),
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
    )
    if stat.st_size <= max_hash_bytes:
        record["sha256"] = _sha256(path)
        record["hash_status"] = "ok"
    else:
        record["hash_status"] = "skipped_large_file"
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"error": f"missing_metrics_json={path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe.strip("_") or "experiment"
