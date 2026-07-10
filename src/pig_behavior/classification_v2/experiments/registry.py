"""Lightweight experiment records for classification_v2.

This registry is intentionally file-based. It records enough provenance for
smoke/baseline runs without introducing a tracking service dependency.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentRecordConfig:
    name: str
    output_dir: Path = Path("outputs/classification_v2/experiment_registry")
    metrics_json: Path | None = None
    artifacts: tuple[Path, ...] = field(default_factory=tuple)
    notes: str = ""
    max_hash_bytes: int = 100_000_000


def write_experiment_record(config: ExperimentRecordConfig) -> dict[str, Any]:
    """Write one immutable experiment record and append it to a JSONL ledger."""
    if not config.name.strip():
        raise ValueError("experiment name must not be empty")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_json(config.metrics_json) if config.metrics_json else None
    artifacts = [_artifact_record(path, max_hash_bytes=config.max_hash_bytes) for path in config.artifacts]
    record = {
        "schema_version": "classification_v2_experiment_record_v1",
        "name": config.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "metrics_json": str(config.metrics_json) if config.metrics_json else None,
        "metrics": metrics,
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


def _artifact_record(path: Path, *, max_hash_bytes: int) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "path": str(path),
        "exists": bool(exists),
        "size_bytes": None,
        "mtime_utc": None,
        "sha256": None,
        "hash_status": "missing",
    }
    if not exists:
        return record
    stat = path.stat()
    record["size_bytes"] = int(stat.st_size)
    record["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    if stat.st_size > max_hash_bytes:
        record["hash_status"] = f"skipped_large_file>{max_hash_bytes}"
        return record
    record["sha256"] = _sha256(path)
    record["hash_status"] = "ok"
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
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
