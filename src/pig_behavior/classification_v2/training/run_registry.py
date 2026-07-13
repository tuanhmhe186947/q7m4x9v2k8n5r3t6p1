"""Append-only registry operations for isolated classification_v2 fold runs."""

from __future__ import annotations

import csv
import json
import os
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA_VERSION = "classification_v2.runs_registry.v1"
REGISTRY_FIELDS = (
    "registry_schema_version",
    "run_id",
    "experiment_name",
    "execution_profile",
    "fold_id",
    "seed",
    "status",
    "failure_reason",
    "code_sha",
    "dirty_worktree",
    "worktree_state_sha256",
    "config_sha256",
    "dataset_snapshot_id",
    "dataset_snapshot_sha256",
    "cache_sha256",
    "fold_manifest_sha256",
    "feature_whitelist_sha256",
    "temporal_view_selection_sha256",
    "fold_event_weight_sha256",
    "architecture_version",
    "backbone_name",
    "pretrained_weight_enum",
    "resolution",
    "temporal_view",
    "temporal_encoder_name",
    "modalities",
    "loss_name",
    "sampler_policy",
    "optimizer_name",
    "precision",
    "augmentation_policy",
    "gpu_model",
    "gpu_vram_bytes",
    "python_version",
    "torch_version",
    "runtime_seconds",
    "peak_vram_bytes",
    "checkpoint_manifest_path",
    "prediction_manifest_path",
    "metric_path",
    "run_manifest_path",
    "completed_at_utc",
)
TERMINAL_REGISTRY_STATUSES = frozenset({"completed", "failed"})
REGISTRY_LOCK_TIMEOUT_SECONDS = 30.0


def validate_registry_entry(entry: dict[str, Any]) -> None:
    """Reject malformed terminal rows before any registry file is mutated."""

    unknown = sorted(set(entry).difference(REGISTRY_FIELDS))
    missing = sorted(set(REGISTRY_FIELDS).difference(entry))
    if unknown or missing:
        raise ValueError(
            "registry entry schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if entry["registry_schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            "registry entry version mismatch="
            f"{entry['registry_schema_version']}"
        )
    if not str(entry["run_id"]).strip():
        raise ValueError("registry entry run_id must not be blank")
    if entry["status"] not in TERMINAL_REGISTRY_STATUSES:
        raise ValueError(f"registry entry is not terminal={entry['status']}")


def ensure_registry_header(path: Path) -> None:
    """Create or validate the exact CSV header under an inter-process lock."""

    with _registry_lock(path):
        _ensure_registry_header_unlocked(path)


def append_registry_entry(path: Path, entry: dict[str, Any]) -> None:
    """Append one validated row while preserving existing immutable run IDs."""

    validate_registry_entry(entry)
    with _registry_lock(path):
        _ensure_registry_header_unlocked(path)
        existing = _read_run_ids(path)
        run_id = str(entry["run_id"])
        if run_id in existing:
            raise FileExistsError(
                f"run_id already registered and immutable={run_id}"
            )
        _append_entries_unlocked(path, [entry])


def registry_registration_errors(
    path: Path,
    entry: dict[str, Any],
) -> list[str]:
    """Check that exactly one immutable CSV row matches its JSON entry."""

    errors: list[str] = []
    if not path.is_file():
        return [f"runs_registry_missing={path}"]
    try:
        _validate_registry_header(path)
    except ValueError as exc:
        return [str(exc)]
    with path.open("r", encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if row.get("run_id") == str(entry.get("run_id", ""))
        ]
    if len(matches) != 1:
        return [
            "runs_registry_run_id_count="
            f"{len(matches)}:{entry.get('run_id')}"
        ]
    observed = matches[0]
    for field in REGISTRY_FIELDS:
        expected_value = entry.get(field)
        expected = "" if expected_value is None else str(expected_value)
        if observed.get(field, "") != expected:
            errors.append(f"runs_registry_field_mismatch={field}")
    return errors


def merge_registry_entries(
    entry_paths: Iterable[Path],
    registry_csv: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate all independent fold rows before one locked append operation."""

    paths = sorted(Path(path) for path in entry_paths)
    if not paths:
        raise ValueError("registry merge requires at least one entry")
    entries = [_read_json(path) for path in paths]
    for entry in entries:
        validate_registry_entry(entry)
    run_ids = [str(entry["run_id"]) for entry in entries]
    duplicate_inputs = sorted(
        {run_id for run_id in run_ids if run_ids.count(run_id) > 1}
    )
    if duplicate_inputs:
        raise ValueError(
            f"registry merge duplicate run_ids={duplicate_inputs}"
        )
    if dry_run:
        existing = _validated_existing_run_ids(registry_csv)
        _raise_on_collisions(existing, run_ids)
    else:
        with _registry_lock(registry_csv):
            _ensure_registry_header_unlocked(registry_csv)
            existing = _read_run_ids(registry_csv)
            _raise_on_collisions(existing, run_ids)
            _append_entries_unlocked(registry_csv, entries)
    return {
        "registry_csv": str(registry_csv),
        "entry_count": len(entries),
        "run_ids": run_ids,
        "dry_run": dry_run,
        "errors": [],
        "valid": True,
    }


def _validated_existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    _validate_registry_header(path)
    return _read_run_ids(path)


def _raise_on_collisions(existing: set[str], run_ids: list[str]) -> None:
    collisions = sorted(existing.intersection(run_ids))
    if collisions:
        raise FileExistsError(
            f"registry merge run_id collisions={collisions}"
        )


def _append_entries_unlocked(
    path: Path,
    entries: list[dict[str, Any]],
) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REGISTRY_FIELDS))
        writer.writerows(entries)
        handle.flush()
        os.fsync(handle.fileno())


def _read_run_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row.get("run_id", "")) for row in csv.DictReader(handle)
        }


def _ensure_registry_header_unlocked(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REGISTRY_FIELDS))
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())
    _validate_registry_header(path)


def _validate_registry_header(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != list(REGISTRY_FIELDS):
        raise ValueError(f"runs registry header mismatch={header}")


@contextmanager
def _registry_lock(path: Path) -> Iterator[None]:
    """Serialize local writers; remote fold registries are merged afterward."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + REGISTRY_LOCK_TIMEOUT_SECONDS
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"runs registry lock timeout={lock_path}"
                ) from None
            time.sleep(0.05)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "append_registry_entry",
    "ensure_registry_header",
    "merge_registry_entries",
    "registry_registration_errors",
    "validate_registry_entry",
]
