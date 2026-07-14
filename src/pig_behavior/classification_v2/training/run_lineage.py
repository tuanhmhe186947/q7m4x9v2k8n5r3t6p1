"""Immutable run packets for reproducible classification_v2 fold execution."""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from pig_behavior.classification_v2.contracts.training_snapshot import (
    check_training_snapshot,
)
from pig_behavior.classification_v2.training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    RUN_IDENTITY_REQUIRED_FIELDS,
)
from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    training_config_to_jsonable,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256 as _file_sha256,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    is_sha256 as _is_sha256,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    payload_sha256 as _payload_sha256,
)
from pig_behavior.classification_v2.training.run_identity import (
    RunIdentity,
    build_run_identity,
)
from pig_behavior.classification_v2.training.run_registry import (
    REGISTRY_SCHEMA_VERSION,
    merge_registry_entries,
)
from pig_behavior.classification_v2.training.run_registry import (
    append_registry_entry as _append_registry_entry,
)
from pig_behavior.classification_v2.training.run_registry import (
    ensure_registry_header as _ensure_registry_header,
)

RUN_MANIFEST_SCHEMA_VERSION = "classification_v2.run_manifest.v2"
ENVIRONMENT_SCHEMA_VERSION = "classification_v2.run_environment.v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "classification_v2.run_artifacts.v1"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "classification_v2.run_checkpoints.v1"
PREDICTION_MANIFEST_SCHEMA_VERSION = "classification_v2.run_predictions.v1"
TERMINAL_STATUSES = frozenset({"completed", "failed"})


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    """One prediction file and its unique producing checkpoint."""

    path: Path
    checkpoint_path: Path
    split: str
    expected_rows: int


@dataclass(slots=True)
class RunLineageSession:
    """Own one isolated fold/run directory and preserve terminal evidence."""

    identity: RunIdentity
    run_dir: Path
    registry_csv: Path
    environment: dict[str, Any]
    input_artifacts: list[dict[str, Any]]
    started_at_utc: str
    started_monotonic: float
    resumed: bool = False
    terminal: bool = False

    def __enter__(self) -> RunLineageSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: object,
    ) -> None:
        if exc is not None and not self.terminal:
            fail_run_lineage(
                self,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )


def initialize_run_lineage(
    config: ClassificationV2TrainingConfig,
    *,
    snapshot_check: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> RunLineageSession:
    """Validate immutable inputs and create or resume one isolated run packet."""

    checked = snapshot_check or check_training_snapshot(
        config.dataset.snapshot_json
    )
    if checked.get("valid") is not True:
        raise ValueError(f"training snapshot invalid={checked.get('errors')}")
    identity = build_run_identity(config, checked)
    current = checked.get("current") or {}
    input_artifacts = _snapshot_artifact_records(current)
    captured_environment = environment or capture_environment()
    fold_dir = config.execution.output_dir / identity.fold_id
    run_dir = fold_dir / identity.run_id
    registry_csv = config.execution.runs_registry_csv
    if run_dir.exists():
        if not config.execution.resume:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        _validate_resume_packet(
            run_dir,
            identity=identity,
            input_artifacts=input_artifacts,
        )
        manifest = _read_json(run_dir / "run_manifest.json")
        status = str(manifest.get("status", ""))
        if status in TERMINAL_STATUSES:
            raise FileExistsError(
                f"terminal run cannot be resumed: {run_dir} status={status}"
            )
        _record_resume_environment(
            run_dir / "environment.json",
            captured_environment,
        )
        _ensure_registry_header(registry_csv)
        return RunLineageSession(
            identity=identity,
            run_dir=run_dir,
            registry_csv=registry_csv,
            environment=captured_environment,
            input_artifacts=input_artifacts,
            started_at_utc=str(manifest["started_at_utc"]),
            started_monotonic=time.perf_counter(),
            resumed=True,
        )

    fold_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    started_at = _utc_now()
    _write_json_atomic(
        run_dir / "resolved_config.json",
        training_config_to_jsonable(config),
    )
    _write_json_atomic(
        run_dir / "environment.json",
        {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "initial": captured_environment,
            "resume_events": [],
        },
    )
    _write_json_atomic(
        run_dir / "artifact_manifest.json",
        {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "run_id": identity.run_id,
            "identity_sha256": identity.identity_sha256,
            "inputs": input_artifacts,
            "outputs": [],
            "status": "initialized",
        },
    )
    _write_empty_output_manifests(run_dir, identity)
    _write_json_atomic(
        run_dir / "run_manifest.json",
        _run_manifest_payload(
            identity,
            status="initialized",
            started_at_utc=started_at,
            resumed=False,
            registry_csv_path=registry_csv,
        ),
    )
    _ensure_registry_header(registry_csv)
    return RunLineageSession(
        identity=identity,
        run_dir=run_dir,
        registry_csv=registry_csv,
        environment=captured_environment,
        input_artifacts=input_artifacts,
        started_at_utc=started_at,
        started_monotonic=time.perf_counter(),
    )


def finalize_run_lineage(
    session: RunLineageSession,
    *,
    checkpoint_paths: Iterable[Path],
    predictions: Iterable[PredictionArtifact],
    metric_paths: Iterable[Path],
    runtime_seconds: float | None = None,
    peak_vram_bytes: int | None = None,
) -> dict[str, Any]:
    """Link outputs, write terminal manifests, and append one immutable row."""

    if session.terminal:
        raise ValueError(f"run session is already terminal: {session.identity.run_id}")
    runtime = _resolved_runtime(session, runtime_seconds)
    checkpoints = [
        _checkpoint_record(path, session.identity)
        for path in checkpoint_paths
    ]
    checkpoint_by_path = {
        str(Path(item["path"]).resolve()): item for item in checkpoints
    }
    prediction_records = [
        _prediction_record(item, checkpoint_by_path, session.identity)
        for item in predictions
    ]
    metric_records = [
        _output_artifact(path, kind="metric") for path in metric_paths
    ]
    completed_at = _utc_now()
    checkpoint_manifest_path = session.run_dir / "checkpoint_manifest.json"
    prediction_manifest_path = session.run_dir / "prediction_manifest.json"
    _write_json_atomic(
        checkpoint_manifest_path,
        {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "run_id": session.identity.run_id,
            "identity_sha256": session.identity.identity_sha256,
            "checkpoints": checkpoints,
            "status": "completed",
            "errors": [],
        },
    )
    _write_json_atomic(
        prediction_manifest_path,
        {
            "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
            "run_id": session.identity.run_id,
            "identity_sha256": session.identity.identity_sha256,
            "predictions": prediction_records,
            "status": "completed",
            "errors": [],
        },
    )
    outputs = [
        *checkpoints,
        *prediction_records,
        *metric_records,
        _output_artifact(
            session.run_dir / "environment.json",
            kind="environment",
        ),
        _output_artifact(
            session.run_dir / "resolved_config.json",
            kind="resolved_config",
        ),
    ]
    _write_json_atomic(
        session.run_dir / "artifact_manifest.json",
        {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "run_id": session.identity.run_id,
            "identity_sha256": session.identity.identity_sha256,
            "inputs": session.input_artifacts,
            "outputs": outputs,
            "status": "completed",
        },
    )
    metric_path = str(metric_records[0]["path"]) if metric_records else ""
    registry_entry = _registry_entry(
        session,
        status="completed",
        failure_reason="",
        completed_at_utc=completed_at,
        runtime_seconds=runtime,
        peak_vram_bytes=peak_vram_bytes,
        metric_path=metric_path,
    )
    registry_entry_path = session.run_dir / "registry_entry.json"
    _write_json_atomic(registry_entry_path, registry_entry)
    run_manifest = _run_manifest_payload(
        session.identity,
        status="completed",
        started_at_utc=session.started_at_utc,
        completed_at_utc=completed_at,
        resumed=session.resumed,
        runtime_seconds=runtime,
        peak_vram_bytes=peak_vram_bytes,
        registry_entry_sha256=_file_sha256(registry_entry_path),
        registry_csv_path=session.registry_csv,
    )
    _write_json_atomic(session.run_dir / "run_manifest.json", run_manifest)
    session.terminal = True
    try:
        _append_registry_entry(session.registry_csv, registry_entry)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "run packet completed but central registry append failed; "
            "merge registry_entry.json after resolving the registry error"
        ) from exc
    return {
        "run_id": session.identity.run_id,
        "run_dir": str(session.run_dir),
        "status": "completed",
        "checkpoint_count": len(checkpoints),
        "prediction_count": len(prediction_records),
        "runtime_seconds": runtime,
        "registry_csv": str(session.registry_csv),
        "errors": [],
        "valid": True,
    }


def fail_run_lineage(
    session: RunLineageSession,
    *,
    failure_reason: str,
    runtime_seconds: float | None = None,
    peak_vram_bytes: int | None = None,
) -> dict[str, Any]:
    """Preserve a terminal failure without deleting partial run artifacts."""

    if session.terminal:
        raise ValueError(f"run session is already terminal: {session.identity.run_id}")
    reason = failure_reason.strip()
    if not reason:
        raise ValueError("failure_reason must not be blank")
    runtime = _resolved_runtime(session, runtime_seconds)
    completed_at = _utc_now()
    for name, schema in [
        ("checkpoint_manifest.json", CHECKPOINT_MANIFEST_SCHEMA_VERSION),
        ("prediction_manifest.json", PREDICTION_MANIFEST_SCHEMA_VERSION),
    ]:
        path = session.run_dir / name
        payload = _read_json(path)
        payload.update(
            {
                "schema_version": schema,
                "run_id": session.identity.run_id,
                "identity_sha256": session.identity.identity_sha256,
                "status": "failed",
                "failure_reason": reason,
            }
        )
        _write_json_atomic(path, payload)
    self_managed = {
        "run_manifest.json",
        "artifact_manifest.json",
        "checkpoint_manifest.json",
        "prediction_manifest.json",
        "registry_entry.json",
    }
    partial_outputs = [
        _output_artifact(path, kind="partial_output")
        for path in sorted(session.run_dir.iterdir())
        if path.is_file() and path.name not in self_managed
    ]
    _write_json_atomic(
        session.run_dir / "artifact_manifest.json",
        {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "run_id": session.identity.run_id,
            "identity_sha256": session.identity.identity_sha256,
            "inputs": session.input_artifacts,
            "outputs": partial_outputs,
            "status": "failed",
            "failure_reason": reason,
        },
    )
    registry_entry = _registry_entry(
        session,
        status="failed",
        failure_reason=reason,
        completed_at_utc=completed_at,
        runtime_seconds=runtime,
        peak_vram_bytes=peak_vram_bytes,
        metric_path="",
    )
    registry_entry_path = session.run_dir / "registry_entry.json"
    _write_json_atomic(registry_entry_path, registry_entry)
    run_manifest = _run_manifest_payload(
        session.identity,
        status="failed",
        started_at_utc=session.started_at_utc,
        completed_at_utc=completed_at,
        resumed=session.resumed,
        runtime_seconds=runtime,
        peak_vram_bytes=peak_vram_bytes,
        failure_reason=reason,
        registry_entry_sha256=_file_sha256(registry_entry_path),
        registry_csv_path=session.registry_csv,
    )
    _write_json_atomic(session.run_dir / "run_manifest.json", run_manifest)
    session.terminal = True
    registry_error = ""
    try:
        _append_registry_entry(session.registry_csv, registry_entry)
    except (OSError, ValueError) as exc:
        registry_error = f"registry_append_failed={type(exc).__name__}: {exc}"
    errors = [reason]
    if registry_error:
        errors.append(registry_error)
    return {
        "run_id": session.identity.run_id,
        "run_dir": str(session.run_dir),
        "status": "failed",
        "failure_reason": reason,
        "runtime_seconds": runtime,
        "registry_csv": str(session.registry_csv),
        "errors": errors,
        "valid": False,
    }


def capture_environment() -> dict[str, Any]:
    """Capture software and hardware without downloading or mutating packages."""

    try:
        import torchvision

        torchvision_version: str | None = torchvision.__version__
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        torchvision_version = f"unavailable:{type(exc).__name__}"
    cuda_available = torch.cuda.is_available()
    gpu_model = torch.cuda.get_device_name(0) if cuda_available else "NONE"
    gpu_vram = (
        int(torch.cuda.get_device_properties(0).total_memory)
        if cuda_available
        else 0
    )
    return {
        "captured_at_utc": _utc_now(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if cuda_available else None
        ),
        "gpu_model": gpu_model,
        "gpu_vram_bytes": gpu_vram,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }


def _record_resume_environment(
    path: Path,
    environment: dict[str, Any],
) -> None:
    """Append one resume environment while the run packet is nonterminal."""

    packet = _read_json(path)
    if packet.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("resume environment schema mismatch")
    events = packet.get("resume_events")
    if not isinstance(events, list) or not isinstance(packet.get("initial"), dict):
        raise ValueError("resume environment packet is malformed")
    packet["resume_events"] = [
        *events,
        {
            "resumed_at_utc": _utc_now(),
            "environment": environment,
        },
    ]
    _write_json_atomic(path, packet)


def _checkpoint_record(path: Path, identity: RunIdentity) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint missing: {resolved}")
    audit_path = resolved.with_suffix(resolved.suffix + ".audit.json")
    if not audit_path.is_file():
        raise FileNotFoundError(f"checkpoint audit missing: {audit_path}")
    audit = _read_json(audit_path)
    if audit.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint audit schema mismatch")
    if audit.get("valid") is not True or audit.get("errors"):
        raise ValueError("checkpoint audit is not valid")
    lineage = audit.get("lineage") or {}
    identity_payload = identity.to_payload()
    required = {
        name: identity_payload[name]
        for name in RUN_IDENTITY_REQUIRED_FIELDS
    }
    mismatches = {
        key: {"expected": value, "observed": lineage.get(key)}
        for key, value in required.items()
        if lineage.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint lineage mismatch={mismatches}")
    return {
        "kind": "checkpoint",
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "audit_path": str(audit_path),
        "audit_sha256": _file_sha256(audit_path),
        "audit_size_bytes": int(audit_path.stat().st_size),
        "lineage": required,
    }


def _prediction_record(
    artifact: PredictionArtifact,
    checkpoint_by_path: dict[str, dict[str, Any]],
    identity: RunIdentity,
) -> dict[str, Any]:
    path = artifact.path.resolve()
    checkpoint_path = artifact.checkpoint_path.resolve()
    checkpoint = checkpoint_by_path.get(str(checkpoint_path))
    if checkpoint is None:
        raise ValueError(
            "prediction checkpoint is not in checkpoint manifest: "
            f"{checkpoint_path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"prediction file missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"window_id", "y_true", "y_pred", "prediction_split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction schema missing columns={missing}")
    if len(frame) != artifact.expected_rows:
        raise ValueError(
            f"prediction count mismatch={len(frame)} expected={artifact.expected_rows}"
        )
    if frame["window_id"].fillna("").astype(str).duplicated().any():
        raise ValueError("prediction manifest contains duplicate window_id")
    observed_splits = sorted(
        frame["prediction_split"].fillna("").astype(str).unique()
    )
    if observed_splits != [artifact.split]:
        raise ValueError(
            f"prediction split mismatch={observed_splits} expected={artifact.split}"
        )
    return {
        "kind": "prediction",
        "path": str(path),
        "sha256": _file_sha256(path),
        "size_bytes": int(path.stat().st_size),
        "rows": int(len(frame)),
        "split": artifact.split,
        "fold_id": identity.fold_id,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint["sha256"],
        "config_sha256": identity.config_sha256,
        "dataset_snapshot_sha256": identity.dataset_snapshot_sha256,
    }


def _validate_resume_packet(
    run_dir: Path,
    *,
    identity: RunIdentity,
    input_artifacts: list[dict[str, Any]],
) -> None:
    required = [
        "run_manifest.json",
        "environment.json",
        "artifact_manifest.json",
        "checkpoint_manifest.json",
        "prediction_manifest.json",
        "resolved_config.json",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"resume run packet missing files={missing}")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    observed_identity = run_manifest.get("identity") or {}
    if observed_identity != identity.to_payload():
        mismatches = {
            key: {
                "expected": value,
                "observed": observed_identity.get(key),
            }
            for key, value in identity.to_payload().items()
            if observed_identity.get(key) != value
        }
        raise ValueError(f"resume run identity mismatch={mismatches}")
    artifact_manifest = _read_json(run_dir / "artifact_manifest.json")
    if artifact_manifest.get("inputs") != input_artifacts:
        raise ValueError("resume input artifact manifest drift")
    resolved_config = _read_json(run_dir / "resolved_config.json")
    if _payload_sha256(resolved_config) != identity.config_sha256:
        raise ValueError("resume resolved config hash drift")
    environment = _read_json(run_dir / "environment.json")
    if environment.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("resume environment schema drift")
    if not isinstance(environment.get("initial"), dict):
        raise ValueError("resume environment initial payload is missing")
    if not isinstance(environment.get("resume_events"), list):
        raise ValueError("resume environment event history is invalid")


def _write_empty_output_manifests(
    run_dir: Path,
    identity: RunIdentity,
) -> None:
    _write_json_atomic(
        run_dir / "checkpoint_manifest.json",
        {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "run_id": identity.run_id,
            "identity_sha256": identity.identity_sha256,
            "checkpoints": [],
            "status": "initialized",
            "errors": [],
        },
    )
    _write_json_atomic(
        run_dir / "prediction_manifest.json",
        {
            "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
            "run_id": identity.run_id,
            "identity_sha256": identity.identity_sha256,
            "predictions": [],
            "status": "initialized",
            "errors": [],
        },
    )


def _run_manifest_payload(
    identity: RunIdentity,
    *,
    status: str,
    started_at_utc: str,
    resumed: bool,
    registry_csv_path: Path,
    completed_at_utc: str | None = None,
    runtime_seconds: float | None = None,
    peak_vram_bytes: int | None = None,
    failure_reason: str = "",
    registry_entry_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": identity.run_id,
        "identity_sha256": identity.identity_sha256,
        "identity": identity.to_payload(),
        "status": status,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "runtime_seconds": runtime_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "failure_reason": failure_reason,
        "resumed": resumed,
        "registry_entry_sha256": registry_entry_sha256,
        "registry_csv_path": str(registry_csv_path.resolve()),
        "manifest_paths": {
            "environment": "environment.json",
            "artifacts": "artifact_manifest.json",
            "checkpoints": "checkpoint_manifest.json",
            "predictions": "prediction_manifest.json",
            "registry_entry": "registry_entry.json",
        },
    }


def _registry_entry(
    session: RunLineageSession,
    *,
    status: str,
    failure_reason: str,
    completed_at_utc: str,
    runtime_seconds: float,
    peak_vram_bytes: int | None,
    metric_path: str,
) -> dict[str, Any]:
    identity = session.identity
    environment = session.environment
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "identity_schema_version": identity.identity_schema_version,
        "run_id": identity.run_id,
        "experiment_name": identity.experiment_name,
        "execution_profile": identity.execution_profile,
        "fold_id": identity.fold_id,
        "seed": identity.seed,
        "status": status,
        "failure_reason": failure_reason,
        "code_sha": identity.code_sha,
        "dirty_worktree": identity.dirty_worktree,
        "worktree_state_sha256": identity.worktree_state_sha256,
        "config_sha256": identity.config_sha256,
        "dataset_snapshot_id": identity.dataset_snapshot_id,
        "dataset_snapshot_sha256": identity.dataset_snapshot_sha256,
        "cache_sha256": identity.cache_sha256,
        "fold_manifest_sha256": identity.fold_manifest_sha256,
        "feature_whitelist_sha256": identity.feature_whitelist_sha256,
        "temporal_view_selection_sha256": (
            identity.temporal_view_selection_sha256
        ),
        "temporal_view_manifest_sha256": (
            identity.temporal_view_manifest_sha256
        ),
        "fold_event_weight_sha256": identity.fold_event_weight_sha256,
        "architecture_version": identity.architecture_version,
        "model_mode": identity.model_mode,
        "backbone_name": identity.backbone_name,
        "pretrained_weight_enum": identity.pretrained_weight_enum,
        "resolution": identity.resolution,
        "visual_freeze_contract_version": (
            identity.visual_freeze_contract_version
        ),
        "visual_freeze_policy": identity.visual_freeze_policy,
        "visual_frozen_warmup_epochs": (
            identity.visual_frozen_warmup_epochs
        ),
        "visual_layer4_only_epochs": identity.visual_layer4_only_epochs,
        "visual_backbone_lr_multiplier": (
            identity.visual_backbone_lr_multiplier
        ),
        "temporal_view": identity.temporal_view,
        "temporal_encoder_name": identity.temporal_encoder_name,
        "modalities": "|".join(identity.modalities),
        "loss_name": identity.loss_name,
        "sampler_policy": identity.sampler_policy,
        "optimizer_name": identity.optimizer_name,
        "precision": identity.precision,
        "augmentation_policy": identity.augmentation_policy,
        "gpu_model": environment.get("gpu_model"),
        "gpu_vram_bytes": environment.get("gpu_vram_bytes"),
        "python_version": environment.get("python_version"),
        "torch_version": environment.get("torch_version"),
        "runtime_seconds": runtime_seconds,
        "peak_vram_bytes": peak_vram_bytes or 0,
        "checkpoint_manifest_path": str(
            (session.run_dir / "checkpoint_manifest.json").resolve()
        ),
        "prediction_manifest_path": str(
            (session.run_dir / "prediction_manifest.json").resolve()
        ),
        "metric_path": metric_path,
        "run_manifest_path": str(
            (session.run_dir / "run_manifest.json").resolve()
        ),
        "completed_at_utc": completed_at_utc,
    }


def _snapshot_artifact_records(current: dict[str, Any]) -> list[dict[str, Any]]:
    """Record present inputs and fail only for absent required artifacts."""

    records: list[dict[str, Any]] = []
    for name, item in sorted((current.get("artifacts") or {}).items()):
        if item.get("exists") is not True:
            if item.get("required") is True:
                raise ValueError(
                    f"required snapshot input artifact is missing={name}"
                )
            continue
        records.append(
            {
                "name": name,
                "path": str(Path(str(item["path"])).resolve()),
                "type": item.get("type"),
                "size_bytes": int(item.get("size_bytes", 0)),
                "sha256": _required_hash(item, f"snapshot artifact {name}"),
            }
        )
    if not records:
        raise ValueError("snapshot contains no input artifacts")
    return records


def _required_hash(item: dict[str, Any], name: str) -> str:
    value = str(item.get("sha256", ""))
    if not _is_sha256(value):
        raise ValueError(f"{name} lacks a valid sha256")
    return value


def _output_artifact(path: Path, *, kind: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"output artifact missing={resolved}")
    return {
        "kind": kind,
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _resolved_runtime(
    session: RunLineageSession,
    runtime_seconds: float | None,
) -> float:
    value = (
        float(runtime_seconds)
        if runtime_seconds is not None
        else float(time.perf_counter() - session.started_monotonic)
    )
    if value < 0.0:
        raise ValueError("runtime_seconds must be non-negative")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "PredictionArtifact",
    "RunIdentity",
    "RunLineageSession",
    "build_run_identity",
    "capture_environment",
    "fail_run_lineage",
    "finalize_run_lineage",
    "initialize_run_lineage",
    "merge_registry_entries",
]
