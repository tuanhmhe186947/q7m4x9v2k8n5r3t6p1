"""Crash-bounded execution and immutable packets for the legacy L5 ladder."""

from __future__ import annotations

import gc
import json
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    CANONICAL_VIEWS,
    LINEAGE_SCOPE,
    SHORT_SCOPE,
    TemporalLadderConfig,
    TemporalLadderOutcome,
    build_temporal_ladder_selection,
    implementation_hashes,
    load_temporal_ladder_view,
    preflight_temporal_ladder_view,
    temporal_ladder_git_guard,
    train_temporal_ladder_core,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
)

DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

RUN_RESULT_SCHEMA = (
    "classification_v2.legacy_development_l5.temporal_ladder_run_result.v1"
)
RUN_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l5.temporal_ladder_run_manifest.v1"
)
ARTIFACT_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l5.temporal_ladder_artifacts.v1"
)
REPEAT_GATE_SCHEMA = (
    "classification_v2.legacy_development_l5.temporal_ladder_repeat_gate.v1"
)
MATRIX_GATE_SCHEMA = (
    "classification_v2.legacy_development_l5.temporal_ladder_short_matrix.v1"
)

ARTIFACT_FILES = {
    "environment": "environment.json",
    "preflight": "preflight.json",
    "selection_manifest": "training_selection_manifest.csv",
    "selection_audit": "training_selection_audit.json",
    "epoch_metrics": "epoch_metrics.csv",
    "window_predictions": "validation_window_predictions.csv",
    "native_predictions": "validation_native_predictions.csv",
    "validation_metrics": "validation_metrics.json",
    "validation_per_class": "validation_per_class.csv",
    "validation_confusion": "validation_confusion.csv",
    "checkpoint": "best_validation_checkpoint.pt",
    "checkpoint_manifest": "checkpoint_manifest.json",
    "prediction_manifest": "prediction_manifest.json",
    "run_result": "run_result.json",
}

_RUN_EXECUTED_IN_PROCESS = False


def run_temporal_ladder_view(
    config: TemporalLadderConfig,
    *,
    view_id: str,
    run_id: str,
) -> dict[str, Any]:
    """Run one exact view once in a fresh process with no OOM retry."""

    global _RUN_EXECUTED_IN_PROCESS
    if _RUN_EXECUTED_IN_PROCESS:
        raise RuntimeError("temporal ladder permits one run per process")
    if not _safe_run_id(run_id):
        raise ValueError(f"unsafe temporal ladder run ID: {run_id!r}")
    if view_id not in CANONICAL_VIEWS:
        raise ValueError(f"unknown temporal ladder view: {view_id}")
    preflight = preflight_temporal_ladder_view(config, view_id)
    if not preflight["gpu_launch_authorized"]:
        raise RuntimeError(f"temporal ladder preflight failed={preflight['errors']}")
    _RUN_EXECUTED_IN_PROCESS = True
    _, view, parent = load_temporal_ladder_view(config, view_id)
    selection = build_temporal_ladder_selection(view, config, view_id)
    run_root = config.output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    paths = _run_paths(run_root)
    git_guard = temporal_ladder_git_guard(config)
    if not git_guard["valid"]:
        raise RuntimeError(f"temporal ladder git guard failed={git_guard['errors']}")
    started_at = _utc_now()
    started = time.perf_counter()
    planned = _planned_manifest(
        config,
        view_id=view_id,
        run_id=run_id,
        selection=selection.audit,
        parent=parent,
        preflight=preflight,
        git_guard=git_guard,
        started_at=started_at,
    )
    _write_json_exclusive(paths["run_manifest"], planned)
    planned_sha = file_sha256(paths["run_manifest"])
    _write_json_exclusive(paths["preflight"], preflight)
    _write_dataframe_exclusive(paths["selection_manifest"], selection.manifest)
    _write_json_exclusive(paths["selection_audit"], selection.audit)
    outcome: TemporalLadderOutcome | None = None
    execution: dict[str, Any]
    failure: dict[str, Any] | None = None
    try:
        outcome, execution = _execute_cuda(
            view,
            selection,
            config,
            view_id=view_id,
        )
    except Exception as error:
        failure = {
            "schema_version": (
                "classification_v2.legacy_development_l5."
                "temporal_ladder_failure.v1"
            ),
            "run_id": run_id,
            "view_id": view_id,
            "process_id": os.getpid(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "oom_retry_performed": False,
            "captured_at_utc": _utc_now(),
        }
        _write_json_exclusive(paths["unexpected_failure"], failure)
        execution = _failed_execution(config, error)
    runtime_seconds = time.perf_counter() - started
    return _finalize_run(
        paths,
        config=config,
        view_id=view_id,
        run_id=run_id,
        planned=planned,
        planned_sha=planned_sha,
        selection=selection.audit,
        outcome=outcome,
        execution=execution,
        failure=failure,
        runtime_seconds=runtime_seconds,
    )


def audit_temporal_ladder_run(
    config: TemporalLadderConfig,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Independently verify one completed packet and all 14 artifacts."""

    resolved_result = result_path.resolve()
    run_root = resolved_result.parent
    if resolved_result.name != ARTIFACT_FILES["run_result"]:
        raise ValueError("temporal ladder result filename mismatch")
    result = _read_json(resolved_result)
    manifest_path = run_root / "run_manifest.json"
    artifact_path = run_root / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    artifact = _read_json(artifact_path)
    errors: list[str] = []
    expected_result = {
        "schema_version": RUN_RESULT_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_TRAINING",
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "config_sha256": config.sha256,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [],
        "valid": True,
    }
    errors.extend(_mapping_errors(result, expected_result, "result"))
    expected_manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": result.get("run_id"),
        "view_id": result.get("view_id"),
        "status": "completed",
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "config_sha256": config.sha256,
        "run_result_sha256": file_sha256(resolved_result),
        "artifact_manifest_sha256": file_sha256(artifact_path),
        "failure_reason": "",
    }
    errors.extend(_mapping_errors(manifest, expected_manifest, "manifest"))
    artifact_audit = _audit_artifacts(run_root, artifact)
    errors.extend(artifact_audit["errors"])
    if result.get("view_id") not in CANONICAL_VIEWS:
        errors.append("unknown_result_view_id")
    execution = result.get("execution") or {}
    execution_expected = {
        "cublas_workspace_config": str(
            config.payload["optimization"]["cublas_workspace_config"]
        ),
        "oom": False,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "errors": [],
        "valid": True,
    }
    errors.extend(_mapping_errors(execution, execution_expected, "execution"))
    return {
        "schema_version": (
            "classification_v2.legacy_development_l5."
            "temporal_ladder_run_audit.v1"
        ),
        "run_id": result.get("run_id"),
        "view_id": result.get("view_id"),
        "result_path": str(resolved_result),
        "result_sha256": file_sha256(resolved_result),
        "run_manifest_sha256": file_sha256(manifest_path),
        "artifact_manifest_sha256": file_sha256(artifact_path),
        "verified_artifacts": artifact_audit["verified_artifacts"],
        "result": result,
        "manifest": manifest,
        "errors": errors,
        "valid": not errors,
    }


def audit_temporal_ladder_repeat_gate(
    config: TemporalLadderConfig,
    *,
    view_id: str,
    primary_result_path: Path,
    repeat_result_path: Path,
) -> dict[str, Any]:
    """Compare two separate short runs and require exact learned outputs."""

    if config.training_scope != SHORT_SCOPE:
        raise ValueError("repeat gate requires short training scope")
    primary = audit_temporal_ladder_run(
        config,
        result_path=primary_result_path,
    )
    repeat = audit_temporal_ladder_run(
        config,
        result_path=repeat_result_path,
    )
    errors = [
        *(f"primary:{value}" for value in primary["errors"]),
        *(f"repeat:{value}" for value in repeat["errors"]),
    ]
    left = primary["result"]
    right = repeat["result"]
    if left.get("view_id") != view_id or right.get("view_id") != view_id:
        errors.append("repeat_gate_view_id_mismatch")
    equality_fields = (
        "config_sha256",
        "implementation_hashes",
        "view_id",
        "training_scope",
        "selection_content_sha256",
        "train_native_unit_sha256",
        "validation_native_unit_sha256",
        "train_native_units",
        "validation_native_units",
        "train_windows",
        "validation_windows",
        "optimizer_steps",
        "best_epoch",
        "parameter_sha256",
        "window_prediction_content_sha256",
        "native_prediction_content_sha256",
        "epoch_metrics_content_sha256",
        "validation_metrics",
    )
    equality: dict[str, bool] = {}
    for field in equality_fields:
        equal = left.get(field) == right.get(field)
        equality[field] = equal
        if not equal:
            errors.append(f"repeat_field_mismatch={field}")
    first_pid = int(left.get("process_id", -1))
    second_pid = int(right.get("process_id", -1))
    if first_pid <= 0 or second_pid <= 0 or first_pid == second_pid:
        errors.append("repeat_process_ids_not_distinct")
    interval = _non_overlapping_intervals(left, right)
    errors.extend(interval["errors"])
    valid = not errors
    return {
        "schema_version": REPEAT_GATE_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_REPEAT_GATE"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_REPEAT_GATE"
        ),
        "view_id": view_id,
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": SHORT_SCOPE,
        "short_config_sha256": config.sha256,
        "implementation_hashes": implementation_hashes(config),
        "primary": _audit_summary(primary),
        "repeat": _audit_summary(repeat),
        "equality_fields": equality,
        "execution_intervals": interval,
        "full_view_expansion_authorized": valid,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }


def write_temporal_ladder_repeat_gate(
    config: TemporalLadderConfig,
    *,
    view_id: str,
    primary_result_path: Path,
    repeat_result_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Write one view-level gate only after a passing independent audit."""

    gate = audit_temporal_ladder_repeat_gate(
        config,
        view_id=view_id,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
    )
    if not gate["valid"]:
        raise ValueError(f"temporal ladder repeat gate failed={gate['errors']}")
    path = config.output_root / _repeat_gate_filename(view_id)
    _write_json_exclusive(path, gate)
    return path, gate


def audit_temporal_ladder_short_matrix(
    config: TemporalLadderConfig,
) -> dict[str, Any]:
    """Require all eight exact repeat gates and one common native universe."""

    if config.training_scope != SHORT_SCOPE:
        raise ValueError("short matrix gate requires short config")
    errors: list[str] = []
    gates: dict[str, Any] = {}
    train_hashes: set[str] = set()
    validation_hashes: set[str] = set()
    for view_id in CANONICAL_VIEWS:
        path = config.output_root / _repeat_gate_filename(view_id)
        if not path.is_file():
            errors.append(f"missing_repeat_gate={view_id}")
            continue
        gate = _read_json(path)
        expected = {
            "status": "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_REPEAT_GATE",
            "view_id": view_id,
            "lineage_scope": LINEAGE_SCOPE,
            "training_scope": SHORT_SCOPE,
            "short_config_sha256": config.sha256,
            "implementation_hashes": implementation_hashes(config),
            "full_view_expansion_authorized": True,
            "valid": True,
        }
        errors.extend(_mapping_errors(gate, expected, f"gate.{view_id}"))
        if gate.get("errors"):
            errors.append(f"gate_has_errors={view_id}")
        primary_result = (gate.get("primary") or {}).get("result") or {}
        train_hashes.add(str(primary_result.get("train_native_unit_sha256")))
        validation_hashes.add(
            str(primary_result.get("validation_native_unit_sha256"))
        )
        gates[view_id] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "primary_result_sha256": (gate.get("primary") or {}).get(
                "result_sha256"
            ),
            "repeat_result_sha256": (gate.get("repeat") or {}).get(
                "result_sha256"
            ),
        }
    if len(train_hashes) != 1:
        errors.append(f"short_train_native_universe_hashes={sorted(train_hashes)}")
    if len(validation_hashes) != 1:
        errors.append(
            "validation_native_universe_hashes="
            f"{sorted(validation_hashes)}"
        )
    valid = not errors and len(gates) == len(CANONICAL_VIEWS)
    return {
        "schema_version": MATRIX_GATE_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_SHORT_MATRIX"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_SHORT_MATRIX"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "short_config_path": str(config.path),
        "short_config_sha256": config.sha256,
        "implementation_hashes": implementation_hashes(config),
        "view_count": len(gates),
        "views": gates,
        "train_native_unit_sha256": (
            next(iter(train_hashes)) if len(train_hashes) == 1 else None
        ),
        "validation_native_unit_sha256": (
            next(iter(validation_hashes)) if len(validation_hashes) == 1 else None
        ),
        "full_expansion_authorized": valid,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }


def write_temporal_ladder_short_matrix(
    config: TemporalLadderConfig,
) -> tuple[Path, dict[str, Any]]:
    """Write the sole full-expansion authority for all eight ladder views."""

    gate = audit_temporal_ladder_short_matrix(config)
    if not gate["valid"]:
        raise ValueError(f"temporal ladder matrix gate failed={gate['errors']}")
    filename = str(config.payload["output"]["matrix_gate_filename"])
    path = config.output_root / filename
    _write_json_exclusive(path, gate)
    return path, gate


def _execute_cuda(
    view: Any,
    selection: Any,
    config: TemporalLadderConfig,
    *,
    view_id: str,
) -> tuple[TemporalLadderOutcome | None, dict[str, Any]]:
    optimization = config.payload["optimization"]
    errors: list[str] = []
    oom = False
    oom_message: str | None = None
    expected_cublas = str(optimization["cublas_workspace_config"])
    observed_cublas = os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        expected_cublas,
    )
    if observed_cublas != expected_cublas:
        errors.append(
            "cublas_workspace_config="
            f"{observed_cublas!r}!={expected_cublas!r}"
        )
    device = torch.device(str(optimization["device"]))
    if device.type != "cuda":
        raise ValueError("temporal ladder production run requires CUDA")
    if torch.cuda.is_initialized():
        raise RuntimeError("temporal ladder did not start in a fresh process")
    if not torch.cuda.is_available():
        raise RuntimeError("temporal ladder requested unavailable CUDA")
    device_index = int(device.index) if device.index is not None else 0
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    actual_total = int(properties.total_memory)
    free_before, mem_info_total = (
        int(value) for value in torch.cuda.mem_get_info(device)
    )
    expected_total = int(optimization["validated_local_gpu_vram_bytes"])
    allocator_limit = int(optimization["allocator_limit_bytes"])
    allocator_fraction = allocator_limit / actual_total
    if actual_total != expected_total or mem_info_total != expected_total:
        errors.append(
            f"gpu_total_vram={actual_total},{mem_info_total}!={expected_total}"
        )
    if free_before < allocator_limit:
        errors.append(f"free_vram={free_before}<{allocator_limit}")
    if allocator_fraction > float(optimization["maximum_peak_vram_fraction"]):
        errors.append("allocator_fraction_exceeds_configured_ceiling")
    allocated_before = int(torch.cuda.memory_allocated(device))
    reserved_before = int(torch.cuda.memory_reserved(device))
    if allocated_before != 0 or reserved_before != 0:
        errors.append("cuda_memory_not_empty_before_model_creation")
    outcome: TemporalLadderOutcome | None = None
    peak_allocated = 0
    peak_reserved = 0
    cleanup_errors: list[str] = []
    cublas_cleared = False
    if not errors:
        torch.cuda.set_per_process_memory_fraction(allocator_fraction, device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            outcome = train_temporal_ladder_core(
                view,
                selection,
                config,
                view_id,
                device=device,
            )
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError as error:
            oom = True
            oom_message = str(error)
            errors.append("cuda_out_of_memory_no_retry")
        finally:
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            gc.collect()
            try:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                clear_workspaces = getattr(
                    torch._C,
                    "_cuda_clearCublasWorkspaces",
                    None,
                )
                if not callable(clear_workspaces):
                    raise RuntimeError("PyTorch cannot clear cuBLAS workspaces")
                clear_workspaces()
                cublas_cleared = True
                torch.cuda.empty_cache()
            except RuntimeError as error:
                cleanup_errors.append(f"cuda_cleanup_error={error}")
    post_allocated = int(torch.cuda.memory_allocated(device))
    post_reserved = int(torch.cuda.memory_reserved(device))
    if peak_allocated > allocator_limit or peak_reserved > allocator_limit:
        errors.append("cuda_peak_exceeds_allocator_limit")
    if post_allocated != 0 or post_reserved != 0:
        errors.append("cuda_memory_not_released_after_training")
    if outcome is not None and outcome.maximum_loaded_batch_bytes > int(
        optimization["maximum_loaded_batch_bytes"]
    ):
        errors.append("loaded_batch_exceeds_frozen_limit")
    errors.extend(cleanup_errors)
    valid = outcome is not None and not errors
    return outcome, {
        "device": str(device),
        "device_name": str(properties.name),
        "process_id": os.getpid(),
        "actual_total_vram_bytes": actual_total,
        "mem_info_total_vram_bytes": mem_info_total,
        "free_vram_before_bytes": free_before,
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "allocator_fraction": allocator_fraction,
        "allocator_limit_bytes": allocator_limit,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "post_cleanup_allocated_bytes": post_allocated,
        "post_cleanup_reserved_bytes": post_reserved,
        "cublas_workspaces_cleared": cublas_cleared,
        "cublas_workspace_config": observed_cublas,
        "precision": "float32",
        "autocast_enabled": False,
        "oom": oom,
        "oom_message": oom_message,
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": errors,
        "valid": valid,
    }


def _finalize_run(
    paths: dict[str, Path],
    *,
    config: TemporalLadderConfig,
    view_id: str,
    run_id: str,
    planned: dict[str, Any],
    planned_sha: str,
    selection: dict[str, Any],
    outcome: TemporalLadderOutcome | None,
    execution: dict[str, Any],
    failure: dict[str, Any] | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    valid = outcome is not None and failure is None and execution.get("valid") is True
    if not valid:
        result = _failure_result(
            config,
            view_id=view_id,
            run_id=run_id,
            planned=planned,
            planned_sha=planned_sha,
            selection=selection,
            execution=execution,
            failure=failure,
            runtime_seconds=runtime_seconds,
        )
        _write_json_exclusive(paths["run_result"], result)
        final_manifest = {
            **planned,
            "status": "failed",
            "completed_at_utc": _utc_now(),
            "runtime_seconds": runtime_seconds,
            "run_result_sha256": file_sha256(paths["run_result"]),
            "artifact_manifest_sha256": None,
            "failure_reason": str((failure or {}).get("error_message", "unknown")),
        }
        _replace_json(paths["run_manifest"], final_manifest)
        return result
    assert outcome is not None
    _write_dataframe_exclusive(paths["epoch_metrics"], outcome.epoch_metrics)
    _write_dataframe_exclusive(
        paths["window_predictions"],
        outcome.window_predictions,
    )
    _write_dataframe_exclusive(
        paths["native_predictions"],
        outcome.native_predictions,
    )
    _write_json_exclusive(paths["validation_metrics"], outcome.metrics)
    _write_dataframe_exclusive(
        paths["validation_per_class"],
        outcome.per_class_metrics,
    )
    _write_dataframe_exclusive(paths["validation_confusion"], outcome.confusion)
    checkpoint_payload = {
        "schema_version": (
            "classification_v2.legacy_development_l5."
            "temporal_ladder_checkpoint.v1"
        ),
        "run_id": run_id,
        "view_id": view_id,
        "training_scope": config.training_scope,
        "config_sha256": config.sha256,
        "implementation_hashes": implementation_hashes(config),
        "selection_content_sha256": selection["selection_content_sha256"],
        "best_epoch": outcome.best_epoch,
        "optimizer_steps": outcome.optimizer_steps,
        "parameter_sha256": outcome.parameter_sha256,
        "model_state": outcome.model_state,
        "optimizer_state": outcome.optimizer_state,
        "lineage_scope": LINEAGE_SCOPE,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
    }
    _write_torch_exclusive(paths["checkpoint"], checkpoint_payload)
    checkpoint_manifest = {
        "schema_version": (
            "classification_v2.legacy_development_l5."
            "temporal_ladder_checkpoint_manifest.v1"
        ),
        "run_id": run_id,
        "view_id": view_id,
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "parameter_sha256": outcome.parameter_sha256,
        "best_epoch": outcome.best_epoch,
        "optimizer_steps": outcome.optimizer_steps,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(paths["checkpoint_manifest"], checkpoint_manifest)
    prediction_manifest = {
        "schema_version": (
            "classification_v2.legacy_development_l5."
            "temporal_ladder_prediction_manifest.v1"
        ),
        "run_id": run_id,
        "view_id": view_id,
        "window_predictions_sha256": file_sha256(paths["window_predictions"]),
        "native_predictions_sha256": file_sha256(paths["native_predictions"]),
        "window_prediction_content_sha256": outcome.window_prediction_sha256,
        "native_prediction_content_sha256": outcome.native_prediction_sha256,
        "window_rows": len(outcome.window_predictions),
        "native_rows": len(outcome.native_predictions),
        "aggregation": "mean_window_probability_per_native_unit_v1",
        "outer_holdout_predictions_created": 0,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(paths["prediction_manifest"], prediction_manifest)
    environment = _environment_payload(execution)
    _write_json_exclusive(paths["environment"], environment)
    result = {
        "schema_version": RUN_RESULT_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_TRAINING",
        "run_id": run_id,
        "view_id": view_id,
        "process_id": os.getpid(),
        "started_at_utc": planned["started_at_utc"],
        "completed_at_utc": _utc_now(),
        "runtime_seconds": runtime_seconds,
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "development_metrics_authorized": True,
        "config_sha256": config.sha256,
        "implementation_hashes": implementation_hashes(config),
        "planned_run_manifest_sha256": planned_sha,
        "selection_content_sha256": selection["selection_content_sha256"],
        "train_native_unit_sha256": selection["train_native_unit_sha256"],
        "validation_native_unit_sha256": selection[
            "validation_native_unit_sha256"
        ],
        "train_native_units": selection["train_native_units"],
        "validation_native_units": selection["validation_native_units"],
        "train_windows": selection["train_windows"],
        "validation_windows": selection["validation_windows"],
        "optimizer_steps": outcome.optimizer_steps,
        "best_epoch": outcome.best_epoch,
        "validation_metrics": outcome.metrics,
        "parameter_sha256": outcome.parameter_sha256,
        "window_prediction_content_sha256": outcome.window_prediction_sha256,
        "native_prediction_content_sha256": outcome.native_prediction_sha256,
        "epoch_metrics_content_sha256": outcome.epoch_metrics_sha256,
        "maximum_loaded_batch_bytes": outcome.maximum_loaded_batch_bytes,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "execution": execution,
        "errors": [],
        "valid": True,
    }
    _write_json_exclusive(paths["run_result"], result)
    artifacts = _build_artifact_manifest(paths, run_id=run_id, view_id=view_id)
    _write_json_exclusive(paths["artifact_manifest"], artifacts)
    final_manifest = {
        **planned,
        "status": "completed",
        "completed_at_utc": result["completed_at_utc"],
        "runtime_seconds": runtime_seconds,
        "best_epoch": outcome.best_epoch,
        "optimizer_steps": outcome.optimizer_steps,
        "validation_metrics": outcome.metrics,
        "run_result_sha256": file_sha256(paths["run_result"]),
        "artifact_manifest_sha256": file_sha256(paths["artifact_manifest"]),
        "checkpoint_manifest_sha256": file_sha256(paths["checkpoint_manifest"]),
        "prediction_manifest_sha256": file_sha256(paths["prediction_manifest"]),
        "failure_reason": "",
    }
    _replace_json(paths["run_manifest"], final_manifest)
    return result


def _planned_manifest(
    config: TemporalLadderConfig,
    *,
    view_id: str,
    run_id: str,
    selection: dict[str, Any],
    parent: dict[str, Any],
    preflight: dict[str, Any],
    git_guard: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    expected = CANONICAL_VIEWS[view_id]
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "view_id": view_id,
        "experiment_name": "legacy_l5_v1_temporal_length_protocol_ladder_v1",
        "status": "planned",
        "started_at_utc": started_at,
        "process_id": os.getpid(),
        "code_sha": git_guard["code_sha"],
        "dirty_worktree": bool(git_guard["dirty_entries"]),
        "dirty_entries": git_guard["dirty_entries"],
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_hashes": implementation_hashes(config),
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "feature_control_id": "V1",
        "temporal_encoder_name": "masked_mean",
        "temporal_view_name": expected["temporal_view_name"],
        "sampling_protocol": expected["sampling_protocol"],
        "sequence_length": expected["sequence_length"],
        "windows_per_native_unit": expected["windows_per_native_unit"],
        "selection_content_sha256": selection["selection_content_sha256"],
        "train_native_unit_sha256": selection["train_native_unit_sha256"],
        "validation_native_unit_sha256": selection[
            "validation_native_unit_sha256"
        ],
        "train_native_units": selection["train_native_units"],
        "validation_native_units": selection["validation_native_units"],
        "train_windows": selection["train_windows"],
        "validation_windows": selection["validation_windows"],
        "consumer_parent": parent["hashes"],
        "preflight_valid": preflight["valid"],
        "precision": "float32",
        "autocast_enabled": False,
        "cublas_workspace_config": config.payload["optimization"][
            "cublas_workspace_config"
        ],
        "oom_retry_allowed": False,
        "source_media_reads": 0,
        "outer_predictions_created": 0,
    }


def _build_artifact_manifest(
    paths: dict[str, Path],
    *,
    run_id: str,
    view_id: str,
) -> dict[str, Any]:
    artifacts = []
    for name, filename in ARTIFACT_FILES.items():
        path = paths[name]
        if path.name != filename or not path.is_file():
            raise ValueError(f"missing temporal ladder artifact: {name}")
        artifacts.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "direction": "output",
            }
        )
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "run_id": run_id,
        "view_id": view_id,
        "status": "completed",
        "artifacts": artifacts,
        "errors": [],
        "valid": True,
    }


def _audit_artifacts(run_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        return {
            "verified_artifacts": 0,
            "errors": ["artifact_rows_not_list"],
            "valid": False,
        }
    by_name = {str(row.get("name")): row for row in rows if isinstance(row, dict)}
    if set(by_name) != set(ARTIFACT_FILES):
        errors.append("artifact_name_set_mismatch")
    verified = 0
    for name, filename in ARTIFACT_FILES.items():
        row = by_name.get(name)
        if row is None:
            continue
        path = run_root / filename
        if Path(str(row.get("path", ""))).resolve() != path.resolve():
            errors.append(f"artifact_path_mismatch={name}")
            continue
        if not path.is_file():
            errors.append(f"artifact_missing={name}")
            continue
        if file_sha256(path) != row.get("sha256"):
            errors.append(f"artifact_hash_mismatch={name}")
            continue
        if path.stat().st_size != int(row.get("size_bytes", -1)):
            errors.append(f"artifact_size_mismatch={name}")
            continue
        if row.get("direction") != "output":
            errors.append(f"artifact_direction_mismatch={name}")
            continue
        verified += 1
    return {
        "verified_artifacts": verified,
        "required_artifacts": len(ARTIFACT_FILES),
        "errors": errors,
        "valid": not errors and verified == len(ARTIFACT_FILES),
    }


def _failure_result(
    config: TemporalLadderConfig,
    *,
    view_id: str,
    run_id: str,
    planned: dict[str, Any],
    planned_sha: str,
    selection: dict[str, Any],
    execution: dict[str, Any],
    failure: dict[str, Any] | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    message = str((failure or {}).get("error_message", "execution invalid"))
    return {
        "schema_version": RUN_RESULT_SCHEMA,
        "status": "FAIL_LEGACY_DEVELOPMENT_L5_TEMPORAL_LADDER_TRAINING",
        "run_id": run_id,
        "view_id": view_id,
        "process_id": os.getpid(),
        "started_at_utc": planned["started_at_utc"],
        "completed_at_utc": _utc_now(),
        "runtime_seconds": runtime_seconds,
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "development_metrics_authorized": False,
        "config_sha256": config.sha256,
        "implementation_hashes": implementation_hashes(config),
        "planned_run_manifest_sha256": planned_sha,
        "selection_content_sha256": selection["selection_content_sha256"],
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "execution": execution,
        "errors": [message],
        "valid": False,
    }


def _failed_execution(
    config: TemporalLadderConfig,
    error: Exception,
) -> dict[str, Any]:
    optimization = config.payload["optimization"]
    return {
        "device": optimization["device"],
        "device_name": "execution_failed_before_complete_report",
        "process_id": os.getpid(),
        "actual_total_vram_bytes": None,
        "mem_info_total_vram_bytes": None,
        "free_vram_before_bytes": None,
        "allocated_before_bytes": None,
        "reserved_before_bytes": None,
        "allocator_fraction": optimization["maximum_peak_vram_fraction"],
        "allocator_limit_bytes": optimization["allocator_limit_bytes"],
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "post_cleanup_allocated_bytes": None,
        "post_cleanup_reserved_bytes": None,
        "cublas_workspaces_cleared": False,
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "precision": "float32",
        "autocast_enabled": False,
        "oom": isinstance(error, torch.cuda.OutOfMemoryError),
        "oom_message": str(error),
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [f"{type(error).__name__}: {error}"],
        "valid": False,
    }


def _environment_payload(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": (
            "classification_v2.legacy_development_l5."
            "temporal_ladder_environment.v1"
        ),
        "captured_at_utc": _utc_now(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "device_name": execution["device_name"],
        "actual_total_vram_bytes": execution["actual_total_vram_bytes"],
        "cublas_workspace_config": execution["cublas_workspace_config"],
        "precision": "float32",
        "autocast_enabled": False,
        "dataloader_num_workers": 0,
        "pin_memory": False,
        "oom_retry_allowed": False,
    }


def _non_overlapping_intervals(
    primary: dict[str, Any],
    repeat: dict[str, Any],
) -> dict[str, Any]:
    try:
        first_start = datetime.fromisoformat(str(primary["started_at_utc"]))
        first_end = datetime.fromisoformat(str(primary["completed_at_utc"]))
        second_start = datetime.fromisoformat(str(repeat["started_at_utc"]))
        second_end = datetime.fromisoformat(str(repeat["completed_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        return {
            "primary_interval": None,
            "repeat_interval": None,
            "errors": [f"invalid_execution_interval={error}"],
            "valid": False,
        }
    non_overlapping = first_end <= second_start or second_end <= first_start
    errors = [] if non_overlapping else ["training_run_intervals_overlap"]
    return {
        "primary_interval": [first_start.isoformat(), first_end.isoformat()],
        "repeat_interval": [second_start.isoformat(), second_end.isoformat()],
        "errors": errors,
        "valid": not errors,
    }


def _audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": audit["run_id"],
        "view_id": audit["view_id"],
        "result_path": audit["result_path"],
        "result_sha256": audit["result_sha256"],
        "run_manifest_sha256": audit["run_manifest_sha256"],
        "artifact_manifest_sha256": audit["artifact_manifest_sha256"],
        "verified_artifacts": audit["verified_artifacts"],
        "result": audit["result"],
        "errors": audit["errors"],
        "valid": audit["valid"],
    }


def _repeat_gate_filename(view_id: str) -> str:
    if view_id not in CANONICAL_VIEWS:
        raise ValueError(f"unknown temporal ladder view: {view_id}")
    return f"legacy_l5_temporal_ladder_{view_id}_short_gate_v1.json"


def _run_paths(root: Path) -> dict[str, Path]:
    paths = {name: root / filename for name, filename in ARTIFACT_FILES.items()}
    paths.update(
        {
            "run_manifest": root / "run_manifest.json",
            "artifact_manifest": root / "artifact_manifest.json",
            "unexpected_failure": root / "unexpected_failure.json",
        }
    )
    return paths


def _safe_run_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value))


def _mapping_errors(
    payload: dict[str, Any],
    expected: dict[str, Any],
    prefix: str,
) -> list[str]:
    return [
        f"{prefix}.{field}={payload.get(field)!r}!={value!r}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    temporary.replace(path)


def _write_dataframe_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n", float_format="%.17g")


def _write_torch_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ARTIFACT_FILES",
    "ARTIFACT_MANIFEST_SCHEMA",
    "MATRIX_GATE_SCHEMA",
    "REPEAT_GATE_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "RUN_RESULT_SCHEMA",
    "audit_temporal_ladder_repeat_gate",
    "audit_temporal_ladder_run",
    "audit_temporal_ladder_short_matrix",
    "run_temporal_ladder_view",
    "write_temporal_ladder_repeat_gate",
    "write_temporal_ladder_short_matrix",
]
