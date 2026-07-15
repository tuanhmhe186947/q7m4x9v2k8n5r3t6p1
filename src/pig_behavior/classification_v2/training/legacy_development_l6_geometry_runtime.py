"""Crash-bounded execution and immutable packets for legacy L6 geometry."""

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

from pig_behavior.classification_v2.training.legacy_development_l6_geometry import (
    LINEAGE_SCOPE,
    MODES,
    SHORT_SCOPE,
    LegacyL6GeometryConfig,
    LegacyL6GeometryOutcome,
    fit_geometry_normalization,
    geometry_training_git_guard,
    implementation_hashes,
    l6_feature_whitelist,
    load_geometry_training_inputs,
    preflight_geometry_mode,
    train_geometry_core,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

RUN_RESULT_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_run_result.v1"
)
RUN_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_run_manifest.v1"
)
ARTIFACT_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_artifacts.v1"
)
REPEAT_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_repeat_gate.v1"
)
MATRIX_GATE_SCHEMA = (
    "classification_v2.legacy_development_l6.geometry_short_matrix.v1"
)

ARTIFACT_FILES = {
    "environment": "environment.json",
    "preflight": "preflight.json",
    "selection_manifest": "training_selection_manifest.csv",
    "selection_audit": "training_selection_audit.json",
    "normalization": "geometry_normalization.json",
    "feature_whitelist": "feature_whitelist.json",
    "epoch_metrics": "epoch_metrics.csv",
    "window_predictions": "validation_window_predictions.csv",
    "native_predictions": "validation_native_predictions.csv",
    "validation_metrics": "validation_metrics.json",
    "validation_per_class": "validation_per_class.csv",
    "validation_confusion": "validation_confusion.csv",
    "confusion_groups": "validation_confusion_groups.csv",
    "missing_window_predictions": (
        "missing_modality_validation_window_predictions.csv"
    ),
    "missing_native_predictions": (
        "missing_modality_validation_native_predictions.csv"
    ),
    "missing_validation_metrics": "missing_modality_validation_metrics.json",
    "missing_confusion_groups": (
        "missing_modality_validation_confusion_groups.csv"
    ),
    "checkpoint": "best_validation_checkpoint.pt",
    "checkpoint_manifest": "checkpoint_manifest.json",
    "prediction_manifest": "prediction_manifest.json",
    "run_result": "run_result.json",
}

_RUN_EXECUTED_IN_PROCESS = False
MAX_WINDOWS_ARTIFACT_PATH_CHARS = 240


def run_geometry_mode(
    config: LegacyL6GeometryConfig,
    *,
    mode: str,
    run_id: str,
) -> dict[str, Any]:
    """Run one mode once in a fresh process, without an OOM retry."""

    global _RUN_EXECUTED_IN_PROCESS
    if _RUN_EXECUTED_IN_PROCESS:
        raise RuntimeError("L6 geometry permits one run per process")
    if mode not in MODES:
        raise ValueError(f"unknown L6 geometry mode={mode}")
    if not _safe_run_id(run_id):
        raise ValueError(f"unsafe L6 geometry run ID={run_id!r}")
    path_length_audit = _validate_run_path_lengths(
        config,
        mode=mode,
        run_id=run_id,
    )
    preflight = preflight_geometry_mode(config, mode)
    if not preflight["gpu_launch_authorized"]:
        raise RuntimeError(f"L6 geometry preflight failed={preflight['errors']}")
    _RUN_EXECUTED_IN_PROCESS = True
    _, base, cache, selection = load_geometry_training_inputs(config)
    normalization = fit_geometry_normalization(cache, selection)
    run_root = config.output_root / mode / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    paths = _run_paths(run_root)
    git_guard = geometry_training_git_guard(config)
    if not git_guard["valid"]:
        raise RuntimeError(f"L6 geometry Git guard failed={git_guard['errors']}")
    started_at = _utc_now()
    started = time.perf_counter()
    planned = _planned_manifest(
        config,
        mode=mode,
        run_id=run_id,
        selection=selection.audit,
        normalization=normalization.to_payload(),
        preflight=preflight,
        git_guard=git_guard,
        path_length_audit=path_length_audit,
        started_at=started_at,
    )
    _write_json_exclusive(paths["run_manifest"], planned)
    planned_sha = file_sha256(paths["run_manifest"])
    _write_json_exclusive(paths["preflight"], preflight)
    _write_dataframe_exclusive(paths["selection_manifest"], selection.manifest)
    _write_json_exclusive(paths["selection_audit"], selection.audit)
    _write_json_exclusive(paths["normalization"], normalization.to_payload())
    _write_json_exclusive(paths["feature_whitelist"], l6_feature_whitelist(mode))
    outcome: LegacyL6GeometryOutcome | None = None
    failure: dict[str, Any] | None = None
    try:
        outcome, execution = _execute_cuda(
            base,
            cache,
            selection,
            config,
            mode=mode,
        )
    except Exception as error:
        failure = {
            "schema_version": (
                "classification_v2.legacy_development_l6.geometry_failure.v1"
            ),
            "run_id": run_id,
            "mode": mode,
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
        mode=mode,
        run_id=run_id,
        planned=planned,
        planned_sha=planned_sha,
        selection=selection.audit,
        outcome=outcome,
        execution=execution,
        failure=failure,
        runtime_seconds=runtime_seconds,
    )


def audit_geometry_run(
    config: LegacyL6GeometryConfig,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Verify one completed packet and every declared artifact hash."""

    resolved = result_path.resolve()
    if resolved.name != ARTIFACT_FILES["run_result"]:
        raise ValueError("L6 geometry result filename mismatch")
    root = resolved.parent
    result = _read_json(resolved)
    manifest = _read_json(root / "run_manifest.json")
    artifact = _read_json(root / "artifact_manifest.json")
    errors: list[str] = []
    expected_result = {
        "schema_version": RUN_RESULT_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_TRAINING",
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
        "mode": result.get("mode"),
        "status": "completed",
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "config_sha256": config.sha256,
        "run_result_sha256": file_sha256(resolved),
        "artifact_manifest_sha256": file_sha256(
            root / "artifact_manifest.json"
        ),
        "failure_reason": "",
    }
    errors.extend(_mapping_errors(manifest, expected_manifest, "manifest"))
    artifact_audit = _audit_artifacts(root, artifact)
    errors.extend(artifact_audit["errors"])
    if result.get("mode") not in MODES:
        errors.append("unknown_geometry_mode")
    execution = _object(result.get("execution"), "execution")
    execution_expected = {
        "cublas_workspaces_cleared": True,
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
            "classification_v2.legacy_development_l6.geometry_run_audit.v1"
        ),
        "run_id": result.get("run_id"),
        "mode": result.get("mode"),
        "result_path": str(resolved),
        "result_sha256": file_sha256(resolved),
        "run_manifest_sha256": file_sha256(root / "run_manifest.json"),
        "artifact_manifest_sha256": file_sha256(
            root / "artifact_manifest.json"
        ),
        "verified_artifacts": artifact_audit["verified_artifacts"],
        "result": result,
        "manifest": manifest,
        "errors": errors,
        "valid": not errors,
    }


def audit_geometry_repeat_gate(
    config: LegacyL6GeometryConfig,
    *,
    mode: str,
    primary_result_path: Path,
    repeat_result_path: Path,
) -> dict[str, Any]:
    """Require two deterministic, fresh, non-overlapping short executions."""

    if config.training_scope != SHORT_SCOPE:
        raise ValueError("L6 repeat gate requires short scope")
    if mode not in MODES:
        raise ValueError(f"unknown L6 geometry mode={mode}")
    primary = audit_geometry_run(config, result_path=primary_result_path)
    repeat = audit_geometry_run(config, result_path=repeat_result_path)
    errors = [*primary["errors"], *repeat["errors"]]
    first = _object(primary.get("result"), "primary result")
    second = _object(repeat.get("result"), "repeat result")
    equality_fields = (
        "config_sha256",
        "mode",
        "selection_content_sha256",
        "normalization_state_sha256",
        "parameter_sha256",
        "window_prediction_sha256",
        "native_prediction_sha256",
        "epoch_metrics_sha256",
        "missing_native_prediction_sha256",
        "best_epoch",
        "optimizer_steps",
    )
    equality: dict[str, bool] = {}
    for field in equality_fields:
        equality[field] = first.get(field) == second.get(field)
        if not equality[field]:
            errors.append(f"repeat_field_differs={field}")
    first_pid = first.get("process_id")
    second_pid = second.get("process_id")
    distinct_processes = first_pid != second_pid
    if not distinct_processes:
        errors.append("repeat_process_ids_not_distinct")
    interval = _non_overlapping_intervals(first, second)
    errors.extend(interval["errors"])
    valid = not errors
    return {
        "schema_version": REPEAT_GATE_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_REPEAT"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_GEOMETRY_REPEAT"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": config.training_scope,
        "mode": mode,
        "short_config_sha256": config.sha256,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "primary": _audit_summary(primary),
        "repeat": _audit_summary(repeat),
        "equality": equality,
        "distinct_process_ids": distinct_processes,
        "non_overlapping_execution": interval,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "full_mode_expansion_authorized": valid,
        "errors": errors,
        "valid": valid,
    }


def write_geometry_repeat_gate(
    config: LegacyL6GeometryConfig,
    *,
    mode: str,
    primary_result_path: Path,
    repeat_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    gate = audit_geometry_repeat_gate(
        config,
        mode=mode,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
    )
    _write_json_exclusive(output_path, gate)
    return gate


def audit_geometry_short_matrix(
    config: LegacyL6GeometryConfig,
    *,
    repeat_gate_paths: dict[str, Path],
) -> dict[str, Any]:
    """Close all three controls before authorizing any full L6 expansion."""

    if config.training_scope != SHORT_SCOPE:
        raise ValueError("L6 short matrix requires short scope")
    if set(repeat_gate_paths) != set(MODES):
        raise ValueError("L6 short matrix mode set drift")
    errors: list[str] = []
    summaries: dict[str, Any] = {}
    process_ids: list[int] = []
    for mode in MODES:
        path = repeat_gate_paths[mode].resolve()
        gate = _read_json(path)
        expected = {
            "schema_version": REPEAT_GATE_SCHEMA,
            "status": "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_REPEAT",
            "lineage_scope": LINEAGE_SCOPE,
            "training_scope": SHORT_SCOPE,
            "mode": mode,
            "short_config_sha256": config.sha256,
            "full_mode_expansion_authorized": True,
            "errors": [],
            "valid": True,
        }
        errors.extend(_mapping_errors(gate, expected, f"repeat gate {mode}"))
        primary = _object(gate.get("primary"), f"{mode}.primary")
        repeat = _object(gate.get("repeat"), f"{mode}.repeat")
        process_ids.extend([int(primary["process_id"]), int(repeat["process_id"])])
        summaries[mode] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "primary": primary,
            "repeat": repeat,
        }
    distinct_processes = len(set(process_ids)) == len(process_ids)
    if not distinct_processes:
        errors.append("short_matrix_process_ids_not_all_distinct")
    valid = not errors
    return {
        "schema_version": MATRIX_GATE_SCHEMA,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_SHORT_MATRIX"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L6_GEOMETRY_SHORT_MATRIX"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "training_scope": SHORT_SCOPE,
        "short_config_sha256": config.sha256,
        "modes": list(MODES),
        "repeat_gates": summaries,
        "all_process_ids_distinct": distinct_processes,
        "all_mode_repeat_gates_pass": valid,
        "full_expansion_authorized": valid,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }


def write_geometry_short_matrix(
    config: LegacyL6GeometryConfig,
    *,
    repeat_gate_paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    matrix = audit_geometry_short_matrix(
        config,
        repeat_gate_paths=repeat_gate_paths,
    )
    _write_json_exclusive(output_path, matrix)
    return matrix


def _execute_cuda(
    base: Any,
    cache: Any,
    selection: Any,
    config: LegacyL6GeometryConfig,
    *,
    mode: str,
) -> tuple[LegacyL6GeometryOutcome, dict[str, Any]]:
    optimization = _object(config.payload["optimization"], "optimization")
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG",
        str(optimization["cublas_workspace_config"]),
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    if torch.cuda.is_initialized():
        raise RuntimeError("L6 CUDA was initialized before bounded execution")
    if not torch.cuda.is_available():
        raise RuntimeError("L6 geometry requires an available CUDA device")
    device = torch.device(str(optimization["device"]))
    torch.cuda.set_device(device)
    device_index = device.index if device.index is not None else 0
    properties = torch.cuda.get_device_properties(device_index)
    total_vram = int(properties.total_memory)
    memory_fraction = float(optimization["maximum_peak_vram_fraction"])
    allocator_limit = int(total_vram * memory_fraction)
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device_index)
    torch.use_deterministic_algorithms(
        bool(optimization["deterministic_algorithms"])
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    outcome: LegacyL6GeometryOutcome | None = None
    peak_allocated = 0
    peak_reserved = 0
    cublas_workspaces_cleared = False
    try:
        outcome = train_geometry_core(
            base,
            cache,
            selection,
            config,
            mode,
            device=device,
        )
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if peak_reserved > allocator_limit:
            raise MemoryError(
                f"L6 peak reserved VRAM={peak_reserved}>{allocator_limit}"
            )
    finally:
        gc.collect()
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
        cublas_workspaces_cleared = True
        torch.cuda.empty_cache()
    post_allocated = int(torch.cuda.memory_allocated(device))
    post_reserved = int(torch.cuda.memory_reserved(device))
    if outcome is None:
        raise RuntimeError("L6 geometry CUDA execution produced no outcome")
    if post_allocated != 0 or post_reserved != 0:
        raise RuntimeError(
            "L6 geometry CUDA cleanup failed "
            f"allocated={post_allocated} reserved={post_reserved}"
        )
    return outcome, {
        "device": str(device),
        "gpu_name": str(properties.name),
        "gpu_total_vram_bytes": total_vram,
        "maximum_peak_vram_fraction": memory_fraction,
        "allocator_limit_bytes": allocator_limit,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "maximum_loaded_batch_bytes": outcome.maximum_loaded_batch_bytes,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pytorch_cuda_alloc_conf": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF"
        ),
        "cublas_workspaces_cleared": cublas_workspaces_cleared,
        "oom": False,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "post_cleanup_allocated_bytes": post_allocated,
        "post_cleanup_reserved_bytes": post_reserved,
        "errors": [],
        "valid": True,
    }


def _finalize_run(
    paths: dict[str, Path],
    *,
    config: LegacyL6GeometryConfig,
    mode: str,
    run_id: str,
    planned: dict[str, Any],
    planned_sha: str,
    selection: dict[str, Any],
    outcome: LegacyL6GeometryOutcome | None,
    execution: dict[str, Any],
    failure: dict[str, Any] | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    completed_at = _utc_now()
    if outcome is not None and failure is None:
        try:
            _write_outcome(paths, config=config, mode=mode, outcome=outcome)
        except Exception as error:
            failure = {
                "schema_version": (
                    "classification_v2.legacy_development_l6."
                    "geometry_failure.v1"
                ),
                "run_id": run_id,
                "mode": mode,
                "process_id": os.getpid(),
                "failure_stage": "artifact_finalization",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "oom_retry_performed": False,
                "captured_at_utc": _utc_now(),
            }
            if not paths["unexpected_failure"].exists():
                _write_json_exclusive(paths["unexpected_failure"], failure)
            execution = {
                **execution,
                "errors": [
                    *execution.get("errors", []),
                    f"artifact_finalization={type(error).__name__}: {error}",
                ],
                "valid": False,
            }
            outcome = None
    if outcome is not None and failure is None:
        result = _success_result(
            config,
            mode=mode,
            run_id=run_id,
            selection=selection,
            outcome=outcome,
            execution=execution,
            started_at=str(planned["started_at_utc"]),
            completed_at=completed_at,
            runtime_seconds=runtime_seconds,
        )
        status = "completed"
        failure_reason = ""
    else:
        result = _failure_result(
            config,
            mode=mode,
            run_id=run_id,
            selection=selection,
            execution=execution,
            failure=failure,
            started_at=str(planned["started_at_utc"]),
            completed_at=completed_at,
            runtime_seconds=runtime_seconds,
        )
        status = "failed"
        failure_reason = str((failure or {}).get("error_message", "unknown"))
    _write_json_exclusive(paths["environment"], _environment_payload(execution))
    _write_json_exclusive(paths["run_result"], result)
    artifact = _build_artifact_manifest(
        paths,
        run_id=run_id,
        mode=mode,
        status=status,
    )
    _write_json_exclusive(paths["artifact_manifest"], artifact)
    final_manifest = {
        **planned,
        "status": status,
        "planned_manifest_sha256": planned_sha,
        "completed_at_utc": completed_at,
        "runtime_seconds": runtime_seconds,
        "run_result_sha256": file_sha256(paths["run_result"]),
        "artifact_manifest_sha256": file_sha256(paths["artifact_manifest"]),
        "failure_reason": failure_reason,
    }
    _replace_json(paths["run_manifest"], final_manifest)
    return result


def _write_outcome(
    paths: dict[str, Path],
    *,
    config: LegacyL6GeometryConfig,
    mode: str,
    outcome: LegacyL6GeometryOutcome,
) -> None:
    _write_dataframe_exclusive(paths["epoch_metrics"], outcome.epoch_metrics)
    _write_dataframe_exclusive(
        paths["window_predictions"],
        outcome.window_predictions,
    )
    _write_dataframe_exclusive(
        paths["native_predictions"],
        outcome.native_predictions,
    )
    _write_json_exclusive(
        paths["validation_metrics"],
        outcome.validation_metrics,
    )
    _write_dataframe_exclusive(
        paths["validation_per_class"],
        outcome.per_class_metrics,
    )
    _write_dataframe_exclusive(
        paths["validation_confusion"],
        outcome.confusion,
    )
    _write_dataframe_exclusive(
        paths["confusion_groups"],
        outcome.confusion_groups,
    )
    _write_dataframe_exclusive(
        paths["missing_window_predictions"],
        outcome.missing_window_predictions,
    )
    _write_dataframe_exclusive(
        paths["missing_native_predictions"],
        outcome.missing_native_predictions,
    )
    _write_json_exclusive(
        paths["missing_validation_metrics"],
        outcome.missing_validation_metrics,
    )
    _write_dataframe_exclusive(
        paths["missing_confusion_groups"],
        outcome.missing_confusion_groups,
    )
    checkpoint = {
        "schema_version": (
            "classification_v2.legacy_development_l6.geometry_checkpoint.v1"
        ),
        "mode": mode,
        "config_sha256": config.sha256,
        "normalization": outcome.normalization.to_payload(),
        "best_epoch": outcome.best_epoch,
        "optimizer_steps": outcome.optimizer_steps,
        "model_state": outcome.model_state,
        "optimizer_state": outcome.optimizer_state,
    }
    _write_torch_exclusive(paths["checkpoint"], checkpoint)
    checkpoint_manifest = {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "geometry_checkpoint_manifest.v1"
        ),
        "mode": mode,
        "config_sha256": config.sha256,
        "checkpoint_filename": paths["checkpoint"].name,
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "parameter_sha256": outcome.parameter_sha256,
        "normalization_state_sha256": outcome.normalization.state_sha256,
        "best_epoch": outcome.best_epoch,
        "optimizer_steps": outcome.optimizer_steps,
    }
    _write_json_exclusive(paths["checkpoint_manifest"], checkpoint_manifest)
    prediction_manifest = {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "geometry_prediction_manifest.v1"
        ),
        "mode": mode,
        "config_sha256": config.sha256,
        "window_prediction_sha256": outcome.window_prediction_sha256,
        "native_prediction_sha256": outcome.native_prediction_sha256,
        "epoch_metrics_sha256": outcome.epoch_metrics_sha256,
        "missing_native_prediction_sha256": (
            outcome.missing_native_prediction_sha256
        ),
        "outer_holdout_predictions_created": 0,
    }
    _write_json_exclusive(paths["prediction_manifest"], prediction_manifest)


def _success_result(
    config: LegacyL6GeometryConfig,
    *,
    mode: str,
    run_id: str,
    selection: dict[str, Any],
    outcome: LegacyL6GeometryOutcome,
    execution: dict[str, Any],
    started_at: str,
    completed_at: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_RESULT_SCHEMA,
        "status": "PASS_LEGACY_DEVELOPMENT_L6_GEOMETRY_TRAINING",
        "run_id": run_id,
        "process_id": os.getpid(),
        "mode": mode,
        "training_scope": config.training_scope,
        "lineage_scope": LINEAGE_SCOPE,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_hashes": implementation_hashes(config),
        "selection_content_sha256": selection["selection_content_sha256"],
        "normalization_state_sha256": outcome.normalization.state_sha256,
        "parameter_sha256": outcome.parameter_sha256,
        "window_prediction_sha256": outcome.window_prediction_sha256,
        "native_prediction_sha256": outcome.native_prediction_sha256,
        "epoch_metrics_sha256": outcome.epoch_metrics_sha256,
        "missing_native_prediction_sha256": (
            outcome.missing_native_prediction_sha256
        ),
        "best_epoch": outcome.best_epoch,
        "optimizer_steps": outcome.optimizer_steps,
        "validation_metrics": outcome.validation_metrics,
        "missing_validation_metrics": outcome.missing_validation_metrics,
        "execution": execution,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "runtime_seconds": runtime_seconds,
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


def _failure_result(
    config: LegacyL6GeometryConfig,
    *,
    mode: str,
    run_id: str,
    selection: dict[str, Any],
    execution: dict[str, Any],
    failure: dict[str, Any] | None,
    started_at: str,
    completed_at: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    error_message = str((failure or {}).get("error_message", "unknown failure"))
    return {
        "schema_version": RUN_RESULT_SCHEMA,
        "status": "FAIL_LEGACY_DEVELOPMENT_L6_GEOMETRY_TRAINING",
        "run_id": run_id,
        "process_id": os.getpid(),
        "mode": mode,
        "training_scope": config.training_scope,
        "lineage_scope": LINEAGE_SCOPE,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "selection_content_sha256": selection.get("selection_content_sha256"),
        "execution": execution,
        "failure": failure,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "runtime_seconds": runtime_seconds,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "errors": [error_message],
        "valid": False,
    }


def _planned_manifest(
    config: LegacyL6GeometryConfig,
    *,
    mode: str,
    run_id: str,
    selection: dict[str, Any],
    normalization: dict[str, Any],
    preflight: dict[str, Any],
    git_guard: dict[str, Any],
    path_length_audit: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "process_id": os.getpid(),
        "mode": mode,
        "status": "planned",
        "training_scope": config.training_scope,
        "lineage_scope": LINEAGE_SCOPE,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_hashes": implementation_hashes(config),
        "selection_content_sha256": selection["selection_content_sha256"],
        "normalization_state_sha256": normalization["state_sha256"],
        "preflight_status": preflight["status"],
        "git_guard": git_guard,
        "path_length_audit": path_length_audit,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "runtime_seconds": None,
        "planned_manifest_sha256": None,
        "run_result_sha256": None,
        "artifact_manifest_sha256": None,
        "failure_reason": None,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
    }


def _failed_execution(
    config: LegacyL6GeometryConfig,
    error: Exception,
) -> dict[str, Any]:
    oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(
        error
    ).lower()
    allocated = 0
    reserved = 0
    if torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()
        device = torch.device(str(config.payload["optimization"]["device"]))
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
    return {
        "device": str(config.payload["optimization"]["device"]),
        "gpu_name": None,
        "gpu_total_vram_bytes": None,
        "maximum_peak_vram_fraction": float(
            config.payload["optimization"]["maximum_peak_vram_fraction"]
        ),
        "allocator_limit_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "maximum_loaded_batch_bytes": None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pytorch_cuda_alloc_conf": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF"
        ),
        "cublas_workspaces_cleared": False,
        "oom": oom,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "post_cleanup_allocated_bytes": allocated,
        "post_cleanup_reserved_bytes": reserved,
        "errors": [f"{type(error).__name__}: {error}"],
        "valid": False,
    }


def _environment_payload(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": (
            "classification_v2.legacy_development_l6.geometry_environment.v1"
        ),
        "process_id": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": execution.get("gpu_name"),
        "gpu_total_vram_bytes": execution.get("gpu_total_vram_bytes"),
        "peak_allocated_bytes": execution.get("peak_allocated_bytes"),
        "peak_reserved_bytes": execution.get("peak_reserved_bytes"),
        "maximum_peak_vram_fraction": execution.get(
            "maximum_peak_vram_fraction"
        ),
        "allocator_limit_bytes": execution.get("allocator_limit_bytes"),
        "cublas_workspace_config": execution.get("cublas_workspace_config"),
        "pytorch_cuda_alloc_conf": execution.get("pytorch_cuda_alloc_conf"),
        "cublas_workspaces_cleared": execution.get(
            "cublas_workspaces_cleared"
        ),
    }


def _build_artifact_manifest(
    paths: dict[str, Path],
    *,
    run_id: str,
    mode: str,
    status: str,
) -> dict[str, Any]:
    excluded = {"artifact_manifest", "run_manifest"}
    artifacts: dict[str, Any] = {}
    for name, path in paths.items():
        if name in excluded or not path.is_file():
            continue
        artifacts[name] = {
            "filename": path.name,
            "sha256": file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "errors": [],
        "valid": True,
    }


def _audit_artifacts(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    verified = 0
    artifacts = _object(manifest.get("artifacts"), "artifacts")
    if manifest.get("artifact_count") != len(artifacts):
        errors.append("artifact_count_mismatch")
    for name, value in artifacts.items():
        spec = _object(value, f"artifact.{name}")
        path = root / str(spec.get("filename", ""))
        if path.parent.resolve() != root.resolve() or not path.is_file():
            errors.append(f"artifact_missing={name}")
            continue
        if file_sha256(path) != spec.get("sha256"):
            errors.append(f"artifact_hash_mismatch={name}")
            continue
        if int(path.stat().st_size) != int(spec.get("size_bytes", -1)):
            errors.append(f"artifact_size_mismatch={name}")
            continue
        verified += 1
    return {
        "verified_artifacts": verified,
        "errors": errors,
        "valid": not errors,
    }


def _non_overlapping_intervals(
    primary: dict[str, Any],
    repeat: dict[str, Any],
) -> dict[str, Any]:
    try:
        primary_start = datetime.fromisoformat(str(primary["started_at_utc"]))
        primary_end = datetime.fromisoformat(str(primary["completed_at_utc"]))
        repeat_start = datetime.fromisoformat(str(repeat["started_at_utc"]))
        repeat_end = datetime.fromisoformat(str(repeat["completed_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        return {
            "primary_interval": None,
            "repeat_interval": None,
            "errors": [f"invalid_execution_interval={error}"],
            "valid": False,
        }
    ordered = primary_end <= repeat_start or repeat_end <= primary_start
    errors = [] if ordered else ["training_run_intervals_overlap"]
    return {
        "primary_interval": [primary_start.isoformat(), primary_end.isoformat()],
        "repeat_interval": [repeat_start.isoformat(), repeat_end.isoformat()],
        "errors": errors,
        "valid": not errors,
    }


def _audit_summary(audit: dict[str, Any]) -> dict[str, Any]:
    result = _object(audit.get("result"), "audit result")
    return {
        "run_id": result.get("run_id"),
        "process_id": result.get("process_id"),
        "result_path": audit.get("result_path"),
        "result_sha256": audit.get("result_sha256"),
        "run_manifest_sha256": audit.get("run_manifest_sha256"),
        "artifact_manifest_sha256": audit.get("artifact_manifest_sha256"),
        "started_at_utc": result.get("started_at_utc"),
        "completed_at_utc": result.get("completed_at_utc"),
        "runtime_seconds": result.get("runtime_seconds"),
        "selection_content_sha256": result.get("selection_content_sha256"),
        "normalization_state_sha256": result.get(
            "normalization_state_sha256"
        ),
        "parameter_sha256": result.get("parameter_sha256"),
        "native_prediction_sha256": result.get("native_prediction_sha256"),
        "valid": audit.get("valid"),
    }


def _mapping_errors(
    payload: dict[str, Any],
    expected: dict[str, Any],
    name: str,
) -> list[str]:
    return [
        f"{name}.{field}={payload.get(field)!r}!={value!r}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]


def _validate_run_path_lengths(
    config: LegacyL6GeometryConfig,
    *,
    mode: str,
    run_id: str,
) -> dict[str, Any]:
    root = config.output_root / mode / run_id
    paths = _run_paths(root)
    lengths = {name: len(str(path)) for name, path in paths.items()}
    longest_name = max(lengths, key=lengths.__getitem__)
    maximum = lengths[longest_name]
    limit_applies = os.name == "nt"
    if limit_applies and maximum > MAX_WINDOWS_ARTIFACT_PATH_CHARS:
        raise ValueError(
            "L6 geometry Windows artifact path too long "
            f"name={longest_name} chars={maximum}>"
            f"{MAX_WINDOWS_ARTIFACT_PATH_CHARS}"
        )
    return {
        "platform": os.name,
        "windows_limit_applies": limit_applies,
        "maximum_allowed_chars": MAX_WINDOWS_ARTIFACT_PATH_CHARS,
        "maximum_observed_chars": maximum,
        "longest_artifact": longest_name,
        "run_root": str(root),
        "valid": True,
    }


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
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value))


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    _write_json_exclusive(temporary, payload)
    temporary.replace(path)


def _write_dataframe_exclusive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, mode="x", lineterminator="\n")


def _write_torch_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid L6 geometry JSON={path}") from error
    return _object(payload, str(path))


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"L6 geometry runtime {name} must be an object")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
