"""Crash-bounded immutable runtime for the legacy L7 loss matrix."""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from pig_behavior.classification_v2.training.imbalance_losses import LOSS_POLICIES
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder_runtime import (
    _non_overlapping_intervals,
    _replace_json,
    _safe_run_id,
    _write_dataframe_exclusive,
    _write_json_exclusive,
    _write_torch_exclusive,
)
from pig_behavior.classification_v2.training.legacy_development_l7_imbalance import (
    fit_full_training_loss,
    train_l7_imbalance_core,
)
from pig_behavior.classification_v2.training.legacy_development_l7_imbalance_config import (
    LegacyL7ImbalanceConfig,
    l7_imbalance_git_guard,
    l7_implementation_hashes,
    load_l7_imbalance_inputs,
    preflight_l7_imbalance_policy,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256

RUN_RESULT_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_run_result.v1"
)
RUN_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_run_manifest.v1"
)
ARTIFACT_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_artifacts.v1"
)
REPEAT_GATE_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_repeat_gate.v1"
)
MATRIX_GATE_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_short_matrix.v1"
)
FAILURE_SCHEMA = "classification_v2.legacy_development_l7.imbalance_failure.v1"
CHECKPOINT_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_checkpoint.v1"
)
PREDICTION_MANIFEST_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_prediction_manifest.v1"
)
ENVIRONMENT_SCHEMA = (
    "classification_v2.legacy_development_l7.imbalance_environment.v1"
)

PASS_TRAINING_STATUS = "PASS_LEGACY_DEVELOPMENT_L7_IMBALANCE_TRAINING"
PASS_REPEAT_STATUS = "PASS_LEGACY_DEVELOPMENT_L7_IMBALANCE_REPEAT"
PASS_MATRIX_STATUS = "PASS_LEGACY_DEVELOPMENT_L7_IMBALANCE_SHORT_MATRIX"

ARTIFACT_FILES = {
    "environment": "environment.json",
    "preflight": "preflight.json",
    "selection_manifest": "training_selection_manifest.csv",
    "selection_audit": "training_selection_audit.json",
    "loss_fit": "loss_fit_audit.json",
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


def run_l7_imbalance_policy(
    config: LegacyL7ImbalanceConfig,
    *,
    policy: str,
    run_id: str,
) -> dict[str, Any]:
    """Run one policy once in a fresh process without an OOM retry."""

    global _RUN_EXECUTED_IN_PROCESS
    if _RUN_EXECUTED_IN_PROCESS:
        raise RuntimeError("L7 permits one policy run per process")
    if policy not in LOSS_POLICIES:
        raise ValueError(f"unknown L7 loss policy={policy}")
    if not _safe_run_id(run_id):
        raise ValueError(f"unsafe L7 run ID={run_id!r}")
    preflight = preflight_l7_imbalance_policy(config, policy)
    if not preflight["gpu_launch_authorized"]:
        raise RuntimeError(f"L7 preflight failed={preflight['errors']}")
    _RUN_EXECUTED_IN_PROCESS = True
    parent, view, selection = load_l7_imbalance_inputs(config)
    loss_fit = fit_full_training_loss(
        view,
        policy=policy,
        effective_number_beta=float(
            config.payload["loss_fit"]["effective_number_beta"]
        ),
    )
    run_root = config.output_root / policy / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    paths = _run_paths(run_root)
    git_guard = l7_imbalance_git_guard(config)
    if not git_guard["valid"]:
        raise RuntimeError(f"L7 Git guard failed={git_guard['errors']}")
    started_at = _utc_now()
    started = time.perf_counter()
    planned = _planned_manifest(
        config,
        policy=policy,
        run_id=run_id,
        selection=selection.audit,
        loss_fit=loss_fit.to_payload(),
        preflight=preflight,
        git_guard=git_guard,
        started_at=started_at,
    )
    _write_json_exclusive(paths["run_manifest"], planned)
    planned_sha = file_sha256(paths["run_manifest"])
    _write_json_exclusive(paths["preflight"], preflight)
    _write_dataframe_exclusive(paths["selection_manifest"], selection.manifest)
    _write_json_exclusive(paths["selection_audit"], selection.audit)
    _write_json_exclusive(paths["loss_fit"], loss_fit.to_payload())
    outcome: Any | None = None
    failure: dict[str, Any] | None = None
    try:
        outcome, execution = _execute_cuda(
            view,
            selection,
            parent,
            policy=policy,
            config=config,
        )
    except Exception as error:
        failure = _failure_packet(
            run_id=run_id,
            policy=policy,
            error=error,
            failure_stage="cuda_execution",
        )
        _write_json_exclusive(paths["unexpected_failure"], failure)
        execution = _failed_execution(config, error)
    runtime_seconds = time.perf_counter() - started
    return _finalize_run(
        paths,
        config=config,
        policy=policy,
        run_id=run_id,
        planned=planned,
        planned_sha=planned_sha,
        selection=selection.audit,
        outcome=outcome,
        execution=execution,
        failure=failure,
        runtime_seconds=runtime_seconds,
    )


def audit_l7_imbalance_run(
    config: LegacyL7ImbalanceConfig,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Verify one run result and every immutable output hash."""

    result_file = result_path.resolve()
    root = result_file.parent
    result = _read_json(result_file)
    errors: list[str] = []
    if result_file.name != ARTIFACT_FILES["run_result"]:
        errors.append("run_result_filename_mismatch")
    expected = {
        "schema_version": RUN_RESULT_SCHEMA,
        "status": PASS_TRAINING_STATUS,
        "config_sha256": config.sha256,
        "training_scope": config.training_scope,
        "loss_policy": result.get("loss_policy"),
        "lineage_scope": "legacy-only-unreviewed-development",
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
    for field, value in expected.items():
        if field == "loss_policy":
            continue
        if result.get(field) != value:
            errors.append(f"result.{field}={result.get(field)!r}!={value!r}")
    if result.get("loss_policy") not in LOSS_POLICIES:
        errors.append("unknown_result_loss_policy")
    manifest_path = root / "run_manifest.json"
    artifact_path = root / "artifact_manifest.json"
    if not manifest_path.is_file() or not artifact_path.is_file():
        errors.append("run_or_artifact_manifest_missing")
        return {
            "schema_version": RUN_RESULT_SCHEMA.replace(
                "run_result", "run_audit"
            ),
            "result_path": str(result_file),
            "errors": errors,
            "valid": False,
        }
    manifest = _read_json(manifest_path)
    artifact = _read_json(artifact_path)
    if manifest.get("run_result_sha256") != file_sha256(result_file):
        errors.append("run_manifest_result_hash_mismatch")
    if manifest.get("status") != "completed":
        errors.append("run_manifest_not_completed")
    artifact_audit = _audit_artifacts(root, artifact)
    errors.extend(artifact_audit["errors"])
    execution = result.get("execution") or {}
    for field, value in {
        "oom": False,
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "errors": [],
        "valid": True,
    }.items():
        if execution.get(field) != value:
            errors.append(f"execution.{field}={execution.get(field)!r}!={value!r}")
    return {
        "schema_version": (
            "classification_v2.legacy_development_l7.imbalance_run_audit.v1"
        ),
        "run_id": result.get("run_id"),
        "loss_policy": result.get("loss_policy"),
        "result_path": str(result_file),
        "result_sha256": file_sha256(result_file),
        "run_manifest_sha256": file_sha256(manifest_path),
        "artifact_manifest_sha256": file_sha256(artifact_path),
        "verified_artifacts": artifact_audit["verified_artifacts"],
        "errors": errors,
        "valid": not errors,
    }


def audit_l7_imbalance_repeat_gate(
    config: LegacyL7ImbalanceConfig,
    *,
    policy: str,
    primary_result_path: Path,
    repeat_result_path: Path,
) -> dict[str, Any]:
    """Require deterministic results from two non-overlapping processes."""

    primary = audit_l7_imbalance_run(config, result_path=primary_result_path)
    repeat = audit_l7_imbalance_run(config, result_path=repeat_result_path)
    errors = [*primary["errors"], *repeat["errors"]]
    left = _read_json(primary_result_path.resolve())
    right = _read_json(repeat_result_path.resolve())
    if left.get("loss_policy") != policy or right.get("loss_policy") != policy:
        errors.append("repeat_policy_mismatch")
    equality_fields = (
        "config_sha256",
        "loss_policy",
        "selection_content_sha256",
        "loss_fit_audit_sha256",
        "loss_fit_source_sha256",
        "loss_state_sha256",
        "parameter_sha256",
        "window_prediction_sha256",
        "native_prediction_sha256",
        "epoch_metrics_sha256",
        "validation_metrics",
        "optimizer_steps",
        "best_epoch",
    )
    equality: dict[str, bool] = {}
    for field in equality_fields:
        equality[field] = left.get(field) == right.get(field)
        if not equality[field]:
            errors.append(f"repeat_field_differs={field}")
    first_pid = int(left.get("process_id", -1))
    second_pid = int(right.get("process_id", -1))
    if first_pid <= 0 or first_pid == second_pid:
        errors.append("repeat_process_ids_not_distinct")
    interval = _non_overlapping_intervals(left, right)
    errors.extend(interval["errors"])
    valid = not errors
    return {
        "schema_version": REPEAT_GATE_SCHEMA,
        "status": PASS_REPEAT_STATUS if valid else "FAIL_" + PASS_REPEAT_STATUS[5:],
        "lineage_scope": "legacy-only-unreviewed-development",
        "training_scope": config.training_scope,
        "loss_policy": policy,
        "short_config_sha256": config.sha256,
        "primary": _summary(primary, left),
        "repeat": _summary(repeat, right),
        "equality": equality,
        "non_overlapping_execution": interval,
        "full_matrix_authorized": valid,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }


def write_l7_imbalance_repeat_gate(
    config: LegacyL7ImbalanceConfig,
    *,
    policy: str,
    primary_result_path: Path,
    repeat_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    gate = audit_l7_imbalance_repeat_gate(
        config,
        policy=policy,
        primary_result_path=primary_result_path,
        repeat_result_path=repeat_result_path,
    )
    _write_json_exclusive(output_path, gate)
    return gate


def audit_l7_imbalance_short_matrix(
    config: LegacyL7ImbalanceConfig,
    *,
    repeat_gate_paths: dict[str, Path],
) -> dict[str, Any]:
    """Close all three loss policies and require common selection evidence."""

    if set(repeat_gate_paths) != set(LOSS_POLICIES):
        raise ValueError("L7 matrix policy set drift")
    errors: list[str] = []
    gates: dict[str, Any] = {}
    selection_hashes: set[str] = set()
    fit_hashes: set[str] = set()
    process_ids: list[int] = []
    for policy in LOSS_POLICIES:
        gate = _read_json(repeat_gate_paths[policy].resolve())
        expected = {
            "schema_version": REPEAT_GATE_SCHEMA,
            "status": PASS_REPEAT_STATUS,
            "loss_policy": policy,
            "short_config_sha256": config.sha256,
            "full_matrix_authorized": True,
            "valid": True,
        }
        for field, value in expected.items():
            if gate.get(field) != value:
                errors.append(
                    f"gate.{policy}.{field}={gate.get(field)!r}!={value!r}"
                )
        left = (gate.get("primary") or {}).get("result") or {}
        right = (gate.get("repeat") or {}).get("result") or {}
        selection_hashes.add(str(left.get("selection_content_sha256")))
        fit_hashes.add(str(left.get("loss_fit_source_sha256")))
        process_ids.extend(
            [int(left.get("process_id", -1)), int(right.get("process_id", -1))]
        )
        gates[policy] = {
            "path": str(repeat_gate_paths[policy].resolve()),
            "sha256": file_sha256(repeat_gate_paths[policy]),
            "primary_result_sha256": (gate.get("primary") or {}).get(
                "result_sha256"
            ),
            "repeat_result_sha256": (gate.get("repeat") or {}).get(
                "result_sha256"
            ),
        }
    if len(selection_hashes) != 1:
        errors.append("L7 policies do not share selection hash")
    if len(fit_hashes) != 1:
        errors.append("L7 policies do not share loss-fit lineage hash")
    if len(process_ids) != len(set(process_ids)):
        errors.append("L7 matrix process IDs are not all distinct")
    valid = not errors
    return {
        "schema_version": MATRIX_GATE_SCHEMA,
        "status": PASS_MATRIX_STATUS if valid else "FAIL_" + PASS_MATRIX_STATUS[5:],
        "lineage_scope": "legacy-only-unreviewed-development",
        "training_scope": config.training_scope,
        "short_config_sha256": config.sha256,
        "loss_policies": list(LOSS_POLICIES),
        "gates": gates,
        "selection_content_sha256": (
            next(iter(selection_hashes)) if len(selection_hashes) == 1 else None
        ),
        "loss_fit_audit_sha256": (
            next(iter(fit_hashes)) if len(fit_hashes) == 1 else None
        ),
        "all_repeat_gates_pass": valid,
        "full_expansion_authorized": valid,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "errors": errors,
        "valid": valid,
    }


def write_l7_imbalance_short_matrix(
    config: LegacyL7ImbalanceConfig,
    *,
    repeat_gate_paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    matrix = audit_l7_imbalance_short_matrix(
        config,
        repeat_gate_paths=repeat_gate_paths,
    )
    _write_json_exclusive(output_path, matrix)
    return matrix


def _execute_cuda(
    view: Any,
    selection: Any,
    parent: Any,
    *,
    policy: str,
    config: LegacyL7ImbalanceConfig,
) -> tuple[Any, dict[str, Any]]:
    optimization = parent.payload["optimization"]
    expected_total = int(optimization["validated_local_gpu_vram_bytes"])
    device = torch.device(str(optimization["device"]))
    if device.type != "cuda" or torch.cuda.is_initialized():
        raise RuntimeError("L7 requires a fresh CUDA process")
    if not torch.cuda.is_available():
        raise RuntimeError("L7 CUDA is unavailable")
    device_index = device.index if device.index is not None else 0
    properties = torch.cuda.get_device_properties(device_index)
    actual_total = int(properties.total_memory)
    free_before, total_from_mem = (
        int(value) for value in torch.cuda.mem_get_info(device)
    )
    allocator_limit = int(optimization["allocator_limit_bytes"])
    fraction = allocator_limit / actual_total
    if actual_total != expected_total or total_from_mem != expected_total:
        raise RuntimeError("L7 GPU total VRAM does not match bound host")
    if free_before < allocator_limit:
        raise RuntimeError("L7 free VRAM is below configured allocator limit")
    if fraction > float(optimization["maximum_peak_vram_fraction"]):
        raise RuntimeError("L7 allocator fraction exceeds configured ceiling")
    torch.cuda.set_per_process_memory_fraction(fraction, device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    outcome: Any | None = None
    cublas_cleared = False
    try:
        outcome = train_l7_imbalance_core(
            view,
            selection,
            parent,
            policy=policy,
            device=device,
        )
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    finally:
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if not callable(clear_workspaces):
            raise RuntimeError("PyTorch cannot clear cuBLAS workspaces")
        clear_workspaces()
        cublas_cleared = True
        torch.cuda.empty_cache()
    post_allocated = int(torch.cuda.memory_allocated(device))
    post_reserved = int(torch.cuda.memory_reserved(device))
    if outcome is None:
        raise RuntimeError("L7 CUDA execution produced no outcome")
    if post_allocated != 0 or post_reserved != 0:
        raise RuntimeError(
            f"L7 CUDA cleanup failed allocated={post_allocated} "
            f"reserved={post_reserved}"
        )
    return outcome, {
        "device": str(device),
        "gpu_name": str(properties.name),
        "gpu_total_vram_bytes": actual_total,
        "free_vram_before_bytes": free_before,
        "allocator_limit_bytes": allocator_limit,
        "maximum_peak_vram_fraction": float(
            optimization["maximum_peak_vram_fraction"]
        ),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "maximum_loaded_batch_bytes": outcome.maximum_loaded_batch_bytes,
        "cublas_workspace_config": optimization["cublas_workspace_config"],
        "cublas_workspaces_cleared": cublas_cleared,
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
    config: LegacyL7ImbalanceConfig,
    policy: str,
    run_id: str,
    planned: dict[str, Any],
    planned_sha: str,
    selection: dict[str, Any],
    outcome: Any | None,
    execution: dict[str, Any],
    failure: dict[str, Any] | None,
    runtime_seconds: float,
) -> dict[str, Any]:
    if outcome is None or failure is not None:
        result = {
            "schema_version": RUN_RESULT_SCHEMA,
            "status": "FAIL_LEGACY_DEVELOPMENT_L7_IMBALANCE_TRAINING",
            "run_id": run_id,
            "process_id": os.getpid(),
            "loss_policy": policy,
            "training_scope": config.training_scope,
            "lineage_scope": "legacy-only-unreviewed-development",
            "config_sha256": config.sha256,
            "selection_content_sha256": selection["selection_content_sha256"],
            "execution": execution,
            "failure": failure,
            "started_at_utc": planned["started_at_utc"],
            "completed_at_utc": _utc_now(),
            "runtime_seconds": runtime_seconds,
            "human_review_complete": False,
            "reviewed_or_final_claim_allowed": False,
            "q2_claim_allowed": False,
            "canonical_full_oof_authorized": False,
            "outer_holdout_predictions_authorized": False,
            "source_media_reads": 0,
            "outer_holdout_predictions_created": 0,
            "errors": [
                str((failure or {}).get("error_message", "L7 execution failed"))
            ],
            "valid": False,
        }
        _write_json_exclusive(paths["environment"], _environment_payload(execution))
        _write_json_exclusive(paths["run_result"], result)
        artifact = _artifact_manifest(
            paths,
            run_id=run_id,
            policy=policy,
            allow_partial=True,
        )
        _write_json_exclusive(paths["artifact_manifest"], artifact)
        _replace_json(
            paths["run_manifest"],
            {
                **planned,
                "status": "failed",
                "completed_at_utc": result["completed_at_utc"],
                "runtime_seconds": runtime_seconds,
                "run_result_sha256": file_sha256(paths["run_result"]),
                "artifact_manifest_sha256": file_sha256(
                    paths["artifact_manifest"]
                ),
                "failure_reason": result["errors"][0],
            },
        )
        return result
    _write_dataframe_exclusive(paths["epoch_metrics"], outcome.epoch_metrics)
    _write_dataframe_exclusive(
        paths["window_predictions"],
        outcome.window_predictions,
    )
    _write_dataframe_exclusive(
        paths["native_predictions"],
        outcome.native_predictions,
    )
    _write_json_exclusive(paths["validation_metrics"], outcome.validation_metrics)
    _write_dataframe_exclusive(
        paths["validation_per_class"],
        outcome.per_class_metrics,
    )
    _write_dataframe_exclusive(paths["validation_confusion"], outcome.confusion)
    _write_torch_exclusive(
        paths["checkpoint"],
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "run_id": run_id,
            "loss_policy": policy,
            "config_sha256": config.sha256,
            "loss_fit": outcome.loss_fit.to_payload(),
            "best_epoch": outcome.best_epoch,
            "optimizer_steps": outcome.optimizer_steps,
            "model_state": outcome.model_state,
            "optimizer_state": outcome.optimizer_state,
        },
    )
    _write_json_exclusive(
        paths["checkpoint_manifest"],
        {
            "schema_version": CHECKPOINT_SCHEMA + ".manifest",
            "run_id": run_id,
            "loss_policy": policy,
            "checkpoint_sha256": file_sha256(paths["checkpoint"]),
            "loss_fit_audit_sha256": outcome.loss_fit.fit_audit_sha256,
            "parameter_sha256": outcome.parameter_sha256,
            "best_epoch": outcome.best_epoch,
            "optimizer_steps": outcome.optimizer_steps,
        },
    )
    _write_json_exclusive(
        paths["prediction_manifest"],
        {
            "schema_version": PREDICTION_MANIFEST_SCHEMA,
            "run_id": run_id,
            "loss_policy": policy,
            "window_prediction_sha256": outcome.window_prediction_sha256,
            "native_prediction_sha256": outcome.native_prediction_sha256,
            "epoch_metrics_sha256": outcome.epoch_metrics_sha256,
            "outer_holdout_predictions_created": 0,
        },
    )
    result = {
        "schema_version": RUN_RESULT_SCHEMA,
        "status": PASS_TRAINING_STATUS,
        "run_id": run_id,
        "process_id": os.getpid(),
        "loss_policy": policy,
        "training_scope": config.training_scope,
        "lineage_scope": "legacy-only-unreviewed-development",
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_hashes": l7_implementation_hashes(config),
        "started_at_utc": planned["started_at_utc"],
        "completed_at_utc": _utc_now(),
        "runtime_seconds": runtime_seconds,
        "selection_content_sha256": selection["selection_content_sha256"],
        "train_native_units": selection["train_native_units"],
        "validation_native_units": selection["validation_native_units"],
        "train_windows": selection["train_windows"],
        "validation_windows": selection["validation_windows"],
        "optimizer_steps": outcome.optimizer_steps,
        "best_epoch": outcome.best_epoch,
        "validation_metrics": outcome.validation_metrics,
        "loss_fit_audit_sha256": outcome.loss_fit.fit_audit_sha256,
        "loss_fit_source_sha256": outcome.loss_fit.source_sha256,
        "loss_state_sha256": outcome.loss_fit.state_sha256,
        "parameter_sha256": outcome.parameter_sha256,
        "window_prediction_sha256": outcome.window_prediction_sha256,
        "native_prediction_sha256": outcome.native_prediction_sha256,
        "epoch_metrics_sha256": outcome.epoch_metrics_sha256,
        "execution": execution,
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
    _write_json_exclusive(paths["environment"], _environment_payload(execution))
    _write_json_exclusive(paths["run_result"], result)
    _write_json_exclusive(
        paths["artifact_manifest"],
        _artifact_manifest(paths, run_id=run_id, policy=policy),
    )
    _replace_json(
        paths["run_manifest"],
        {
            **planned,
            "status": "completed",
            "completed_at_utc": result["completed_at_utc"],
            "runtime_seconds": runtime_seconds,
            "run_result_sha256": file_sha256(paths["run_result"]),
            "artifact_manifest_sha256": file_sha256(
                paths["artifact_manifest"]
            ),
            "failure_reason": "",
        },
    )
    return result


def _planned_manifest(
    config: LegacyL7ImbalanceConfig,
    *,
    policy: str,
    run_id: str,
    selection: dict[str, Any],
    loss_fit: dict[str, Any],
    preflight: dict[str, Any],
    git_guard: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "run_id": run_id,
        "loss_policy": policy,
        "status": "planned",
        "process_id": os.getpid(),
        "training_scope": config.training_scope,
        "lineage_scope": "legacy-only-unreviewed-development",
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "implementation_hashes": l7_implementation_hashes(config),
        "selection_content_sha256": selection["selection_content_sha256"],
        "loss_fit_audit_sha256": loss_fit["fit_audit_sha256"],
        "preflight_valid": preflight["valid"],
        "git_guard": git_guard,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "runtime_seconds": None,
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


def _artifact_manifest(
    paths: dict[str, Path],
    *,
    run_id: str,
    policy: str,
    allow_partial: bool = False,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name, filename in ARTIFACT_FILES.items():
        path = paths[name]
        if not path.is_file():
            if allow_partial:
                continue
            raise ValueError(f"L7 artifact missing={name}")
        if path.name != filename:
            raise ValueError(f"L7 artifact filename mismatch={name}")
        artifacts[name] = {
            "filename": filename,
            "sha256": file_sha256(path),
            "size_bytes": int(path.stat().st_size),
        }
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA,
        "run_id": run_id,
        "loss_policy": policy,
        "status": "completed",
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "errors": [],
        "valid": True,
    }


def _audit_artifacts(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return {"verified_artifacts": 0, "errors": ["artifacts_not_object"]}
    verified = 0
    for name, filename in ARTIFACT_FILES.items():
        spec = artifacts.get(name)
        path = root / filename
        if not isinstance(spec, dict) or not path.is_file():
            errors.append(f"artifact_missing={name}")
            continue
        if spec.get("filename") != filename:
            errors.append(f"artifact_filename_mismatch={name}")
            continue
        if spec.get("sha256") != file_sha256(path):
            errors.append(f"artifact_hash_mismatch={name}")
            continue
        if int(spec.get("size_bytes", -1)) != int(path.stat().st_size):
            errors.append(f"artifact_size_mismatch={name}")
            continue
        verified += 1
    if manifest.get("artifact_count") != len(ARTIFACT_FILES):
        errors.append("artifact_count_mismatch")
    return {"verified_artifacts": verified, "errors": errors}


def _failure_packet(
    *,
    run_id: str,
    policy: str,
    error: Exception,
    failure_stage: str,
) -> dict[str, Any]:
    return {
        "schema_version": FAILURE_SCHEMA,
        "run_id": run_id,
        "loss_policy": policy,
        "process_id": os.getpid(),
        "failure_stage": failure_stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "oom_retry_performed": False,
        "captured_at_utc": _utc_now(),
    }


def _failed_execution(
    config: LegacyL7ImbalanceConfig,
    error: Exception,
) -> dict[str, Any]:
    return {
        "device": "cuda:0",
        "gpu_name": None,
        "gpu_total_vram_bytes": None,
        "maximum_peak_vram_fraction": 0.7,
        "allocator_limit_bytes": None,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "maximum_loaded_batch_bytes": None,
        "cublas_workspace_config": None,
        "cublas_workspaces_cleared": False,
        "oom": isinstance(error, torch.cuda.OutOfMemoryError),
        "oom_retry_count": 0,
        "source_media_reads": 0,
        "outer_holdout_predictions_created": 0,
        "post_cleanup_allocated_bytes": 0,
        "post_cleanup_reserved_bytes": 0,
        "errors": [f"{type(error).__name__}: {error}"],
        "valid": False,
    }


def _environment_payload(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
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
        "post_cleanup_allocated_bytes": execution.get(
            "post_cleanup_allocated_bytes"
        ),
        "post_cleanup_reserved_bytes": execution.get(
            "post_cleanup_reserved_bytes"
        ),
        "cublas_workspace_config": execution.get("cublas_workspace_config"),
        "cublas_workspaces_cleared": execution.get(
            "cublas_workspaces_cleared"
        ),
    }


def _summary(audit: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_path": audit.get("result_path"),
        "result_sha256": audit.get("result_sha256"),
        "run_manifest_sha256": audit.get("run_manifest_sha256"),
        "artifact_manifest_sha256": audit.get("artifact_manifest_sha256"),
        "result": result,
        "errors": audit.get("errors", []),
        "valid": audit.get("valid", False),
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
