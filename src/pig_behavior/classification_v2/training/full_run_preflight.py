"""No-training preflight for an explicit, cache-only, reproducible full OOF run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from pig_behavior.classification_v2.contracts.training_snapshot import check_training_snapshot
from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    build_full_multimodal_oof_run_plan,
    full_run_config_fingerprint,
)


def build_full_run_preflight(
    config: FullMultimodalOofConfig,
    *,
    snapshot_json: Path,
    runtime_benchmark_audit_json: Path,
) -> dict[str, Any]:
    """Validate immutable data, runtime policy, CUDA, Git, and full workload without training."""

    plan = build_full_multimodal_oof_run_plan(config)
    snapshot = check_training_snapshot(snapshot_json)
    runtime = json.loads(runtime_benchmark_audit_json.read_text(encoding="utf-8"))
    git_state = _git_state()
    errors: list[str] = []
    warnings: list[str] = []
    if config.run_mode != "full":
        errors.append(f"preflight_requires_run_mode_full={config.run_mode}")
    if not plan.get("valid") or not plan.get("paper_facing_candidate_plan"):
        errors.append(f"invalid_full_workload_plan={plan.get('errors')}")
    if snapshot.get("valid") is not True:
        errors.append(f"invalid_training_snapshot={snapshot.get('errors')}")
    if not config.require_cached_images or config.packed_image_cache_npy is None:
        errors.append("full_run_requires_strict_packed_image_cache")
    for path_name, path in (
        ("packed_image_cache_npy", config.packed_image_cache_npy),
        ("packed_image_cache_index_csv", config.packed_image_cache_index_csv),
    ):
        if path is None or not path.exists():
            errors.append(f"missing_{path_name}={path}")
    errors.extend(_runtime_match_errors(config, runtime))
    if config.precision == "amp" and not torch.cuda.is_available():
        errors.append("amp_full_run_requires_cuda")
    if config.device not in {"cuda", "auto"}:
        errors.append(f"full_run_device_must_be_cuda_or_auto={config.device}")
    if git_state["dirty"] is not False:
        errors.append(f"full_run_requires_clean_git={git_state['dirty']}")
    if not git_state["commit"]:
        errors.append("full_run_requires_git_commit")

    recommendation = runtime.get("recommended_runtime_config") or {}
    throughput = float(recommendation.get("throughput_rows_per_sec", 0.0))
    total_training_rows = sum(
        int(fold.get("effective_training_steps", 0)) * int(fold.get("train_batch_size", 0))
        for fold in plan.get("folds", [])
    )
    estimated_training_seconds = float(total_training_rows / throughput) if throughput > 0.0 else None
    warnings.append("Estimated runtime excludes evaluation, bootstrap metrics, startup, and checkpoint IO.")
    return {
        "schema_version": "classification_v2_full_run_preflight_v1",
        "config_sha256": full_run_config_fingerprint(config),
        "config": plan.get("config"),
        "git_commit": git_state["commit"],
        "git_dirty": git_state["dirty"],
        "snapshot_json": str(snapshot_json),
        "snapshot_id": snapshot.get("expected_snapshot_id"),
        "snapshot_valid": bool(snapshot.get("valid")),
        "runtime_benchmark_audit_json": str(runtime_benchmark_audit_json),
        "runtime_recommendation": recommendation,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "available_fold_count": plan.get("available_fold_count"),
        "selected_fold_count": plan.get("selected_fold_count"),
        "total_eval_rows": plan.get("total_eval_rows"),
        "total_training_steps": plan.get("total_train_steps"),
        "estimated_training_seconds_excluding_eval": estimated_training_seconds,
        "workload_plan": plan,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def _runtime_match_errors(config: FullMultimodalOofConfig, runtime: dict[str, Any]) -> list[str]:
    """Require the full config to use a measured memory-safe runtime recommendation."""

    errors: list[str] = []
    if runtime.get("valid") is not True or runtime.get("errors"):
        errors.append(f"invalid_runtime_benchmark={runtime.get('errors')}")
        return errors
    recommendation = runtime.get("recommended_runtime_config") or {}
    if recommendation.get("precision") != config.precision:
        errors.append(
            f"runtime_precision_mismatch=recommended:{recommendation.get('precision')},config:{config.precision}"
        )
    if int(recommendation.get("train_batch_size", -1)) != int(config.train_batch_size):
        errors.append(
            "runtime_batch_size_mismatch="
            f"recommended:{recommendation.get('train_batch_size')},config:{config.train_batch_size}"
        )
    return errors


def _git_state() -> dict[str, Any]:
    """Read the commit and dirty state bound to the future full-run artifact."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except Exception:
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}
