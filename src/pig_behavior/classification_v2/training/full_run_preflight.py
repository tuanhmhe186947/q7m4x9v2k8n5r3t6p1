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

RUNTIME_RELEVANT_PATH_PREFIXES = (
    "configs/classification_v2/",
    "src/pig_behavior/classification_v2/contracts/",
    "src/pig_behavior/classification_v2/evaluation/",
    "src/pig_behavior/classification_v2/training/",
    "scripts/classification_v2/06_full_oof_training/classification_v2_run_full_multimodal_oof.py",
    "scripts/classification_v2/05_preflight_authorization/"
    "preflight_classification_v2_full_multimodal_oof.py",
    "scripts/classification_v2/04_baselines_smokes/"
    "summarize_classification_v2_runtime_benchmark.py",
)
RUNTIME_AUDIT_ONLY_PATHS = {
    "src/pig_behavior/classification_v2/training/full_run_preflight.py",
    "scripts/classification_v2/05_preflight_authorization/"
    "preflight_classification_v2_full_multimodal_oof.py",
}
AUTH_GATE_ONLY_RUNTIME_PATHS = {
    "scripts/classification_v2/06_full_oof_training/classification_v2_run_full_multimodal_oof.py",
}
AUTH_GATE_ONLY_DIFF_MARKERS = (
    "FULL_RUN_AUTHORIZATION",
    "authorization",
    "full_run_authorization",
    "reviewer",
    "reviewed_at",
    "schema_version",
)
AUTH_GATE_ONLY_STRUCTURAL_LINES = {
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    "errors.append(",
}


def build_full_run_preflight(
    config: FullMultimodalOofConfig,
    *,
    snapshot_json: Path,
    runtime_benchmark_audit_json: Path,
    feature_whitelist_audit_json: Path = Path(
        "outputs/classification_v2/model_design/q2_feature_whitelist_audit.json"
    ),
) -> dict[str, Any]:
    """Validate immutable data, runtime policy, CUDA, Git, and full workload without training."""

    plan = build_full_multimodal_oof_run_plan(config)
    snapshot = check_training_snapshot(snapshot_json)
    runtime = json.loads(runtime_benchmark_audit_json.read_text(encoding="utf-8"))
    feature_whitelist = _read_optional_json(feature_whitelist_audit_json)
    git_state = current_git_state()
    errors: list[str] = []
    warnings: list[str] = []
    if config.run_mode != "full":
        errors.append(f"preflight_requires_run_mode_full={config.run_mode}")
    if not plan.get("valid") or not plan.get("paper_facing_candidate_plan"):
        errors.append(f"invalid_full_workload_plan={plan.get('errors')}")
    if snapshot.get("valid") is not True:
        errors.append(f"invalid_training_snapshot={snapshot.get('errors')}")
    errors.extend(_feature_whitelist_audit_errors(feature_whitelist))
    if not config.require_cached_images or config.packed_image_cache_npy is None:
        errors.append("full_run_requires_strict_packed_image_cache")
    if not config.require_packed_visual_context or config.visual_context_packed_cache_npy is None:
        errors.append("full_run_requires_strict_packed_visual_context")
    errors.extend(_canonical_full_run_path_errors(config))
    for path_name, path in (
        ("packed_image_cache_npy", config.packed_image_cache_npy),
        ("packed_image_cache_index_csv", config.packed_image_cache_index_csv),
        ("visual_context_cache_manifest_csv", config.visual_context_cache_manifest_csv),
        ("visual_context_packed_cache_npy", config.visual_context_packed_cache_npy),
        ("visual_context_packed_cache_index_csv", config.visual_context_packed_cache_index_csv),
    ):
        if path is None or not path.exists():
            errors.append(f"missing_{path_name}={path}")
    runtime_match = _runtime_match_audit(
        config,
        runtime,
        expected_git_commit=git_state["commit"],
    )
    errors.extend(runtime_match["errors"])
    warnings.extend(runtime_match["warnings"])
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
    estimated_training_seconds = (
        float(total_training_rows / throughput) if throughput > 0.0 else None
    )
    warnings.append(
        "Estimated runtime excludes evaluation, bootstrap metrics, startup, and checkpoint IO."
    )
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
        "runtime_match_audit": runtime_match,
        "feature_whitelist_audit_json": str(feature_whitelist_audit_json),
        "feature_whitelist_valid": feature_whitelist.get("valid"),
        "feature_whitelist_contract_version": feature_whitelist.get("contract_version"),
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


def _feature_whitelist_audit_errors(audit: dict[str, Any]) -> list[str]:
    """Require the Q2 feature whitelist leakage audit before full OOF."""

    errors: list[str] = []
    if audit.get("missing") is True:
        errors.append(f"missing_feature_whitelist_audit={audit.get('path')}")
        return errors
    if audit.get("valid") is not True or audit.get("errors"):
        errors.append(f"invalid_feature_whitelist_audit={audit.get('errors')}")
    if audit.get("never_use_all_numeric_columns") is not True:
        errors.append("feature_whitelist_must_block_all_numeric_columns")
    if audit.get("fail_closed_on_unknown_columns") is not True:
        errors.append("feature_whitelist_must_fail_closed_on_unknown_columns")
    if audit.get("forbidden_probe_columns_not_blocked") not in ([], None):
        errors.append(
            f"feature_whitelist_probe_leakage={audit.get('forbidden_probe_columns_not_blocked')}"
        )
    return errors


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path), "valid": False}
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_match_audit(
    config: FullMultimodalOofConfig,
    runtime: dict[str, Any],
    *,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Require the full config to use a measured memory-safe runtime recommendation."""

    errors: list[str] = []
    warnings: list[str] = []
    if runtime.get("valid") is not True or runtime.get("errors"):
        errors.append(f"invalid_runtime_benchmark={runtime.get('errors')}")
        return {
            "errors": errors,
            "warnings": warnings,
            "runtime_git_commit_check": {},
        }
    recommendation = runtime.get("recommended_runtime_config") or {}
    if recommendation.get("model_architecture_version") != config.model_architecture_version:
        errors.append(
            "runtime_architecture_mismatch="
            f"recommended:{recommendation.get('model_architecture_version')},"
            f"config:{config.model_architecture_version}"
        )
    git_check = _runtime_git_commit_check(
        recommended_git_commit=recommendation.get("git_commit"),
        current_git_commit=expected_git_commit,
    )
    if git_check["changed"] and not git_check["allowed_without_rebenchmark"]:
        errors.append(
            "runtime_git_commit_mismatch_relevant_changes="
            f"recommended:{recommendation.get('git_commit')},"
            f"current:{expected_git_commit},"
            f"relevant_paths:{git_check['runtime_relevant_changed_paths']}"
        )
    elif git_check["changed"]:
        warnings.append(
            "runtime_git_commit_mismatch_allowed_audit_only_changes="
            f"recommended:{recommendation.get('git_commit')},"
            f"current:{expected_git_commit}"
        )
    if recommendation.get("precision") != config.precision:
        errors.append(
            "runtime_precision_mismatch="
            f"recommended:{recommendation.get('precision')},"
            f"config:{config.precision}"
        )
    if int(recommendation.get("train_batch_size", -1)) != int(config.train_batch_size):
        errors.append(
            "runtime_batch_size_mismatch="
            f"recommended:{recommendation.get('train_batch_size')},config:{config.train_batch_size}"
        )
    return {
        "errors": errors,
        "warnings": warnings,
        "runtime_git_commit_check": git_check,
    }


def _runtime_match_errors(
    config: FullMultimodalOofConfig,
    runtime: dict[str, Any],
    *,
    expected_git_commit: str | None = None,
) -> list[str]:
    """Return only errors for callers using the pre-audit compatibility API."""
    return _runtime_match_audit(
        config,
        runtime,
        expected_git_commit=expected_git_commit,
    )["errors"]


def _runtime_git_commit_check(
    *,
    recommended_git_commit: Any,
    current_git_commit: str | None,
) -> dict[str, Any]:
    """Classify benchmark commit drift by runtime-relevant source changes."""

    recommended = str(recommended_git_commit or "")
    current = str(current_git_commit or "")
    if not recommended or not current:
        return {
            "changed": recommended != current,
            "allowed_without_rebenchmark": False,
            "changed_paths": [],
            "runtime_relevant_changed_paths": [],
            "error": "missing_git_commit_for_runtime_benchmark_check",
        }
    if recommended == current:
        return {
            "changed": False,
            "allowed_without_rebenchmark": True,
            "changed_paths": [],
            "runtime_relevant_changed_paths": [],
            "error": None,
        }
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", f"{recommended}..{current}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return {
            "changed": True,
            "allowed_without_rebenchmark": False,
            "changed_paths": [],
            "runtime_relevant_changed_paths": [],
            "error": f"git_diff_failed={exc}",
        }
    changed_paths = [
        path.strip().replace("\\", "/") for path in diff.stdout.splitlines() if path.strip()
    ]
    relevant_paths = [
        path
        for path in changed_paths
        if _runtime_relevant_path(path)
        and not _changed_only_auth_gate(
            recommended=recommended,
            current=current,
            path=path,
        )
    ]
    return {
        "changed": True,
        "allowed_without_rebenchmark": not relevant_paths,
        "changed_paths": changed_paths,
        "runtime_relevant_changed_paths": relevant_paths,
        "error": None,
    }


def _runtime_relevant_path(path: str) -> bool:
    """Return whether a changed file can invalidate runtime benchmark evidence."""

    normalized = path.replace("\\", "/").lower()
    if normalized in RUNTIME_AUDIT_ONLY_PATHS:
        return False
    return any(normalized.startswith(prefix.lower()) for prefix in RUNTIME_RELEVANT_PATH_PREFIXES)


def _changed_only_auth_gate(
    *,
    recommended: str,
    current: str,
    path: str,
) -> bool:
    """Allow auth-only runner changes without invalidating runtime benchmarks."""

    normalized = path.replace("\\", "/").lower()
    if normalized not in AUTH_GATE_ONLY_RUNTIME_PATHS:
        return False
    try:
        diff = subprocess.run(
            ["git", "diff", "-U0", f"{recommended}..{current}", "--", path],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    hunks = _changed_diff_hunks(diff.stdout)
    if not hunks:
        return False
    return all(_auth_gate_hunk_allowed(hunk) for hunk in hunks)


def _changed_diff_hunks(diff_text: str) -> list[list[str]]:
    """Extract changed source lines grouped by zero-context diff hunk."""

    hunks: list[list[str]] = []
    current_hunk: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = []
            continue
        if line.startswith(("+++", "---")):
            continue
        if not line.startswith(("+", "-")):
            continue
        changed = line[1:].strip()
        if changed:
            current_hunk.append(changed)
    if current_hunk:
        hunks.append(current_hunk)
    return hunks


def _auth_gate_hunk_allowed(hunk: list[str]) -> bool:
    """Require every allowed hunk to be clearly tied to authorization logic."""

    has_auth_marker = any(_line_has_auth_gate_marker(line) for line in hunk)
    if not has_auth_marker:
        return False
    return all(
        _line_has_auth_gate_marker(line) or line in AUTH_GATE_ONLY_STRUCTURAL_LINES for line in hunk
    )


def _line_has_auth_gate_marker(line: str) -> bool:
    return any(marker in line for marker in AUTH_GATE_ONLY_DIFF_MARKERS)


def _canonical_full_run_path_errors(config: FullMultimodalOofConfig) -> list[str]:
    """Fail closed on ad hoc cache/output roots before a long full OOF run."""

    errors: list[str] = []
    output_parts = {part.lower() for part in config.output_dir.parts}
    if {"model_smoke", "smoke", "pilot", "resume_smoke"}.intersection(output_parts):
        errors.append(f"full_run_output_dir_must_not_be_smoke_or_pilot={config.output_dir}")
    expected_actor_root = Path("outputs/classification_v2/image_cache_v2_letterbox")
    expected_actor_tensor = expected_actor_root / f"packed_rgb_{config.image_size}_letterbox.npy"
    expected_actor_index = expected_actor_root / "packed_image_cache_index.csv"
    if config.packed_image_cache_npy is not None and _norm(config.packed_image_cache_npy) != _norm(
        expected_actor_tensor
    ):
        errors.append(
            "packed_actor_cache_must_use_canonical_letterbox_tensor="
            f"expected:{expected_actor_tensor},actual:{config.packed_image_cache_npy}"
        )
    if config.packed_image_cache_index_csv is not None and _norm(
        config.packed_image_cache_index_csv
    ) != _norm(expected_actor_index):
        errors.append(
            "packed_actor_cache_index_must_use_canonical_letterbox_index="
            f"expected:{expected_actor_index},"
            f"actual:{config.packed_image_cache_index_csv}"
        )
    expected_visual_root = Path("outputs/classification_v2/visual_interaction_cache")
    expected_visual_manifest = expected_visual_root / "visual_context_manifest.csv"
    expected_visual_tensor = expected_visual_root / f"packed_rgb_{config.image_size}_letterbox.npy"
    expected_visual_index = expected_visual_root / "packed_image_cache_index.csv"
    if config.visual_context_cache_manifest_csv is not None and _norm(
        config.visual_context_cache_manifest_csv
    ) != _norm(expected_visual_manifest):
        errors.append(
            "visual_context_manifest_must_use_canonical_cache="
            f"expected:{expected_visual_manifest},actual:{config.visual_context_cache_manifest_csv}"
        )
    if config.visual_context_packed_cache_npy is not None and _norm(
        config.visual_context_packed_cache_npy
    ) != _norm(expected_visual_tensor):
        errors.append(
            "packed_visual_context_must_use_canonical_letterbox_tensor="
            f"expected:{expected_visual_tensor},"
            f"actual:{config.visual_context_packed_cache_npy}"
        )
    if config.visual_context_packed_cache_index_csv is not None and _norm(
        config.visual_context_packed_cache_index_csv
    ) != _norm(expected_visual_index):
        errors.append(
            "packed_visual_context_index_must_use_canonical_letterbox_index="
            f"expected:{expected_visual_index},"
            f"actual:{config.visual_context_packed_cache_index_csv}"
        )
    return errors


def _norm(path: Path) -> str:
    return path.as_posix().lower().rstrip("/")


def validate_preflight_for_execution(
    config: FullMultimodalOofConfig,
    preflight: dict[str, Any],
    *,
    git_state: dict[str, Any] | None = None,
) -> list[str]:
    """Reject stale, dirty, invalid, or config-mismatched preflight payloads."""

    state = git_state or current_git_state()
    errors: list[str] = []
    if preflight.get("valid") is not True or preflight.get("errors"):
        errors.append(f"full_run_preflight_invalid={preflight.get('errors')}")
    expected = full_run_config_fingerprint(config)
    if preflight.get("config_sha256") != expected:
        errors.append(
            "full_run_config_fingerprint_mismatch="
            f"expected:{expected},"
            f"preflight:{preflight.get('config_sha256')}"
        )
    if preflight.get("git_dirty") is not False:
        errors.append(f"preflight_git_dirty={preflight.get('git_dirty')}")
    if state.get("dirty") is not False:
        errors.append(f"current_git_dirty={state.get('dirty')}")
    if not state.get("commit") or state.get("commit") != preflight.get("git_commit"):
        errors.append(
            "preflight_git_commit_mismatch="
            f"preflight:{preflight.get('git_commit')},"
            f"current:{state.get('commit')}"
        )
    return errors


def current_git_state() -> dict[str, Any]:
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
