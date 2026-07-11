"""Audit matched CUDA runtime pilots without treating speed runs as model evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def summarize_runtime_benchmarks(
    audit_paths: list[Path],
    *,
    max_reserved_memory_mb: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare matched one-fold pilots and select the fastest memory-safe runtime config."""

    if not audit_paths:
        raise ValueError("at least one benchmark audit is required")
    if max_reserved_memory_mb <= 0.0:
        raise ValueError("max_reserved_memory_mb must be positive")

    rows = [_read_benchmark(path) for path in audit_paths]
    table = pd.DataFrame(rows).sort_values(["precision", "train_batch_size"], kind="mergesort")
    errors = _matched_workload_errors(table)
    valid_candidate = (
        table["audit_valid"]
        & table["cache_only"]
        & table["git_clean"]
        & table["throughput_rows_per_sec"].gt(0.0)
        & table["peak_reserved_memory_mb"].le(float(max_reserved_memory_mb))
    )
    table["valid_runtime_candidate"] = valid_candidate
    recommended: dict[str, Any] | None = None
    if not errors and valid_candidate.any():
        best = table.loc[valid_candidate].sort_values(
            ["throughput_rows_per_sec", "peak_reserved_memory_mb"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        recommended = {
            "model_architecture_version": str(best["model_architecture_version"]),
            "git_commit": str(best["git_commit"]),
            "precision": str(best["precision"]),
            "train_batch_size": int(best["train_batch_size"]),
            "throughput_rows_per_sec": float(best["throughput_rows_per_sec"]),
            "peak_reserved_memory_mb": float(best["peak_reserved_memory_mb"]),
            "audit_json": str(best["audit_json"]),
        }
    if recommended is None:
        errors.append("no_matched_memory_safe_runtime_candidate")

    summary = {
        "schema_version": "classification_v2_runtime_benchmark_v1",
        "benchmark_count": int(len(table)),
        "max_reserved_memory_mb": float(max_reserved_memory_mb),
        "recommended_runtime_config": recommended,
        "errors": errors,
        "warnings": [
            "Runtime pilots compare throughput and memory only; losses and predictions are not paper metrics.",
            "Re-benchmark when image size, model architecture, sequence contract, or GPU changes.",
        ],
        "valid": not errors,
    }
    return table, summary


def write_runtime_benchmark(
    audit_paths: list[Path],
    output_dir: Path,
    *,
    max_reserved_memory_mb: float,
) -> dict[str, Any]:
    """Write a machine-readable benchmark table and decision audit."""

    table, summary = summarize_runtime_benchmarks(
        audit_paths,
        max_reserved_memory_mb=max_reserved_memory_mb,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "runtime_benchmark.csv"
    json_path = output_dir / "runtime_benchmark_audit.json"
    table.to_csv(csv_path, index=False)
    summary["runtime_benchmark_csv"] = str(csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["runtime_benchmark_audit_json"] = str(json_path)
    return summary


def _read_benchmark(path: Path) -> dict[str, Any]:
    """Extract comparable fields from one learned-OOF pilot audit."""

    audit = json.loads(path.read_text(encoding="utf-8"))
    folds = audit.get("fold_audits", [])
    if len(folds) != 1:
        raise ValueError(f"runtime benchmark requires exactly one fold: {path}")
    fold = folds[0]
    config = audit.get("config", {})
    image_audit = audit.get("image_load_audit", {})
    throughput = float(fold.get("training_rows_per_sec", 0.0))
    peak_reserved = float(fold.get("cuda_peak_memory_reserved_mb", 0.0))
    if not math.isfinite(throughput) or not math.isfinite(peak_reserved):
        raise ValueError(f"non-finite benchmark metric: {path}")
    return {
        "audit_json": str(path),
        "precision": str(config.get("precision", "")),
        "train_batch_size": int(config.get("train_batch_size", 0)),
        "training_steps": int(fold.get("training_steps_completed", 0)),
        "training_elapsed_sec": float(fold.get("training_elapsed_sec", 0.0)),
        "throughput_steps_per_sec": float(fold.get("optimizer_steps_per_sec", 0.0)),
        "throughput_rows_per_sec": throughput,
        "peak_allocated_memory_mb": float(fold.get("cuda_peak_memory_allocated_mb", 0.0)),
        "peak_reserved_memory_mb": peak_reserved,
        "train_indices_sha256": str(fold.get("train_indices_sha256", "")),
        "eval_indices_sha256": str(fold.get("eval_indices_sha256", "")),
        "device": str(audit.get("device", "")),
        "git_commit": str(audit.get("git_commit", "")),
        "git_clean": bool(audit.get("git_dirty") is False and audit.get("git_commit")),
        "image_size": int(config.get("image_size", 0)),
        "hidden_dim": int(config.get("hidden_dim", 0)),
        "model_architecture_version": str(config.get("model_architecture_version", "")),
        "ablation_variant": str(config.get("ablation_variant", "")),
        "sample_weight_policy": str(config.get("sample_weight_policy", "")),
        "seed": int(config.get("seed", 0)),
        "cache_only": bool(
            image_audit.get("require_cached_images") is True
            and int(image_audit.get("disk_image_cache_misses", -1)) == 0
            and int(image_audit.get("source_image_loads", -1)) == 0
        ),
        "audit_valid": bool(audit.get("valid") is True and not audit.get("errors")),
    }


def _matched_workload_errors(table: pd.DataFrame) -> list[str]:
    """Reject benchmark tables whose rows differ beyond precision or batch size."""

    errors: list[str] = []
    matched_fields = (
        "training_steps",
        "train_indices_sha256",
        "eval_indices_sha256",
        "device",
        "git_commit",
        "image_size",
        "hidden_dim",
        "model_architecture_version",
        "ablation_variant",
        "sample_weight_policy",
        "seed",
    )
    for field in matched_fields:
        if table[field].nunique(dropna=False) != 1:
            errors.append(f"benchmark_workload_mismatch={field}")
    if not table["cache_only"].all():
        errors.append("benchmark_contains_non_cache_only_run")
    if not table["audit_valid"].all():
        errors.append("benchmark_contains_invalid_run")
    if not table["git_clean"].all():
        errors.append("benchmark_contains_dirty_or_uncommitted_run")
    return errors
