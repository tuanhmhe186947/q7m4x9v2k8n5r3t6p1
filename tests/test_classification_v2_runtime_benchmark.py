import json
from pathlib import Path

from pig_behavior.classification_v2.training.runtime_benchmark import summarize_runtime_benchmarks


def test_runtime_benchmark_selects_fastest_matched_memory_safe_candidate(tmp_path: Path) -> None:
    """Runtime selection must honor both matched lineage and the explicit VRAM budget."""

    slow = _write_audit(tmp_path / "slow.json", batch=32, throughput=100.0, reserved_mb=200.0)
    fast_too_large = _write_audit(tmp_path / "large.json", batch=128, throughput=400.0, reserved_mb=900.0)
    balanced = _write_audit(tmp_path / "balanced.json", batch=64, throughput=250.0, reserved_mb=400.0)

    _, summary = summarize_runtime_benchmarks(
        [slow, fast_too_large, balanced],
        max_reserved_memory_mb=500.0,
    )

    assert summary["valid"] is True
    assert summary["recommended_runtime_config"]["train_batch_size"] == 64


def test_runtime_benchmark_rejects_mismatched_fold_lineage(tmp_path: Path) -> None:
    """Different fold rows cannot be compared as if only runtime settings changed."""

    first = _write_audit(tmp_path / "first.json", batch=32, throughput=100.0, reserved_mb=200.0)
    second = _write_audit(
        tmp_path / "second.json",
        batch=64,
        throughput=200.0,
        reserved_mb=300.0,
        train_hash="different",
    )

    _, summary = summarize_runtime_benchmarks([first, second], max_reserved_memory_mb=500.0)

    assert summary["valid"] is False
    assert "benchmark_workload_mismatch=train_indices_sha256" in summary["errors"]


def _write_audit(
    path: Path,
    *,
    batch: int,
    throughput: float,
    reserved_mb: float,
    train_hash: str = "train-hash",
) -> Path:
    audit = {
        "valid": True,
        "errors": [],
        "device": "cuda",
        "config": {
            "precision": "amp",
            "train_batch_size": batch,
            "image_size": 64,
            "hidden_dim": 48,
            "ablation_variant": "full",
            "sample_weight_policy": "event_class",
            "seed": 7,
        },
        "image_load_audit": {
            "require_cached_images": True,
            "disk_image_cache_misses": 0,
            "source_image_loads": 0,
        },
        "fold_audits": [
            {
                "training_steps_completed": 100,
                "training_elapsed_sec": 1.0,
                "optimizer_steps_per_sec": 100.0,
                "training_rows_per_sec": throughput,
                "cuda_peak_memory_allocated_mb": reserved_mb / 2,
                "cuda_peak_memory_reserved_mb": reserved_mb,
                "train_indices_sha256": train_hash,
                "eval_indices_sha256": "eval-hash",
            }
        ],
    }
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path
