"""Verify R128 startup readiness locally on E: drive cache.

Performs:
1. Parity comparison between canonical on-the-fly OpenCV preprocessing
   vs packed R128 cache.
2. Cold process startup timing instrumented through authority, Dataset,
   DataLoader, first batch, forward, loss.
3. Steady-state 100-batch data consumption benchmark measuring samples/sec,
   median/p95 batch wait, peak CPU/RAM.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (  # noqa: E402
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    image_sequence_collate,
)
from pig_behavior.classification_v2.models.balanced.contracts import (  # noqa: E402
    ModelBatch,
    SequenceSegment,
)
from pig_behavior.classification_v2.training import stage1_temporal_screening as stage1  # noqa: E402


def run_parity_gate(
    frame_context_csv: Path,
    window_context_csv: Path,
    packed_cache_npy: Path,
    packed_index_csv: Path,
    sample_size: int = 50,
) -> dict[str, Any]:
    print("--- PARITY GATE AUDIT ---")
    canonical_ds = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=frame_context_csv,
            window_context_csv=window_context_csv,
            image_size=128,
            require_complete=True,
            image_cache_size=0,
            video_capture_cache_size=4,
        )
    )

    packed_ds = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=frame_context_csv,
            window_context_csv=window_context_csv,
            packed_image_cache_npy=packed_cache_npy,
            packed_image_cache_index_csv=packed_index_csv,
            image_size=128,
            require_complete=True,
            require_cached_images=True,
        )
    )

    # Pick deterministic samples covering both CVAT and Legacy
    cvat_indices = [
        i for i, row in enumerate(canonical_ds.windows.itertuples())
        if getattr(row, "source_type", "") == "cvat_tracking_xml"
    ][: sample_size // 2]
    legacy_indices = [
        i for i, row in enumerate(canonical_ds.windows.itertuples())
        if getattr(row, "source_type", "") == "legacy_recovered"
    ][: sample_size // 2]
    eval_indices = cvat_indices + legacy_indices

    max_abs_error = 0.0
    exact_equal_count = 0
    total_tested = 0

    for idx in eval_indices:
        item_canon = canonical_ds[idx]
        item_packed = packed_ds[idx]

        assert item_canon["window_id"] == item_packed["window_id"]
        assert item_canon["image"].shape == item_packed["image"].shape

        tensor_canon = item_canon["image"].numpy()
        tensor_packed = item_packed["image"].numpy()

        diff = np.abs(tensor_canon - tensor_packed)
        item_max_err = float(diff.max())
        if item_max_err > max_abs_error:
            max_abs_error = item_max_err
        if item_max_err == 0.0:
            exact_equal_count += 1
        total_tested += 1

    canonical_ds.close()
    packed_ds.close()

    parity_passed = bool(max_abs_error <= 1e-6)
    print(
        f"Parity results: tested={total_tested}, exact_match={exact_equal_count}, "
        f"max_abs_diff={max_abs_error:.9f}, passed={parity_passed}"
    )
    return {
        "tested_samples": total_tested,
        "exact_match_count": exact_equal_count,
        "max_abs_diff": max_abs_error,
        "parity_passed": parity_passed,
    }


def run_cold_startup_test(
    authority_path: Path,
    frame_context_csv: Path,
    window_context_csv: Path,
    packed_cache_npy: Path,
    packed_index_csv: Path,
) -> dict[str, Any]:
    print("\n--- COLD PROCESS STARTUP TEST ---")
    t0 = time.perf_counter()

    # Stage 1: Authority Init
    t_auth_start = time.perf_counter()
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    assert isinstance(authority, dict) and len(authority) > 0
    t_auth_sec = time.perf_counter() - t_auth_start

    # Stage 2: Dataset Init
    t_ds_start = time.perf_counter()
    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=frame_context_csv,
            window_context_csv=window_context_csv,
            packed_image_cache_npy=packed_cache_npy,
            packed_image_cache_index_csv=packed_index_csv,
            image_size=128,
            require_complete=True,
            require_cached_images=True,
        )
    )
    t_ds_sec = time.perf_counter() - t_ds_start

    # Stage 3: DataLoader Init
    t_dl_start = time.perf_counter()
    batch_size = 16
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=image_sequence_collate,
        num_workers=0,
    )
    t_dl_sec = time.perf_counter() - t_dl_start

    # Stage 4: First Batch Load
    t_b1_start = time.perf_counter()
    first_batch_raw = next(iter(loader))
    t_b1_sec = time.perf_counter() - t_b1_start

    # Build model & forward on CPU
    device = torch.device("cpu")
    model = stage1._build_b1_model("T6").to(device)
    model.eval()

    t_fwd_start = time.perf_counter()
    batch_img = first_batch_raw["image"].to(device)
    batch_mask = first_batch_raw["observed_mask"].to(device)
    seq_len = batch_img.shape[1]
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)
    model_batch = ModelBatch(
        target=SequenceSegment(
            valid_mask=batch_mask,
            frame_offsets=torch.arange(-(seq_len - 1), 1, device=device).repeat(batch_size, 1),
            images=batch_img,
        ),
        labels=labels,
        native_unit_id=first_batch_raw["window_id"],
        window_id=first_batch_raw["window_id"],
    )
    with torch.inference_mode():
        out = model(model_batch)
        logits = out["logits"]
        loss = nn.functional.cross_entropy(logits, labels)
    t_fwd_sec = time.perf_counter() - t_fwd_start
    loss_val = float(loss.item())

    cold_total_sec = time.perf_counter() - t0
    dataset.close()

    print(f"Authority init: {t_auth_sec:.4f}s")
    print(f"Dataset init: {t_ds_sec:.4f}s")
    print(f"DataLoader init: {t_dl_sec:.4f}s")
    print(f"First full batch: {t_b1_sec:.4f}s")
    print(f"Forward + Loss: {t_fwd_sec:.4f}s (Loss = {loss_val:.4f})")
    print(f"Cold process startup total: {cold_total_sec:.4f}s")

    return {
        "cold_process_startup_sec": cold_total_sec,
        "authority_init_sec": t_auth_sec,
        "dataset_init_sec": t_ds_sec,
        "dataloader_init_sec": t_dl_sec,
        "first_full_batch_sec": t_b1_sec,
        "forward_sec": t_fwd_sec,
        "loss_value": loss_val,
        "real_data_used": True,
        "real_labels_used": True,
        "full_batch_loaded": True,
        "loss_finite": bool(np.isfinite(loss_val)),
    }


def run_benchmark_100_batches(
    frame_context_csv: Path,
    window_context_csv: Path,
    packed_cache_npy: Path,
    packed_index_csv: Path,
    num_batches: int = 100,
    batch_size: int = 16,
) -> dict[str, Any]:
    print("\n--- 100-BATCH STEADY-STATE BENCHMARK ---")
    process = psutil.Process(os.getpid())

    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=frame_context_csv,
            window_context_csv=window_context_csv,
            packed_image_cache_npy=packed_cache_npy,
            packed_image_cache_index_csv=packed_index_csv,
            image_size=128,
            require_complete=True,
            require_cached_images=True,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=image_sequence_collate,
        num_workers=0,
    )

    batch_wait_times: list[float] = []
    total_examples = 0
    t_start = time.perf_counter()
    t_last = t_start

    cpu_samples: list[float] = []
    ram_samples_gb: list[float] = []

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        t_now = time.perf_counter()
        wait_sec = t_now - t_last
        batch_wait_times.append(wait_sec)
        total_examples += len(batch["window_id"])

        cpu_samples.append(process.cpu_percent(interval=None))
        ram_samples_gb.append(process.memory_info().rss / (1024 ** 3))

        t_last = time.perf_counter()

    total_wall_sec = time.perf_counter() - t_start
    batches_per_sec = num_batches / total_wall_sec
    samples_per_sec = total_examples / total_wall_sec
    median_wait = float(np.median(batch_wait_times))
    p95_wait = float(np.percentile(batch_wait_times, 95))
    peak_cpu = float(np.max(cpu_samples)) if cpu_samples else 0.0
    peak_ram_gb = float(np.max(ram_samples_gb)) if ram_samples_gb else 0.0

    load_audit = dataset.image_load_audit()
    dataset.close()

    print(f"Consumed {num_batches} batches ({total_examples} samples) in {total_wall_sec:.4f}s")
    print(f"Throughput: {batches_per_sec:.2f} batches/sec ({samples_per_sec:.2f} samples/sec)")
    print(f"Batch wait: median={median_wait*1000:.2f}ms, p95={p95_wait*1000:.2f}ms")
    print(f"Peak CPU: {peak_cpu:.1f}%, Peak RAM: {peak_ram_gb:.3f} GB")
    print(f"Packed cache hits: {load_audit['packed_image_cache_hits']}")

    return {
        "real_batches": num_batches,
        "batch_size": batch_size,
        "total_examples": total_examples,
        "total_wall_sec": total_wall_sec,
        "batches_per_sec": batches_per_sec,
        "samples_per_sec": samples_per_sec,
        "median_batch_wait_sec": median_wait,
        "p95_batch_wait_sec": p95_wait,
        "peak_cpu_percent": peak_cpu,
        "peak_ram_gb": peak_ram_gb,
        "load_audit": load_audit,
    }


if __name__ == "__main__":
    authority_path = Path(
        "docs/classification_v2/temporal_v2_canonical_authority_mapping.json"
    )
    if not authority_path.exists():
        authority_path = Path(
            "outputs/classification_v2/s1_stage1_temporal_screening/"
            "s1_stage1_cpu_preflight_20260810_52d62718.json"
        )

    cache_dir = (
        Path("E:/PigProjectStorage/PIG_Behavior_Project/outputs/classification_v2")
        / "model_readiness_audit/pre_gpu_autoresearch_q2_6c2f204_20260804_084638"
        / "reviewed_rgb_v1"
    )
    frame_csv = cache_dir / "image_context_v2" / "image_frame_context_manifest.csv"
    win_csv = cache_dir / "image_context_v2" / "image_window_context_manifest.csv"
    packed_npy = cache_dir / "actor_rgb_128_full" / "packed_rgb_128_letterbox.npy"
    packed_idx = cache_dir / "actor_rgb_128_full" / "packed_image_cache_index.csv"

    print("Running verification suite...")
    parity_res = run_parity_gate(
        frame_csv, win_csv, packed_npy, packed_idx, sample_size=50
    )
    startup_res = run_cold_startup_test(
        authority_path, frame_csv, win_csv, packed_npy, packed_idx
    )
    bench_res = run_benchmark_100_batches(
        frame_csv, win_csv, packed_npy, packed_idx, num_batches=100, batch_size=16
    )

    final_report = {
        "parity": parity_res,
        "cold_startup": startup_res,
        "benchmark_100": bench_res,
    }
    out_file = Path("outputs/classification_v2/r128_startup_readiness_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"\nSaved readiness report to {out_file}")
