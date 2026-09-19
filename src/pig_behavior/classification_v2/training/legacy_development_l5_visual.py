"""Pretrained-weight and 4 GiB VRAM gates for legacy L5 visuals."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights

from pig_behavior.classification_v2.models.visual_backbones import (
    build_visual_frame_encoder,
    visual_backbone_contract,
)
from pig_behavior.classification_v2.training.legacy_development_l5 import (
    LINEAGE_SCOPE,
    LegacyL5Config,
    git_state,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
)

L5_WEIGHT_AUDIT_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.pretrained_weights.v1"
)
L5_VRAM_PROBE_SCHEMA_VERSION = (
    "classification_v2.legacy_development_l5.vram_probe.v1"
)
GIB = 1024**3
GPU_CONTROL_IDS = ("V0", "V1", "V2")
EXPECTED_GPU_GIB = 4
VRAM_CAP_FRACTION = 0.7
MAX_WINDOWS_WEIGHT_FILE_PATH_CHARS = 210

_WEIGHT_ENUMS: dict[str, Any] = {
    "ResNet18_Weights.IMAGENET1K_V1": ResNet18_Weights.IMAGENET1K_V1,
    "ResNet34_Weights.IMAGENET1K_V1": ResNet34_Weights.IMAGENET1K_V1,
}


@dataclass(frozen=True, slots=True)
class LegacyVisualProbeControl:
    """One exact visual control and its bounded frame batch."""

    control_id: str
    backbone_name: str
    pretrained_weight_enum: str
    image_size: int
    frame_batch_size: int


def legacy_l5_visual_probe_controls(
    config: LegacyL5Config,
) -> tuple[LegacyVisualProbeControl, ...]:
    """Resolve the frozen V0/V1/V2 matrix without constructing a model."""

    feature_cache = _object(config.payload["feature_cache"], "feature_cache")
    batch_sizes = {
        "resnet18": int(feature_cache["resnet18_frame_batch_size"]),
        "resnet34": int(feature_cache["resnet34_frame_batch_size"]),
    }
    rows = {
        str(row["control_id"]): row
        for row in config.payload["visual_controls"]
        if str(row["control_id"]) in GPU_CONTROL_IDS
    }
    if tuple(rows) != GPU_CONTROL_IDS:
        raise ValueError("legacy L5 GPU visual control order drift")
    return tuple(
        LegacyVisualProbeControl(
            control_id=control_id,
            backbone_name=str(rows[control_id]["backbone_name"]),
            pretrained_weight_enum=str(
                rows[control_id]["pretrained_weight_enum"]
            ),
            image_size=int(rows[control_id]["image_size"]),
            frame_batch_size=batch_sizes[
                str(rows[control_id]["backbone_name"])
            ],
        )
        for control_id in GPU_CONTROL_IDS
    )


def prepare_legacy_l5_pretrained_weights(
    config: LegacyL5Config,
    *,
    readiness_audit_path: Path,
    weight_cache_root: Path,
    allow_download: bool,
) -> dict[str, Any]:
    """Prepare exact torchvision weights on CPU and bind their hashes."""

    readiness = _read_json(readiness_audit_path)
    _validate_readiness_parent(config, readiness)
    cache_root = weight_cache_root.resolve()
    development_root = config.development_root.resolve()
    if not cache_root.is_relative_to(development_root):
        raise ValueError("legacy L5 weight cache must stay under its lane root")
    hub_dir = cache_root / "hub"
    checkpoint_dir = hub_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))
    cuda_initialized_before = bool(torch.cuda.is_initialized())
    reports: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    unique_controls = _unique_weight_controls(
        legacy_l5_visual_probe_controls(config)
    )
    for control in unique_controls:
        enum = _weight_enum(control.pretrained_weight_enum)
        filename, expected_prefix = _weight_filename_and_prefix(enum.url)
        path = checkpoint_dir / filename
        _validate_windows_weight_path(path)
        existed_before = path.is_file()
        if not existed_before and not allow_download:
            errors.append(
                "pretrained_weight_not_cached="
                f"{control.pretrained_weight_enum}:{path}"
            )
            reports[control.pretrained_weight_enum] = {
                "backbone_name": control.backbone_name,
                "pretrained_weight_enum": control.pretrained_weight_enum,
                "url": enum.url,
                "cache_path": str(path),
                "existed_before": False,
                "downloaded_now": False,
                "valid": False,
            }
            continue
        encoder, contract = build_visual_frame_encoder(
            control.backbone_name,
            control.pretrained_weight_enum,
        )
        parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
        del encoder
        gc.collect()
        report = _cached_weight_report(
            path,
            expected_sha256_prefix=expected_prefix,
        )
        report.update(
            {
                "backbone_name": control.backbone_name,
                "pretrained_weight_enum": control.pretrained_weight_enum,
                "url": enum.url,
                "existed_before": existed_before,
                "downloaded_now": not existed_before,
                "output_dim": contract.output_dim,
                "parameter_count": int(parameter_count),
            }
        )
        reports[control.pretrained_weight_enum] = report
        errors.extend(
            f"{control.pretrained_weight_enum}:{error}"
            for error in report["errors"]
        )
    cuda_initialized_after = bool(torch.cuda.is_initialized())
    if cuda_initialized_before or cuda_initialized_after:
        errors.append("pretrained_weight_prepare_initialized_cuda")
    valid = not errors
    return {
        "schema_version": L5_WEIGHT_AUDIT_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_PRETRAINED_WEIGHTS"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_PRETRAINED_WEIGHTS"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "readiness_audit_path": str(readiness_audit_path),
        "readiness_audit_sha256": file_sha256(readiness_audit_path),
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "git_state": git_state(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "weight_cache_root": str(cache_root),
        "torch_hub_dir": str(hub_dir),
        "windows_weight_file_path_limit_chars": (
            MAX_WINDOWS_WEIGHT_FILE_PATH_CHARS
        ),
        "network_download_allowed": bool(allow_download),
        "pretrained_weight_downloads": sum(
            int(report.get("downloaded_now", False))
            for report in reports.values()
        ),
        "cuda_initialized_before": cuda_initialized_before,
        "cuda_initialized_after": cuda_initialized_after,
        "artifacts": reports,
        "vram_probe_authorized": valid,
        "feature_cache_expansion_authorized": False,
        "accuracy_f1_computed": False,
        "optimizer_steps": 0,
        "errors": errors,
        "valid": valid,
    }


def _unique_weight_controls(
    controls: tuple[LegacyVisualProbeControl, ...],
) -> tuple[LegacyVisualProbeControl, ...]:
    unique: dict[str, LegacyVisualProbeControl] = {}
    for control in controls:
        unique.setdefault(control.pretrained_weight_enum, control)
    return tuple(unique.values())


def _weight_enum(name: str) -> Any:
    try:
        return _WEIGHT_ENUMS[name]
    except KeyError as error:
        raise ValueError(f"unsupported L5 pretrained weight enum: {name}") from error


def _weight_filename_and_prefix(url: str) -> tuple[str, str]:
    filename = Path(urlparse(url).path).name
    match = re.search(r"-([0-9a-f]{8,64})\.pth$", filename)
    if not filename or match is None:
        raise ValueError(f"unhashable torchvision weight URL: {url}")
    return filename, match.group(1)


def _validate_windows_weight_path(path: Path) -> None:
    if os.name == "nt" and len(str(path)) > MAX_WINDOWS_WEIGHT_FILE_PATH_CHARS:
        raise ValueError(
            "legacy L5 pretrained-weight path is unsafe for Windows partial "
            f"files: {len(str(path))}>{MAX_WINDOWS_WEIGHT_FILE_PATH_CHARS}"
        )


def _cached_weight_report(
    path: Path,
    *,
    expected_sha256_prefix: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if not path.is_file():
        errors.append("cached_weight_file_missing")
        return {
            "cache_path": str(path),
            "size_bytes": 0,
            "sha256": None,
            "expected_sha256_prefix": expected_sha256_prefix,
            "errors": errors,
            "valid": False,
        }
    digest = file_sha256(path)
    if not digest.startswith(expected_sha256_prefix):
        errors.append("cached_weight_sha256_prefix_mismatch")
    return {
        "cache_path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest,
        "expected_sha256_prefix": expected_sha256_prefix,
        "errors": errors,
        "valid": not errors,
    }


def run_legacy_l5_vram_probe(
    config: LegacyL5Config,
    *,
    full_cache_audit_path: Path,
    weights_audit_path: Path,
    device_name: str,
) -> dict[str, Any]:
    """Run bounded pretrained forwards under a hard 4 GiB allocator cap."""

    full_cache = _read_json(full_cache_audit_path)
    weights = _read_json(weights_audit_path)
    _validate_full_cache_parent(config, full_cache)
    _validate_weights_parent(config, weights)
    readiness_path = Path(str(full_cache["readiness_audit_path"]))
    readiness = _read_json(readiness_path)
    _validate_readiness_parent(config, readiness)
    if file_sha256(readiness_path) != full_cache["readiness_audit_sha256"]:
        raise ValueError("legacy L5 full-cache readiness hash drift")
    live_weight_errors = _live_weight_errors(weights)
    if live_weight_errors:
        raise ValueError(f"legacy L5 pretrained weights drift: {live_weight_errors}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("legacy L5 VRAM probe requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("legacy L5 VRAM probe requested unavailable CUDA")
    optimization = _object(config.payload["optimization"], "optimization")
    declared_gib = int(optimization["declared_local_gpu_vram_gib"])
    maximum_fraction = float(optimization["maximum_peak_vram_fraction"])
    if declared_gib != EXPECTED_GPU_GIB or maximum_fraction > VRAM_CAP_FRACTION:
        raise ValueError("legacy L5 four-GiB VRAM safety contract drift")
    device_index = (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    actual_total = int(properties.total_memory)
    free_before, mem_info_total = (
        int(value) for value in torch.cuda.mem_get_info(device)
    )
    declared_bytes = declared_gib * GIB
    budget_bytes = _vram_budget_bytes(
        declared_bytes=declared_bytes,
        actual_total_bytes=actual_total,
        maximum_fraction=maximum_fraction,
    )
    allocator_fraction = budget_bytes / actual_total
    preflight_errors = _device_preflight_errors(
        declared_gib=declared_gib,
        actual_total_bytes=actual_total,
        mem_info_total_bytes=mem_info_total,
        free_bytes=free_before,
        budget_bytes=budget_bytes,
    )
    controls: dict[str, dict[str, Any]] = {}
    if not preflight_errors:
        torch.cuda.set_per_process_memory_fraction(allocator_fraction, device)
        _seed_cuda(int(config.payload["optimization"]["seeds"][0]))
        torch.hub.set_dir(str(weights["torch_hub_dir"]))
        for control in legacy_l5_visual_probe_controls(config):
            report = _probe_control(
                config,
                control=control,
                device=device,
                budget_bytes=budget_bytes,
                full_cache=full_cache,
                readiness=readiness,
            )
            controls[control.control_id] = report
            if report["oom"]:
                break
    missing_controls = sorted(set(GPU_CONTROL_IDS).difference(controls))
    errors = list(preflight_errors)
    errors.extend(
        f"{control_id}:{error}"
        for control_id, report in controls.items()
        for error in report["errors"]
    )
    if missing_controls:
        errors.append(f"unexecuted_gpu_controls={missing_controls}")
    free_after, _ = (int(value) for value in torch.cuda.mem_get_info(device))
    valid = not errors
    return {
        "schema_version": L5_VRAM_PROBE_SCHEMA_VERSION,
        "status": (
            "PASS_LEGACY_DEVELOPMENT_L5_VRAM_PROBE"
            if valid
            else "FAIL_LEGACY_DEVELOPMENT_L5_VRAM_PROBE"
        ),
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config.path),
        "config_sha256": config.sha256,
        "full_cache_audit_path": str(full_cache_audit_path),
        "full_cache_audit_sha256": file_sha256(full_cache_audit_path),
        "weights_audit_path": str(weights_audit_path),
        "weights_audit_sha256": file_sha256(weights_audit_path),
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "git_state": git_state(),
        "device": str(device),
        "device_name": str(properties.name),
        "declared_gpu_vram_gib": declared_gib,
        "actual_total_vram_bytes": actual_total,
        "mem_info_total_vram_bytes": mem_info_total,
        "free_vram_before_bytes": free_before,
        "free_vram_after_bytes": free_after,
        "maximum_peak_vram_fraction": maximum_fraction,
        "configured_allocator_fraction": allocator_fraction,
        "allocator_limit_policy": "torch_per_process_fraction_v1",
        "allocator_limit_bytes": budget_bytes,
        "oom_retry_allowed": False,
        "oom_retry_count": 0,
        "precision": "float32",
        "autocast_enabled": False,
        "gradient_enabled": False,
        "controls": controls,
        "feature_cache_expansion_authorized": valid,
        "accuracy_f1_computed": False,
        "optimizer_steps": 0,
        "errors": errors,
        "valid": valid,
    }


def _probe_control(
    config: LegacyL5Config,
    *,
    control: LegacyVisualProbeControl,
    device: torch.device,
    budget_bytes: int,
    full_cache: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    sample_rows = int(config.payload["feature_cache"]["repeat_sample_rows"])
    cache = _cache_lineage(
        config,
        image_size=control.image_size,
        full_cache=full_cache,
        readiness=readiness,
    )
    sample = _load_probe_sample(
        tensor_path=Path(cache["packed_tensor_path"]),
        index_path=Path(cache["packed_index_path"]),
        expected_rows=int(cache["expected_rows"]),
        image_size=control.image_size,
        sample_rows=sample_rows,
    )
    errors: list[str] = []
    oom = False
    oom_message: str | None = None
    encoder: nn.Module | None = None
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        encoder, contract = build_visual_frame_encoder(
            control.backbone_name,
            control.pretrained_weight_enum,
        )
        encoder.eval()
        encoder.to(device)
        first = _feature_digest_pass(
            encoder,
            images=sample["images"],
            control=control,
            device=device,
        )
        second = _feature_digest_pass(
            encoder,
            images=sample["images"],
            control=control,
            device=device,
        )
        if first["feature_sha256"] != second["feature_sha256"]:
            errors.append("pretrained_feature_repeat_hash_mismatch")
        if first["nonfinite_values"] or second["nonfinite_values"]:
            errors.append("pretrained_feature_nonfinite_values")
        if first["feature_dim"] != contract.output_dim:
            errors.append("pretrained_feature_output_dim_mismatch")
    except torch.cuda.OutOfMemoryError as error:
        oom = True
        oom_message = str(error)
        first = None
        second = None
        errors.append("cuda_out_of_memory_no_retry")
    finally:
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if encoder is not None:
            encoder.to("cpu")
        del encoder
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    post_allocated = int(torch.cuda.memory_allocated(device))
    post_reserved = int(torch.cuda.memory_reserved(device))
    if peak_allocated > budget_bytes:
        errors.append("peak_allocated_exceeds_allocator_limit")
    if peak_reserved > budget_bytes:
        errors.append("peak_reserved_exceeds_allocator_limit")
    if post_allocated != 0:
        errors.append("cuda_allocation_not_released_after_control")
    if post_reserved != 0:
        errors.append("cuda_reservation_not_released_after_control")
    return {
        "control_id": control.control_id,
        "backbone_name": control.backbone_name,
        "pretrained_weight_enum": control.pretrained_weight_enum,
        "image_size": control.image_size,
        "frame_batch_size": control.frame_batch_size,
        "sample_rows": sample_rows,
        "sample_packed_rows": sample["packed_rows"],
        "sample_context_id_sha256": sample["context_id_sha256"],
        "sample_rgb_sha256": sample["rgb_sha256"],
        "packed_tensor_path": cache["packed_tensor_path"],
        "packed_tensor_parent_sha256": cache["packed_tensor_sha256"],
        "packed_index_path": cache["packed_index_path"],
        "packed_index_parent_sha256": cache["packed_index_sha256"],
        "repeat_pass_1": first,
        "repeat_pass_2": second,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "allocator_limit_bytes": budget_bytes,
        "post_cleanup_allocated_bytes": post_allocated,
        "post_cleanup_reserved_bytes": post_reserved,
        "runtime_sec": float(time.perf_counter() - started),
        "oom": oom,
        "oom_message": oom_message,
        "oom_retry_count": 0,
        "errors": errors,
        "valid": not errors,
    }


def _feature_digest_pass(
    encoder: nn.Module,
    *,
    images: np.ndarray,
    control: LegacyVisualProbeControl,
    device: torch.device,
) -> dict[str, Any]:
    contract = visual_backbone_contract(
        control.backbone_name,
        control.pretrained_weight_enum,
    )
    digest = hashlib.sha256()
    nonfinite = 0
    feature_rows = 0
    feature_dim: int | None = None
    mean = torch.tensor(contract.input_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(contract.input_std, dtype=torch.float32).view(1, 3, 1, 1)
    with torch.inference_mode():
        for start in range(0, len(images), control.frame_batch_size):
            raw = torch.from_numpy(images[start : start + control.frame_batch_size])
            batch = raw.permute(0, 3, 1, 2).contiguous().to(torch.float32)
            batch = (batch / 255.0 - mean) / std
            features = encoder(batch.to(device, non_blocking=False))
            if features.ndim != 2:
                raise RuntimeError(
                    f"legacy L5 feature rank drift: {tuple(features.shape)}"
                )
            feature_dim = int(features.shape[1])
            cpu = features.detach().to("cpu").contiguous().numpy()
            nonfinite += int((~np.isfinite(cpu)).sum())
            feature_rows += int(cpu.shape[0])
            digest.update(cpu.astype(np.float32, copy=False).tobytes())
            del raw, batch, features, cpu
    torch.cuda.synchronize(device)
    return {
        "feature_rows": feature_rows,
        "feature_dim": feature_dim,
        "feature_dtype": "float32",
        "feature_sha256": digest.hexdigest(),
        "nonfinite_values": nonfinite,
    }


def _cache_lineage(
    config: LegacyL5Config,
    *,
    image_size: int,
    full_cache: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    expected_rows = int(config.payload["expected_counts"]["image_context_rows"])
    if image_size == 224:
        root = Path(str(full_cache["cache_root"]))
        hashes = _object(full_cache["cache_artifact_hashes"], "cache hashes")
        return {
            "expected_rows": expected_rows,
            "packed_tensor_path": str(root / "packed_rgb_224_letterbox.npy"),
            "packed_tensor_sha256": str(hashes["packed_tensor"]),
            "packed_index_path": str(root / "packed_image_cache_index.csv"),
            "packed_index_sha256": str(hashes["packed_index"]),
        }
    if image_size != 160:
        raise ValueError(f"legacy L5 unsupported probe image size: {image_size}")
    artifacts = _object(readiness["input_hash_audit"]["artifacts"], "input hashes")
    tensor = _object(artifacts["cache_160_packed_tensor"], "160 tensor")
    index = _object(artifacts["cache_160_packed_index"], "160 index")
    return {
        "expected_rows": expected_rows,
        "packed_tensor_path": str(tensor["path"]),
        "packed_tensor_sha256": str(tensor["sha256"]),
        "packed_index_path": str(index["path"]),
        "packed_index_sha256": str(index["sha256"]),
    }


def _load_probe_sample(
    *,
    tensor_path: Path,
    index_path: Path,
    expected_rows: int,
    image_size: int,
    sample_rows: int,
) -> dict[str, Any]:
    if sample_rows <= 0 or sample_rows > expected_rows:
        raise ValueError("legacy L5 probe sample row count is invalid")
    packed_rows = _spread_rows(expected_rows, sample_rows)
    tensor = np.load(tensor_path, mmap_mode="r")
    try:
        if tuple(tensor.shape) != (
            expected_rows,
            image_size,
            image_size,
            3,
        ):
            raise ValueError(
                f"legacy L5 packed probe shape drift: {tuple(tensor.shape)}"
            )
        if tensor.dtype != np.uint8:
            raise ValueError(f"legacy L5 packed probe dtype drift: {tensor.dtype}")
        images = np.array(tensor[packed_rows], dtype=np.uint8, copy=True)
    finally:
        _close_memmap(tensor)
    context_ids = _context_ids_for_rows(index_path, packed_rows)
    return {
        "images": images,
        "packed_rows": packed_rows.tolist(),
        "context_id_sha256": _string_sequence_sha256(context_ids),
        "rgb_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
    }


def _spread_rows(total_rows: int, sample_rows: int) -> np.ndarray:
    if total_rows <= 0 or sample_rows <= 0 or sample_rows > total_rows:
        raise ValueError("invalid legacy L5 spread-row request")
    rows = np.linspace(0, total_rows - 1, sample_rows, dtype=np.int64)
    if len(np.unique(rows)) != sample_rows:
        raise ValueError("legacy L5 spread-row selection contains duplicates")
    return rows


def _context_ids_for_rows(
    index_path: Path,
    packed_rows: np.ndarray,
) -> list[str]:
    positions = {int(row): index for index, row in enumerate(packed_rows)}
    context_ids = [""] * len(packed_rows)
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_context_id", "packed_row"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("legacy L5 packed index columns drift")
        for row in reader:
            packed_row = int(row["packed_row"])
            position = positions.get(packed_row)
            if position is not None:
                context_ids[position] = str(row["image_context_id"])
    if any(not context_id for context_id in context_ids):
        raise ValueError("legacy L5 packed index is missing probe rows")
    return context_ids


def _string_sequence_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _vram_budget_bytes(
    *,
    declared_bytes: int,
    actual_total_bytes: int,
    maximum_fraction: float,
) -> int:
    if declared_bytes <= 0 or actual_total_bytes <= 0:
        raise ValueError("legacy L5 VRAM totals must be positive")
    if not 0.0 < maximum_fraction <= VRAM_CAP_FRACTION:
        raise ValueError("legacy L5 VRAM fraction exceeds its safety cap")
    return int(min(declared_bytes, actual_total_bytes) * maximum_fraction)


def _device_preflight_errors(
    *,
    declared_gib: int,
    actual_total_bytes: int,
    mem_info_total_bytes: int,
    free_bytes: int,
    budget_bytes: int,
) -> list[str]:
    errors: list[str] = []
    actual_gib = actual_total_bytes / GIB
    if declared_gib != EXPECTED_GPU_GIB:
        errors.append(f"declared_gpu_vram_gib={declared_gib}")
    if not 3.5 <= actual_gib <= 4.25:
        errors.append(f"actual_gpu_vram_gib={actual_gib:.6f}")
    total_delta = abs(actual_total_bytes - mem_info_total_bytes)
    if total_delta > 64 * 1024**2:
        errors.append(f"cuda_total_memory_delta_bytes={total_delta}")
    if free_bytes < budget_bytes:
        errors.append(
            f"free_vram_below_allocator_budget={free_bytes}<{budget_bytes}"
        )
    return errors


def _seed_cuda(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _live_weight_errors(weights: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    artifacts = _object(weights["artifacts"], "weight artifacts")
    for name, value in artifacts.items():
        report = _object(value, f"weight artifact {name}")
        path = Path(str(report["cache_path"]))
        if not path.is_file():
            errors.append(f"missing={name}:{path}")
            continue
        observed = file_sha256(path)
        if observed != report.get("sha256"):
            errors.append(f"sha256={name}:{observed}!={report.get('sha256')}")
    return errors


def _validate_readiness_parent(
    config: LegacyL5Config,
    readiness: dict[str, Any],
) -> None:
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L5_READINESS",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "pretrained_weight_prepare_authorized": True,
        "config_sha256": config.sha256,
        "valid": True,
    }
    _require_parent(readiness, expected, "readiness")


def _validate_full_cache_parent(
    config: LegacyL5Config,
    full_cache: dict[str, Any],
) -> None:
    expected = {
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHE_FULL",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "mode": "full",
        "config_sha256": config.sha256,
        "pretrained_feature_cache_authorized": True,
        "valid": True,
    }
    _require_parent(full_cache, expected, "full cache")
    if Path(str(full_cache["cache_root"])).resolve() != (
        config.full_cache_224_root.resolve()
    ):
        raise ValueError("legacy L5 full-cache root drift")


def _validate_weights_parent(
    config: LegacyL5Config,
    weights: dict[str, Any],
) -> None:
    expected = {
        "schema_version": L5_WEIGHT_AUDIT_SCHEMA_VERSION,
        "status": "PASS_LEGACY_DEVELOPMENT_L5_PRETRAINED_WEIGHTS",
        "lineage_scope": LINEAGE_SCOPE,
        "human_review_complete": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_sha256": config.sha256,
        "implementation_source_sha256": file_sha256(Path(__file__)),
        "vram_probe_authorized": True,
        "valid": True,
    }
    _require_parent(weights, expected, "pretrained weights")
    cache_root = Path(str(weights["weight_cache_root"])).resolve()
    hub_dir = Path(str(weights["torch_hub_dir"])).resolve()
    if not cache_root.is_relative_to(config.development_root.resolve()):
        raise ValueError("legacy L5 pretrained-weight root escaped its lane")
    if hub_dir != cache_root / "hub":
        raise ValueError("legacy L5 torch hub path drift")
    checkpoint_root = hub_dir / "checkpoints"
    artifacts = _object(weights["artifacts"], "weight artifacts")
    for name, value in artifacts.items():
        report = _object(value, f"weight artifact {name}")
        path = Path(str(report["cache_path"])).resolve()
        if not path.is_relative_to(checkpoint_root):
            raise ValueError(
                f"legacy L5 weight artifact escaped checkpoint root: {name}"
            )
        _validate_windows_weight_path(path)


def _require_parent(
    payload: dict[str, Any],
    expected: dict[str, Any],
    name: str,
) -> None:
    errors = [
        f"{field}:{payload.get(field)!r}!={value!r}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]
    if payload.get("errors"):
        errors.append(f"declared_errors={payload['errors']}")
    if errors:
        raise ValueError(f"legacy L5 {name} parent mismatch: {errors}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"legacy L5 JSON must be an object: {path}")
    return payload


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"legacy L5 {name} must be an object")
    return value


def _close_memmap(array: np.ndarray) -> None:
    mmap_handle = getattr(array, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


__all__ = [
    "L5_VRAM_PROBE_SCHEMA_VERSION",
    "L5_WEIGHT_AUDIT_SCHEMA_VERSION",
    "LegacyVisualProbeControl",
    "legacy_l5_visual_probe_controls",
    "prepare_legacy_l5_pretrained_weights",
    "run_legacy_l5_vram_probe",
]
