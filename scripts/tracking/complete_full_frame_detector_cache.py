"""Complete the frozen R0 detector cache using odd-frame inference only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.tracking import (  # noqa: E402
    generate_current_main_baseline_caches as baseline,
)

from pig_behavior.tracking.detector_cache import (  # noqa: E402
    DETECTOR_CACHE_SCHEMA_VERSION,
    DetectorCacheIdentity,
    DetectorEvidenceCache,
    ReplayDetector,
)

STARTING_MAIN_SHA = "f9c698075358ceca4215d8a98b8fe5ed8887c67c"
R0_CACHE_INVENTORY_SHA256 = (
    "8790e8b9bb05fd94733951998ce190490dece89dae6a4c04edee1a058c81e16f"
)
R0_WEIGHT_SHA256 = (
    "6b57d95b82f8715ab7525efe7524feab6d55a50bc0376355dc7ea208ada49fed"
)
R0_DETECTOR_CONFIG_SHA256 = (
    "2b50d8afa950626e2bed6b41807cb602a01a90e66baf7529fa08945d3d676ef8"
)
R0_CONFIG_SHA256 = (
    "9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc3458eac207808d"
)
R0_CREATION_AUTHORITY = baseline.CREATION_AUTHORITY
FULL_CREATION_AUTHORITY = (
    "user-authorized full-frame detector cache completion for locked full-13"
)
ODD_CREATION_AUTHORITY = (
    "user-authorized odd-frame-only detector inference for locked full-13"
)
DATE_STAMP = "20260728"
EXPECTED_VIDEOS = 13
EXPECTED_FRAMES = 1800
EXPECTED_SUBSET_FRAMES = 900
EXPECTED_SUBSET_TOTAL = 11700
EXPECTED_FULL_TOTAL = 23400
HEARTBEAT_BATCH = 50
MINIMUM_FREE_BYTES = 4 * 1024**3
CONTENT_HASH_CONTRACT = "detector_cache_frame_content_sha256_v1"
AUTHORITY_FILENAMES = (
    f"R0_EVEN_FRAME_CACHE_FREEZE_{DATE_STAMP}.json",
    f"FULL_FRAME_DETECTOR_CONFIG_AUTHORITY_{DATE_STAMP}.json",
    f"FULL_FRAME_DETECTOR_MISSING_FRAME_MANIFEST_{DATE_STAMP}.json",
    f"FULL_FRAME_CACHE_PROFILE_CONSUMPTION_CONTRACT_{DATE_STAMP}.json",
    f"FULL_FRAME_DETECTOR_CACHE_AUTHORITY_{DATE_STAMP}.json",
    f"FULL_FRAME_DETECTOR_CACHE_INVENTORY_{DATE_STAMP}.csv",
    f"FULL_FRAME_DETECTOR_CACHE_DECISION_{DATE_STAMP}.json",
)


class FullFrameCacheError(RuntimeError):
    """Fail-closed full-frame cache completion error."""


def utc_now() -> str:
    """Return a stable UTC timestamp for audit-only run state."""

    return datetime.now(UTC).isoformat()


def canonical_hash(payload: Any) -> str:
    """Hash JSON-compatible authority content deterministically."""

    return baseline.canonical_hash(payload)


def sha256_file(path: Path) -> str:
    """Hash one file without loading it entirely into memory."""

    return baseline.sha256_file(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically replace a machine-readable run-state or authority file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    for attempt in range(100):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.05)


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: list[str],
) -> None:
    """Write a deterministic CSV artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args: str) -> str:
    """Run a read-only Git query in the producer worktree."""

    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
        encoding="utf-8",
    ).strip()


def require_clean_producer() -> tuple[str, str]:
    """Bind the producer commit while tolerating only ignored run caches."""

    head = git_output("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_MAIN_SHA, head],
        cwd=REPO,
        check=False,
    )
    if ancestor.returncode:
        raise FullFrameCacheError("starting main is not an ancestor of producer")
    if git_output("status", "--short"):
        raise FullFrameCacheError("cache producer worktree must be clean")
    return head, git_output("rev-parse", "HEAD:src/pig_behavior/tracking")


def expected_even_indices(frame_count: int = EXPECTED_FRAMES) -> tuple[int, ...]:
    """Return the exact frozen R0 detector cadence."""

    return tuple(range(0, frame_count, 2))


def expected_odd_indices(frame_count: int = EXPECTED_FRAMES) -> tuple[int, ...]:
    """Return the exact missing detector population."""

    return tuple(range(1, frame_count, 2))


def derive_missing_indices(
    existing_indices: Iterable[int],
    frame_count: int = EXPECTED_FRAMES,
) -> tuple[int, ...]:
    """Derive and validate the odd-only completion set."""

    existing = tuple(int(index) for index in existing_indices)
    if existing != expected_even_indices(frame_count):
        raise FullFrameCacheError("existing cache is not the exact even subset")
    missing = tuple(index for index in range(frame_count) if index not in existing)
    if missing != expected_odd_indices(frame_count):
        raise FullFrameCacheError("derived missing set is not the exact odd subset")
    return missing


def source_cache_path(root: Path, video_key: str) -> Path:
    """Resolve one frozen R0 cache partition."""

    return root / "partitions" / video_key / "detector_evidence.npz"


def odd_cache_path(root: Path, video_key: str) -> Path:
    """Resolve one newly inferred odd-only cache partition."""

    return root / "odd_inference" / "partitions" / video_key / "detector_evidence.npz"


def full_cache_path(root: Path, video_key: str) -> Path:
    """Resolve one combined full-frame cache partition."""

    return root / "full" / "partitions" / video_key / "detector_evidence.npz"


def checkpoint_path(root: Path, stage: str, video_key: str) -> Path:
    """Resolve one per-video transaction checkpoint."""

    return root / "checkpoints" / stage / f"{video_key}.json"


def _update_hash_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _update_hash_array(digest: Any, array: np.ndarray | None) -> None:
    if array is None:
        _update_hash_text(digest, "NONE")
        return
    contiguous = np.ascontiguousarray(array)
    _update_hash_text(digest, contiguous.dtype.str)
    _update_hash_text(digest, json.dumps(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))


def cache_content_hash(
    cache: DetectorEvidenceCache,
    frame_indices: Iterable[int] | None = None,
) -> str:
    """Hash names and exact frame evidence independently of cache identity."""

    indices = (
        tuple(sorted(cache.frames))
        if frame_indices is None
        else tuple(sorted(frame_indices))
    )
    digest = hashlib.sha256()
    _update_hash_text(digest, CONTENT_HASH_CONTRACT)
    _update_hash_text(
        digest,
        json.dumps(cache.names, sort_keys=True, separators=(",", ":")),
    )
    for frame_index in indices:
        if frame_index not in cache.frames:
            raise FullFrameCacheError(f"missing hash frame: {frame_index}")
        entry = cache.frames[frame_index]
        _update_hash_text(digest, str(frame_index))
        _update_hash_text(
            digest,
            json.dumps(list(entry["original_frame_dimensions"])),
        )
        _update_hash_text(digest, repr(entry["frame_timestamp"]))
        for key in ("xyxy", "conf", "cls", "id", "masks", "appearance"):
            _update_hash_text(digest, key)
            _update_hash_array(digest, entry[key])
    return digest.hexdigest()


def _copy_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        key: None if value is None else value.copy()
        for key, value in entry.items()
        if key
        in {
            "xyxy",
            "conf",
            "cls",
            "id",
            "masks",
            "appearance",
        }
    } | {
        "original_frame_dimensions": tuple(entry["original_frame_dimensions"]),
        "frame_timestamp": entry["frame_timestamp"],
    }


def compare_frame_entries(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    """Compare all serialized detector evidence for one frame."""

    if first["original_frame_dimensions"] != second["original_frame_dimensions"]:
        return False
    if first["frame_timestamp"] != second["frame_timestamp"]:
        return False
    for key in ("xyxy", "conf", "cls", "id", "masks", "appearance"):
        left = first[key]
        right = second[key]
        if (left is None) != (right is None):
            return False
        if left is not None and not np.array_equal(left, right):
            return False
    return True


def combine_caches(
    even: DetectorEvidenceCache,
    odd: DetectorEvidenceCache,
    identity: DetectorCacheIdentity,
) -> DetectorEvidenceCache:
    """Build one full cache without mutating either source cache."""

    if tuple(even.frames) != expected_even_indices():
        raise FullFrameCacheError("even cache coverage is invalid")
    if tuple(odd.frames) != expected_odd_indices():
        raise FullFrameCacheError("odd cache coverage is invalid")
    if even.names != odd.names:
        raise FullFrameCacheError("detector class-name mapping differs")
    full = DetectorEvidenceCache(identity=identity, names=dict(even.names))
    for frame_index in range(EXPECTED_FRAMES):
        source = even if frame_index % 2 == 0 else odd
        full.frames[frame_index] = _copy_entry(source.frames[frame_index])
        full._validate_frame(frame_index, full.frames[frame_index])
    if tuple(full.frames) != tuple(range(EXPECTED_FRAMES)):
        raise FullFrameCacheError("combined cache coverage is invalid")
    return full


def assert_even_subset_parity(
    even: DetectorEvidenceCache,
    full: DetectorEvidenceCache,
) -> str:
    """Prove the combined cache preserved every even-frame record exactly."""

    if even.names != full.names:
        raise FullFrameCacheError("even subset class-name mapping changed")
    for frame_index in expected_even_indices():
        if not compare_frame_entries(
            even.frames[frame_index],
            full.frames[frame_index],
        ):
            raise FullFrameCacheError(
                f"even detector evidence changed at frame {frame_index}"
            )
    source_hash = cache_content_hash(even)
    subset_hash = cache_content_hash(full, expected_even_indices())
    if source_hash != subset_hash:
        raise FullFrameCacheError("even subset canonical content hash changed")
    return source_hash


def r0_identity(
    video: Any,
    r0_preflight: dict[str, Any],
) -> DetectorCacheIdentity:
    """Reconstruct the exact immutable R0 cache identity."""

    return DetectorCacheIdentity(
        video_key=video.video_key,
        source_video_sha256=video.video_sha256,
        detector_weight_sha256=r0_preflight["detector_weight_sha256"],
        detector_semantic_config_sha256=r0_preflight[
            "detector_semantic_config_sha256"
        ],
        producer_code_sha=r0_preflight["producer_code_sha"],
        creation_authority=R0_CREATION_AUTHORITY,
    )


def generated_identity(
    video: Any,
    producer_sha: str,
    creation_authority: str,
) -> DetectorCacheIdentity:
    """Create an odd or full cache identity under the frozen detector authority."""

    return DetectorCacheIdentity(
        video_key=video.video_key,
        source_video_sha256=video.video_sha256,
        detector_weight_sha256=R0_WEIGHT_SHA256,
        detector_semantic_config_sha256=R0_DETECTOR_CONFIG_SHA256,
        producer_code_sha=producer_sha,
        creation_authority=creation_authority,
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullFrameCacheError(f"invalid JSON authority: {path}") from exc
    if not isinstance(payload, dict):
        raise FullFrameCacheError(f"JSON authority is not an object: {path}")
    return payload


def environment_payload() -> dict[str, Any]:
    """Capture exact detector software and CUDA semantics."""

    import cv2
    import numpy
    import torch
    import ultralytics

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "ultralytics": ultralytics.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def assert_environment_matches_r0(
    current: dict[str, Any],
    frozen: dict[str, Any],
) -> None:
    """Fail when the executable detector stack differs from R0."""

    keys = (
        "python",
        "platform",
        "numpy",
        "opencv",
        "torch",
        "torch_cuda",
        "cuda_available",
        "gpu",
        "ultralytics",
    )
    mismatches = {
        key: {"expected": frozen.get(key), "actual": current.get(key)}
        for key in keys
        if current.get(key) != frozen.get(key)
    }
    if mismatches:
        raise FullFrameCacheError(
            f"detector environment differs from R0: {mismatches}"
        )
    if not current["cuda_available"]:
        raise FullFrameCacheError("authorized detector GPU is unavailable")


def model_authority(model: Any) -> dict[str, Any]:
    """Describe architecture restored from the exact frozen weight artifact."""

    torch_model = model.model
    yaml_payload = getattr(torch_model, "yaml", {})
    return {
        "wrapper_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_class": (
            f"{type(torch_model).__module__}.{type(torch_model).__qualname__}"
        ),
        "task": str(getattr(model, "task", "")),
        "names": {
            str(key): str(value)
            for key, value in sorted(dict(getattr(model, "names", {})).items())
        },
        "parameter_count": sum(
            int(parameter.numel()) for parameter in torch_model.parameters()
        ),
        "architecture_yaml_sha256": canonical_hash(yaml_payload),
    }


def verify_r0_root(
    videos: list[Any],
    r0_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Verify and inventory the original R0 cache without writing it."""

    inventory_path = r0_root / "ARTIFACT_SHA256.json"
    if sha256_file(inventory_path) != R0_CACHE_INVENTORY_SHA256:
        raise FullFrameCacheError("R0 cache inventory hash changed")
    decision = load_json(
        r0_root / "CURRENT_MAIN_DETECTOR_CACHE_GENERATION_DECISION.json"
    )
    preflight = load_json(
        r0_root / "CURRENT_MAIN_DETECTOR_CACHE_PREFLIGHT.json"
    )
    if decision.get("decision") != "PASS_EXACT_CURRENT_MAIN_BASELINE_CACHES_READY":
        raise FullFrameCacheError("R0 cache generation authority is not PASS")
    if decision.get("cache_schema") != DETECTOR_CACHE_SCHEMA_VERSION:
        raise FullFrameCacheError("R0 cache schema changed")
    if decision.get("detector_weight_sha256") != R0_WEIGHT_SHA256:
        raise FullFrameCacheError("R0 detector weight authority changed")
    if decision.get("detector_semantic_config_sha256") != R0_DETECTOR_CONFIG_SHA256:
        raise FullFrameCacheError("R0 detector semantic authority changed")
    if decision.get("effective_realtime_fast_config_sha256") != R0_CONFIG_SHA256:
        raise FullFrameCacheError("R0 effective profile authority changed")
    rows: list[dict[str, Any]] = []
    for video in videos:
        expected = decision["cache_artifacts"].get(video.video_key)
        if not isinstance(expected, dict):
            raise FullFrameCacheError(f"missing R0 partition: {video.video_key}")
        path = source_cache_path(r0_root, video.video_key)
        actual_sha = sha256_file(path)
        if actual_sha != expected["sha256"]:
            raise FullFrameCacheError(f"R0 cache bytes changed: {video.video_key}")
        loaded = DetectorEvidenceCache.load(
            path,
            expected_identity=r0_identity(video, preflight),
        )
        indices = tuple(loaded.frames)
        if indices != expected_even_indices():
            raise FullFrameCacheError(f"R0 cadence changed: {video.video_key}")
        rows.append(
            {
                "video_key": video.video_key,
                "source_video_sha256": video.video_sha256,
                "cache_path": str(path),
                "cache_file_sha256": actual_sha,
                "detector_record_count": len(indices),
                "frame_indices": list(indices),
                "minimum_frame": min(indices),
                "maximum_frame": max(indices),
                "detector_weights_sha256": R0_WEIGHT_SHA256,
                "detector_semantic_config_sha256": R0_DETECTOR_CONFIG_SHA256,
                "cache_schema": DETECTOR_CACHE_SCHEMA_VERSION,
                "canonical_content_hash": cache_content_hash(loaded),
            }
        )
    if len(rows) != EXPECTED_VIDEOS:
        raise FullFrameCacheError("R0 cache does not contain 13 partitions")
    if sum(row["detector_record_count"] for row in rows) != EXPECTED_SUBSET_TOTAL:
        raise FullFrameCacheError("R0 cache does not contain 11,700 records")
    return decision, preflight, rows


def detector_authority(
    source_repo: Path,
    environment: dict[str, Any],
    model_payload: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Reproduce and freeze the exact detector invocation contract."""

    cfg, detector_payload = baseline.detector_configuration(source_repo)
    detector_sha = canonical_hash(detector_payload)
    weights_sha = sha256_file(Path(cfg.weights_path))
    config_sha = canonical_hash(baseline.effective_config_payload())
    if detector_sha != R0_DETECTOR_CONFIG_SHA256:
        raise FullFrameCacheError("detector semantic configuration changed")
    if weights_sha != R0_WEIGHT_SHA256:
        raise FullFrameCacheError("detector weights changed")
    if config_sha != R0_CONFIG_SHA256:
        raise FullFrameCacheError("effective R0 configuration changed")
    preprocessing_payload = {
        "preprocessing": detector_payload["preprocessing"],
        "mask_sha256": detector_payload["mask_sha256"],
        "mask_input_frame": detector_payload["mask_input_frame"],
        "imgsz": detector_payload["imgsz"],
        "masks_py_sha256": sha256_file(
            REPO / "src" / "pig_behavior" / "tracking" / "masks.py"
        ),
        "baseline_generator_sha256": sha256_file(
            REPO
            / "scripts"
            / "tracking"
            / "generate_current_main_baseline_caches.py"
        ),
    }
    authority = {
        "schema_version": "tracking.full_frame_detector_config_authority.v1",
        "date": DATE_STAMP,
        "detector_authority_match": "PASS",
        "detector_weights_path": str(Path(cfg.weights_path)),
        "detector_weights_sha256": weights_sha,
        "detector_semantic_config": detector_payload,
        "detector_semantic_config_sha256": detector_sha,
        "effective_realtime_fast_config_sha256": config_sha,
        "model_architecture": model_payload,
        "model_architecture_sha256": canonical_hash(model_payload),
        "preprocessing_sha256": canonical_hash(preprocessing_payload),
        "preprocessing_authority": preprocessing_payload,
        "image_size": cfg.imgsz,
        "confidence_threshold": cfg.det_conf,
        "nms_iou": cfg.nms_iou,
        "maximum_detections": cfg.max_raw_detections,
        "class_policy": {
            "class_id": cfg.class_id,
            "allowed_class_name": cfg.allowed_class_name,
        },
        "precision_mode": "FP32",
        "device": cfg.device,
        "bbox_coordinate_convention": (
            "xyxy_absolute_float32_in_original_decoded_frame_coordinates"
        ),
        "output_sorting": "ultralytics_native_result_order",
        "cache_serialization": DETECTOR_CACHE_SCHEMA_VERSION,
        "environment": environment,
    }
    return cfg, authority


def profile_consumption_contract() -> dict[str, Any]:
    """Declare the scientifically permitted cache consumers."""

    return {
        "schema_version": "tracking.full_frame_cache_consumption_contract.v1",
        "date": DATE_STAMP,
        "B0_PROFILE": "bytetrack_raw",
        "B0_DETECTOR_CADENCE": "EVERY_FRAME",
        "B0_CACHE_CONSUMPTION": "FULL_FRAME_CACHE",
        "B1_PROFILE": "hybrid_bytetrack",
        "B1_DETECTOR_CADENCE": "EVERY_FRAME",
        "B1_CACHE_CONSUMPTION": "FULL_FRAME_CACHE",
        "R0_PROFILE": "realtime_fast",
        "R0_DETECTOR_CADENCE": "EVERY_2_FRAMES",
        "R0_CACHE_CONSUMPTION": "FROZEN_EVEN_SUBSET",
        "R0_RERUN_REQUIRED": "NO",
        "R1_PROFILE": "rf_hybrid_offline",
        "R1_DETECTOR_CADENCE": "EVERY_2_FRAMES",
        "R1_CACHE_CONSUMPTION": "FROZEN_EVEN_SUBSET",
        "CROSS_CORE_COMPARISON_SCOPE": (
            "WHOLE_PIPELINE_EFFECT_INCLUDING_DETECTOR_CADENCE"
        ),
        "PURE_ASSOCIATION_CORE_EFFECT_CLAIM_AUTHORIZED": "NO",
        "B1_MINUS_B0_DETECTOR_CADENCE_MATCHED": "YES",
        "R1_MINUS_R0_DETECTOR_CADENCE_MATCHED": "YES",
        "all_profiles_consume_identical_detector_rows": False,
    }


def load_model(cfg: Any) -> Any:
    """Load the exact frozen detector without invoking inference."""

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise FullFrameCacheError("CUDA is unavailable")
    model = YOLO(str(cfg.weights_path))
    model.to(cfg.device)
    return model


def inspect_prior_attempt(
    prior_root: Path | None,
    videos: list[Any],
) -> tuple[dict[str, Any] | None, list[tuple[Any, DetectorEvidenceCache]]]:
    """Validate reusable committed odd caches and count failed retry calls."""

    if prior_root is None:
        return None, []
    preflight = load_json(prior_root / "FULL_FRAME_CACHE_PREFLIGHT.json")
    state = load_json(prior_root / "FULL_FRAME_CACHE_RUN_STATE.json")
    stderr_path = prior_root / "generation.stderr.log"
    stderr_text = stderr_path.read_text(encoding="utf-8")
    if "PermissionError" not in stderr_text or "record_odd_video" not in stderr_text:
        raise FullFrameCacheError("prior attempt failure is not the known heartbeat lock")
    if state.get("status") != "RUNNING":
        raise FullFrameCacheError("prior attempt state is not the frozen failed state")
    prior_producer = str(preflight["producer_code_sha"])
    videos_by_key = {video.video_key: video for video in videos}
    imported: list[tuple[Any, DetectorEvidenceCache]] = []
    committed_calls = 0
    for checkpoint in sorted(
        (prior_root / "checkpoints" / "odd_inference").glob("*.json")
    ):
        payload = load_json(checkpoint)
        if payload.get("status") != "COMMITTED":
            raise FullFrameCacheError("prior odd checkpoint is not committed")
        video_key = str(payload["video_key"])
        video = videos_by_key.get(video_key)
        if video is None:
            raise FullFrameCacheError("prior checkpoint video is outside population")
        path = odd_cache_path(prior_root, video_key)
        cache = DetectorEvidenceCache.load(
            path,
            expected_identity=generated_identity(
                video,
                prior_producer,
                ODD_CREATION_AUTHORITY,
            ),
        )
        if tuple(cache.frames) != expected_odd_indices():
            raise FullFrameCacheError("prior odd checkpoint coverage changed")
        if sha256_file(path) != payload["cache_artifact_sha256"]:
            raise FullFrameCacheError("prior odd checkpoint artifact changed")
        if cache_content_hash(cache) != payload["canonical_content_hash"]:
            raise FullFrameCacheError("prior odd checkpoint content changed")
        committed_calls += int(payload["detector_inference_calls"])
        imported.append((video, cache))
    if committed_calls != len(imported) * EXPECTED_SUBSET_FRAMES:
        raise FullFrameCacheError("prior committed inference count is inconsistent")
    last_heartbeat_calls = int(state["completed_odd_frames"])
    physical_calls = last_heartbeat_calls + HEARTBEAT_BATCH
    if physical_calls < committed_calls:
        raise FullFrameCacheError("prior physical inference count is inconsistent")
    authority = {
        "prior_attempt_root": str(prior_root),
        "prior_attempt_producer_code_sha": prior_producer,
        "prior_attempt_run_state_sha256": sha256_file(
            prior_root / "FULL_FRAME_CACHE_RUN_STATE.json"
        ),
        "prior_attempt_stderr_sha256": sha256_file(stderr_path),
        "prior_attempt_last_successful_heartbeat_calls": last_heartbeat_calls,
        "prior_attempt_physical_odd_calls": physical_calls,
        "prior_committed_unique_odd_records": committed_calls,
        "prior_uncommitted_retry_calls": physical_calls - committed_calls,
        "imported_video_keys": [
            video.video_key for video, _ in imported
        ],
        "retry_policy": "SAME_ODD_FRAMES_SAME_DETECTOR_AUTHORITY",
    }
    return authority, imported


def import_prior_odd_caches(
    output_root: Path,
    producer_sha: str,
    imported: list[tuple[Any, DetectorEvidenceCache]],
) -> None:
    """Rebind committed prior odd evidence without detector inference."""

    for video, source in imported:
        destination = odd_cache_path(output_root, video.video_key)
        if destination.exists() or destination.with_suffix(".sha256.json").exists():
            raise FullFrameCacheError("refusing imported odd-cache overwrite")
        rebound = DetectorEvidenceCache(
            identity=generated_identity(
                video,
                producer_sha,
                ODD_CREATION_AUTHORITY,
            ),
            names=dict(source.names),
        )
        for frame_index, entry in source.frames.items():
            rebound.frames[frame_index] = _copy_entry(entry)
            rebound._validate_frame(frame_index, rebound.frames[frame_index])
        if cache_content_hash(rebound) != cache_content_hash(source):
            raise FullFrameCacheError("imported odd cache content changed")
        artifact_sha = rebound.save(destination)
        atomic_write_json(
            checkpoint_path(
                output_root,
                "odd_inference",
                video.video_key,
            ),
            {
                "schema_version": "tracking.full_frame_odd_checkpoint.v1",
                "status": "COMMITTED",
                "video_key": video.video_key,
                "source_video_sha256": video.video_sha256,
                "frame_indices": list(expected_odd_indices()),
                "detector_inference_calls": EXPECTED_SUBSET_FRAMES,
                "even_frame_detector_inference_calls": 0,
                "cache_artifact_sha256": artifact_sha,
                "canonical_content_hash": cache_content_hash(rebound),
                "provenance": "IMPORTED_COMMITTED_PRIOR_ODD_INFERENCE",
                "committed_at": utc_now(),
            },
        )


def preflight(
    source_repo: Path,
    lineage_manifest: Path,
    r0_root: Path,
    output_root: Path,
    prior_attempt_root: Path | None = None,
) -> None:
    """Freeze inputs, authority, and the odd-only execution plan."""

    if output_root.exists():
        raise FullFrameCacheError(f"refusing existing output root: {output_root}")
    producer_sha, tracking_tree = require_clean_producer()
    videos, lineage_file_sha = baseline.load_population(
        source_repo,
        lineage_manifest,
    )
    if len(videos) != EXPECTED_VIDEOS:
        raise FullFrameCacheError("locked population is not 13 videos")
    r0_decision, r0_preflight, freeze_rows = verify_r0_root(videos, r0_root)
    environment = environment_payload()
    frozen_environment = load_json(
        r0_root / "CURRENT_MAIN_DETECTOR_CACHE_ENVIRONMENT.json"
    )
    assert_environment_matches_r0(environment, frozen_environment)
    cfg, _ = baseline.detector_configuration(source_repo)
    model = load_model(cfg)
    cfg, config_authority = detector_authority(
        source_repo,
        environment,
        model_authority(model),
    )
    del model
    if cfg.detect_every_n_frames != 2:
        raise FullFrameCacheError("frozen R0 detector cadence changed")
    if shutil.disk_usage(output_root.parent).free < MINIMUM_FREE_BYTES:
        raise FullFrameCacheError("insufficient disk space")
    prior_authority, imported = inspect_prior_attempt(
        prior_attempt_root,
        videos,
    )

    missing_videos = []
    for video, freeze in zip(videos, freeze_rows, strict=True):
        missing = derive_missing_indices(freeze["frame_indices"])
        missing_videos.append(
            {
                "video_key": video.video_key,
                "source_video_path": str(video.video_path),
                "source_video_sha256": video.video_sha256,
                "frame_start": 0,
                "frame_end": video.frame_count - 1,
                "expected_total_frames": video.frame_count,
                "existing_even_frame_count": len(freeze["frame_indices"]),
                "missing_frame_indices": list(missing),
                "missing_frame_count": len(missing),
                "frame_parity": "ODD_ONLY",
                "extraction_decode_policy": (
                    "OpenCV VideoCapture sequential decode; infer only exact "
                    "manifest indices; verify CAP_PROP_POS_FRAMES after read"
                ),
            }
        )
    if sum(row["missing_frame_count"] for row in missing_videos) != (
        EXPECTED_SUBSET_TOTAL
    ):
        raise FullFrameCacheError("missing-frame manifest is not 11,700 frames")

    output_root.mkdir(parents=True)
    import_prior_odd_caches(output_root, producer_sha, imported)
    freeze_payload = {
        "schema_version": "tracking.r0_even_frame_cache_freeze.v1",
        "date": DATE_STAMP,
        "source_r0_cache_root": str(r0_root),
        "source_r0_cache_inventory_sha256": R0_CACHE_INVENTORY_SHA256,
        "video_count": len(freeze_rows),
        "even_records_per_video": EXPECTED_SUBSET_FRAMES,
        "total_existing_records": EXPECTED_SUBSET_TOTAL,
        "odd_frame_records_existing": 0,
        "detector_weights_sha256": R0_WEIGHT_SHA256,
        "detector_semantic_config_sha256": R0_DETECTOR_CONFIG_SHA256,
        "preprocessing_sha256": config_authority["preprocessing_sha256"],
        "cache_schema": DETECTOR_CACHE_SCHEMA_VERSION,
        "content_hash_contract": CONTENT_HASH_CONTRACT,
        "partitions": freeze_rows,
    }
    missing_payload = {
        "schema_version": "tracking.full_frame_missing_manifest.v1",
        "date": DATE_STAMP,
        "selection_policy": "EXACT_COMPLEMENT_OF_FROZEN_EVEN_SUBSET",
        "expected_total_frames_per_video": EXPECTED_FRAMES,
        "existing_even_frames_per_video": EXPECTED_SUBSET_FRAMES,
        "missing_odd_frames_per_video": EXPECTED_SUBSET_FRAMES,
        "missing_odd_frames_total": EXPECTED_SUBSET_TOTAL,
        "videos": missing_videos,
    }
    atomic_write_json(
        output_root / AUTHORITY_FILENAMES[0],
        freeze_payload,
    )
    atomic_write_json(
        output_root / AUTHORITY_FILENAMES[1],
        config_authority,
    )
    atomic_write_json(
        output_root / AUTHORITY_FILENAMES[2],
        missing_payload,
    )
    atomic_write_json(
        output_root / AUTHORITY_FILENAMES[3],
        profile_consumption_contract(),
    )
    atomic_write_json(
        output_root / "FULL_FRAME_CACHE_ENVIRONMENT.json",
        environment,
    )
    planned = {
        "schema_version": "tracking.full_frame_cache_preflight.v1",
        "status": "PASS_READY_FOR_ODD_ONLY_INFERENCE",
        "created_at": utc_now(),
        "starting_main_sha": STARTING_MAIN_SHA,
        "producer_code_sha": producer_sha,
        "tracking_tree_object": tracking_tree,
        "source_repo": str(source_repo),
        "source_lineage_manifest": str(lineage_manifest),
        "source_lineage_file_sha256": lineage_file_sha,
        "source_r0_cache_root": str(r0_root),
        "source_r0_cache_inventory_sha256": R0_CACHE_INVENTORY_SHA256,
        "output_root": str(output_root),
        "detector_authority_sha256": sha256_file(
            output_root / AUTHORITY_FILENAMES[1]
        ),
        "missing_manifest_sha256": sha256_file(
            output_root / AUTHORITY_FILENAMES[2]
        ),
        "r0_generation_decision_sha256": sha256_file(
            r0_root / "CURRENT_MAIN_DETECTOR_CACHE_GENERATION_DECISION.json"
        ),
        "r0_producer_code_sha": r0_preflight["producer_code_sha"],
        "r0_cache_artifacts": r0_decision["cache_artifacts"],
        "selected_skills": [
            "tracking-experiment-guardian",
            "experiment-lineage-reproducibility",
            "computer-vision-opencv",
        ],
        "expected_odd_detector_calls": EXPECTED_SUBSET_TOTAL,
        "expected_even_detector_calls": 0,
        "prior_attempt": prior_authority,
        "new_odd_detector_calls_expected": (
            EXPECTED_SUBSET_TOTAL
            - len(imported) * EXPECTED_SUBSET_FRAMES
        ),
        "tracker_executions": 0,
        "metric_runs": 0,
        "mp4_count": 0,
    }
    atomic_write_json(output_root / "FULL_FRAME_CACHE_PREFLIGHT.json", planned)
    atomic_write_json(
        output_root / "FULL_FRAME_CACHE_RUN_STATE.json",
        {
            "schema_version": "tracking.full_frame_cache_run_state.v1",
            "status": "PLANNED",
            "phase": "AWAITING_ODD_ONLY_INFERENCE",
            "completed_odd_frames": 0,
            "total_odd_frames": EXPECTED_SUBSET_TOTAL,
            "last_update": utc_now(),
        },
    )
    (output_root / "FULL_FRAME_CACHE_COMMANDS.txt").write_text(
        "PREFLIGHT\n"
        + subprocess.list2cmdline(sys.argv)
        + "\nEVEN_FRAME_DETECTOR_INFERENCE_CALLS=0\n"
        + "TRACKER_EXECUTIONS=0\nMETRIC_RUNS=0\n",
        encoding="utf-8",
    )


def verify_preflight(
    source_repo: Path,
    lineage_manifest: Path,
    r0_root: Path,
    output_root: Path,
) -> tuple[list[Any], Any, dict[str, Any], dict[str, Any], str]:
    """Revalidate all preflight hashes before any detector invocation."""

    preflight_payload = load_json(output_root / "FULL_FRAME_CACHE_PREFLIGHT.json")
    if preflight_payload.get("status") != "PASS_READY_FOR_ODD_ONLY_INFERENCE":
        raise FullFrameCacheError("full-frame cache preflight did not pass")
    producer_sha, _ = require_clean_producer()
    if preflight_payload["producer_code_sha"] != producer_sha:
        raise FullFrameCacheError("producer commit changed after preflight")
    videos, lineage_file_sha = baseline.load_population(
        source_repo,
        lineage_manifest,
    )
    if lineage_file_sha != preflight_payload["source_lineage_file_sha256"]:
        raise FullFrameCacheError("source lineage changed after preflight")
    _, r0_preflight, _ = verify_r0_root(videos, r0_root)
    environment = environment_payload()
    assert_environment_matches_r0(
        environment,
        load_json(r0_root / "CURRENT_MAIN_DETECTOR_CACHE_ENVIRONMENT.json"),
    )
    cfg, _ = baseline.detector_configuration(source_repo)
    if canonical_hash(baseline.detector_configuration(source_repo)[1]) != (
        R0_DETECTOR_CONFIG_SHA256
    ):
        raise FullFrameCacheError("detector configuration changed after preflight")
    if sha256_file(Path(cfg.weights_path)) != R0_WEIGHT_SHA256:
        raise FullFrameCacheError("detector weights changed after preflight")
    missing_path = output_root / AUTHORITY_FILENAMES[2]
    if sha256_file(missing_path) != preflight_payload["missing_manifest_sha256"]:
        raise FullFrameCacheError("missing-frame manifest changed")
    missing_payload = load_json(missing_path)
    return videos, cfg, r0_preflight, missing_payload, producer_sha


def update_run_state(
    output_root: Path,
    *,
    status: str,
    phase: str,
    completed: int,
    video_key: str | None = None,
    frame_index: int | None = None,
    failure: str | None = None,
) -> None:
    """Refresh the atomic progress heartbeat."""

    atomic_write_json(
        output_root / "FULL_FRAME_CACHE_RUN_STATE.json",
        {
            "schema_version": "tracking.full_frame_cache_run_state.v1",
            "status": status,
            "phase": phase,
            "completed_odd_frames": completed,
            "total_odd_frames": EXPECTED_SUBSET_TOTAL,
            "current_video_key": video_key,
            "last_processed_frame": frame_index,
            "failure": failure,
            "last_update": utc_now(),
        },
    )


def record_odd_video(
    video: Any,
    cfg: Any,
    model: Any,
    output_root: Path,
    producer_sha: str,
    missing_indices: tuple[int, ...],
    completed_before: int,
) -> tuple[DetectorEvidenceCache, int]:
    """Infer and transactionally freeze exactly one video's odd frames."""

    import cv2

    from pig_behavior.tracking.masks import apply_mask_to_frame, load_mask

    if missing_indices != expected_odd_indices(video.frame_count):
        raise FullFrameCacheError(f"odd manifest changed: {video.video_key}")
    output = odd_cache_path(output_root, video.video_key)
    checkpoint = checkpoint_path(output_root, "odd_inference", video.video_key)
    if output.exists() or output.with_suffix(".sha256.json").exists():
        if not checkpoint.is_file():
            raise FullFrameCacheError(
                f"partial odd cache lacks checkpoint: {video.video_key}"
            )
        loaded = DetectorEvidenceCache.load(
            output,
            expected_identity=generated_identity(
                video,
                producer_sha,
                ODD_CREATION_AUTHORITY,
            ),
        )
        if tuple(loaded.frames) != missing_indices:
            raise FullFrameCacheError(
                f"checkpoint odd coverage changed: {video.video_key}"
            )
        payload = load_json(checkpoint)
        if payload.get("detector_inference_calls") != EXPECTED_SUBSET_FRAMES:
            raise FullFrameCacheError(
                f"checkpoint call count changed: {video.video_key}"
            )
        if payload.get("cache_artifact_sha256") != sha256_file(output):
            raise FullFrameCacheError(
                f"checkpoint cache hash changed: {video.video_key}"
            )
        return loaded, 0

    cache = DetectorEvidenceCache(
        identity=generated_identity(
            video,
            producer_sha,
            ODD_CREATION_AUTHORITY,
        )
    )
    capture = cv2.VideoCapture(str(video.video_path))
    if not capture.isOpened():
        raise FullFrameCacheError(f"cannot open source: {video.video_key}")
    mask = load_mask(Path(cfg.mask_path), video.width, video.height, cfg)
    missing_set = frozenset(missing_indices)
    calls = 0
    try:
        for frame_index in range(video.frame_count):
            ok, frame = capture.read()
            if not ok:
                raise FullFrameCacheError(
                    f"decode failed: {video.video_key} frame {frame_index}"
                )
            decoded_position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if decoded_position != frame_index + 1:
                raise FullFrameCacheError(
                    f"decoder index drift: {video.video_key} frame {frame_index}"
                )
            if frame_index not in missing_set:
                continue
            if frame_index % 2 == 0:
                raise FullFrameCacheError(
                    f"refusing even-frame inference: {video.video_key}"
                )
            detector_frame = (
                apply_mask_to_frame(frame, mask)
                if cfg.mask_input_frame and mask is not None
                else frame
            )
            result = baseline.invoke_detector(model, detector_frame, cfg)
            cache.record(
                frame_index,
                result,
                original_frame_dimensions=(video.height, video.width),
            )
            calls += 1
            if calls % HEARTBEAT_BATCH == 0:
                update_run_state(
                    output_root,
                    status="RUNNING",
                    phase="ODD_ONLY_INFERENCE",
                    completed=completed_before + calls,
                    video_key=video.video_key,
                    frame_index=frame_index,
                )
    finally:
        capture.release()
    if calls != EXPECTED_SUBSET_FRAMES:
        raise FullFrameCacheError(
            f"odd inference coverage mismatch: {video.video_key}"
        )
    if tuple(cache.frames) != missing_indices:
        raise FullFrameCacheError(
            f"odd cache frame order changed: {video.video_key}"
        )
    artifact_sha = cache.save(output)
    checkpoint_payload = {
        "schema_version": "tracking.full_frame_odd_checkpoint.v1",
        "status": "COMMITTED",
        "video_key": video.video_key,
        "source_video_sha256": video.video_sha256,
        "frame_indices": list(missing_indices),
        "detector_inference_calls": calls,
        "even_frame_detector_inference_calls": 0,
        "cache_artifact_sha256": artifact_sha,
        "canonical_content_hash": cache_content_hash(cache),
        "committed_at": utc_now(),
    }
    atomic_write_json(checkpoint, checkpoint_payload)
    return cache, calls


def replay_cache(cache: DetectorEvidenceCache) -> int:
    """Replay every exact frame without detector inference."""

    replay = ReplayDetector(cache)
    for frame_index, entry in cache.frames.items():
        replay.set_frame_context(
            frame_index,
            entry["original_frame_dimensions"],
        )
        replay.predict()
    return replay.invocations


def validate_full_partition(
    video: Any,
    full: DetectorEvidenceCache,
) -> dict[str, Any]:
    """Validate numeric, coordinate, coverage, and deterministic replay gates."""

    indices = tuple(full.frames)
    if indices != tuple(range(EXPECTED_FRAMES)):
        raise FullFrameCacheError(f"full coverage failed: {video.video_key}")
    zero_detection_frames = 0
    for frame_index, entry in full.frames.items():
        xyxy = entry["xyxy"]
        if xyxy.shape[0] == 0:
            zero_detection_frames += 1
        if np.any(xyxy[:, 0] < 0) or np.any(xyxy[:, 1] < 0):
            raise FullFrameCacheError(
                f"negative bbox coordinate: {video.video_key} {frame_index}"
            )
        if np.any(xyxy[:, 2] > video.width) or np.any(
            xyxy[:, 3] > video.height
        ):
            raise FullFrameCacheError(
                f"bbox exceeds frame: {video.video_key} {frame_index}"
            )
        if np.any(xyxy[:, 2] < xyxy[:, 0]) or np.any(
            xyxy[:, 3] < xyxy[:, 1]
        ):
            raise FullFrameCacheError(
                f"invalid bbox ordering: {video.video_key} {frame_index}"
            )
    first_replay = replay_cache(full)
    first_hash = cache_content_hash(full)
    second_replay = replay_cache(full)
    second_hash = cache_content_hash(full)
    if first_replay != EXPECTED_FRAMES or second_replay != EXPECTED_FRAMES:
        raise FullFrameCacheError(f"cache replay failed: {video.video_key}")
    if first_hash != second_hash:
        raise FullFrameCacheError(f"replay determinism failed: {video.video_key}")
    return {
        "video_key": video.video_key,
        "frame_count": len(indices),
        "minimum_frame": min(indices),
        "maximum_frame": max(indices),
        "zero_detection_frames": zero_detection_frames,
        "first_replay_frames": first_replay,
        "second_replay_frames": second_replay,
        "canonical_content_hash": first_hash,
        "determinism": "PASS",
    }


def write_provenance(
    output_root: Path,
    videos: list[Any],
    r0_decision: dict[str, Any],
) -> None:
    """Bind every full-cache record to even reuse or odd inference."""

    rows = []
    for video in videos:
        even_sha = r0_decision["cache_artifacts"][video.video_key]["sha256"]
        odd_sha = sha256_file(odd_cache_path(output_root, video.video_key))
        for frame_index in range(EXPECTED_FRAMES):
            is_even = frame_index % 2 == 0
            rows.append(
                {
                    "video_key": video.video_key,
                    "source_video_sha256": video.video_sha256,
                    "frame_index": frame_index,
                    "parity": "EVEN" if is_even else "ODD",
                    "provenance": (
                        "FROZEN_R0_EVEN_RECORD"
                        if is_even
                        else "ACTUAL_ODD_FRAME_DETECTOR_INFERENCE"
                    ),
                    "source_cache_sha256": even_sha if is_even else odd_sha,
                }
            )
    write_csv(
        output_root / "FULL_FRAME_RECORD_PROVENANCE.csv",
        rows,
        [
            "video_key",
            "source_video_sha256",
            "frame_index",
            "parity",
            "provenance",
            "source_cache_sha256",
        ],
    )


def artifact_inventory(
    root: Path,
    *,
    excluded_names: set[str],
) -> list[dict[str, Any]]:
    """Inventory every retained file except declared self-referential records."""

    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name not in excluded_names
    ]


def generate(
    source_repo: Path,
    lineage_manifest: Path,
    r0_root: Path,
    output_root: Path,
) -> None:
    """Run odd-only inference, merge, and freeze the full cache authority."""

    completed = 0
    try:
        videos, cfg, r0_preflight, missing_payload, producer_sha = (
            verify_preflight(
                source_repo,
                lineage_manifest,
                r0_root,
                output_root,
            )
        )
        r0_decision = load_json(
            r0_root / "CURRENT_MAIN_DETECTOR_CACHE_GENERATION_DECISION.json"
        )
        run_preflight = load_json(
            output_root / "FULL_FRAME_CACHE_PREFLIGHT.json"
        )
        prior_attempt = run_preflight.get("prior_attempt") or {}
        prior_physical_calls = int(
            prior_attempt.get("prior_attempt_physical_odd_calls", 0)
        )
        expected_new_calls = int(
            run_preflight["new_odd_detector_calls_expected"]
        )
        model = load_model(cfg)
        configured_model = model_authority(model)
        frozen_config = load_json(output_root / AUTHORITY_FILENAMES[1])
        if canonical_hash(configured_model) != frozen_config[
            "model_architecture_sha256"
        ]:
            raise FullFrameCacheError("loaded model architecture changed")
        missing_by_video = {
            row["video_key"]: tuple(row["missing_frame_indices"])
            for row in missing_payload["videos"]
        }
        per_video: list[dict[str, Any]] = []
        actual_calls = 0
        for video in videos:
            update_run_state(
                output_root,
                status="RUNNING",
                phase="ODD_ONLY_INFERENCE",
                completed=completed,
                video_key=video.video_key,
            )
            odd, calls = record_odd_video(
                video,
                cfg,
                model,
                output_root,
                producer_sha,
                missing_by_video[video.video_key],
                completed,
            )
            actual_calls += calls
            completed += EXPECTED_SUBSET_FRAMES
            even_path = source_cache_path(r0_root, video.video_key)
            even = DetectorEvidenceCache.load(
                even_path,
                expected_identity=r0_identity(video, r0_preflight),
            )
            full_identity = generated_identity(
                video,
                producer_sha,
                FULL_CREATION_AUTHORITY,
            )
            full = combine_caches(even, odd, full_identity)
            even_hash = assert_even_subset_parity(even, full)
            output = full_cache_path(output_root, video.video_key)
            checkpoint = checkpoint_path(
                output_root,
                "full_cache",
                video.video_key,
            )
            if output.exists() or output.with_suffix(".sha256.json").exists():
                if not checkpoint.is_file():
                    raise FullFrameCacheError(
                        f"partial full cache lacks checkpoint: {video.video_key}"
                    )
                full = DetectorEvidenceCache.load(
                    output,
                    expected_identity=full_identity,
                )
                assert_even_subset_parity(even, full)
            else:
                full_sha = full.save(output)
                atomic_write_json(
                    checkpoint,
                    {
                        "schema_version": (
                            "tracking.full_frame_partition_checkpoint.v1"
                        ),
                        "status": "COMMITTED",
                        "video_key": video.video_key,
                        "cache_artifact_sha256": full_sha,
                        "canonical_content_hash": cache_content_hash(full),
                        "even_subset_content_hash": even_hash,
                        "odd_subset_content_hash": cache_content_hash(odd),
                        "committed_at": utc_now(),
                    },
                )
            validation = validate_full_partition(video, full)
            full_sha = sha256_file(output)
            checkpoint_payload = load_json(checkpoint)
            if checkpoint_payload["cache_artifact_sha256"] != full_sha:
                raise FullFrameCacheError(
                    f"full checkpoint hash changed: {video.video_key}"
                )
            per_video.append(
                validation
                | {
                    "source_r0_even_cache_sha256": sha256_file(even_path),
                    "odd_cache_sha256": sha256_file(
                        odd_cache_path(output_root, video.video_key)
                    ),
                    "full_cache_sha256": full_sha,
                    "even_subset_content_hash": even_hash,
                    "odd_subset_content_hash": cache_content_hash(odd),
                    "even_subset_parity": "PASS",
                }
            )
            update_run_state(
                output_root,
                status="RUNNING",
                phase="FULL_CACHE_VALIDATION",
                completed=completed,
                video_key=video.video_key,
                frame_index=EXPECTED_FRAMES - 1,
            )
        del model
        if completed != EXPECTED_SUBSET_TOTAL:
            raise FullFrameCacheError("odd completion count is not 11,700")
        if actual_calls != expected_new_calls:
            raise FullFrameCacheError("new odd detector-call count is incorrect")
        if sum(row["frame_count"] for row in per_video) != EXPECTED_FULL_TOTAL:
            raise FullFrameCacheError("full cache count is not 23,400")
        verify_r0_root(videos, r0_root)
        write_provenance(output_root, videos, r0_decision)
        source_hashes = {
            video.video_key: video.video_sha256 for video in videos
        }
        physical_odd_calls = prior_physical_calls + actual_calls
        retry_odd_calls = physical_odd_calls - EXPECTED_SUBSET_TOTAL
        if retry_odd_calls < 0:
            raise FullFrameCacheError("physical odd detector-call count is invalid")
        full_authority = {
            "schema_version": "tracking.full_frame_detector_cache_authority.v1",
            "date": DATE_STAMP,
            "status": "ESTABLISHED",
            "retention_class": "NON_DISPOSABLE_FROZEN_DETECTOR_AUTHORITY",
            "deletion_allowed": "NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT",
            "producer_code_sha": producer_sha,
            "source_r0_cache_root": str(r0_root),
            "source_r0_cache_inventory_sha256": R0_CACHE_INVENTORY_SHA256,
            "full_frame_cache_root": str(output_root),
            "detector_weights_sha256": R0_WEIGHT_SHA256,
            "detector_semantic_config_sha256": R0_DETECTOR_CONFIG_SHA256,
            "detector_config_authority_sha256": sha256_file(
                output_root / AUTHORITY_FILENAMES[1]
            ),
            "missing_frame_manifest_sha256": sha256_file(
                output_root / AUTHORITY_FILENAMES[2]
            ),
            "source_video_hashes": source_hashes,
            "video_count": len(videos),
            "full_cache_frames": EXPECTED_FULL_TOTAL,
            "existing_even_records_preserved": EXPECTED_SUBSET_TOTAL,
            "new_odd_records": EXPECTED_SUBSET_TOTAL,
            "even_frame_detector_inference_calls": 0,
            "unique_odd_frame_inference_records": EXPECTED_SUBSET_TOTAL,
            "odd_frame_detector_inference_calls": physical_odd_calls,
            "odd_frame_retry_inference_calls": retry_odd_calls,
            "prior_attempt": prior_attempt or None,
            "detector_calls_in_this_invocation": actual_calls,
            "even_subset_parity": "PASS",
            "full_frame_coverage": "PASS",
            "cache_replay": "PASS",
            "determinism": "PASS",
            "content_hash_contract": CONTENT_HASH_CONTRACT,
            "canonical_full_population_hash": canonical_hash(
                [
                    {
                        "video_key": row["video_key"],
                        "canonical_content_hash": row[
                            "canonical_content_hash"
                        ],
                    }
                    for row in per_video
                ]
            ),
            "per_video": per_video,
            "tracker_executions": 0,
            "standard_v2_metric_runs": 0,
            "legacy_metric_runs": 0,
            "unseen_videos_accessed": False,
            "run_root_mp4_count": len(list(output_root.rglob("*.mp4"))),
        }
        if full_authority["run_root_mp4_count"]:
            raise FullFrameCacheError("cache authority root contains MP4")
        with (output_root / "FULL_FRAME_CACHE_COMMANDS.txt").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "ODD_ONLY_GENERATION_AND_VALIDATION\n"
                + subprocess.list2cmdline(sys.argv)
                + "\nEVEN_FRAME_DETECTOR_INFERENCE_CALLS=0\n"
                + f"ODD_FRAME_DETECTOR_INFERENCE_CALLS={physical_odd_calls}\n"
                + f"ODD_FRAME_RETRY_INFERENCE_CALLS={retry_odd_calls}\n"
                + "TRACKER_EXECUTIONS=0\nMETRIC_RUNS=0\n"
            )
        update_run_state(
            output_root,
            status="COMPLETED",
            phase="FULL_CACHE_FROZEN",
            completed=EXPECTED_SUBSET_TOTAL,
        )
        excluded = {
            "ARTIFACT_SHA256.json",
            AUTHORITY_FILENAMES[4],
            AUTHORITY_FILENAMES[5],
            AUTHORITY_FILENAMES[6],
        }
        inventory = artifact_inventory(output_root, excluded_names=excluded)
        write_csv(
            output_root / AUTHORITY_FILENAMES[5],
            inventory,
            ["relative_path", "size_bytes", "sha256"],
        )
        inventory_hash = sha256_file(output_root / AUTHORITY_FILENAMES[5])
        full_authority["artifact_inventory_sha256"] = inventory_hash
        full_authority_path = output_root / AUTHORITY_FILENAMES[4]
        atomic_write_json(full_authority_path, full_authority)
        authority_sha = sha256_file(full_authority_path)
        decision = {
            "schema_version": "tracking.full_frame_detector_cache_decision.v1",
            "date": DATE_STAMP,
            "decision": "PASS_FULL_FRAME_DETECTOR_CACHE_FROZEN",
            "authority_sha256": authority_sha,
            "artifact_inventory_sha256": inventory_hash,
            "video_count": EXPECTED_VIDEOS,
            "full_cache_frames": EXPECTED_FULL_TOTAL,
            "existing_even_records_preserved": EXPECTED_SUBSET_TOTAL,
            "new_odd_records": EXPECTED_SUBSET_TOTAL,
            "even_frame_detector_inference_calls": 0,
            "unique_odd_frame_inference_records": EXPECTED_SUBSET_TOTAL,
            "odd_frame_detector_inference_calls": physical_odd_calls,
            "odd_frame_retry_inference_calls": retry_odd_calls,
            "retry_policy_compliance": (
                "PASS_SAME_ODD_FRAMES_SAME_DETECTOR_AUTHORITY"
            ),
            "even_subset_parity": "PASS",
            "detector_authority_match": "PASS",
            "cache_replay": "PASS",
            "full_frame_coverage": "PASS",
            "tracker_executions": 0,
            "metric_runs": 0,
            "unseen_videos_accessed": False,
            "run_root_mp4_count": 0,
            "ready_for_b0_b1_prediction_regeneration": True,
        }
        atomic_write_json(output_root / AUTHORITY_FILENAMES[6], decision)
        atomic_write_json(
            output_root / "ARTIFACT_SHA256.json",
            {
                "schema_version": "tracking.full_frame_cache_inventory.v1",
                "inventory_excludes_itself": True,
                "authority_sha256": authority_sha,
                "decision_sha256": sha256_file(
                    output_root / AUTHORITY_FILENAMES[6]
                ),
                "inventory_csv_sha256": inventory_hash,
                "artifacts": artifact_inventory(
                    output_root,
                    excluded_names={"ARTIFACT_SHA256.json"},
                ),
            },
        )
    except Exception as exc:
        if output_root.exists():
            try:
                update_run_state(
                    output_root,
                    status="FAILED",
                    phase="ODD_ONLY_CACHE_COMPLETION",
                    completed=completed,
                    failure=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                pass
        raise


def finalize_docs(output_root: Path, docs_root: Path) -> None:
    """Copy only finalized small authority records into the Git worktree."""

    decision = load_json(output_root / AUTHORITY_FILENAMES[6])
    if decision.get("decision") != "PASS_FULL_FRAME_DETECTOR_CACHE_FROZEN":
        raise FullFrameCacheError("cannot finalize docs before a PASS decision")
    if docs_root.exists():
        raise FullFrameCacheError(f"refusing existing docs root: {docs_root}")
    docs_root.mkdir(parents=True)
    for name in AUTHORITY_FILENAMES:
        source = output_root / name
        if not source.is_file():
            raise FullFrameCacheError(f"missing finalized authority: {name}")
        shutil.copyfile(source, docs_root / name)
    atomic_write_json(
        docs_root / f"FULL_FRAME_DETECTOR_CACHE_GIT_BINDING_{DATE_STAMP}.json",
        {
            "schema_version": "tracking.full_frame_cache_git_binding.v1",
            "date": DATE_STAMP,
            "full_frame_cache_root": str(output_root),
            "full_frame_cache_authority_sha256": sha256_file(
                output_root / AUTHORITY_FILENAMES[4]
            ),
            "full_frame_cache_decision_sha256": sha256_file(
                output_root / AUTHORITY_FILENAMES[6]
            ),
            "artifact_inventory_sha256": sha256_file(
                output_root / "ARTIFACT_SHA256.json"
            ),
            "retention_class": "NON_DISPOSABLE_FROZEN_DETECTOR_AUTHORITY",
            "deletion_allowed": "NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("preflight", "generate", "finalize-docs"),
        required=True,
    )
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--lineage-manifest", type=Path)
    parser.add_argument("--r0-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path)
    parser.add_argument("--prior-attempt-root", type=Path)
    return parser.parse_args()


def require_argument(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if value is None:
        raise FullFrameCacheError(f"--{name.replace('_', '-')} is required")
    return Path(value).resolve()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if args.phase == "finalize-docs":
        finalize_docs(
            output_root,
            require_argument(args, "docs_root"),
        )
        return 0
    source_repo = require_argument(args, "source_repo")
    lineage_manifest = require_argument(args, "lineage_manifest")
    r0_root = require_argument(args, "r0_cache_root")
    if args.phase == "preflight":
        preflight(
            source_repo,
            lineage_manifest,
            r0_root,
            output_root,
            (
                None
                if args.prior_attempt_root is None
                else args.prior_attempt_root.resolve()
            ),
        )
    else:
        generate(source_repo, lineage_manifest, r0_root, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
