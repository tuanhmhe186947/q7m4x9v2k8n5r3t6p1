"""Freeze a full-frame 0.20/64 detector cache for historical H5b/H4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
for import_root in (REPO, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pig_behavior.tracking.detector_cache import (  # noqa: E402
    DETECTOR_CACHE_SCHEMA_VERSION,
    DetectorCacheIdentity,
    DetectorEvidenceCache,
    ReplayDetector,
)
from pig_behavior.tracking.masks import (  # noqa: E402
    apply_mask_to_frame,
    load_mask,
)

DATE = "20260728"
RESOLVED_STARTING_MAIN_SHA = "b189a863fc9b1f60c71682fa734de7f65818ef6b"
REQUESTED_STARTING_MAIN_SHA = "b189a8659bd2dca1135b46cc53e338aebef26a6e"
HISTORICAL_RUN_ID = "20260719_h5b_h4_full13_combined_v2"
HISTORICAL_SOURCE_SHA = "31d360ba96b4065ce5125c0d88765531cc5898ae"
WEIGHTS_SHA256 = (
    "6b57d95b82f8715ab7525efe7524feab6d55a50bc0376355dc7ea208ada49fed"
)
MASK_SHA256 = (
    "b59b998ef49335b730c5f117e7161f24ccd277d3b5130c0e640dab7bbb980658"
)
CONFIDENCE = 0.20
MAX_RAW_DETECTIONS = 64
NMS_IOU = 0.50
IMAGE_SIZE = 640
EXPECTED_VIDEOS = 13
EXPECTED_FRAMES = 1800
EXPECTED_TOTAL = EXPECTED_VIDEOS * EXPECTED_FRAMES
PRODUCER_TOPOLOGY = "CURRENT_PRODUCER_SEMANTICALLY_EQUIVALENT"
CREATION_AUTHORITY = (
    "historical H5b/H4 0.20/64 detector-cache reproduction candidate"
)
CONTENT_HASH_CONTRACT = "detector_cache_frame_content_sha256_v1"
CURRENT_CACHE_ROOT = Path(
    "C:/Users/ironh/Downloads/PIG_Behavior_Project/outputs/tracking/"
    "detector_cache_full_frame_standard_v2_20260728_retry1"
)
FIXED_REPEAT_FRAMES = (
    0,
    1,
    2,
    224,
    225,
    450,
    728,
    735,
    790,
    846,
    899,
    900,
    901,
    1079,
    1092,
    1188,
    1195,
    1350,
    1797,
    1798,
    1799,
)
DETERMINISM_ATOL = 1e-5


class HistoricalCacheError(RuntimeError):
    """Fail closed when a frozen detector-cache authority gate changes."""


@dataclass(frozen=True)
class VideoAuthority:
    """One exact source and population-binding record."""

    video_key: str
    video_path: Path
    video_sha256: str
    gt_path: Path
    gt_sha256: str
    frame_count: int
    width: int
    height: int


def sha256_file(path: Path) -> str:
    """Hash one file without mutating it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    """Hash one JSON-compatible semantic payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HistoricalCacheError(f"Expected JSON object: {path}")
    return payload


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write a deterministic CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_output(repo: Path, *args: str) -> str:
    """Run one read-only Git query."""

    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def utc_now() -> str:
    """Return a stable UTC timestamp string."""

    return datetime.now(UTC).isoformat()


def script_sha256() -> str:
    """Hash this exact producer helper."""

    return sha256_file(Path(__file__).resolve())


def environment_payload() -> dict[str, Any]:
    """Capture the complete executable detector environment."""

    import cv2
    import torch
    import ultralytics

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "ultralytics": ultralytics.__version__,
        "historical_run_python": (
            "3.11.9 (tags/v3.11.9:de54cf5, Apr 2 2024, 10:12:12) "
            "[MSC v.1938 64 bit (AMD64)]"
        ),
        "historical_ultralytics_version": "UNRESOLVED",
        "historical_torch_cuda_cudnn_versions": "UNRESOLVED",
    }


def effective_config(source_repo: Path, environment: dict[str, Any]) -> dict[str, Any]:
    """Return the full detector-stage semantic configuration."""

    weights = source_repo / "models" / "detector" / "pig_detector_yolov8.pt"
    mask = source_repo / "data" / "annotations" / "scene" / "mask.png"
    return {
        "schema_version": "tracking.historical_h5b_h4.detector_config.v1",
        "date": DATE,
        "historical_best_run_id": HISTORICAL_RUN_ID,
        "historical_best_source_sha": HISTORICAL_SOURCE_SHA,
        "detector_producer_topology": PRODUCER_TOPOLOGY,
        "producer_git_sha": git_output(REPO, "rev-parse", "HEAD"),
        "producer_script_path": str(Path(__file__).resolve()),
        "producer_script_sha256": script_sha256(),
        "detector_code_authority": {
            "historical_masks_git_blob": (
                "5083aafae073b8f6ad113ce2f51d5ec044bb4297"
            ),
            "current_masks_git_blob": (
                "5083aafae073b8f6ad113ce2f51d5ec044bb4297"
            ),
            "historical_runner_git_blob": (
                "186c3924539bc10c3192600441fd5e334398de11"
            ),
            "current_runner_git_blob": (
                "2b7f451a7333125accb3f9f081a29011d675b362"
            ),
            "historical_hybrid_profile_git_blob": (
                "97470ab98db2bcb77564c2a666b9fe2cafd822e7"
            ),
            "detector_cache_schema_sha256": sha256_file(
                REPO / "src" / "pig_behavior" / "tracking" / "detector_cache.py"
            ),
            "mask_code_sha256": sha256_file(
                REPO / "src" / "pig_behavior" / "tracking" / "masks.py"
            ),
        },
        "weights": {
            "path": str(weights),
            "sha256": sha256_file(weights),
        },
        "mask": {
            "path": str(mask),
            "sha256": sha256_file(mask),
            "read_mode": "cv2.IMREAD_GRAYSCALE",
            "threshold": 127,
            "dilation": {
                "kernel": "MORPH_ELLIPSE",
                "roi_dilate_px": 8,
                "iterations": 1,
            },
            "resize": "INTER_NEAREST_WHEN_REQUIRED",
            "application": "cv2.bitwise_and",
            "mask_input_frame": True,
        },
        "detector": {
            "call": "ultralytics.YOLO.predict",
            "confidence_threshold": CONFIDENCE,
            "maximum_raw_detections": MAX_RAW_DETECTIONS,
            "nms_iou": NMS_IOU,
            "image_size": IMAGE_SIZE,
            "class_policy": "ALL_MODEL_CLASSES_SINGLE_PIG_MODEL",
            "class_id_filter": None,
            "allowed_class_name": None,
            "frame_cadence": "EVERY_FRAME",
            "device": "cuda:0",
            "precision": "FP32",
            "half": False,
            "sorting": "ULTRALYTICS_NATIVE_RESULT_ORDER",
            "bbox_coordinates": (
                "xyxy_absolute_float32_original_decoded_frame_coordinates"
            ),
            "preprocessing": {
                "decode": "OpenCV_VideoCapture_sequential",
                "source_color_order": "OpenCV_BGR_as_read",
                "letterbox": "ultralytics_library_default_for_imgsz_640",
                "normalization": "ultralytics_library_default",
                "autocast": "library_default_with_half_false",
            },
        },
        "cache": {
            "schema": DETECTOR_CACHE_SCHEMA_VERSION,
            "zero_detection_representation": (
                "empty float32 xyxy/conf arrays and int64 cls array"
            ),
            "frame_indexing": "ZERO_BASED_INCLUSIVE_0_TO_1799",
            "frame_timestamps": "NOT_USED",
            "serialization": "numpy savez_compressed deterministic schema",
        },
        "environment": environment,
    }


def _video_population_document(worktree_repo: Path) -> dict[str, Any]:
    path = (
        worktree_repo
        / "docs"
        / "tracking"
        / "development_2x2_standard_v2"
        / "DEVELOPMENT_2X2_POPULATION_MANIFEST_20260728.json"
    )
    return load_json(path)


def _population_path_authority(
    worktree_repo: Path,
) -> dict[str, dict[str, Any]]:
    path = (
        worktree_repo
        / "docs"
        / "tracking"
        / "b0_b1_r0_standard_v2"
        / "B0_B1_R0_STANDARD_V2_POPULATION_MANIFEST_20260728.json"
    )
    payload = load_json(path)
    return {
        str(record["video_key"]): record
        for record in payload["videos"]
    }


def load_population(
    source_repo: Path,
    worktree_repo: Path,
) -> list[VideoAuthority]:
    """Load and revalidate the historical Standard-V2 full-13 population."""

    import cv2

    payload = _video_population_document(worktree_repo)
    path_authority = _population_path_authority(worktree_repo)
    if payload.get("video_count") != EXPECTED_VIDEOS:
        raise HistoricalCacheError("Population is not exact full-13")
    rows: list[VideoAuthority] = []
    for record in payload["videos"]:
        video_key = str(record["video_key"])
        authority_record = path_authority[video_key]
        video_path = (
            source_repo
            / "data"
            / "videos"
            / Path(authority_record["source_video_path"]).name
        )
        gt_path = (
            source_repo
            / "data"
            / "annotations"
            / "tracking"
            / Path(authority_record["gt_path"]).name
        )
        if sha256_file(video_path) != record["source_video_sha256"]:
            raise HistoricalCacheError(f"Source hash mismatch: {video_key}")
        if sha256_file(gt_path) != record["gt_sha256"]:
            raise HistoricalCacheError(f"GT hash mismatch: {video_key}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise HistoricalCacheError(f"Cannot decode: {video_key}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if (
            count != EXPECTED_FRAMES
            or int(record["frame_start"]) != 0
            or int(record["frame_end"]) != EXPECTED_FRAMES - 1
            or width <= 0
            or height <= 0
        ):
            raise HistoricalCacheError(f"Frame authority mismatch: {video_key}")
        rows.append(
            VideoAuthority(
                video_key=video_key,
                video_path=video_path,
                video_sha256=str(record["source_video_sha256"]),
                gt_path=gt_path,
                gt_sha256=str(record["gt_sha256"]),
                frame_count=count,
                width=width,
                height=height,
            )
        )
    return sorted(rows, key=lambda item: item.video_key)


def detector_cfg(config: dict[str, Any]) -> SimpleNamespace:
    """Build only the proven mask and detector settings."""

    return SimpleNamespace(
        weights_path=Path(config["weights"]["path"]),
        mask_path=Path(config["mask"]["path"]),
        use_mask=True,
        mask_input_frame=True,
        roi_dilate_px=8,
        det_conf=CONFIDENCE,
        max_raw_detections=MAX_RAW_DETECTIONS,
        nms_iou=NMS_IOU,
        imgsz=IMAGE_SIZE,
        class_id=None,
        allowed_class_name=None,
        device="cuda:0",
        half=False,
    )


def cache_identity(
    video: VideoAuthority,
    config_sha256: str,
    producer_git_sha: str,
) -> DetectorCacheIdentity:
    """Bind one cache partition to exact source, config, and producer."""

    return DetectorCacheIdentity(
        video_key=video.video_key,
        source_video_sha256=video.video_sha256,
        detector_weight_sha256=WEIGHTS_SHA256,
        detector_semantic_config_sha256=config_sha256,
        producer_code_sha=producer_git_sha,
        creation_authority=CREATION_AUTHORITY,
    )


def cache_path(root: Path, video_key: str) -> Path:
    """Return one stable cache partition path."""

    return root / "cache" / video_key / "detector_evidence.npz"


def checkpoint_path(root: Path, video_key: str) -> Path:
    """Return one completed-partition checkpoint path."""

    return root / "manifests" / "checkpoints" / f"{video_key}.json"


def current_cache_path(video_key: str) -> Path:
    """Return the frozen current 0.25/32 full-cache partition."""

    return (
        CURRENT_CACHE_ROOT
        / "full"
        / "partitions"
        / video_key
        / "detector_evidence.npz"
    )


def invoke_detector(model: Any, frame: np.ndarray, cfg: Any) -> Any:
    """Run exactly one frozen detector invocation."""

    results = model.predict(
        source=frame,
        conf=cfg.det_conf,
        iou=cfg.nms_iou,
        max_det=cfg.max_raw_detections,
        imgsz=cfg.imgsz,
        verbose=False,
        device=cfg.device,
        half=cfg.half,
    )
    if not isinstance(results, (list, tuple)) or len(results) != 1:
        raise HistoricalCacheError("Detector must return exactly one result")
    return results[0]


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


def cache_content_hash(cache: DetectorEvidenceCache) -> str:
    """Hash exact detector evidence independent of cache identity."""

    digest = hashlib.sha256()
    _update_hash_text(digest, CONTENT_HASH_CONTRACT)
    _update_hash_text(
        digest,
        json.dumps(cache.names, sort_keys=True, separators=(",", ":")),
    )
    for frame_index, entry in cache.frames.items():
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


def _pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """Return the strict upper-triangle pairwise IoU values."""

    if len(boxes) < 2:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    union = areas[:, None] + areas[None, :] - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
    return iou[np.triu_indices(len(boxes), k=1)]


def validate_cache(
    video: VideoAuthority,
    cache: DetectorEvidenceCache,
) -> dict[str, Any]:
    """Validate coverage, confidence, NMS, ordering, and coordinates."""

    if tuple(cache.frames) != tuple(range(EXPECTED_FRAMES)):
        raise HistoricalCacheError(f"Coverage failed: {video.video_key}")
    zero_frames = 0
    total_detections = 0
    maximum_count = 0
    cap_frames = 0
    for frame_index, entry in cache.frames.items():
        boxes = entry["xyxy"]
        confidence = entry["conf"]
        classes = entry["cls"]
        count = len(boxes)
        total_detections += count
        maximum_count = max(maximum_count, count)
        zero_frames += int(count == 0)
        cap_frames += int(count >= MAX_RAW_DETECTIONS)
        if count > MAX_RAW_DETECTIONS:
            raise HistoricalCacheError(
                f"max_det exceeded: {video.video_key} {frame_index}"
            )
        if (
            not np.all(np.isfinite(boxes))
            or not np.all(np.isfinite(confidence))
            or not np.all(np.isfinite(classes))
        ):
            raise HistoricalCacheError(
                f"Non-finite detector row: {video.video_key} {frame_index}"
            )
        if np.any(confidence < CONFIDENCE - 1e-6):
            raise HistoricalCacheError(
                f"Confidence contract failed: {video.video_key} {frame_index}"
            )
        if (
            np.any(boxes[:, 0] < 0)
            or np.any(boxes[:, 1] < 0)
            or np.any(boxes[:, 2] > video.width)
            or np.any(boxes[:, 3] > video.height)
            or np.any(boxes[:, 2] < boxes[:, 0])
            or np.any(boxes[:, 3] < boxes[:, 1])
        ):
            raise HistoricalCacheError(
                f"Bbox contract failed: {video.video_key} {frame_index}"
            )
        for class_value in np.unique(classes):
            class_boxes = boxes[classes == class_value]
            if np.any(_pairwise_iou(class_boxes) > NMS_IOU + 1e-4):
                raise HistoricalCacheError(
                    f"NMS contract failed: {video.video_key} {frame_index}"
                )
    replay = ReplayDetector(cache)
    for frame_index, entry in cache.frames.items():
        replay.set_frame_context(
            frame_index,
            entry["original_frame_dimensions"],
        )
        replay.predict()
    if replay.invocations != EXPECTED_FRAMES:
        raise HistoricalCacheError(f"Replay failed: {video.video_key}")
    return {
        "video_key": video.video_key,
        "frame_count": len(cache.frames),
        "minimum_frame": min(cache.frames),
        "maximum_frame": max(cache.frames),
        "total_detections": total_detections,
        "maximum_detections_in_frame": maximum_count,
        "zero_detection_frames": zero_frames,
        "frames_at_64_candidate_cap": cap_frames,
        "canonical_content_hash": cache_content_hash(cache),
        "cache_replay": "PASS",
        "schema_validation": "PASS",
        "confidence_validation": "PASS",
        "nms_validation": "PASS",
        "bbox_validation": "PASS",
    }


def producer_audit_document() -> dict[str, Any]:
    """Record the detector-stage historical/current semantic audit."""

    return {
        "schema_version": "tracking.historical_h5b_h4.producer_audit.v1",
        "date": DATE,
        "historical_source_sha": HISTORICAL_SOURCE_SHA,
        "topology": PRODUCER_TOPOLOGY,
        "scope": "RAW_YOLO_DETECTOR_STAGE_BEFORE_TRACKER_ASSOCIATION",
        "operations": [
            {
                "operation": "sequential OpenCV video decode",
                "historical": "cv2.VideoCapture/read zero-based loop",
                "current": "cv2.VideoCapture/read zero-based loop",
                "status": "SEMANTIC_MATCH",
            },
            {
                "operation": "mask load, threshold, dilation, resize, apply",
                "historical": "Git blob 5083aafae073b8f6ad113ce2f51d5ec044bb4297",
                "current": "Git blob 5083aafae073b8f6ad113ce2f51d5ec044bb4297",
                "status": "EXACT_SOURCE_BLOB_MATCH",
            },
            {
                "operation": "YOLO preprocessing and coordinate restoration",
                "historical": "Ultralytics library detector stage",
                "current": "Ultralytics library detector stage",
                "status": "SEMANTIC_MATCH_VERSION_BOUND_CURRENT_RUN",
            },
            {
                "operation": "confidence, NMS, max detections, image size",
                "historical": "0.20 / 0.50 / 64 / 640 authority",
                "current": "explicit 0.20 / 0.50 / 64 / 640 arguments",
                "status": "EXACT_EFFECTIVE_CONFIG_MATCH",
            },
            {
                "operation": "precision and device",
                "historical": "FP32 automatic CUDA; exact packages unresolved",
                "current": "FP32 cuda:0 with fully recorded packages",
                "status": "SEMANTIC_MATCH_ENVIRONMENT_PARITY_UNPROVEN",
            },
            {
                "operation": "cache serialization",
                "historical": "raw cache unavailable",
                "current": DETECTOR_CACHE_SCHEMA_VERSION,
                "status": "NEW_REPRODUCTION_AUTHORITY_REPRESENTATION",
            },
        ],
        "scientific_limitations": [
            "The original historical raw detector cache is unavailable.",
            "Historical package versions beyond Python are unresolved.",
            "The original hybrid runner called model.track; this cache freezes "
            "the raw detector stage only and invokes no tracker.",
            "Exact historical detector-row parity is not proven.",
            "Final historical XML parity is deferred to a separate task.",
        ],
    }


def preflight(
    source_repo: Path,
    worktree_repo: Path,
    historical_repo: Path,
    output_root: Path,
    docs_root: Path,
) -> dict[str, Any]:
    """Freeze producer, config, population, and planned run before inference."""

    if output_root.exists():
        raise HistoricalCacheError(f"Refusing existing root: {output_root}")
    head = git_output(worktree_repo, "rev-parse", "HEAD")
    if git_output(worktree_repo, "status", "--short"):
        raise HistoricalCacheError("Producer worktree must be clean")
    if git_output(historical_repo, "status", "--porcelain=v1", "-uall"):
        raise HistoricalCacheError("Historical worktree is unexpectedly dirty")
    historical_head = git_output(historical_repo, "rev-parse", "HEAD")
    environment = environment_payload()
    if not environment["cuda_available"]:
        raise HistoricalCacheError("CUDA detector environment is unavailable")
    videos = load_population(source_repo, worktree_repo)
    config = effective_config(source_repo, environment)
    if config["weights"]["sha256"] != WEIGHTS_SHA256:
        raise HistoricalCacheError("Detector weights authority mismatch")
    if config["mask"]["sha256"] != MASK_SHA256:
        raise HistoricalCacheError("Detector mask authority mismatch")
    config_sha = canonical_hash(config)
    execution_rows = [
        {
            "video_key": video.video_key,
            "source_path": str(video.video_path),
            "source_sha256": video.video_sha256,
            "frame_start": 0,
            "frame_end": EXPECTED_FRAMES - 1,
            "expected_frame_count": EXPECTED_FRAMES,
            "decoder_policy": "OPENCV_SEQUENTIAL_NO_CONTENT_SKIP",
            "historical_gt_sha256_population_binding_only": video.gt_sha256,
            "inclusion_role": "LOCKED_DEVELOPMENT_FULL13",
        }
        for video in videos
    ]
    execution = {
        "schema_version": "tracking.historical_h5b_h4.execution_manifest.v1",
        "date": DATE,
        "video_count": len(videos),
        "frames_per_video": EXPECTED_FRAMES,
        "total_frames": sum(video.frame_count for video in videos),
        "common_source_video_authority": "PASS",
        "common_frame_authority": "PASS",
        "gt_consumed_by_detector_inference": False,
        "determinism_policy": {
            "status_target": "STRATIFIED_REPEAT_DETERMINISM",
            "fixed_frames_per_video": list(FIXED_REPEAT_FRAMES),
            "structural_additions": [
                "all zero-detection frames",
                "all frames with at least 60 detections",
            ],
            "selection_uses_tracking_metrics": False,
            "bbox_confidence_absolute_tolerance": DETERMINISM_ATOL,
            "count_class_order_tolerance": "EXACT",
        },
        "videos": execution_rows,
    }
    output_root.mkdir(parents=True)
    for directory in ("cache", "manifests", "audits", "commands", "environment"):
        (output_root / directory).mkdir()
    audit = producer_audit_document()
    preflight_payload = {
        "schema_version": "tracking.historical_h5b_h4.cache_preflight.v1",
        "date": DATE,
        "status": "PASS",
        "requested_starting_main_sha": REQUESTED_STARTING_MAIN_SHA,
        "resolved_starting_main_sha": RESOLVED_STARTING_MAIN_SHA,
        "producer_git_sha": head,
        "producer_script_sha256": script_sha256(),
        "historical_worktree_head": historical_head,
        "historical_worktree_status": "",
        "detector_producer_topology": PRODUCER_TOPOLOGY,
        "effective_config_sha256": config_sha,
        "video_count": len(videos),
        "total_frames": sum(video.frame_count for video in videos),
        "output_root": str(output_root),
        "cache_root_absent_before_preflight": True,
        "selected_skills": [
            "tracking-experiment-guardian",
            "experiment-lineage-reproducibility",
            "computer-vision-opencv",
        ],
        "execution_counts": {
            "tracker_executions": 0,
            "repair_invocations": 0,
            "evaluator_runs": 0,
            "prediction_files_generated": 0,
            "unseen_files_accessed": 0,
        },
    }
    write_json(output_root / "manifests" / "PREFLIGHT.json", preflight_payload)
    write_json(
        output_root
        / "manifests"
        / f"HISTORICAL_H5B_H4_DETECTOR_EFFECTIVE_CONFIG_{DATE}.json",
        config,
    )
    write_json(
        output_root
        / "manifests"
        / f"HISTORICAL_H5B_H4_DETECTOR_EXECUTION_MANIFEST_{DATE}.json",
        execution,
    )
    write_json(
        output_root
        / "audits"
        / f"HISTORICAL_DETECTOR_PRODUCER_AUDIT_{DATE}.json",
        audit,
    )
    write_json(
        output_root / "environment" / "DETECTOR_ENVIRONMENT.json",
        environment,
    )
    (output_root / "commands" / "COMMANDS.txt").write_text(
        "PREFLIGHT\n"
        + subprocess.list2cmdline(sys.argv)
        + "\nTRACKER_EXECUTIONS=0\nREPAIR_INVOCATIONS=0\n"
        + "EVALUATOR_RUNS=0\nPREDICTION_FILES_GENERATED=0\n",
        encoding="utf-8",
    )
    docs_root.mkdir(parents=True, exist_ok=True)
    write_json(
        docs_root / f"HISTORICAL_DETECTOR_PRODUCER_AUDIT_{DATE}.json",
        audit,
    )
    write_json(
        docs_root
        / f"HISTORICAL_H5B_H4_DETECTOR_EFFECTIVE_CONFIG_{DATE}.json",
        config | {"effective_config_sha256": config_sha},
    )
    write_json(
        docs_root
        / f"HISTORICAL_H5B_H4_DETECTOR_EXECUTION_MANIFEST_{DATE}.json",
        execution,
    )
    return preflight_payload


def load_preflight(
    source_repo: Path,
    worktree_repo: Path,
    output_root: Path,
) -> tuple[list[VideoAuthority], dict[str, Any], str]:
    """Revalidate all frozen preflight inputs."""

    preflight_payload = load_json(output_root / "manifests" / "PREFLIGHT.json")
    config_path = (
        output_root
        / "manifests"
        / f"HISTORICAL_H5B_H4_DETECTOR_EFFECTIVE_CONFIG_{DATE}.json"
    )
    config = load_json(config_path)
    if sha256_file(Path(config["weights"]["path"])) != WEIGHTS_SHA256:
        raise HistoricalCacheError("Weights changed after preflight")
    if sha256_file(Path(config["mask"]["path"])) != MASK_SHA256:
        raise HistoricalCacheError("Mask changed after preflight")
    if canonical_hash(config) != preflight_payload["effective_config_sha256"]:
        raise HistoricalCacheError("Effective config changed after preflight")
    if script_sha256() != preflight_payload["producer_script_sha256"]:
        raise HistoricalCacheError("Producer script changed after preflight")
    videos = load_population(source_repo, worktree_repo)
    return videos, config, str(preflight_payload["producer_git_sha"])


def load_cache(
    path: Path,
    video: VideoAuthority,
    config_sha: str,
    producer_sha: str,
) -> DetectorEvidenceCache:
    """Load a cache under exact historical reproduction identity."""

    return DetectorEvidenceCache.load(
        path,
        expected_identity=cache_identity(video, config_sha, producer_sha),
    )


def record_video(
    video: VideoAuthority,
    cfg: Any,
    model: Any,
    output_root: Path,
    config_sha: str,
    producer_sha: str,
    attempts_before: int,
) -> tuple[DetectorEvidenceCache, int]:
    """Infer and transactionally save one complete video cache."""

    import cv2

    output = cache_path(output_root, video.video_key)
    checkpoint = checkpoint_path(output_root, video.video_key)
    if output.exists() or output.with_suffix(".sha256.json").exists():
        if not checkpoint.is_file():
            raise HistoricalCacheError(
                f"Partial cache lacks checkpoint: {video.video_key}"
            )
        cache = load_cache(
            output,
            video,
            config_sha,
            producer_sha,
        )
        if tuple(cache.frames) != tuple(range(EXPECTED_FRAMES)):
            raise HistoricalCacheError(
                f"Completed cache coverage changed: {video.video_key}"
            )
        return cache, 0
    cache = DetectorEvidenceCache(
        identity=cache_identity(video, config_sha, producer_sha)
    )
    capture = cv2.VideoCapture(str(video.video_path))
    if not capture.isOpened():
        raise HistoricalCacheError(f"Cannot open source: {video.video_key}")
    mask = load_mask(cfg.mask_path, video.width, video.height, cfg)
    calls = 0
    try:
        for frame_index in range(EXPECTED_FRAMES):
            ok, frame = capture.read()
            if not ok:
                raise HistoricalCacheError(
                    f"Decode failed: {video.video_key} {frame_index}"
                )
            decoded_position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if decoded_position != frame_index + 1:
                raise HistoricalCacheError(
                    f"Decoder index drift: {video.video_key} {frame_index}"
                )
            detector_frame = apply_mask_to_frame(frame, mask)
            result = invoke_detector(model, detector_frame, cfg)
            cache.record(
                frame_index,
                result,
                original_frame_dimensions=(video.height, video.width),
            )
            calls += 1
            if calls % 50 == 0:
                write_json(
                    output_root / "manifests" / "RUN_STATE.json",
                    {
                        "status": "RUNNING",
                        "phase": "FULL_FRAME_INFERENCE",
                        "video_key": video.video_key,
                        "last_processed_frame": frame_index,
                        "detector_inference_attempts": attempts_before + calls,
                        "updated_at": utc_now(),
                    },
                )
    finally:
        capture.release()
    if calls != EXPECTED_FRAMES:
        raise HistoricalCacheError(f"Inference coverage failed: {video.video_key}")
    artifact_sha = cache.save(output)
    write_json(
        checkpoint,
        {
            "status": "COMMITTED",
            "video_key": video.video_key,
            "frame_count": len(cache.frames),
            "cache_artifact_sha256": artifact_sha,
            "canonical_content_hash": cache_content_hash(cache),
            "detector_inference_calls": calls,
            "committed_at": utc_now(),
        },
    )
    return cache, calls


def _current_identity(video: VideoAuthority) -> DetectorCacheIdentity:
    return DetectorCacheIdentity(
        video_key=video.video_key,
        source_video_sha256=video.video_sha256,
        detector_weight_sha256=WEIGHTS_SHA256,
        detector_semantic_config_sha256=(
            "2b50d8afa950626e2bed6b41807cb602a01a90e66baf7529fa08945d3d676ef8"
        ),
        producer_code_sha="aadbf7787902e23b74d66353a076601059865a29",
        creation_authority=(
            "user-authorized full-frame detector cache completion for locked full-13"
        ),
    )


def difference_audit(
    videos: list[VideoAuthority],
    caches: dict[str, DetectorEvidenceCache],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Describe structural differences from current 0.25/32 evidence."""

    rows: list[dict[str, Any]] = []
    additional_confidences: list[float] = []
    affected_frames = 0
    cap_affected_frames = 0
    total_additional = 0
    for video in videos:
        current = DetectorEvidenceCache.load(
            current_cache_path(video.video_key),
            expected_identity=_current_identity(video),
        )
        historical = caches[video.video_key]
        for frame_index in range(EXPECTED_FRAMES):
            old_entry = current.frames[frame_index]
            new_entry = historical.frames[frame_index]
            old_count = len(old_entry["conf"])
            new_count = len(new_entry["conf"])
            additional = max(0, new_count - old_count)
            if additional:
                affected_frames += 1
                total_additional += additional
                ordered = np.sort(new_entry["conf"])
                additional_values = ordered[:additional]
                additional_confidences.extend(
                    float(value) for value in additional_values
                )
            cap_effect = new_count > 32
            cap_affected_frames += int(cap_effect)
            if additional or cap_effect:
                rows.append(
                    {
                        "video_key": video.video_key,
                        "frame_index": frame_index,
                        "current_025_32_count": old_count,
                        "historical_020_64_count": new_count,
                        "additional_candidate_count": additional,
                        "affected_by_32_vs_64_cap": cap_effect,
                    }
                )
    distribution = (
        {
            "count": len(additional_confidences),
            "minimum": min(additional_confidences),
            "median": float(np.median(additional_confidences)),
            "p95": float(np.quantile(additional_confidences, 0.95)),
            "maximum": max(additional_confidences),
        }
        if additional_confidences
        else {
            "count": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    )
    return (
        {
            "schema_version": "tracking.historical_h5b_h4.difference_audit.v1",
            "comparison_role": "DESCRIPTIVE_GT_FREE_ONLY",
            "quality_interpretation_authorized": False,
            "frames_with_more_detections": affected_frames,
            "total_additional_candidate_detections": total_additional,
            "additional_confidence_distribution": distribution,
            "frames_affected_by_32_vs_64_cap": cap_affected_frames,
        },
        rows,
    )


def repeat_frames(cache: DetectorEvidenceCache) -> tuple[int, ...]:
    """Apply the predeclared structural determinism subset policy."""

    selected = set(FIXED_REPEAT_FRAMES)
    for frame_index, entry in cache.frames.items():
        count = len(entry["conf"])
        if count == 0 or count >= 60:
            selected.add(frame_index)
    return tuple(sorted(selected))


def deterministic_repeat(
    video: VideoAuthority,
    cfg: Any,
    model: Any,
    cache: DetectorEvidenceCache,
) -> dict[str, Any]:
    """Repeat the predeclared subset and compare frozen detector rows."""

    import cv2

    indices = repeat_frames(cache)
    capture = cv2.VideoCapture(str(video.video_path))
    if not capture.isOpened():
        raise HistoricalCacheError(f"Repeat cannot open: {video.video_key}")
    mask = load_mask(cfg.mask_path, video.width, video.height, cfg)
    maximum_bbox_delta = 0.0
    maximum_conf_delta = 0.0
    try:
        for frame_index in range(EXPECTED_FRAMES):
            ok, frame = capture.read()
            if not ok:
                raise HistoricalCacheError(
                    f"Repeat decode failed: {video.video_key} {frame_index}"
                )
            if frame_index not in indices:
                continue
            result = invoke_detector(model, apply_mask_to_frame(frame, mask), cfg)
            boxes = result.boxes
            repeated_xyxy = np.asarray(
                boxes.xyxy.detach().cpu().numpy(), dtype=np.float32
            )
            repeated_conf = np.asarray(
                boxes.conf.detach().cpu().numpy(), dtype=np.float32
            )
            repeated_cls = np.asarray(
                boxes.cls.detach().cpu().numpy(), dtype=np.int64
            )
            frozen = cache.frames[frame_index]
            if (
                repeated_xyxy.shape != frozen["xyxy"].shape
                or repeated_conf.shape != frozen["conf"].shape
                or not np.array_equal(repeated_cls, frozen["cls"])
            ):
                raise HistoricalCacheError(
                    f"Repeat count/class/order failed: "
                    f"{video.video_key} {frame_index}"
                )
            if repeated_xyxy.size:
                bbox_delta = float(
                    np.max(np.abs(repeated_xyxy - frozen["xyxy"]))
                )
                conf_delta = float(
                    np.max(np.abs(repeated_conf - frozen["conf"]))
                )
                maximum_bbox_delta = max(maximum_bbox_delta, bbox_delta)
                maximum_conf_delta = max(maximum_conf_delta, conf_delta)
                if (
                    bbox_delta > DETERMINISM_ATOL
                    or conf_delta > DETERMINISM_ATOL
                ):
                    raise HistoricalCacheError(
                        f"Repeat tolerance failed: "
                        f"{video.video_key} {frame_index}"
                    )
    finally:
        capture.release()
    return {
        "video_key": video.video_key,
        "repeat_frame_count": len(indices),
        "repeat_frames": list(indices),
        "maximum_bbox_absolute_delta": maximum_bbox_delta,
        "maximum_confidence_absolute_delta": maximum_conf_delta,
        "count_class_order": "EXACT",
        "tolerance": DETERMINISM_ATOL,
        "status": "PASS",
    }


def artifact_inventory(
    root: Path,
    excluded_names: set[str],
) -> list[dict[str, Any]]:
    """Inventory all non-self-referential stable artifacts."""

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
    worktree_repo: Path,
    historical_repo: Path,
    output_root: Path,
    docs_root: Path,
) -> dict[str, Any]:
    """Generate, replay, repeat, compare, and freeze the cache authority."""

    import torch
    from ultralytics import YOLO

    started = utc_now()
    videos, config, producer_sha = load_preflight(
        source_repo,
        worktree_repo,
        output_root,
    )
    config_sha = canonical_hash(config)
    cfg = detector_cfg(config)
    if not torch.cuda.is_available():
        raise HistoricalCacheError("CUDA unavailable")
    model = YOLO(str(cfg.weights_path))
    model.to(cfg.device)
    attempts_before = 0
    state_path = output_root / "manifests" / "RUN_STATE.json"
    if state_path.is_file():
        attempts_before = int(
            load_json(state_path).get("detector_inference_attempts", 0)
        )
    caches: dict[str, DetectorEvidenceCache] = {}
    validations: list[dict[str, Any]] = []
    new_calls = 0
    for video in videos:
        print(f"CACHE_BEGIN {video.video_key}", flush=True)
        cache, calls = record_video(
            video,
            cfg,
            model,
            output_root,
            config_sha,
            producer_sha,
            attempts_before + new_calls,
        )
        new_calls += calls
        caches[video.video_key] = cache
        validation = validate_cache(video, cache)
        validation["cache_artifact_sha256"] = sha256_file(
            cache_path(output_root, video.video_key)
        )
        validations.append(validation)
        print(
            f"CACHE_END {video.video_key} frames={len(cache.frames)}",
            flush=True,
        )
    if sum(row["frame_count"] for row in validations) != EXPECTED_TOTAL:
        raise HistoricalCacheError("Unique frame count is not 23,400")
    difference, difference_rows = difference_audit(videos, caches)
    write_json(
        output_root / "audits" / "CURRENT_025_32_DIFFERENCE_SUMMARY.json",
        difference,
    )
    write_csv(
        output_root / "audits" / "CURRENT_025_32_DIFFERENCE_FRAMES.csv",
        difference_rows,
        [
            "video_key",
            "frame_index",
            "current_025_32_count",
            "historical_020_64_count",
            "additional_candidate_count",
            "affected_by_32_vs_64_cap",
        ],
    )
    repeats = []
    repeat_calls = 0
    for video in videos:
        print(f"REPEAT_BEGIN {video.video_key}", flush=True)
        result = deterministic_repeat(video, cfg, model, caches[video.video_key])
        repeats.append(result)
        repeat_calls += result["repeat_frame_count"]
        print(
            f"REPEAT_END {video.video_key} frames={result['repeat_frame_count']}",
            flush=True,
        )
    del model
    torch.cuda.empty_cache()
    determinism = {
        "schema_version": "tracking.historical_h5b_h4.determinism.v1",
        "status": "STRATIFIED_REPEAT_DETERMINISM=PASS",
        "policy": "FIXED_ALL13_PLUS_STRUCTURAL_ZERO_OR_NEAR_CAP",
        "tolerance": DETERMINISM_ATOL,
        "repeat_detector_inference_calls": repeat_calls,
        "per_video": repeats,
    }
    write_json(output_root / "audits" / "DETERMINISM.json", determinism)
    write_json(
        output_root / "audits" / "CACHE_VALIDATION.json",
        {
            "status": "PASS",
            "video_count": len(validations),
            "unique_authoritative_frame_records": sum(
                row["frame_count"] for row in validations
            ),
            "missing_frame_records": 0,
            "duplicate_authoritative_frame_records": 0,
            "per_video": validations,
        },
    )
    total_attempts = attempts_before + new_calls + repeat_calls
    retry_attempts = max(0, attempts_before)
    cache_manifest = {
        "schema_version": "tracking.historical_h5b_h4.cache_manifest.v1",
        "date": DATE,
        "status": "ESTABLISHED",
        "cache_root": str(output_root),
        "video_count": len(validations),
        "unique_authoritative_frame_records": EXPECTED_TOTAL,
        "missing_frame_records": 0,
        "duplicate_authoritative_frame_records": 0,
        "detector_inference_attempts": total_attempts,
        "primary_generation_inference_calls": new_calls,
        "determinism_repeat_inference_calls": repeat_calls,
        "detector_retry_attempts": retry_attempts,
        "effective_config_sha256": config_sha,
        "per_video": validations,
    }
    write_json(
        output_root
        / "manifests"
        / f"HISTORICAL_H5B_H4_DETECTOR_CACHE_MANIFEST_{DATE}.json",
        cache_manifest,
    )
    ended = utc_now()
    run_manifest = {
        "schema_version": "tracking.historical_h5b_h4.cache_run.v1",
        "status": "PASS",
        "started_at": started,
        "ended_at": ended,
        "exact_command": subprocess.list2cmdline(sys.argv),
        "producer_git_sha": producer_sha,
        "producer_script_sha256": script_sha256(),
        "effective_config_sha256": config_sha,
        "detector_inference_attempts": total_attempts,
        "primary_generation_inference_calls": new_calls,
        "determinism_repeat_inference_calls": repeat_calls,
        "detector_retry_attempts": retry_attempts,
        "errors": [],
        "execution_counts": {
            "tracker_executions": 0,
            "repair_invocations": 0,
            "evaluator_runs": 0,
            "prediction_files_generated": 0,
            "unseen_files_accessed": 0,
        },
    }
    write_json(output_root / "manifests" / "RUN_MANIFEST.json", run_manifest)
    with (output_root / "commands" / "COMMANDS.txt").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            "GENERATION_VALIDATION_AND_REPEAT\n"
            + subprocess.list2cmdline(sys.argv)
            + f"\nDETECTOR_INFERENCE_ATTEMPTS={total_attempts}\n"
            + f"DETECTOR_RETRY_ATTEMPTS={retry_attempts}\n"
            + "TRACKER_EXECUTIONS=0\nREPAIR_INVOCATIONS=0\n"
            + "EVALUATOR_RUNS=0\nPREDICTION_FILES_GENERATED=0\n"
        )
    excluded = {
        f"HISTORICAL_H5B_H4_DETECTOR_CACHE_INVENTORY_{DATE}.csv",
        f"HISTORICAL_H5B_H4_DETECTOR_CACHE_AUTHORITY_{DATE}.json",
        f"HISTORICAL_H5B_H4_DETECTOR_CACHE_DECISION_{DATE}.json",
    }
    inventory = artifact_inventory(output_root, excluded)
    inventory_path = (
        output_root
        / "manifests"
        / f"HISTORICAL_H5B_H4_DETECTOR_CACHE_INVENTORY_{DATE}.csv"
    )
    write_csv(
        inventory_path,
        inventory,
        ["relative_path", "size_bytes", "sha256"],
    )
    cache_authority_hash = canonical_hash(
        [
            {
                "video_key": row["video_key"],
                "cache_artifact_sha256": row["cache_artifact_sha256"],
                "canonical_content_hash": row["canonical_content_hash"],
            }
            for row in validations
        ]
    )
    authority = {
        "schema_version": "tracking.historical_h5b_h4.cache_authority.v1",
        "date": DATE,
        "status": "ESTABLISHED",
        "retention_class": (
            "NON_DISPOSABLE_FROZEN_HISTORICAL_METHOD_REPRODUCTION_INPUT"
        ),
        "deletion_allowed": "NO_WITHOUT_EXPLICIT_AUTHORITY_RETIREMENT",
        "historical_best_run_id": HISTORICAL_RUN_ID,
        "historical_best_source_sha": HISTORICAL_SOURCE_SHA,
        "detector_producer_topology": PRODUCER_TOPOLOGY,
        "cache_root": str(output_root),
        "cache_authority_sha256": cache_authority_hash,
        "effective_config_sha256": config_sha,
        "inventory_sha256": sha256_file(inventory_path),
        "video_count": len(validations),
        "unique_authoritative_frame_records": EXPECTED_TOTAL,
        "missing_frame_records": 0,
        "duplicate_authoritative_frame_records": 0,
        "detector_inference_attempts": total_attempts,
        "detector_retry_attempts": retry_attempts,
        "cache_validation": "PASS",
        "cache_replay": "PASS",
        "determinism": "STRATIFIED_REPEAT_DETERMINISM=PASS",
        "original_historical_raw_cache_available": False,
        "exact_historical_detector_row_parity_proven": False,
        "deterministic_config_reproduction_authority_established": True,
        "historical_final_prediction_parity": "DEFERRED_SEPARATE_TASK",
        "per_video": validations,
        "execution_counts": run_manifest["execution_counts"]
        | {"run_root_mp4_count": 0},
        "scientific_limitations": [
            "Original historical raw detector rows are unavailable.",
            "Historical package versions beyond Python are unresolved.",
            "This authority proves deterministic configured reproduction, "
            "not exact historical detector-row parity.",
            "Tracker, repair, evaluator, and prediction parity remain deferred.",
        ],
    }
    decision = {
        "schema_version": "tracking.historical_h5b_h4.cache_decision.v1",
        "date": DATE,
        "decision": "PASS_HISTORICAL_DETECTOR_CACHE_REPRODUCTION_FROZEN",
        "deterministic_config_reproduction_authority_established": True,
        "ready_for_historical_h5b_h4_executable_reproduction": True,
        "ready_for_unseen_data_authority_freeze": False,
        "ready_for_unseen_evaluation": False,
        "ready_to_promote": False,
        "blockers": [
            "Historical H5b/H4 tracker and repair prediction parity is not yet tested.",
            "Unseen data remains unauthorized.",
        ],
    }
    authority_path = (
        output_root
        / "manifests"
        / f"HISTORICAL_H5B_H4_DETECTOR_CACHE_AUTHORITY_{DATE}.json"
    )
    decision_path = (
        output_root
        / "manifests"
        / f"HISTORICAL_H5B_H4_DETECTOR_CACHE_DECISION_{DATE}.json"
    )
    write_json(authority_path, authority)
    write_json(decision_path, decision)
    write_json(
        docs_root / f"HISTORICAL_H5B_H4_DETECTOR_CACHE_MANIFEST_{DATE}.json",
        cache_manifest,
    )
    write_csv(
        docs_root / f"HISTORICAL_H5B_H4_DETECTOR_CACHE_INVENTORY_{DATE}.csv",
        [
            {
                "video_key": row["video_key"],
                "cache_artifact_sha256": row["cache_artifact_sha256"],
                "canonical_content_hash": row["canonical_content_hash"],
                "frame_count": row["frame_count"],
                "total_detections": row["total_detections"],
            }
            for row in validations
        ],
        [
            "video_key",
            "cache_artifact_sha256",
            "canonical_content_hash",
            "frame_count",
            "total_detections",
        ],
    )
    write_json(
        docs_root / f"HISTORICAL_H5B_H4_DETECTOR_CACHE_AUTHORITY_{DATE}.json",
        authority,
    )
    write_json(
        docs_root / f"HISTORICAL_H5B_H4_DETECTOR_CACHE_DECISION_{DATE}.json",
        decision,
    )
    if any(path.suffix.lower() == ".mp4" for path in output_root.rglob("*")):
        raise HistoricalCacheError("Cache authority root contains MP4")
    historical_head_after = git_output(historical_repo, "rev-parse", "HEAD")
    historical_status_after = git_output(
        historical_repo, "status", "--porcelain=v1", "-uall"
    )
    if (
        historical_head_after
        != load_json(output_root / "manifests" / "PREFLIGHT.json")[
            "historical_worktree_head"
        ]
        or historical_status_after
    ):
        raise HistoricalCacheError("Historical worktree changed")
    write_json(
        output_root / "manifests" / "RUN_STATE.json",
        {
            "status": "COMPLETED",
            "phase": "CACHE_AUTHORITY_FROZEN",
            "detector_inference_attempts": total_attempts,
            "updated_at": utc_now(),
        },
    )
    marker = output_root / "NON_DISPOSABLE_FROZEN_AUTHORITY_DO_NOT_DELETE.txt"
    marker.write_text(
        "NON_DISPOSABLE_FROZEN_HISTORICAL_METHOD_REPRODUCTION_INPUT\n"
        "Deletion requires explicit authority retirement.\n",
        encoding="utf-8",
    )
    for path in output_root.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
    return {
        "decision": decision,
        "authority": authority,
        "difference_audit": difference,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "generate"), required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--worktree-repo", type=Path, default=REPO)
    parser.add_argument("--historical-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=(
            REPO
            / "docs"
            / "tracking"
            / "historical_h5b_h4_reproduction"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise HistoricalCacheError("PYTHONDONTWRITEBYTECODE=1 is required")
    if args.phase == "preflight":
        result = preflight(
            args.source_repo.resolve(),
            args.worktree_repo.resolve(),
            args.historical_repo.resolve(),
            args.output_root.resolve(),
            args.docs_root.resolve(),
        )
    else:
        result = generate(
            args.source_repo.resolve(),
            args.worktree_repo.resolve(),
            args.historical_repo.resolve(),
            args.output_root.resolve(),
            args.docs_root.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
