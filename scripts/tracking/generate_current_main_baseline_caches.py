"""Freeze and generate exact detector caches for the current-main full baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

STARTING_MAIN_SHA = "64d835cbf1b25ecdef3a777a50f0b46db6c93f61"
SOURCE_LINEAGE_SHA256 = (
    "0cfb26acc7766e05c497d9efdfafa40dc92f2d5c527e0338b89602eef0838dfc"
)
CREATION_AUTHORITY = (
    "user-authorized current-main realtime_fast full-13 baseline cache generation"
)
MINIMUM_FREE_BYTES = 4 * 1024**3


class BaselineCacheError(RuntimeError):
    """Fail-closed cache freeze or generation error."""


@dataclass(frozen=True, slots=True)
class VideoAuthority:
    video_key: str
    video_path: Path
    video_sha256: str
    gt_path: Path
    gt_sha256: str
    frame_count: int
    width: int
    height: int
    gt_authority: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def locked_lineage_payload_hash(payload: dict[str, Any]) -> str:
    authority = payload.get("manifest_sha256")
    if not isinstance(authority, str):
        raise BaselineCacheError("locked lineage manifest lacks payload authority")
    without_self_hash = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    calculated = canonical_hash(without_self_hash)
    if calculated != authority or calculated != SOURCE_LINEAGE_SHA256:
        raise BaselineCacheError("locked source-lineage payload hash mismatch")
    return calculated


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO,
        text=True,
        encoding="utf-8",
    ).strip()


def require_lineage() -> tuple[str, str]:
    head = git_output("rev-parse", "HEAD")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_MAIN_SHA, head],
        cwd=REPO,
        check=False,
    )
    if result.returncode != 0:
        raise BaselineCacheError("starting main is not an ancestor of producer")
    starting_tree = git_output(
        "rev-parse",
        f"{STARTING_MAIN_SHA}:src/pig_behavior/tracking",
    )
    current_tree = git_output("rev-parse", "HEAD:src/pig_behavior/tracking")
    if current_tree != starting_tree:
        raise BaselineCacheError("tracking tree changed after baseline authority")
    if git_output("status", "--short"):
        raise BaselineCacheError("cache producer worktree must be clean")
    return head, current_tree


def detector_configuration(source_repo: Path) -> tuple[Any, dict[str, Any]]:
    from pig_behavior.tracking.config import TrackingConfig, validate_config
    from pig_behavior.tracking.profiles.realtime import EVAL_CONFIGS

    cfg = TrackingConfig(
        mode="realtime",
        video_path=source_repo / "data" / "videos" / "Pigs281119_000085_30fps.mp4",
        weights_path=source_repo / "models" / "detector" / "pig_detector_yolov8.pt",
        mask_path=source_repo / "data" / "annotations" / "scene" / "mask.png",
        output_dir=source_repo / "outputs" / "unused_baseline_cache",
        device="cuda:0",
        half=False,
        write_output_video=False,
        **EVAL_CONFIGS["realtime_fast"],
    )
    validate_config(cfg)
    if not cfg.weights_path.is_file():
        raise BaselineCacheError(f"detector weights missing: {cfg.weights_path}")
    if not cfg.mask_path or not Path(cfg.mask_path).is_file():
        raise BaselineCacheError("scene mask is missing")
    payload = {
        "mode": cfg.mode,
        "det_conf": cfg.det_conf,
        "nms_iou": cfg.nms_iou,
        "max_raw_detections": cfg.max_raw_detections,
        "imgsz": cfg.imgsz,
        "class_id": cfg.class_id,
        "allowed_class_name": cfg.allowed_class_name,
        "use_mask": cfg.use_mask,
        "mask_input_frame": cfg.mask_input_frame,
        "mask_sha256": sha256_file(Path(cfg.mask_path)),
        "roi_dilate_px": cfg.roi_dilate_px,
        "detect_every_n_frames": cfg.detect_every_n_frames,
        "half": cfg.half,
        "preprocessing": {
            "source_color_order": "OpenCV_BGR_as_read",
            "mask_read_mode": "IMREAD_GRAYSCALE",
            "mask_threshold": 127,
            "mask_dilation": "MORPH_ELLIPSE",
            "mask_application": "cv2.bitwise_and",
            "ultralytics_letterbox": "library_default_for_imgsz",
        },
        "detector_call": "ultralytics.YOLO.predict",
    }
    return cfg, payload


def effective_config_payload() -> dict[str, Any]:
    from pig_behavior.tracking.profiles.realtime import EVAL_CONFIGS

    config = EVAL_CONFIGS["realtime_fast"]
    return {key: config[key] for key in sorted(config)}


def load_population(
    source_repo: Path,
    lineage_manifest: Path,
) -> tuple[list[VideoAuthority], str]:
    import cv2

    payload = json.loads(lineage_manifest.read_text(encoding="utf-8"))
    locked_lineage_payload_hash(payload)
    if len(payload.get("videos", [])) != 13:
        raise BaselineCacheError("locked baseline population is not 13 videos")
    videos: list[VideoAuthority] = []
    for row in payload["videos"]:
        video_key = str(row["canonical_video_key"])
        video_path = source_repo / "data" / "videos" / f"{video_key}.mp4"
        gt_name = Path(str(row["gt_path"])).name
        gt_path = source_repo / "data" / "annotations" / "tracking" / gt_name
        if not video_path.is_file() or sha256_file(video_path) != row["video_sha256"]:
            raise BaselineCacheError(f"source authority mismatch: {video_key}")
        if not gt_path.is_file() or sha256_file(gt_path) != row["gt_sha256"]:
            raise BaselineCacheError(f"GT authority mismatch: {video_key}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise BaselineCacheError(f"cannot open source video: {video_key}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if frame_count != 1800 or width <= 0 or height <= 0:
            raise BaselineCacheError(f"unexpected media authority: {video_key}")
        videos.append(
            VideoAuthority(
                video_key=video_key,
                video_path=video_path,
                video_sha256=str(row["video_sha256"]),
                gt_path=gt_path,
                gt_sha256=str(row["gt_sha256"]),
                frame_count=frame_count,
                width=width,
                height=height,
                gt_authority=(
                    "UNRESOLVED_EXCLUDE_FROM_MECHANISM_RANKING"
                    if "_000216_" in video_key
                    else "AUTHORITATIVE_FOR_MECHANISTIC_CONCLUSIONS"
                ),
            )
        )
    return sorted(videos, key=lambda item: item.video_key), sha256_file(
        lineage_manifest
    )


def cache_path(cache_root: Path, video_key: str) -> Path:
    return cache_root / "partitions" / video_key / "detector_evidence.npz"


def frame_indices(video: VideoAuthority, cadence: int) -> tuple[int, ...]:
    return tuple(range(0, video.frame_count, cadence))


def environment_payload() -> dict[str, Any]:
    import cv2
    import numpy
    import torch
    import ultralytics

    return {
        "python": sys.version,
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
    }


def preflight(
    source_repo: Path,
    lineage_manifest: Path,
    cache_root: Path,
    baseline_root: Path,
) -> None:
    from pig_behavior.tracking.detector_cache import DETECTOR_CACHE_SCHEMA_VERSION

    if cache_root.exists():
        raise BaselineCacheError(f"refusing existing cache root: {cache_root}")
    if baseline_root.exists():
        raise BaselineCacheError(f"refusing existing baseline root: {baseline_root}")
    producer_sha, tracking_tree = require_lineage()
    videos, lineage_sha = load_population(source_repo, lineage_manifest)
    cfg, detector_payload = detector_configuration(source_repo)
    environment = environment_payload()
    if not environment["cuda_available"]:
        raise BaselineCacheError("authorized cache-generation GPU is unavailable")
    if shutil.disk_usage(cache_root.parent).free < MINIMUM_FREE_BYTES:
        raise BaselineCacheError("insufficient disk space")
    cache_root.mkdir(parents=True)
    coverage = []
    partitions = []
    for video in videos:
        indices = frame_indices(video, cfg.detect_every_n_frames)
        partition_id = video.video_key
        for frame_index in indices:
            coverage.append(
                {
                    "video_key": video.video_key,
                    "source_video_sha256": video.video_sha256,
                    "frame_index": frame_index,
                    "detector_frame_required": "true",
                    "cache_partition_id": partition_id,
                }
            )
        partitions.append(
            {
                "cache_partition_id": partition_id,
                "video_key": video.video_key,
                "source_video_sha256": video.video_sha256,
                "first_frame_index": indices[0],
                "last_frame_index": indices[-1],
                "requested_frame_count": len(indices),
                "coverage_sha256": canonical_hash(indices),
                "cache_path": str(cache_path(cache_root, video.video_key)),
                "cache_artifact_sha256": "",
                "status": "PLANNED",
            }
        )
    write_csv(
        cache_root / "CURRENT_MAIN_DETECTOR_CACHE_COVERAGE.csv",
        coverage,
        [
            "video_key",
            "source_video_sha256",
            "frame_index",
            "detector_frame_required",
            "cache_partition_id",
        ],
    )
    write_csv(
        cache_root / "CURRENT_MAIN_DETECTOR_CACHE_PARTITIONS.csv",
        partitions,
        list(partitions[0]),
    )
    population = [
        {
            "video_key": video.video_key,
            "source_video_path": str(video.video_path),
            "source_video_sha256": video.video_sha256,
            "gt_path": str(video.gt_path),
            "gt_sha256": video.gt_sha256,
            "start_frame": 0,
            "end_frame": video.frame_count - 1,
            "aggregate_role": "LOCKED_FULL13_BASELINE",
            "gt_authority": video.gt_authority,
            "mechanism_ranking_excluded": "_000216_" in video.video_key,
            "cache_authority": "CURRENT_MAIN_EXACT_FULL_VIDEO_CACHE",
        }
        for video in videos
    ]
    write_csv(
        cache_root / "CURRENT_MAIN_BASELINE_FROZEN_VIDEO_MANIFEST.csv",
        population,
        list(population[0]),
    )
    detector_sha = canonical_hash(detector_payload)
    weights_sha = sha256_file(Path(cfg.weights_path))
    config_payload = effective_config_payload()
    preflight_payload = {
        "schema_version": "tracking.current_main_baseline_cache_preflight.v1",
        "cache_preflight": "PASS",
        "starting_main_sha": STARTING_MAIN_SHA,
        "producer_code_sha": producer_sha,
        "tracking_tree_object": tracking_tree,
        "source_lineage_sha256": lineage_sha,
        "baseline_videos_resolved": "13/13",
        "cache_partitions": 13,
        "cache_frames_requested": len(coverage),
        "detector_weight_sha256": weights_sha,
        "detector_semantic_config": detector_payload,
        "detector_semantic_config_sha256": detector_sha,
        "effective_realtime_fast_config": config_payload,
        "effective_realtime_fast_config_sha256": canonical_hash(config_payload),
        "cache_schema": DETECTOR_CACHE_SCHEMA_VERSION,
        "include_hidden": True,
        "output_timing_contract": "causal_framewise",
        "output_delay_frames": 0,
        "future_frames_used": False,
        "offline_repair": False,
        "post_video_smoothing": False,
        "h1_h2_validation_roles_consumed": False,
        "h1_h2_validation_execution": False,
        "population_use": "ROLE_BLIND_CURRENT_MAIN_BASELINE_ONLY",
        "mechanism_ranking_exclusions": ["Pigs281119_000216_30fps"],
        "baseline_root_absent": True,
        "cache_root_absent_before_preflight": True,
    }
    write_json(
        cache_root / "CURRENT_MAIN_DETECTOR_CACHE_PREFLIGHT.json",
        preflight_payload,
    )
    write_json(
        cache_root / "CURRENT_MAIN_DETECTOR_CACHE_ENVIRONMENT.json",
        environment,
    )
    (cache_root / "CURRENT_MAIN_DETECTOR_CACHE_COMMANDS.txt").write_text(
        "PREFLIGHT\n"
        + subprocess.list2cmdline(sys.argv)
        + "\nTRACKING_RUNS=0\nVALIDATION_EXECUTIONS=0\n",
        encoding="utf-8",
    )


def load_preflight(
    source_repo: Path,
    lineage_manifest: Path,
    cache_root: Path,
) -> tuple[list[VideoAuthority], Any, dict[str, Any]]:
    payload = json.loads(
        (cache_root / "CURRENT_MAIN_DETECTOR_CACHE_PREFLIGHT.json").read_text(
            encoding="utf-8"
        )
    )
    if payload.get("cache_preflight") != "PASS":
        raise BaselineCacheError("cache preflight did not pass")
    producer_sha, tracking_tree = require_lineage()
    if payload["producer_code_sha"] != producer_sha:
        raise BaselineCacheError("producer SHA changed after preflight")
    if payload["tracking_tree_object"] != tracking_tree:
        raise BaselineCacheError("tracking tree changed after preflight")
    videos, lineage_sha = load_population(source_repo, lineage_manifest)
    if payload["source_lineage_sha256"] != lineage_sha:
        raise BaselineCacheError("source lineage changed after preflight")
    cfg, detector_payload = detector_configuration(source_repo)
    if canonical_hash(detector_payload) != payload[
        "detector_semantic_config_sha256"
    ]:
        raise BaselineCacheError("detector semantic configuration changed")
    if sha256_file(Path(cfg.weights_path)) != payload["detector_weight_sha256"]:
        raise BaselineCacheError("detector weights changed")
    return videos, cfg, payload


def invoke_detector(model: Any, frame: np.ndarray, cfg: Any) -> Any:
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
        raise BaselineCacheError("detector must return exactly one result")
    return results[0]


def record_video(
    video: VideoAuthority,
    cfg: Any,
    model: Any,
    cache_root: Path,
    producer_sha: str,
    detector_sha: str,
    weights_sha: str,
) -> tuple[str, int]:
    import cv2

    from pig_behavior.tracking.detector_cache import (
        DetectorCacheIdentity,
        DetectorEvidenceCache,
    )
    from pig_behavior.tracking.masks import apply_mask_to_frame, load_mask

    output = cache_path(cache_root, video.video_key)
    if output.exists() or output.with_suffix(".sha256.json").exists():
        raise BaselineCacheError(f"refusing cache overwrite: {output}")
    identity = DetectorCacheIdentity(
        video_key=video.video_key,
        source_video_sha256=video.video_sha256,
        detector_weight_sha256=weights_sha,
        detector_semantic_config_sha256=detector_sha,
        producer_code_sha=producer_sha,
        creation_authority=CREATION_AUTHORITY,
    )
    cache = DetectorEvidenceCache(identity=identity)
    capture = cv2.VideoCapture(str(video.video_path))
    if not capture.isOpened():
        raise BaselineCacheError(f"cannot open source: {video.video_key}")
    mask = load_mask(Path(cfg.mask_path), video.width, video.height, cfg)
    calls = 0
    try:
        for frame_index in range(video.frame_count):
            ok, frame = capture.read()
            if not ok:
                raise BaselineCacheError(
                    f"decode failed: {video.video_key} frame {frame_index}"
                )
            if frame_index % cfg.detect_every_n_frames:
                continue
            detector_frame = (
                apply_mask_to_frame(frame, mask)
                if cfg.mask_input_frame and mask is not None
                else frame
            )
            result = invoke_detector(model, detector_frame, cfg)
            cache.record(
                frame_index,
                result,
                original_frame_dimensions=(video.height, video.width),
            )
            calls += 1
    finally:
        capture.release()
    expected = len(frame_indices(video, cfg.detect_every_n_frames))
    if calls != expected:
        raise BaselineCacheError(f"inference coverage mismatch: {video.video_key}")
    return cache.save(output), calls


def load_identity(
    video: VideoAuthority,
    preflight_payload: dict[str, Any],
) -> Any:
    from pig_behavior.tracking.detector_cache import DetectorCacheIdentity

    return DetectorCacheIdentity(
        video_key=video.video_key,
        source_video_sha256=video.video_sha256,
        detector_weight_sha256=preflight_payload["detector_weight_sha256"],
        detector_semantic_config_sha256=preflight_payload[
            "detector_semantic_config_sha256"
        ],
        producer_code_sha=preflight_payload["producer_code_sha"],
        creation_authority=CREATION_AUTHORITY,
    )


def replay_validate(
    videos: list[VideoAuthority],
    cache_root: Path,
    preflight_payload: dict[str, Any],
) -> dict[str, Any]:
    from pig_behavior.tracking.detector_cache import (
        DetectorCacheError,
        DetectorEvidenceCache,
        ReplayDetector,
    )

    rows = []
    total = 0
    for video in videos:
        path = cache_path(cache_root, video.video_key)
        identity = load_identity(video, preflight_payload)
        loaded = DetectorEvidenceCache.load(path, expected_identity=identity)
        expected = frame_indices(video, 2)
        if tuple(loaded.frames) != expected:
            raise BaselineCacheError(f"cache coverage mismatch: {video.video_key}")
        replay = ReplayDetector(loaded)
        for frame_index in expected:
            dimensions = loaded.frames[frame_index][
                "original_frame_dimensions"
            ]
            replay.set_frame_context(frame_index, dimensions)
            replay.predict()
        if replay.invocations != len(expected):
            raise BaselineCacheError("replay count mismatch")
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / path.name
            sidecar = corrupt.with_suffix(".sha256.json")
            shutil.copy2(path, corrupt)
            shutil.copy2(path.with_suffix(".sha256.json"), sidecar)
            with corrupt.open("ab") as handle:
                handle.write(b"corruption")
            try:
                DetectorEvidenceCache.load(corrupt, expected_identity=identity)
            except DetectorCacheError:
                corrupted_copy = "PASS"
            else:
                raise BaselineCacheError("corrupted cache did not fail closed")
        rows.append(
            {
                "video_key": video.video_key,
                "frames_loaded": len(expected),
                "frame_indices_exact": True,
                "replay_without_inference": True,
                "corrupted_copy_negative_test": corrupted_copy,
            }
        )
        total += len(expected)
    return {
        "schema_version": "tracking.current_main_cache_replay_validation.v1",
        "result": "PASS",
        "partitions": rows,
        "cache_frames_loaded": total,
        "cache_frames_replayed": total,
        "detector_inference_during_replay": 0,
    }


def update_partition_manifest(
    cache_root: Path,
    hashes: dict[str, str],
) -> None:
    path = cache_root / "CURRENT_MAIN_DETECTOR_CACHE_PARTITIONS.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        row["cache_artifact_sha256"] = hashes[row["video_key"]]
        row["status"] = "IMMUTABLE_READY"
    write_csv(path, rows, fields)


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "ARTIFACT_SHA256.json"
    ]


def generate(
    source_repo: Path,
    lineage_manifest: Path,
    cache_root: Path,
) -> None:
    import torch
    from ultralytics import YOLO

    videos, cfg, preflight_payload = load_preflight(
        source_repo,
        lineage_manifest,
        cache_root,
    )
    if not torch.cuda.is_available():
        raise BaselineCacheError("CUDA unavailable for authorized generation")
    model = YOLO(str(cfg.weights_path))
    model.to("cuda:0")
    cache_hashes = {}
    calls = 0
    if any(cache_root.glob("partitions/*/detector_evidence.npz")):
        raise BaselineCacheError("refusing partial or existing cache population")
    for video in videos:
        print(f"CACHE_BEGIN {video.video_key}", flush=True)
        digest, video_calls = record_video(
            video,
            cfg,
            model,
            cache_root,
            preflight_payload["producer_code_sha"],
            preflight_payload["detector_semantic_config_sha256"],
            preflight_payload["detector_weight_sha256"],
        )
        cache_hashes[video.video_key] = digest
        calls += video_calls
        print(
            f"CACHE_END {video.video_key} frames={video_calls}",
            flush=True,
        )
    update_partition_manifest(cache_root, cache_hashes)
    replay = replay_validate(videos, cache_root, preflight_payload)
    if calls != 11700 or replay["cache_frames_loaded"] != 11700:
        raise BaselineCacheError("full-13 cache population is not 11,700")
    write_json(
        cache_root / "CURRENT_MAIN_DETECTOR_CACHE_REPLAY_VALIDATION.json",
        replay,
    )
    decision = {
        "schema_version": "tracking.current_main_baseline_cache_decision.v1",
        "decision": "PASS_EXACT_CURRENT_MAIN_BASELINE_CACHES_READY",
        "starting_main_sha": STARTING_MAIN_SHA,
        "producer_code_sha": preflight_payload["producer_code_sha"],
        "tracking_tree_object": preflight_payload["tracking_tree_object"],
        "source_lineage_sha256": preflight_payload["source_lineage_sha256"],
        "detector_weight_sha256": preflight_payload["detector_weight_sha256"],
        "detector_semantic_config_sha256": preflight_payload[
            "detector_semantic_config_sha256"
        ],
        "effective_realtime_fast_config_sha256": preflight_payload[
            "effective_realtime_fast_config_sha256"
        ],
        "cache_schema": preflight_payload["cache_schema"],
        "cache_partitions_created": 13,
        "cache_frames_recorded": calls,
        "cache_replay_validation": "PASS",
        "detector_inference_calls_for_cache_generation": calls,
        "gpu_inference_runs": 1,
        "tracking_runs": 0,
        "h1_h2_validation_execution": False,
        "run_root_mp4_count": len(list(cache_root.rglob("*.mp4"))),
        "cache_artifacts": {
            key: {
                "sha256": cache_hashes[key],
                "relative_path": cache_path(cache_root, key)
                .relative_to(cache_root)
                .as_posix(),
                "frame_count": 900,
            }
            for key in sorted(cache_hashes)
        },
    }
    if decision["run_root_mp4_count"]:
        raise BaselineCacheError("cache generation produced MP4")
    write_json(
        cache_root / "CURRENT_MAIN_DETECTOR_CACHE_GENERATION_DECISION.json",
        decision,
    )
    with (
        cache_root / "CURRENT_MAIN_DETECTOR_CACHE_COMMANDS.txt"
    ).open("a", encoding="utf-8") as handle:
        handle.write(
            "GENERATION_AND_REPLAY\n"
            + subprocess.list2cmdline(sys.argv)
            + f"\nDETECTOR_INFERENCE_CALLS={calls}\n"
            + "TRACKING_RUNS=0\nVALIDATION_EXECUTIONS=0\n"
        )
    write_json(
        cache_root / "ARTIFACT_SHA256.json",
        {
            "schema_version": "tracking.current_main_cache_inventory.v1",
            "inventory_excludes_itself": True,
            "artifacts": artifact_inventory(cache_root),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "generate"), required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--lineage-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_repo = args.source_repo.resolve()
    lineage_manifest = args.lineage_manifest.resolve()
    cache_root = args.cache_root.resolve()
    baseline_root = args.baseline_root.resolve()
    if args.phase == "preflight":
        preflight(source_repo, lineage_manifest, cache_root, baseline_root)
    else:
        generate(source_repo, lineage_manifest, cache_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
