"""Lock an immutable RF_ACC23 evaluation lineage manifest.

The manifest records exactly what a reproduction of the RF_ACC23 tracking
baseline would consume: the 13 evaluation videos, their ground truth, the
detector weight, the mask, the resolved semantic configuration, and the
environment. Nothing here runs a tracker, a detector, or an evaluation.

The tool refuses to overwrite an existing manifest so a locked lineage cannot
be silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "tracking.rf_acc23_lineage_manifest.v1"

#: The 13 development/evaluation videos. This is NOT an unbiased final test
#: set; it is the development/evaluation set used throughout tracking work.
EXPECTED_VIDEO_COUNT = 13

#: Hard6 as stated by the task brief. Membership is recorded as REPORTED until
#: an artifact or memory entry confirms it; see ``hard6_authority``.
HARD6_REPORTED_SUFFIXES: tuple[str, ...] = (
    "000114",
    "000231",
    "000233",
    "000263",
    "000327",
    "000302",
)

#: Videos whose ground-truth authority is flagged in project memory.
GT_AUTHORITY_FLAGS: dict[str, str] = {
    "000216": "GT_AUTHORITY_FLAGGED_IN_MEMORY_VERIFY_BEFORE_USE",
}


class LineageLockError(RuntimeError):
    """Raised when the lineage cannot be locked safely."""


@dataclass(frozen=True, slots=True)
class LockRequest:
    repo_root: Path
    output_path: Path
    video_dir: Path
    gt_dir: Path
    detector_weight: Path
    mask_path: Path | None
    allow_overwrite: bool = False


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_video_key(path: Path) -> str:
    return path.stem


def gt_path_for(video: Path, gt_dir: Path) -> Path | None:
    """Resolve the GT XML for a video, tolerating the known name variant."""

    direct = gt_dir / f"{video.stem}.xml"
    if direct.is_file():
        return direct
    prefixed = gt_dir / f"Tracking_annotation_{video.stem}.xml"
    if prefixed.is_file():
        return prefixed
    return None


def video_media_properties(path: Path) -> dict[str, Any]:
    """Read frame count, FPS and resolution without decoding the whole file."""

    try:
        import cv2
    except ImportError:
        return {"probe_status": "OPENCV_UNAVAILABLE"}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"probe_status": "OPEN_FAILED"}
    try:
        return {
            "probe_status": "OK",
            "frame_count_metadata": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps_metadata": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count_is_container_metadata_not_decoded": True,
        }
    finally:
        capture.release()


def resolved_semantic_config() -> dict[str, Any]:
    """Export the effective RF_ACC23 semantic configuration from code."""

    from pig_behavior.tracking.profiles.realtime import (
        EVAL_CONFIGS,
        PRESENTATION_PROFILES,
    )

    config = dict(EVAL_CONFIGS["realtime_fast"])
    payload = {
        "eval_config_name": "realtime_fast",
        "presentation_profile_realtime": dict(PRESENTATION_PROFILES["realtime"]),
        "presentation_profile_realtime_fast": dict(
            PRESENTATION_PROFILES["realtime_fast"]
        ),
        "config": {key: config[key] for key in sorted(config)},
    }
    payload["config_sha256"] = _payload_sha256(payload["config"])
    return payload


def resolved_timing_contract() -> dict[str, Any]:
    """Resolve the output timing contract for RF_ACC23 through real code."""

    try:
        from pig_behavior.tracking.config import TrackingConfig
        from pig_behavior.tracking.telemetry import resolve_output_timing_contract
    except ImportError as exc:
        return {"status": "IMPORT_FAILED", "detail": str(exc)}
    from pig_behavior.tracking.profiles.realtime import EVAL_CONFIGS

    try:
        cfg = TrackingConfig(mode="realtime", **EVAL_CONFIGS["realtime_fast"])
    except TypeError:
        try:
            cfg = TrackingConfig(mode="realtime")
            for key, value in EVAL_CONFIGS["realtime_fast"].items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        except Exception as exc:  # noqa: BLE001 - report, never guess
            return {"status": "CONFIG_CONSTRUCTION_FAILED", "detail": str(exc)}
    contract, delay = resolve_output_timing_contract(cfg)
    return {
        "status": "RESOLVED",
        "output_timing_contract": contract,
        "output_delay_frames": int(delay),
        "is_causal_delay_zero": contract == "causal_framewise" and int(delay) == 0,
    }


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    for module_name, key in (
        ("torch", "torch"),
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv"),
        ("numpy", "numpy"),
    ):
        try:
            module = __import__(module_name)
            snapshot[f"{key}_version"] = getattr(module, "__version__", "unknown")
        except ImportError:
            snapshot[f"{key}_version"] = "MISSING"
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["cuda_built_version"] = torch.version.cuda
        snapshot["torch_build_is_cpu_only"] = torch.version.cuda is None
        if torch.cuda.is_available():
            snapshot["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        snapshot["cuda_available"] = None
    snapshot["nvidia_smi"] = _nvidia_smi()
    return snapshot


def _nvidia_smi() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "UNAVAILABLE", "detail": str(exc)}
    if result.returncode != 0:
        return {"status": "FAILED", "returncode": result.returncode}
    return {"status": "OK", "gpus": result.stdout.strip().splitlines()}


def git_state(repo_root: Path) -> dict[str, Any]:
    def _git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "code_sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "tracking_tree_object": _git("rev-parse", "HEAD:src/pig_behavior/tracking"),
        "tracking_last_change_sha": _git(
            "log", "-1", "--format=%H", "--", "src/pig_behavior/tracking"
        ),
        "worktree_clean_tracked": not _git("status", "--porcelain", "-uno"),
    }


def build_manifest(request: LockRequest) -> dict[str, Any]:
    if request.output_path.exists() and not request.allow_overwrite:
        raise LineageLockError(
            f"refusing to overwrite an existing lineage manifest: "
            f"{request.output_path}. A locked lineage is immutable; write a new "
            "versioned path instead."
        )
    if not request.video_dir.is_dir():
        raise LineageLockError(f"video_dir is not a directory: {request.video_dir}")
    if not request.gt_dir.is_dir():
        raise LineageLockError(f"gt_dir is not a directory: {request.gt_dir}")
    if not request.detector_weight.is_file():
        raise LineageLockError(
            f"detector weight is not a file: {request.detector_weight}"
        )

    videos = sorted(request.video_dir.glob("*.mp4"))
    if len(videos) != EXPECTED_VIDEO_COUNT:
        raise LineageLockError(
            f"expected {EXPECTED_VIDEO_COUNT} evaluation videos, found "
            f"{len(videos)}: {[v.name for v in videos]}"
        )

    entries: list[dict[str, Any]] = []
    missing_gt: list[str] = []
    for video in videos:
        key = canonical_video_key(video)
        gt = gt_path_for(video, request.gt_dir)
        if gt is None:
            missing_gt.append(key)
        suffix = key.split("_")[1] if "_" in key else ""
        entries.append(
            {
                "canonical_video_key": key,
                "video_path": str(video),
                "video_size_bytes": video.stat().st_size,
                "video_sha256": sha256_file(video),
                "gt_path": None if gt is None else str(gt),
                "gt_sha256": None if gt is None else sha256_file(gt),
                "gt_filename_variant": (
                    None if gt is None else gt.name != f"{key}.xml"
                ),
                "media": video_media_properties(video),
                "in_hard6_reported": suffix in HARD6_REPORTED_SUFFIXES,
                "gt_authority_status": GT_AUTHORITY_FLAGS.get(suffix, "NOT_FLAGGED"),
            }
        )
    if missing_gt:
        raise LineageLockError(f"videos without resolvable ground truth: {missing_gt}")

    hard6 = [entry["canonical_video_key"] for entry in entries if entry["in_hard6_reported"]]
    semantic = resolved_semantic_config()
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_purpose": "RF_ACC23 tracking evaluation lineage lock",
        "population_status": {
            "video_count": len(entries),
            "is_final_unbiased_test_set": False,
            "population_role": "DEVELOPMENT_AND_EVALUATION_SET",
            "note": (
                "These 13 videos have been used for development and tuning. "
                "They are not an unbiased final test set."
            ),
        },
        "videos": entries,
        "hard6_reported": hard6,
        "hard6_authority": (
            "REPORTED_BY_TASK_BRIEF_NOT_CONFIRMED_BY_REPOSITORY_ARTIFACT"
        ),
        "detector_weight": {
            "path": str(request.detector_weight),
            "size_bytes": request.detector_weight.stat().st_size,
            "sha256": sha256_file(request.detector_weight),
        },
        "mask": _mask_entry(request.mask_path),
        "semantic_config": semantic,
        "timing_contract": resolved_timing_contract(),
        "git": git_state(request.repo_root),
        "environment": environment_snapshot(),
        "evidence_status": {
            "rf_acc23_metrics_included": False,
            "reason": (
                "No RF_ACC23 metrics artifact was found in the repository. This "
                "manifest locks inputs only; it asserts no quality metric."
            ),
        },
    }
    manifest["manifest_sha256"] = _payload_sha256(manifest)
    return manifest


def _mask_entry(mask_path: Path | None) -> dict[str, Any]:
    if mask_path is None:
        return {"status": "NOT_SUPPLIED"}
    if not mask_path.is_file():
        return {"status": "MISSING", "path": str(mask_path)}
    entry: dict[str, Any] = {
        "status": "OK",
        "path": str(mask_path),
        "size_bytes": mask_path.stat().st_size,
        "sha256": sha256_file(mask_path),
    }
    try:
        import cv2

        image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            entry["dimensions"] = list(image.shape)
    except ImportError:
        entry["dimensions"] = "OPENCV_UNAVAILABLE"
    return entry


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=Path("data/videos"))
    parser.add_argument(
        "--gt-dir", type=Path, default=Path("data/annotations/tracking")
    )
    parser.add_argument(
        "--detector-weight",
        type=Path,
        default=Path("models/detector/pig_detector_yolov8.pt"),
    )
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = LockRequest(
        repo_root=args.repo_root,
        output_path=args.output,
        video_dir=args.video_dir,
        gt_dir=args.gt_dir,
        detector_weight=args.detector_weight,
        mask_path=args.mask_path,
        allow_overwrite=args.allow_overwrite,
    )
    try:
        manifest = build_manifest(request)
    except LineageLockError as exc:
        print(f"LINEAGE_LOCK_REFUSED: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"LINEAGE_LOCKED: {args.output}")
    print(f"MANIFEST_SHA256={manifest['manifest_sha256']}")
    print(f"SEMANTIC_CONFIG_SHA256={manifest['semantic_config']['config_sha256']}")
    timing = manifest["timing_contract"]
    print(f"TIMING_CONTRACT={timing.get('output_timing_contract')}")
    print(f"OUTPUT_DELAY_FRAMES={timing.get('output_delay_frames')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
