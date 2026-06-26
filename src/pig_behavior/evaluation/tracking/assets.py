"""Logic for pair discovery, metadata, and prediction lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pig_behavior.output_layout import prediction_xml_candidates

from .cvat_io import read_task_name


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent project root."""
    current = Path.cwd() if start is None else Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DATA_DIR = PROJECT_ROOT / "data"
TRACKING_GT_DIR = DATA_DIR / "annotations" / "tracking"
VIDEO_DIR = DATA_DIR / "videos"
PREDICTION_ROOT = PROJECT_ROOT / "outputs" / "pred"
EVAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "eval"
DETECTOR_WEIGHTS_V8 = PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov8.pt"
DETECTOR_WEIGHTS_V26 = PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov26.pt"
DETECTOR_WEIGHTS = DETECTOR_WEIGHTS_V8


@dataclass(slots=True)
class TrackingPair:
    """Matched ground-truth and prediction assets for one video."""

    video_stem: str
    video_path: Path
    gt_xml: Path
    pred_xml: Path | None = None


def normalize_key(text: str) -> str:
    """Normalize names for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def video_metadata(video_path: Path) -> dict[str, Any]:
    """Read optional video metadata when OpenCV is available."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}

    metadata = {
        "video_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "video_fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "video_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "video_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    capture.release()
    return metadata


def find_prediction_xml(
    video_stem: str,
    prediction_root: Path,
    preferred_mode: str | None = None,
) -> Path | None:
    """Find prediction CVAT XML for a given video stem."""
    for candidate in prediction_xml_candidates(
        prediction_root,
        video_stem,
        preferred_mode=preferred_mode,
    ):
        if candidate.exists():
            return candidate

    legacy_mode_scoped = sorted(
        prediction_root.glob(f"{video_stem}/*/annotations_cvat_video_1_1.xml")
    )
    if legacy_mode_scoped:
        return legacy_mode_scoped[0]

    canonical_mode_scoped = sorted(
        prediction_root.glob(f"*/{video_stem}/annotations_cvat_video_1_1.xml")
    )
    if canonical_mode_scoped:
        return canonical_mode_scoped[0]

    video_key = video_stem.lower()
    candidates = sorted(
        path
        for path in prediction_root.rglob("*.xml")
        if (
            video_key in path.name.lower()
            or any(video_key in part.lower() for part in path.parts)
        )
        and "cvat_video" in path.name.lower()
        and ".bak_" not in path.name.lower()
    )
    return candidates[0] if candidates else None


def list_tracking_pairs(
    *,
    tracking_gt_dir: Path = TRACKING_GT_DIR,
    video_dir: Path = VIDEO_DIR,
    prediction_root: Path = PREDICTION_ROOT,
    preferred_mode: str | None = None,
) -> list[TrackingPair]:
    """Match GT XML files to videos and optional prediction XML files."""
    videos = [path for path in video_dir.glob("*") if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    video_by_key = {normalize_key(path.stem): path for path in videos}
    pairs: list[TrackingPair] = []

    for gt_xml in sorted(tracking_gt_dir.glob("*.xml")):
        matched_video: Path | None = None
        gt_text = gt_xml.stem

        for key, video in video_by_key.items():
            if key in normalize_key(gt_text):
                matched_video = video
                break

        if matched_video is None:
            task_name = read_task_name(gt_xml)
            for key, video in video_by_key.items():
                if key in normalize_key(task_name):
                    matched_video = video
                    break

        if matched_video is None:
            continue

        pred_xml = find_prediction_xml(
            matched_video.stem,
            prediction_root,
            preferred_mode=preferred_mode,
        )
        pairs.append(
            TrackingPair(
                video_stem=matched_video.stem,
                video_path=matched_video,
                gt_xml=gt_xml,
                pred_xml=pred_xml,
            )
        )

    return pairs


def resolve_mask_path() -> Path | None:
    """Return the default scene mask when available."""
    mask_path = DATA_DIR / "annotations" / "scene" / "mask.png"
    return mask_path if mask_path.exists() else None


__all__ = [
    "DATA_DIR",
    "DETECTOR_WEIGHTS",
    "DETECTOR_WEIGHTS_V8",
    "DETECTOR_WEIGHTS_V26",
    "EVAL_OUTPUT_ROOT",
    "PREDICTION_ROOT",
    "PROJECT_ROOT",
    "TRACKING_GT_DIR",
    "VIDEO_DIR",
    "TrackingPair",
    "find_prediction_xml",
    "find_project_root",
    "list_tracking_pairs",
    "normalize_key",
    "resolve_mask_path",
    "video_metadata",
]
