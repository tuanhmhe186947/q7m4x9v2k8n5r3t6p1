"""Logic for pair discovery, video metadata, prediction lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cvat_io import read_task_name


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing project metadata."""
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
PREDICTION_ROOT = PROJECT_ROOT / "outputs" / "id_tracking"
EVAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluation" / "tracking_metrics"
DETECTOR_WEIGHTS_V8 = PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov8.pt"
DETECTOR_WEIGHTS_V26 = PROJECT_ROOT / "models" / "detector" / "pig_detector_yolov26.pt"
DETECTOR_WEIGHTS = DETECTOR_WEIGHTS_V8


@dataclass(slots=True)
class TrackingPair:
    """Matched ground-truth/prediction assets for one video."""

    video_stem: str
    video_path: Path
    gt_xml: Path
    pred_xml: Path | None = None


def normalize_key(text: str) -> str:
    """Normalize file names for robust matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def video_metadata(video_path: Path) -> dict[str, Any]:
    """Read optional video metadata with OpenCV if available."""
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


def find_prediction_xml(video_stem: str, prediction_root: Path) -> Path | None:
    """Find a prediction CVAT video XML for a video stem."""
    preferred = (
        prediction_root
        / video_stem
        / f"{video_stem}_annotations_cvat_video_1_1.xml"
    )
    if preferred.exists():
        return preferred

    candidates = sorted(
        path
        for path in prediction_root.rglob("*.xml")
        if video_stem.lower() in path.name.lower()
        and "cvat_video" in path.name.lower()
        and ".bak_" not in path.name.lower()
    )
    return candidates[0] if candidates else None


def list_tracking_pairs(
    *,
    tracking_gt_dir: Path = TRACKING_GT_DIR,
    video_dir: Path = VIDEO_DIR,
    prediction_root: Path = PREDICTION_ROOT,
) -> list[TrackingPair]:
    """Match GT XML files to videos and prediction XML files."""
    videos = [p for p in video_dir.glob("*") if p.suffix.lower() in {".mp4", ".avi"}]
    video_by_key = {normalize_key(p.stem): p for p in videos}
    pairs = []

    for gt_xml in sorted(tracking_gt_dir.glob("*.xml")):
        gt_text = gt_xml.stem
        matched_video = None
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

        pred_xml = find_prediction_xml(matched_video.stem, prediction_root)
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
    """Find the pen mask after annotations were split into subfolders."""
    candidates = [
        DATA_DIR / "annotations" / "scene" / "mask.png",
        DATA_DIR / "annotations" / "mask.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None
