from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def safe_video_id(source_folder: str, source_video: str) -> str:
    folder = Path(source_folder)
    if folder.name:
        return folder.name
    return Path(source_video).stem


def frame_filename(frame_index: int) -> str:
    return f"f{frame_index:06d}.jpg"


def clamp_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(ix1 + 1, min(width, int(round(x2))))
    iy2 = max(iy1 + 1, min(height, int(round(y2))))
    return ix1, iy1, ix2, iy2


def output_dir(
    output_root: Path,
    kind: str,
    window_type: str,
    day: str,
    video_id: str,
    group_id: str,
    pig_id: str,
) -> Path:
    return output_root / kind / window_type / str(day) / str(video_id) / str(group_id) / str(pig_id)


def write_crop(frame: np.ndarray, bbox: tuple[float, float, float, float], path: Path) -> str:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    crop = frame[y1:y2, x1:x2]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
    return str(path)


def write_full_frame(frame: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    return str(path)
