from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import Detection
from .tracker import BBox


@dataclass(frozen=True)
class BBoxMaskMetrics:
    center_in_mask: bool
    bbox_mask_coverage: float


def load_scene_mask(mask_path: str | Path, frame_width: int, frame_height: int) -> np.ndarray:
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Scene mask does not exist: {path}")

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read scene mask as grayscale image: {path}")

    if mask.shape[:2] != (frame_height, frame_width):
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    return mask > 0


def bbox_mask_metrics(scene_mask: np.ndarray, bbox: BBox) -> BBoxMaskMetrics:
    height, width = scene_mask.shape[:2]
    x1, y1, x2, y2 = bbox

    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round((y1 + y2) / 2.0))
    center_in_bounds = 0 <= center_x < width and 0 <= center_y < height
    center_in_mask = bool(center_in_bounds and scene_mask[center_y, center_x])

    ix1 = max(0, min(width, int(np.floor(x1))))
    iy1 = max(0, min(height, int(np.floor(y1))))
    ix2 = max(0, min(width, int(np.ceil(x2))))
    iy2 = max(0, min(height, int(np.ceil(y2))))
    bbox_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    if ix2 <= ix1 or iy2 <= iy1:
        return BBoxMaskMetrics(center_in_mask=center_in_mask, bbox_mask_coverage=0.0)

    valid_pixels = int(scene_mask[iy1:iy2, ix1:ix2].sum())
    return BBoxMaskMetrics(center_in_mask=center_in_mask, bbox_mask_coverage=valid_pixels / bbox_area)


def filter_detections_by_mask(
    detections: list[Detection],
    scene_mask: np.ndarray,
    *,
    min_bbox_coverage: float,
    require_center_inside: bool,
) -> tuple[list[Detection], list[Detection]]:
    kept: list[Detection] = []
    rejected: list[Detection] = []
    for detection in detections:
        metrics = bbox_mask_metrics(scene_mask, detection.bbox)
        center_rejected = require_center_inside and not metrics.center_in_mask
        coverage_rejected = metrics.bbox_mask_coverage < min_bbox_coverage
        if center_rejected or coverage_rejected:
            rejected.append(detection)
        else:
            kept.append(detection)
    return kept, rejected
