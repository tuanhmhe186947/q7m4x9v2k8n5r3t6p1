"""Logic for IoU and Hungarian frame matching."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cvat_io import TrackingObject


def iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union for xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def match_frame(
    gt_objects: list[TrackingObject],
    pred_objects: list[TrackingObject],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Match one frame with Hungarian assignment maximizing IoU."""
    if not gt_objects or not pred_objects:
        return []

    ious = np.zeros((len(gt_objects), len(pred_objects)), dtype=float)
    for i, gt in enumerate(gt_objects):
        for j, pred in enumerate(pred_objects):
            ious[i, j] = iou_xyxy(gt.bbox, pred.bbox)

    row_ind, col_ind = linear_sum_assignment(-ious)
    matches = []
    for row, col in zip(row_ind, col_ind, strict=False):
        iou = float(ious[row, col])
        if iou >= iou_threshold:
            matches.append((int(row), int(col), iou))
    return matches
