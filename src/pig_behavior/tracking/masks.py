"""Mask, ROI, and mask-overlap helpers for pig tracking."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.geometry import (
    bbox_center,
    bbox_iou,
    center_distance_norm,
    clip_box,
)
from pig_behavior.tracking.schemas import Detection, FixedTrack


def load_mask(
    mask_path: Path | None,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> np.ndarray | None:
    if not cfg.use_mask or mask_path is None:
        return None

    import cv2

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if cfg.roi_dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.roi_dilate_px, cfg.roi_dilate_px),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def apply_mask_to_frame(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame

    import cv2

    return cv2.bitwise_and(frame, frame, mask=mask)


def shade_outside_roi(frame: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return frame.copy()
    shaded = (frame.astype(np.float32) * 0.35).astype(np.uint8)
    out = frame.copy()
    out[mask == 0] = shaded[mask == 0]
    return out


def roi_keep(mask: np.ndarray | None, box: np.ndarray, cfg: TrackingConfig) -> bool:
    if mask is None:
        return True

    height, width = mask.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height).astype(int)
    if x2 <= x1 or y2 <= y1:
        return False

    if cfg.roi_mode == "center":
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)
        return bool(mask[cy, cx] == 255)

    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    cover = np.count_nonzero(roi == 255) / float(roi.size)
    return cover >= cfg.roi_min_cover


def mask_anchor_boxes(
    mask: np.ndarray | None,
    width: int,
    height: int,
    count: int,
    median_box: np.ndarray | None,
) -> list[np.ndarray]:
    """Create hidden fallback boxes so every frame can contain 8 shapes."""
    if mask is not None and np.count_nonzero(mask) > 0:
        ys, xs = np.where(mask > 0)
        rx1, rx2 = float(xs.min()), float(xs.max())
        ry1, ry2 = float(ys.min()), float(ys.max())
    else:
        rx1, ry1, rx2, ry2 = 0.0, 0.0, float(width - 1), float(height - 1)

    roi_w = max(1.0, rx2 - rx1)
    roi_h = max(1.0, ry2 - ry1)
    if median_box is not None:
        bw = max(24.0, float(median_box[2] - median_box[0]))
        bh = max(24.0, float(median_box[3] - median_box[1]))
    else:
        bw = max(24.0, roi_w * 0.18)
        bh = max(24.0, roi_h * 0.22)

    cols = 4
    rows = int(math.ceil(count / cols))
    boxes: list[np.ndarray] = []
    for row in range(rows):
        for col in range(cols):
            if len(boxes) == count:
                break
            cx = rx1 + (col + 0.5) * roi_w / cols
            cy = ry1 + (row + 0.5) * roi_h / rows
            box = np.array(
                [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                dtype=np.float32,
            )
            boxes.append(clip_box(box, width, height))
    return boxes


def mask_area(mask: np.ndarray | None) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(mask))


def mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None or first.shape != second.shape:
        return None
    first_bool = first.astype(bool, copy=False)
    second_bool = second.astype(bool, copy=False)
    inter = int(np.logical_and(first_bool, second_bool).sum())
    union = int(np.logical_or(first_bool, second_bool).sum())
    if union <= 0:
        return None
    return float(inter / union)


def shift_mask(
    mask: np.ndarray | None,
    dx: float,
    dy: float,
) -> np.ndarray | None:
    """Translate a binary mask by a small integer offset without wraparound."""
    if mask is None:
        return None
    shift_x = int(round(dx))
    shift_y = int(round(dy))
    if shift_x == 0 and shift_y == 0:
        return mask

    height, width = mask.shape[:2]
    shifted = np.zeros_like(mask, dtype=bool)
    src_x1 = max(0, -shift_x)
    src_x2 = min(width, width - shift_x)
    dst_x1 = max(0, shift_x)
    dst_x2 = min(width, width + shift_x)
    src_y1 = max(0, -shift_y)
    src_y2 = min(height, height - shift_y)
    dst_y1 = max(0, shift_y)
    dst_y2 = min(height, height + shift_y)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return shifted
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return shifted


def track_mask_for_box(
    track: FixedTrack,
    predicted_box: np.ndarray,
    cfg: TrackingConfig,
) -> np.ndarray | None:
    if (
        not cfg.use_mask_iou
        or track.last_mask is None
        or track.missed > cfg.mask_iou_max_missed
        or mask_area(track.last_mask) < cfg.mask_iou_min_area
    ):
        return None
    last_cx, last_cy = bbox_center(track.last_box)
    pred_cx, pred_cy = bbox_center(predicted_box)
    return shift_mask(track.last_mask, pred_cx - last_cx, pred_cy - last_cy)


def detection_overlap_score(
    first: Detection,
    second: Detection,
    cfg: TrackingConfig,
) -> float:
    if (
        cfg.use_mask_iou
        and mask_area(first.mask) >= cfg.mask_iou_min_area
        and mask_area(second.mask) >= cfg.mask_iou_min_area
    ):
        score = mask_iou(first.mask, second.mask)
        if score is not None:
            return score
    return bbox_iou(first.box, second.box)


def track_detection_overlap_score(
    track: FixedTrack,
    predicted_box: np.ndarray,
    det: Detection,
    cfg: TrackingConfig,
) -> float:
    if cfg.use_mask_iou and mask_area(det.mask) >= cfg.mask_iou_min_area:
        score = mask_iou(track_mask_for_box(track, predicted_box, cfg), det.mask)
        if score is not None:
            return score
    return bbox_iou(predicted_box, det.box)


def nearest_anchor_for_detection(
    anchors: list[np.ndarray],
    det: Detection,
    width: int,
    height: int,
) -> tuple[float, int]:
    """Return the nearest anchor index and normalized distance for one detection."""
    distances = [
        (center_distance_norm(anchor, det.box, width, height), idx)
        for idx, anchor in enumerate(anchors)
    ]
    return min(distances, key=lambda item: item[0])


__all__ = [
    "apply_mask_to_frame",
    "detection_overlap_score",
    "load_mask",
    "mask_anchor_boxes",
    "mask_area",
    "mask_iou",
    "nearest_anchor_for_detection",
    "roi_keep",
    "shade_outside_roi",
    "shift_mask",
    "track_detection_overlap_score",
    "track_mask_for_box",
]
