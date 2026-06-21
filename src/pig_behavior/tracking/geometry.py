"""Bounding-box geometry and smoothing utilities for pig tracking."""

from __future__ import annotations

import math

import numpy as np

from pig_behavior.tracking.config import TrackingConfig


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(box, dtype=np.float32).copy()
    out[0] = max(0.0, min(float(width - 1), float(out[0])))
    out[1] = max(0.0, min(float(height - 1), float(out[1])))
    out[2] = max(out[0] + 1.0, min(float(width), float(out[2])))
    out[3] = max(out[1] + 1.0, min(float(height), float(out[3])))
    return out


def bbox_area(box: np.ndarray) -> float:
    return max(1.0, float(box[2] - box[0])) * max(1.0, float(box[3] - box[1]))


def bbox_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)


def bbox_size(box: np.ndarray) -> tuple[float, float]:
    return max(1.0, float(box[2] - box[0])), max(1.0, float(box[3] - box[1]))


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(first) + bbox_area(second) - inter
    return inter / max(union, 1e-6)


def bbox_iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pairwise IoU for two ``xyxy`` box arrays."""
    if first.size == 0 or second.size == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)

    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    x1 = np.maximum(first[:, None, 0], second[None, :, 0])
    y1 = np.maximum(first[:, None, 1], second[None, :, 1])
    x2 = np.minimum(first[:, None, 2], second[None, :, 2])
    y2 = np.minimum(first[:, None, 3], second[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    first_area = np.maximum(1.0, first[:, 2] - first[:, 0]) * np.maximum(
        1.0,
        first[:, 3] - first[:, 1],
    )
    second_area = np.maximum(1.0, second[:, 2] - second[:, 0]) * np.maximum(
        1.0,
        second[:, 3] - second[:, 1],
    )
    union = first_area[:, None] + second_area[None, :] - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


def bbox_iom_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-min-area for two ``xyxy`` box arrays."""
    if first.size == 0 or second.size == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)

    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    x1 = np.maximum(first[:, None, 0], second[None, :, 0])
    y1 = np.maximum(first[:, None, 1], second[None, :, 1])
    x2 = np.minimum(first[:, None, 2], second[None, :, 2])
    y2 = np.minimum(first[:, None, 3], second[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    first_area = np.maximum(1.0, first[:, 2] - first[:, 0]) * np.maximum(
        1.0,
        first[:, 3] - first[:, 1],
    )
    second_area = np.maximum(1.0, second[:, 2] - second[:, 0]) * np.maximum(
        1.0,
        second[:, 3] - second[:, 1],
    )
    min_area = np.minimum(first_area[:, None], second_area[None, :])
    return (inter / np.maximum(min_area, 1e-6)).astype(np.float32)


def center_distance_norm_matrix(
    first: np.ndarray,
    second: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Pairwise normalized center distance for two ``xyxy`` box arrays."""
    if first.size == 0 or second.size == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)

    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    first_centers = np.column_stack(
        ((first[:, 0] + first[:, 2]) / 2.0, (first[:, 1] + first[:, 3]) / 2.0)
    )
    second_centers = np.column_stack(
        (
            (second[:, 0] + second[:, 2]) / 2.0,
            (second[:, 1] + second[:, 3]) / 2.0,
        )
    )
    deltas = first_centers[:, None, :] - second_centers[None, :, :]
    diag = math.sqrt(width * width + height * height)
    return (np.linalg.norm(deltas, axis=2) / max(diag, 1e-6)).astype(np.float32)


def bbox_intersection_area(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iom(first: np.ndarray, second: np.ndarray) -> float:
    """Intersection over the smaller box area, useful for occlusion detection."""
    inter = bbox_intersection_area(first, second)
    return inter / max(min(bbox_area(first), bbox_area(second)), 1e-6)


def center_distance_norm(
    first: np.ndarray,
    second: np.ndarray,
    width: int,
    height: int,
) -> float:
    cx1, cy1 = bbox_center(first)
    cx2, cy2 = bbox_center(second)
    diag = math.sqrt(width * width + height * height)
    return math.dist((cx1, cy1), (cx2, cy2)) / max(diag, 1e-6)


def area_log_ratio(first: np.ndarray, second: np.ndarray) -> float:
    return abs(math.log((bbox_area(second) + 1e-6) / (bbox_area(first) + 1e-6)))


def smooth_alpha_for_score(score: float, cfg: TrackingConfig) -> float:
    if score >= cfg.review_conf:
        return cfg.high_conf_smooth_alpha
    if score >= cfg.track_high_conf:
        return cfg.mid_conf_smooth_alpha
    return cfg.low_conf_smooth_alpha


def smooth_detected_box(
    previous_box: np.ndarray,
    detected_box: np.ndarray,
    score: float,
    missed_frames: int,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> np.ndarray:
    """Limit sudden box-size jumps while keeping motion responsive."""
    previous_box = clip_box(previous_box, width, height)
    detected_box = clip_box(detected_box, width, height)
    prev_cx, prev_cy = bbox_center(previous_box)
    det_cx, det_cy = bbox_center(detected_box)
    prev_w, prev_h = bbox_size(previous_box)
    det_w, det_h = bbox_size(detected_box)

    max_scale_change = (
        cfg.max_box_scale_change_after_gap
        if missed_frames > 0
        else cfg.max_box_scale_change_per_frame
    )
    min_scale = max(0.05, 1.0 - max_scale_change)
    max_scale = 1.0 + max_scale_change
    limited_w = float(np.clip(det_w, prev_w * min_scale, prev_w * max_scale))
    limited_h = float(np.clip(det_h, prev_h * min_scale, prev_h * max_scale))

    alpha = smooth_alpha_for_score(score, cfg)
    center_alpha = max(alpha, 0.80)
    cx = center_alpha * det_cx + (1.0 - center_alpha) * prev_cx
    cy = center_alpha * det_cy + (1.0 - center_alpha) * prev_cy
    smooth_w = alpha * limited_w + (1.0 - alpha) * prev_w
    smooth_h = alpha * limited_h + (1.0 - alpha) * prev_h

    smoothed = np.array(
        [
            cx - smooth_w / 2.0,
            cy - smooth_h / 2.0,
            cx + smooth_w / 2.0,
            cy + smooth_h / 2.0,
        ],
        dtype=np.float32,
    )
    return clip_box(smoothed, width, height)


__all__ = [
    "area_log_ratio",
    "bbox_area",
    "bbox_center",
    "bbox_intersection_area",
    "bbox_iom",
    "bbox_iom_matrix",
    "bbox_iou",
    "bbox_iou_matrix",
    "bbox_size",
    "center_distance_norm",
    "center_distance_norm_matrix",
    "clip_box",
    "smooth_alpha_for_score",
    "smooth_detected_box",
]
