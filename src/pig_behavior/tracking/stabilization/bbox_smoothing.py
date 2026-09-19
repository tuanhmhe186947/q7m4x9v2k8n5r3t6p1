"""Bounding box smoothing for stable annotation outputs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pig_behavior.tracking.geometry import bbox_center, bbox_size, clip_box
from pig_behavior.tracking.stabilization.config import AnnotationStableConfig


def smooth_trajectory_boxes(
    boxes: NDArray[np.float32],
    config: AnnotationStableConfig,
    width: int,
    height: int,
) -> NDArray[np.float32]:
    """Smooths a sequence of bounding boxes [N, 4] (xyxy) using median or EMA.

    Asserts that boxes dimension is 2D: [N, 4]
    """
    assert boxes.ndim == 2, f"Expected 2D boxes array, got shape {boxes.shape}"
    assert boxes.shape[1] == 4, f"Expected xyxy format, got columns {boxes.shape[1]}"

    if len(boxes) < 3 or not config.smooth_bbox:
        return boxes.copy()

    # Convert xyxy to cx, cy, w, h
    cx_cy_w_h = np.zeros_like(boxes)
    for i, box in enumerate(boxes):
        cx, cy = bbox_center(box)
        w, h = bbox_size(box)
        cx_cy_w_h[i] = [cx, cy, w, h]

    smoothed = cx_cy_w_h.copy()

    if config.smooth_method == "median":
        window = config.smooth_bbox_window
        half_w = window // 2
        n = len(boxes)
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            window_data = cx_cy_w_h[start:end]
            smoothed[i] = np.median(window_data, axis=0)
    elif config.smooth_method == "ema":
        # simple EMA smoothing
        alpha = 0.5  # default factor
        for i in range(1, len(boxes)):
            smoothed[i] = alpha * cx_cy_w_h[i] + (1.0 - alpha) * smoothed[i - 1]

    # Convert back to xyxy and limit maximum shift from original center to avoid drifting
    final_boxes = np.zeros_like(boxes)
    max_shift = float(config.max_smoothing_shift_px)

    for i in range(len(boxes)):
        orig_cx, orig_cy, orig_w, orig_h = cx_cy_w_h[i]
        scx, scy, sw, sh = smoothed[i]

        # Clamp center shift
        dx = scx - orig_cx
        dy = scy - orig_cy
        dist = np.hypot(dx, dy)
        if dist > max_shift and dist > 0:
            scale = max_shift / dist
            scx = orig_cx + dx * scale
            scy = orig_cy + dy * scale

        # Clamp size shift (at most 25% change)
        sw = float(np.clip(sw, orig_w * 0.75, orig_w * 1.25))
        sh = float(np.clip(sh, orig_h * 0.75, orig_h * 1.25))

        x1 = scx - sw / 2.0
        y1 = scy - sh / 2.0
        x2 = scx + sw / 2.0
        y2 = scy + sh / 2.0

        box_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
        final_boxes[i] = clip_box(box_xyxy, width, height)

    return final_boxes
