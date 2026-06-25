# ruff: noqa
"""Depth-based occlusion inference for RGB-D tracking.

# ruff: noqa

Determines which detections are *occluded candidates* by combining 2-D
bounding-box overlap with depth ordering.  In a pair of overlapping
detections, the one farther from the camera is marked as the occluded
candidate.
"""

from __future__ import annotations

import logging

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.schemas import BEVTrackState, Detection3D

logger = logging.getLogger(__name__)


def _bbox_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """IoU between two ``(x1, y1, x2, y2)`` bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / max(union, 1e-8)


def infer_occlusions(
    detections: list[Detection3D],
    cfg: RGBDTrackingConfig,
) -> dict[int, bool]:
    """Return a mapping ``{detection_index: is_occluded_candidate}``.

    For each pair of detections with IoU above *occlusion_iou_threshold*,
    the detection **farther** from the camera (larger depth when
    ``larger_depth_is_farther=True``) is marked as an occluded candidate.

    Detections with invalid depth are not involved in occlusion inference
    but their entry in the returned dict is ``False``.
    """
    n = len(detections)
    occluded: dict[int, bool] = {i: False for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            di = detections[i]
            dj = detections[j]

            iou = _bbox_iou(di.detection_2d.bbox, dj.detection_2d.bbox)
            if iou < cfg.occlusion_iou_threshold:
                continue

            # Both must have valid depth for depth ordering
            if not di.depth_valid or not dj.depth_valid:
                continue
            if di.depth_m is None or dj.depth_m is None:
                continue

            if cfg.larger_depth_is_farther:
                farther_idx = i if di.depth_m > dj.depth_m else j
            else:
                farther_idx = i if di.depth_m < dj.depth_m else j

            occluded[farther_idx] = True

    return occluded


def update_occlusion_age(
    bev_states: dict[int, BEVTrackState],
    matched_track_ids: set[int],
    occluded_track_ids: set[int],
    cfg: RGBDTrackingConfig,
) -> None:
    """Increment or reset ``occluded_age`` on BEV track states.

    Tracks whose matched detection is itself an occluded candidate have
    their ``occluded_age`` incremented.  Tracks that were successfully
    matched to a non-occluded detection are reset to zero.  Unmatched
    tracks have ``occluded_age`` incremented if they exceed the max
    occlusion age they transition to ``lost``.
    """
    for fixed_id, bev in bev_states.items():
        if fixed_id in matched_track_ids and fixed_id not in occluded_track_ids:
            bev.occluded_age = 0
            if bev.state == "lost":
                bev.state = "confirmed"
        else:
            bev.occluded_age += 1
            if bev.occluded_age > cfg.max_occlusion_age:
                bev.state = "lost"


def track_is_occluded(bev_state: BEVTrackState) -> bool:
    """Return ``True`` if the track is currently marked as occluded."""
    return bev_state.occluded_age > 0


__all__ = [
    "infer_occlusions",
    "track_is_occluded",
    "update_occlusion_age",
]
