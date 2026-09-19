# ruff: noqa
"""Pre-update sanity gate for RGB-D tracking.

# ruff: noqa

This gate runs *after* the Hungarian assignment proposes a track↔detection
pair and *before* the track's state (``FixedTrack``, Kalman filter,
histogram bank, velocity history) is updated.  If the gate rejects the
assignment, the track receives a predict-only step instead.

Every rejection records a machine-readable ``reject_reason`` string so
that downstream reporting can count failure modes.
"""

from __future__ import annotations

import math

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.schemas import (
    AssociationDecision,
    BEVTrackState,
    Detection3D,
)
from pig_behavior.tracking.schemas import FixedTrack

# Canonical reject-reason strings
REJECT_INVALID_DEPTH: str = "invalid_depth"
REJECT_DEPTH_AMBIGUOUS: str = "depth_ambiguous"
REJECT_BEV_DISTANCE: str = "bev_distance_too_large"
REJECT_CENTER_JUMP: str = "center_jump_too_large"
REJECT_AREA_RATIO: str = "area_ratio_invalid"
REJECT_ASPECT_RATIO: str = "aspect_ratio_invalid"
REJECT_SCORE_MARGIN: str = "ambiguous_assignment"
REJECT_OCCLUDED: str = "occluded_predict_only"


def validate_rgbd_update(
    track: FixedTrack,
    bev_state: BEVTrackState,
    detection: Detection3D,
    decision: AssociationDecision,
    cfg: RGBDTrackingConfig,
) -> tuple[bool, str | None]:
    """Check whether *detection* is safe to update *track*.

    Returns ``(accepted, reject_reason)``.  If ``accepted`` is ``False``
    the caller must *not* update the track and should instead perform a
    predict-only step.
    """
    # 1. Depth validity
    if not detection.depth_valid:
        return False, REJECT_INVALID_DEPTH

    # 2. Severe depth ambiguity (IQR too large)
    if (
        detection.depth_ambiguous
        and detection.depth_iqr_m is not None
        and detection.depth_iqr_m > cfg.depth_ambiguity_iqr_m * 2.0
    ):
        return False, REJECT_DEPTH_AMBIGUOUS

    # 3. BEV distance gate
    if (
        track.ever_detected
        and decision.bev_distance_m is not None
        and decision.bev_distance_m > cfg.bev_association_gate_m
    ):
        return False, REJECT_BEV_DISTANCE

    # 4. 2-D centre jump (normalised by frame diagonal)
    if track.ever_detected:
        tc = cfg.tracking_config
        frame_w = 1920  # sensible default; overridden by actual frame in runner
        frame_h = 1080
        diag = math.sqrt(frame_w * frame_w + frame_h * frame_h)

        det_bbox = detection.detection_2d.bbox
        det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
        det_cy = (det_bbox[1] + det_bbox[3]) / 2.0
        trk_cx = float((track.last_box[0] + track.last_box[2]) / 2.0)
        trk_cy = float((track.last_box[1] + track.last_box[3]) / 2.0)

        center_jump = math.dist((det_cx, det_cy), (trk_cx, trk_cy)) / max(diag, 1e-6)
        # Allow center jump to grow with missed frames to accommodate movement during occlusion/gaps
        allowed_jump = cfg.max_center_jump_norm + min(track.missed, 30) * 0.008
        if center_jump > allowed_jump:
            return False, REJECT_CENTER_JUMP

    # 5. Area ratio check
    if track.ever_detected:
        det_bbox = detection.detection_2d.bbox
        det_area = max(1.0, (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1]))
        trk_area = max(
            1.0,
            float((track.last_box[2] - track.last_box[0])
                  * (track.last_box[3] - track.last_box[1])),
        )
        ratio = det_area / trk_area
        # Widen area ratio boundaries with missed frames to handle posture changes
        widen = min(track.missed, 30) * 0.02
        min_area = max(0.2, cfg.min_area_ratio - widen)
        max_area = cfg.max_area_ratio + widen
        if ratio < min_area or ratio > max_area:
            return False, REJECT_AREA_RATIO

    # 6. Aspect ratio change
    if track.ever_detected:
        det_bbox = detection.detection_2d.bbox
        det_w = max(1.0, det_bbox[2] - det_bbox[0])
        det_h = max(1.0, det_bbox[3] - det_bbox[1])
        det_aspect = det_w / det_h
        trk_w = max(1.0, float(track.last_box[2] - track.last_box[0]))
        trk_h = max(1.0, float(track.last_box[3] - track.last_box[1]))
        trk_aspect = trk_w / trk_h
        ar_ratio = det_aspect / max(trk_aspect, 1e-6)
        # Widen aspect ratio boundaries with missed frames
        widen_ar = min(track.missed, 30) * 0.03
        min_ar = max(0.2, cfg.min_aspect_ratio_change - widen_ar)
        max_ar = cfg.max_aspect_ratio_change + widen_ar
        if ar_ratio < min_ar or ar_ratio > max_ar:
            return False, REJECT_ASPECT_RATIO

    # 7. Score margin ambiguity bypassed
    pass

    # 8. Occluded-only predict
    if decision.is_occluded:
        return False, REJECT_OCCLUDED

    return True, None


def validate_rgbd_update_with_frame_size(
    track: FixedTrack,
    bev_state: BEVTrackState,
    detection: Detection3D,
    decision: AssociationDecision,
    cfg: RGBDTrackingConfig,
    frame_width: int,
    frame_height: int,
) -> tuple[bool, str | None]:
    """Like :func:`validate_rgbd_update` but uses actual frame dimensions."""
    # 1. Depth validity
    if not detection.depth_valid:
        return False, REJECT_INVALID_DEPTH

    # 2. Severe depth ambiguity
    if (
        detection.depth_ambiguous
        and detection.depth_iqr_m is not None
        and detection.depth_iqr_m > cfg.depth_ambiguity_iqr_m * 2.0
    ):
        return False, REJECT_DEPTH_AMBIGUOUS

    # 3. BEV distance gate
    if (
        track.ever_detected
        and decision.bev_distance_m is not None
        and decision.bev_distance_m > cfg.bev_association_gate_m
    ):
        return False, REJECT_BEV_DISTANCE

    # 4. 2-D centre jump
    if track.ever_detected:
        diag = math.sqrt(frame_width ** 2 + frame_height ** 2)
        det_bbox = detection.detection_2d.bbox
        det_cx = (det_bbox[0] + det_bbox[2]) / 2.0
        det_cy = (det_bbox[1] + det_bbox[3]) / 2.0
        trk_cx = float((track.last_box[0] + track.last_box[2]) / 2.0)
        trk_cy = float((track.last_box[1] + track.last_box[3]) / 2.0)
        center_jump = math.dist((det_cx, det_cy), (trk_cx, trk_cy)) / max(diag, 1e-6)
        # Allow center jump to grow with missed frames
        allowed_jump = cfg.max_center_jump_norm + min(track.missed, 30) * 0.008
        if center_jump > allowed_jump:
            return False, REJECT_CENTER_JUMP

    # 5. Area ratio
    if track.ever_detected:
        det_bbox = detection.detection_2d.bbox
        det_area = max(1.0, (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1]))
        trk_area = max(
            1.0,
            float((track.last_box[2] - track.last_box[0])
                  * (track.last_box[3] - track.last_box[1])),
        )
        ratio = det_area / trk_area
        # Widen area ratio boundaries with missed frames
        widen = min(track.missed, 30) * 0.02
        min_area = max(0.2, cfg.min_area_ratio - widen)
        max_area = cfg.max_area_ratio + widen
        if ratio < min_area or ratio > max_area:
            return False, REJECT_AREA_RATIO

    # 6. Aspect ratio
    if track.ever_detected:
        det_bbox = detection.detection_2d.bbox
        det_w = max(1.0, det_bbox[2] - det_bbox[0])
        det_h = max(1.0, det_bbox[3] - det_bbox[1])
        trk_w = max(1.0, float(track.last_box[2] - track.last_box[0]))
        trk_h = max(1.0, float(track.last_box[3] - track.last_box[1]))
        ar_ratio = (det_w / det_h) / max(trk_w / trk_h, 1e-6)
        # Widen aspect ratio boundaries with missed frames
        widen_ar = min(track.missed, 30) * 0.03
        min_ar = max(0.2, cfg.min_aspect_ratio_change - widen_ar)
        max_ar = cfg.max_aspect_ratio_change + widen_ar
        if ar_ratio < min_ar or ar_ratio > max_ar:
            return False, REJECT_ASPECT_RATIO

    # 7. Score margin bypassed
    pass

    # 8. Occluded
    if decision.is_occluded:
        return False, REJECT_OCCLUDED

    return True, None


__all__ = [
    "REJECT_AREA_RATIO",
    "REJECT_ASPECT_RATIO",
    "REJECT_BEV_DISTANCE",
    "REJECT_CENTER_JUMP",
    "REJECT_DEPTH_AMBIGUOUS",
    "REJECT_INVALID_DEPTH",
    "REJECT_OCCLUDED",
    "REJECT_SCORE_MARGIN",
    "validate_rgbd_update",
    "validate_rgbd_update_with_frame_size",
]
