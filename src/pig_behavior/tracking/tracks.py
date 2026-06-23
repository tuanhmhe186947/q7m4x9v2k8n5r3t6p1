"""Track lifecycle helpers and CVAT shape conversion."""

from __future__ import annotations

from typing import Any

import numpy as np

from pig_behavior.tracking.config import TrackingConfig, tracking_rule_flags_enabled
from pig_behavior.tracking.geometry import center_distance_norm, clip_box
from pig_behavior.tracking.masks import mask_anchor_boxes
from pig_behavior.tracking.schemas import Detection, FixedTrack


def lk_predict_box(
    prev_frame: np.ndarray | None,
    frame: np.ndarray,
    last_box: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    """Predict a missing box with Lucas-Kanade optical flow."""
    if prev_frame is None:
        return None

    import cv2

    x1, y1, x2, y2 = clip_box(last_box, width, height).astype(int)
    if x2 <= x1 or y2 <= y1:
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = prev_gray[y1:y2, x1:x2]
    if roi.size < 64:
        return None

    points = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=60,
        qualityLevel=0.01,
        minDistance=4,
    )
    if points is None:
        return None
    points = points.reshape(-1, 1, 2)
    points[:, :, 0] += x1
    points[:, :, 1] += y1

    next_points, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        points,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    if next_points is None or status is None:
        return None
    good = status.reshape(-1) == 1
    if not np.any(good):
        return None

    delta = (next_points[good] - points[good]).reshape(-1, 2)
    dx = float(np.median(delta[:, 0]))
    dy = float(np.median(delta[:, 1]))
    predicted = last_box.copy()
    predicted[[0, 2]] += dx
    predicted[[1, 3]] += dy
    return clip_box(predicted, width, height)


def initialize_tracks(
    detections: list[Detection],
    mask: np.ndarray | None,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> dict[int, FixedTrack]:
    init_detections = [
        det for det in detections if det.score >= cfg.initial_track_conf
    ]
    selected = init_detections[: cfg.expected_pigs]
    median_box = None
    if selected:
        widths = [det.box[2] - det.box[0] for det in selected]
        heights = [det.box[3] - det.box[1] for det in selected]
        median_w = float(np.median(widths))
        median_h = float(np.median(heights))
        median_box = np.array([0.0, 0.0, median_w, median_h], dtype=np.float32)

    anchors = mask_anchor_boxes(mask, width, height, cfg.expected_pigs, median_box)
    anchor_detection_pairs = sorted(
        (
            (
                center_distance_norm(anchor, det.box, width, height),
                anchor_idx,
                det_idx,
            )
            for anchor_idx, anchor in enumerate(anchors)
            for det_idx, det in enumerate(selected)
        ),
        key=lambda item: item[0],
    )
    used_anchor_idx: set[int] = set()
    used_det_idx: set[int] = set()
    tracks: dict[int, FixedTrack] = {}

    for _cost, anchor_idx, det_idx in anchor_detection_pairs:
        if anchor_idx in used_anchor_idx or det_idx in used_det_idx:
            continue
        fixed_id = anchor_idx + 1
        det = selected[det_idx]
        used_anchor_idx.add(anchor_idx)
        used_det_idx.add(det_idx)
        track = FixedTrack(fixed_id=fixed_id, last_box=det.box.copy())
        track.update_detected(det, width, height, cfg)
        tracks[fixed_id] = track

    for fixed_id in range(1, cfg.expected_pigs + 1):
        if fixed_id not in tracks:
            tracks[fixed_id] = FixedTrack(
                fixed_id=fixed_id,
                last_box=anchors[fixed_id - 1].copy(),
            )

    return tracks


def detection_needs_motion_gate(det: Detection, cfg: TrackingConfig) -> bool:
    return cfg.low_conf_motion_gate and det.score < cfg.motion_gate_confidence


def track_is_visible_for_association(track: FixedTrack) -> bool:
    return (
        track.ever_detected
        and track.missed == 0
        and track.last_source == "detected"
        and not track.last_ambiguous
    )


def track_is_lost_for_association(track: FixedTrack) -> bool:
    return track.ever_detected and not track_is_visible_for_association(track)


def track_is_hidden(track: FixedTrack, cfg: TrackingConfig) -> bool:
    if not track.ever_detected:
        return True
    if (
        track.last_source == "occlusion_hold"
        and track.occlusion_hold_frames >= cfg.occlusion_hold_hidden_frames
    ):
        return True
    if track.missed >= cfg.hidden_missed_frames:
        return True
    return track.last_source == "predicted" and (
        track.last_score < cfg.hidden_score_threshold
    )


def shape_for_track(
    track: FixedTrack,
    frame_index: int,
    cfg: TrackingConfig,
) -> dict[str, Any]:
    hidden = "Yes" if track_is_hidden(track, cfg) else "No"
    track_state = track.get_state()
    legacy_mode = cfg.mode == "legacy_bytetrack"
    # MISSING still has a tracker-produced box, so keep it evaluable. Only an
    # uninitialized placeholder or an expired LOST track is truly absent.
    is_outside = False if legacy_mode else (
        not track.ever_detected or track_state == "LOST"
    )
    is_occluded = hidden == "Yes" if legacy_mode else (
        track_state == "OCCLUDED" or hidden == "Yes"
    )
    needs_review = is_outside or is_occluded or (
        track.last_source != "detected" or track.last_score < cfg.review_conf
    )
    x1, y1, x2, y2 = [round(float(value), 2) for value in track.last_box]
    shape = {
        "type": "rectangle",
        "occluded": bool(is_occluded),
        "outside": bool(is_outside),
        "z_order": 0,
        "rotation": 0.0,
        "points": [x1, y1, x2, y2],
        "group": 0,
        "source": "file",
        "frame": int(frame_index),
        "attributes": [
            {"value": f"ID_{track.fixed_id}", "name": "ID"},
            {"value": cfg.default_behavior, "name": "Behavior"},
            {"value": "Yes" if is_occluded else "No", "name": "Hidden"},
        ],
        "score": round(float(track.last_score), 4),
        "elements": [],
        "label": f"Pig_{track.fixed_id}",
        "_track_source": track.last_source,
        "_track_state": track_state,
        "_state_reason": track.state_reason,
        "_missed_frames": int(track.missed),
        "_needs_review": bool(needs_review),
        "_raw_track_id": track.top_raw_id(),
        "_ever_detected": bool(track.ever_detected),
        "_ambiguous_occlusion": bool(track.last_ambiguous),
        "_occlusion_hold": track.last_source == "occlusion_hold",
        "_motion_state": track.motion_state,
    }
    if tracking_rule_flags_enabled(cfg):
        shape.update(
            {
                "_area_occluded": bool(track.is_area_occluded),
                "_area_occlusion_frames": int(track.area_occlusion_frames),
                "_merged_box_split": bool(track.last_merged_split),
            }
        )
    return shape


def frame_shapes(
    tracks: dict[int, FixedTrack],
    frame_index: int,
    cfg: TrackingConfig,
) -> list[dict[str, Any]]:
    shapes = [
        shape_for_track(tracks[idx], frame_index, cfg)
        for idx in range(1, cfg.expected_pigs + 1)
    ]
    if len(shapes) != cfg.expected_pigs:
        raise RuntimeError(f"Expected {cfg.expected_pigs} shapes, got {len(shapes)}")
    return shapes


__all__ = [
    "detection_needs_motion_gate",
    "frame_shapes",
    "initialize_tracks",
    "lk_predict_box",
    "shape_for_track",
    "track_is_hidden",
    "track_is_lost_for_association",
    "track_is_visible_for_association",
]
