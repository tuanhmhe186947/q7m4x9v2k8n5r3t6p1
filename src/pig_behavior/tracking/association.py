"""Detection-to-track association for fixed-ID pig tracking."""

from __future__ import annotations

import math

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.detections import hist_distance
from pig_behavior.tracking.geometry import (
    area_log_ratio,
    bbox_center,
    bbox_iou_matrix,
    bbox_size,
    center_distance_norm,
    clip_box,
)
from pig_behavior.tracking.masks import track_detection_overlap_score
from pig_behavior.tracking.occlusion import (
    apply_iou_fallback,
    apply_merged_box_splits,
    area_occlusion_should_freeze,
    assignment_is_occlusion_ambiguous,
    build_occlusion_context,
    detect_merged_box_splits,
    detection_is_reserved_for_active_track,
    freeze_area_occluded_track,
    occlusion_assignment_penalty,
    should_hold_occluded_track_box,
)
from pig_behavior.tracking.schemas import (
    Detection,
    FixedTrack,
    OcclusionContext,
    TrackingRuntimeState,
)
from pig_behavior.tracking.tracks import (
    detection_needs_motion_gate,
    lk_predict_box,
    track_is_lost_for_association,
    track_is_visible_for_association,
)


def association_reference_box(
    track: FixedTrack,
    det: Detection,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> np.ndarray:
    if detection_needs_motion_gate(det, cfg) and track.ever_detected:
        reference = (
            track.reliable_box if track.reliable_box is not None else track.last_box
        )
        return clip_box(reference.copy(), width, height)
    return track.predicted_box(width, height)


def track_y_velocity_for_directional_prior(track: FixedTrack) -> float:
    """Prefer stable motion history when available, otherwise use last-frame speed."""
    if (
        track.motion_state == "moving"
        and len(track.reliable_velocity_history) > 0
        and np.linalg.norm(track.reliable_velocity_xy) > 0.0
    ):
        return float(track.reliable_velocity_xy[1])
    return float(track.velocity_xy[1])


def apply_directional_y_prior_to_costs(
    costs: np.ndarray,
    candidate_tracks: list[FixedTrack],
    detection_indices: list[int],
    detections: list[Detection],
    occlusion_context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> None:
    """Softly discourage assignments that reverse a track's Y momentum in overlap."""
    if (
        not cfg.directional_y_prior
        or costs.size == 0
        or cfg.directional_y_penalty_weight <= 0.0
    ):
        return

    track_boxes = np.stack(
        [
            occlusion_context.predicted_boxes.get(
                track.fixed_id,
                track.predicted_box(width, height),
            )
            for track in candidate_tracks
        ],
        axis=0,
    ).astype(np.float32)
    det_boxes = np.stack(
        [detections[det_idx].box for det_idx in detection_indices],
        axis=0,
    ).astype(np.float32)

    track_center_y = (track_boxes[:, 1] + track_boxes[:, 3]) * 0.5
    det_center_y = (det_boxes[:, 1] + det_boxes[:, 3]) * 0.5
    delta_y = det_center_y[None, :] - track_center_y[:, None]

    y_velocity = np.array(
        [track_y_velocity_for_directional_prior(track) for track in candidate_tracks],
        dtype=np.float32,
    )
    active_track = np.array(
        [
            track.ever_detected
            and not track_is_lost_for_association(track)
            and track.hits >= 2
            for track in candidate_tracks
        ],
        dtype=bool,
    )
    velocity_epsilon = float(cfg.directional_y_velocity_epsilon_px)
    margin = float(cfg.directional_y_margin_px)

    moving_up = active_track & (y_velocity < -velocity_epsilon)
    moving_down = active_track & (y_velocity > velocity_epsilon)
    against_momentum = (
        (moving_up[:, None] & (delta_y > margin))
        | (moving_down[:, None] & (delta_y < -margin))
    )
    in_conflict_zone = bbox_iou_matrix(track_boxes, det_boxes) > 0.0
    costs += (
        against_momentum
        & in_conflict_zone
        & np.isfinite(costs)
        & (costs < 1_000_000.0)
    ).astype(np.float32) * float(cfg.directional_y_penalty_weight)


def low_conf_detection_is_plausible(
    track: FixedTrack,
    det: Detection,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> bool:
    """Reject low-confidence detections that jump far from a known track."""
    if not detection_needs_motion_gate(det, cfg):
        return True
    if not track.ever_detected:
        return det.score >= cfg.initial_track_conf

    reference = association_reference_box(track, det, width, height, cfg)
    if track_is_lost_for_association(track):
        if cfg.mode == "legacy_bytetrack":
            top_raw_id = track.top_raw_id()
            if (
                det.raw_id is not None
                and top_raw_id is not None
                and det.raw_id == top_raw_id
            ):
                return True
        if hist_distance(track.mean_hist(), det.hist) <= (
            cfg.lost_track_reid_appearance_threshold
        ):
            return True

    iou_score = track_detection_overlap_score(track, reference, det, cfg)
    if iou_score >= cfg.low_conf_min_iou:
        return True

    center_norm = center_distance_norm(reference, det.box, width, height)
    missed_growth = 0.008 if track_is_lost_for_association(track) else 0.004
    allowed_norm = cfg.low_conf_max_center_jump + min(track.missed, 30) * missed_growth
    if center_norm <= allowed_norm:
        return True

    pred_cx, pred_cy = bbox_center(reference)
    det_cx, det_cy = bbox_center(det.box)
    pred_w, pred_h = bbox_size(reference)
    allowed_px = cfg.low_conf_max_box_jump_scale * math.hypot(pred_w, pred_h)
    center_px = math.dist((pred_cx, pred_cy), (det_cx, det_cy))
    return center_px <= allowed_px


def track_detection_cost(
    track: FixedTrack,
    det: Detection,
    det_index: int,
    occlusion_context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
    raw_owner: dict[int, int] | None = None,
) -> float:
    if not low_conf_detection_is_plausible(track, det, width, height, cfg):
        return 1_000_000.0
    if detection_is_reserved_for_active_track(
        track,
        det,
        det_index,
        occlusion_context,
        width,
        height,
        cfg,
    ):
        return 1_000_000.0

    predicted = association_reference_box(track, det, width, height, cfg)
    iou_score = track_detection_overlap_score(track, predicted, det, cfg)
    center_cost = center_distance_norm(predicted, det.box, width, height)
    app_cost = hist_distance(track.mean_hist(), det.hist)
    area_cost = min(area_log_ratio(predicted, det.box), 2.0) / 2.0

    raw_penalty = 0.0
    if cfg.mode == "legacy_bytetrack" and det.raw_id is not None:
        owner = raw_owner.get(det.raw_id) if raw_owner is not None else None
        if owner is not None and owner != track.fixed_id:
            raw_penalty += 0.18
        elif track.top_raw_id() is not None and track.top_raw_id() != det.raw_id:
            raw_penalty += 0.05

    if track_is_lost_for_association(track):
        cost = (
            0.18 * (1.0 - iou_score)
            + 0.08 * min(center_cost, 1.0)
            + 0.52 * app_cost
            + 0.12 * area_cost
            + raw_penalty
        )
    else:
        cost = (
            0.42 * (1.0 - iou_score)
            + 0.22 * center_cost
            + 0.26 * app_cost
            + 0.10 * area_cost
            + raw_penalty
        )
    cost += occlusion_assignment_penalty(
        track,
        det,
        det_index,
        occlusion_context,
        width,
        height,
        cfg,
    )

    search_radius = 0.08 + min(track.missed, 60) / 60.0 * 0.22
    if (
        track.ever_detected
        and not track_is_lost_for_association(track)
        and iou_score < 0.01
        and center_cost > search_radius
    ):
        cost += 1.0
    return float(cost)


def association_cost_threshold(track: FixedTrack, cfg: TrackingConfig) -> float:
    if not track.ever_detected:
        return cfg.unseen_track_cost_threshold
    if track_is_lost_for_association(track):
        return cfg.lost_track_cost_threshold
    return cfg.match_cost_threshold


def match_and_update_tracks(
    tracks: dict[int, FixedTrack],
    detections: list[Detection],
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    cfg: TrackingConfig,
    runtime: TrackingRuntimeState | None = None,
) -> None:
    from scipy.optimize import linear_sum_assignment

    height, width = frame.shape[:2]
    ordered_tracks = [tracks[idx] for idx in range(1, cfg.expected_pigs + 1)]
    merged_split_boxes, ignored_detection_indices = detect_merged_box_splits(
        tracks,
        detections,
        width,
        height,
        cfg,
        runtime,
    )
    raw_owner: dict[int, int] = {}
    if cfg.mode == "legacy_bytetrack":
        for track in ordered_tracks:
            raw_id = track.top_raw_id()
            if raw_id is not None:
                raw_owner[raw_id] = track.fixed_id

    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    matched_detections.update(ignored_detection_indices)
    apply_merged_box_splits(
        tracks,
        merged_split_boxes,
        matched_tracks,
        width,
        height,
    )
    occlusion_context = build_occlusion_context(
        ordered_tracks,
        detections,
        width,
        height,
        cfg,
    )

    def run_matching_phase(
        candidate_tracks: list[FixedTrack],
        detection_indices: list[int],
    ) -> None:
        if not candidate_tracks or not detection_indices:
            return

        costs = np.zeros(
            (len(candidate_tracks), len(detection_indices)),
            dtype=np.float32,
        )
        for row, track in enumerate(candidate_tracks):
            for col, det_idx in enumerate(detection_indices):
                costs[row, col] = track_detection_cost(
                    track,
                    detections[det_idx],
                    det_idx,
                    occlusion_context,
                    width,
                    height,
                    cfg,
                    raw_owner,
                )

        apply_directional_y_prior_to_costs(
            costs,
            candidate_tracks,
            detection_indices,
            detections,
            occlusion_context,
            width,
            height,
            cfg,
        )
        rows, cols = linear_sum_assignment(costs)
        for row, col in zip(rows, cols, strict=True):
            track = candidate_tracks[row]
            det_idx = detection_indices[col]
            if (
                track.fixed_id in matched_tracks
                or det_idx in matched_detections
                or costs[row, col] > association_cost_threshold(track, cfg)
            ):
                continue
            ambiguous = assignment_is_occlusion_ambiguous(
                track,
                det_idx,
                occlusion_context,
                cfg,
            )
            in_split_recovery = (
                runtime is not None
                and track.fixed_id in runtime.current_recovery_track_ids
            )
            ambiguous = ambiguous or in_split_recovery
            if area_occlusion_should_freeze(
                track,
                detections[det_idx],
                det_idx,
                detections,
                occlusion_context,
                width,
                height,
                cfg,
            ):
                freeze_area_occluded_track(track, width, height, cfg)
                matched_tracks.add(track.fixed_id)
                matched_detections.add(det_idx)
                continue
            learn_identity = not (cfg.freeze_identity_in_occlusion and ambiguous)
            track.update_detected(
                detections[det_idx],
                width,
                height,
                cfg,
                learn_identity=learn_identity,
                ambiguous=ambiguous,
            )
            matched_tracks.add(track.fixed_id)
            matched_detections.add(det_idx)

    if detections:
        visible_tracks = [
            track for track in ordered_tracks if track_is_visible_for_association(track)
        ]
        reid_tracks = [
            track
            for track in ordered_tracks
            if not track_is_visible_for_association(track)
        ]
        all_detection_indices = [
            idx
            for idx in range(len(detections))
            if idx not in ignored_detection_indices
        ]

        if cfg.mode == "legacy_bytetrack":
            run_matching_phase(visible_tracks, all_detection_indices)
            remaining_detection_indices = [
                idx
                for idx in all_detection_indices
                if idx not in matched_detections
            ]
            run_matching_phase(reid_tracks, remaining_detection_indices)
        else:
            high_conf_indices = [
                idx
                for idx in all_detection_indices
                if detections[idx].score >= cfg.track_high_conf
            ]
            low_conf_indices = [
                idx
                for idx in all_detection_indices
                if cfg.det_conf <= detections[idx].score < cfg.track_high_conf
            ]

            run_matching_phase(visible_tracks, high_conf_indices)
            remaining_high_conf_indices = [
                idx for idx in high_conf_indices if idx not in matched_detections
            ]
            run_matching_phase(reid_tracks, remaining_high_conf_indices)

            unmatched_active_tracks = [
                track
                for track in ordered_tracks
                if track.fixed_id not in matched_tracks and track.ever_detected
            ]
            remaining_low_conf_indices = [
                idx for idx in low_conf_indices if idx not in matched_detections
            ]
            run_matching_phase(
                unmatched_active_tracks,
                remaining_low_conf_indices,
            )

        # Run IoU fallback on the remaining unmatched tracks and detections
        apply_iou_fallback(
            tracks,
            detections,
            matched_tracks,
            matched_detections,
            width,
            height,
            cfg,
        )

    # Unmatched high-confidence detections can initialize hidden placeholder IDs.
    unseen_tracks = [
        track
        for track in ordered_tracks
        if track.fixed_id not in matched_tracks and not track.ever_detected
    ]
    remaining_dets = [
        (idx, det)
        for idx, det in enumerate(detections)
        if idx not in matched_detections and det.score >= cfg.initial_track_conf
    ]
    for track, (det_idx, det) in zip(unseen_tracks, remaining_dets, strict=False):
        track.update_detected(det, width, height, cfg)
        matched_tracks.add(track.fixed_id)
        matched_detections.add(det_idx)

    for track in ordered_tracks:
        if track.fixed_id in matched_tracks:
            continue
        if (
            (
                cfg.USE_AREA_OCCLUSION_FREEZE
                or cfg.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE
            )
            and track.is_area_occluded
            and track.area_occlusion_frames < cfg.area_occlusion_freeze_frames
        ):
            freeze_area_occluded_track(track, width, height, cfg)
            continue
        if should_hold_occluded_track_box(track, detections, occlusion_context, cfg):
            hold_box = track.hidden_motion_box(width, height, cfg)
            track.update_predicted(
                hold_box,
                width,
                height,
                ambiguous=True,
                hold=True,
                cfg=cfg,
            )
            continue

        lk_box = lk_predict_box(prev_frame, frame, track.last_box, width, height)
        if lk_box is None:
            lk_box = track.predicted_box(width, height)
        if track.missed > cfg.max_missing_frames:
            lk_box = 0.7 * track.last_box + 0.3 * lk_box
        track.update_predicted(lk_box, width, height, cfg=cfg)


__all__ = [
    "apply_directional_y_prior_to_costs",
    "association_cost_threshold",
    "association_reference_box",
    "low_conf_detection_is_plausible",
    "match_and_update_tracks",
    "track_detection_cost",
    "track_y_velocity_for_directional_prior",
]
