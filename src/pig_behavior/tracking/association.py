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
        if cfg.mode in {"bytetrack", "hybrid_bytetrack"}:
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


def lost_track_fast_motion_owner_bypass_is_plausible(
    track: FixedTrack,
    owner_track: FixedTrack | None,
    det: Detection,
    cfg: TrackingConfig,
    width: int,
    height: int,
    distance: float,
    same_raw_id: bool,
) -> bool:
    if owner_track is None:
        return False
    if track_is_visible_for_association(owner_track):
        return False
    if not same_raw_id:
        return False
    if track.missed > cfg.lost_track_fast_motion_owner_grace:
        return False
    if not (
        cfg.lost_track_fast_motion_min_center_jump
        <= distance
        <= cfg.lost_track_fast_motion_max_center_jump
    ):
        return False
    mean_hist = track.mean_hist()
    if mean_hist is None:
        return False
    if hist_distance(mean_hist, det.hist) > cfg.lost_track_fast_motion_appearance_threshold:
        return False

    owner_gap = center_distance_norm(track.last_box, owner_track.last_box, width, height)
    return owner_gap <= cfg.lost_track_fast_motion_owner_max_gap


def lost_track_same_raw_appearance_bypass_is_plausible(
    track: FixedTrack,
    det: Detection,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.lost_track_reacquire_same_raw_appearance_bypass:
        return False

    mean_hist = track.mean_hist()
    if mean_hist is None:
        return False

    return (
        hist_distance(mean_hist, det.hist)
        <= cfg.lost_track_reacquire_same_raw_appearance_threshold
    )


def lost_track_raw_owner_transfer_is_plausible(
    track: FixedTrack,
    owner_track: FixedTrack | None,
    det: Detection,
    cfg: TrackingConfig,
    width: int,
    height: int,
) -> bool:
    if owner_track is None:
        return False
    if track.top_raw_id() != det.raw_id:
        return False

    track_hist = track.mean_hist()
    if track_hist is None:
        return False
    track_app = hist_distance(track_hist, det.hist)
    if track_app > cfg.lost_track_raw_owner_transfer_appearance_threshold:
        return False

    track_ref = association_reference_box(track, det, width, height, cfg)
    owner_ref = association_reference_box(owner_track, det, width, height, cfg)
    track_distance = center_distance_norm(track_ref, det.box, width, height)
    owner_distance = center_distance_norm(owner_ref, det.box, width, height)
    if (
        track_distance + cfg.lost_track_raw_owner_transfer_min_center_gain
        >= owner_distance
    ):
        return False

    owner_hist = owner_track.mean_hist()
    if owner_hist is None:
        return True
    owner_app = hist_distance(owner_hist, det.hist)
    return (
        track_app + cfg.lost_track_raw_owner_transfer_min_appearance_gain
        < owner_app
    )


def lost_track_different_raw_hidden_owner_bypass_is_plausible(
    track: FixedTrack,
    owner_track: FixedTrack | None,
    det: Detection,
    cfg: TrackingConfig,
    width: int,
    height: int,
) -> bool:
    if not cfg.lost_track_different_raw_hidden_owner_bypass:
        return False
    if track.top_raw_id() == det.raw_id:
        return False
    if owner_track is None:
        return False
    if track_is_visible_for_association(owner_track):
        return False
    if owner_track.missed < cfg.lost_track_different_raw_hidden_owner_min_missed:
        return False

    track_hist = track.mean_hist()
    if track_hist is None:
        return False
    if (
        hist_distance(track_hist, det.hist)
        > cfg.lost_track_different_raw_hidden_owner_appearance_threshold
    ):
        return False

    track_ref = association_reference_box(track, det, width, height, cfg)
    owner_ref = association_reference_box(owner_track, det, width, height, cfg)
    track_distance = center_distance_norm(track_ref, det.box, width, height)
    owner_distance = center_distance_norm(owner_ref, det.box, width, height)
    return (
        track_distance + cfg.lost_track_different_raw_hidden_owner_min_center_gain
        < owner_distance
    )


def lost_track_detection_is_plausible(
    track: FixedTrack,
    det: Detection,
    cfg: TrackingConfig,
    width: int,
    height: int,
    raw_owner: dict[int, int] | None = None,
    raw_owner_tracks: dict[int, FixedTrack] | None = None,
) -> bool:
    if not cfg.lost_track_reacquire_guard:
        return True
    if not track_is_lost_for_association(track):
        return True

    top_raw_id = track.top_raw_id()
    same_raw_id = top_raw_id == det.raw_id
    reference = association_reference_box(track, det, width, height, cfg)
    distance = center_distance_norm(reference, det.box, width, height)
    max_jump = cfg.lost_track_reacquire_max_center_jump + min(track.missed, 8) * 0.015
    same_raw_max_jump = (
        cfg.lost_track_reacquire_same_raw_max_center_jump + min(track.missed, 8) * 0.02
    )
    if (
        cfg.lost_track_reacquire_same_raw_distance_guard
        and same_raw_id
        and distance > same_raw_max_jump
        and not lost_track_same_raw_appearance_bypass_is_plausible(track, det, cfg)
    ):
        return False

    owner = None
    if raw_owner is not None and det.raw_id is not None:
        owner = raw_owner.get(det.raw_id)
    if owner is not None and owner != track.fixed_id:
        owner_guard_enabled = cfg.lost_track_reacquire_raw_owner_guard and (
            cfg.lost_track_reacquire_same_raw_owner_guard
            if same_raw_id
            else cfg.lost_track_reacquire_different_raw_owner_guard
        )
        if owner_guard_enabled:
            owner_track = (
                raw_owner_tracks.get(owner) if raw_owner_tracks is not None else None
            )
            if not lost_track_fast_motion_owner_bypass_is_plausible(
                track,
                owner_track,
                det,
                cfg,
                width,
                height,
                distance,
                same_raw_id,
            ) and not lost_track_raw_owner_transfer_is_plausible(
                track,
                owner_track,
                det,
                cfg,
                width,
                height,
            ) and not lost_track_different_raw_hidden_owner_bypass_is_plausible(
                track,
                owner_track,
                det,
                cfg,
                width,
                height,
            ):
                return False

    if (
        cfg.lost_track_reacquire_non_same_raw_distance_guard
        and distance > max_jump
        and not same_raw_id
    ):
        return False

    return True


def visible_raw_owner_transfer_is_plausible(
    track: FixedTrack,
    det: Detection,
    cfg: TrackingConfig,
    width: int,
    height: int,
    raw_owner_tracks: dict[int, FixedTrack] | None = None,
) -> bool:
    if not cfg.identity_swap_guard:
        return True
    if det.raw_id is None or raw_owner_tracks is None:
        return True
    if not track_is_visible_for_association(track):
        return True

    owner_track = raw_owner_tracks.get(det.raw_id)
    if owner_track is None or owner_track.fixed_id == track.fixed_id:
        return True
    if not track_is_visible_for_association(owner_track):
        return True

    current_ref = association_reference_box(track, det, width, height, cfg)
    owner_ref = association_reference_box(owner_track, det, width, height, cfg)
    current_distance = center_distance_norm(current_ref, det.box, width, height)
    owner_distance = center_distance_norm(owner_ref, det.box, width, height)
    return current_distance + cfg.visible_raw_owner_transfer_min_gain < owner_distance


def track_detection_cost(
    track: FixedTrack,
    det: Detection,
    det_index: int,
    occlusion_context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
    raw_owner: dict[int, int] | None = None,
    raw_owner_tracks: dict[int, FixedTrack] | None = None,
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
    if not lost_track_detection_is_plausible(
        track,
        det,
        cfg,
        width,
        height,
        raw_owner,
        raw_owner_tracks,
    ):
        return 1_000_000.0
    if not visible_raw_owner_transfer_is_plausible(
        track, det, cfg, width, height, raw_owner_tracks
    ):
        return 1_000_000.0

    predicted = association_reference_box(track, det, width, height, cfg)
    iou_score = track_detection_overlap_score(track, predicted, det, cfg)
    center_cost = center_distance_norm(predicted, det.box, width, height)
    app_cost = hist_distance(track.mean_hist(), det.hist)
    area_cost = min(area_log_ratio(predicted, det.box), 2.0) / 2.0

    raw_penalty = 0.0
    if cfg.mode in {"bytetrack", "hybrid_bytetrack"} and det.raw_id is not None:
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


def append_association_debug_event(
    runtime: TrackingRuntimeState | None,
    cfg: TrackingConfig,
    event: dict[str, object],
) -> None:
    if runtime is None or not cfg.association_debug:
        return
    runtime.association_debug_events.append(event)


def track_debug_state(track: FixedTrack) -> dict[str, object]:
    return {
        "track_id": track.fixed_id,
        "track_state": track.get_state(),
        "track_reason": track.state_reason,
        "track_missed": track.missed,
        "track_hits": track.hits,
        "track_source": track.last_source,
        "track_top_raw_id": track.top_raw_id(),
        "track_last_score": round(float(track.last_score), 6),
    }


def detection_debug_state(det: Detection, det_idx: int) -> dict[str, object]:
    return {
        "det_idx": det_idx,
        "det_raw_id": det.raw_id,
        "det_score": round(float(det.score), 6),
        "det_x1": round(float(det.box[0]), 3),
        "det_y1": round(float(det.box[1]), 3),
        "det_x2": round(float(det.box[2]), 3),
        "det_y2": round(float(det.box[3]), 3),
    }


def raw_owner_conflict_is_ambiguous(
    track: FixedTrack,
    owner_track: FixedTrack | None,
    det: Detection,
    selected_cost: float,
    owner_cost: float | None,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.ambiguity_owner_guard:
        return False
    if det.raw_id is None or owner_track is None:
        return False
    if owner_track.fixed_id == track.fixed_id:
        return False
    if track.top_raw_id() == det.raw_id:
        return False
    if owner_cost is None or not np.isfinite(owner_cost):
        return False
    if owner_cost >= 1_000_000.0:
        return False
    return (
        owner_cost - selected_cost
        <= float(cfg.ambiguity_owner_guard_cost_margin)
    )


def hidden_owner_conflict_should_freeze_identity(
    track: FixedTrack,
    owner_track: FixedTrack | None,
    det: Detection,
    selected_cost: float,
    owner_cost: float | None,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.hidden_owner_guard:
        return False
    if det.raw_id is None or owner_track is None:
        return False
    if owner_track.fixed_id == track.fixed_id:
        return False
    if track.top_raw_id() == det.raw_id:
        return False
    owner_is_hidden_or_lost = (
        owner_track.missed >= cfg.hidden_owner_guard_min_missed
        or owner_track.get_state() in {"LOST", "MISSING"}
        or owner_track.state_reason in {"prediction_only", "occlusion_hold"}
    )
    if not owner_is_hidden_or_lost:
        return False
    if owner_cost is None or not np.isfinite(owner_cost) or owner_cost >= 1_000_000.0:
        return True
    return (
        owner_cost - selected_cost
        <= float(cfg.hidden_owner_guard_cost_margin)
    )


def reentry_ambiguous_assignment_should_hold(
    track: FixedTrack,
    ambiguous: bool,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.reentry_ambiguous_hold:
        return False
    if not ambiguous:
        return False
    if not track.ever_detected or track.hits < cfg.reentry_ambiguous_hold_min_hits:
        return False
    return (
        track.get_state() in {"OCCLUDED", "LOST"}
        or track.missed >= cfg.reentry_ambiguous_hold_min_missed
        or track.state_reason in {"prediction_only", "occlusion_hold"}
    )


def match_and_update_tracks(
    tracks: dict[int, FixedTrack],
    detections: list[Detection],
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    cfg: TrackingConfig,
    runtime: TrackingRuntimeState | None = None,
    frame_index: int | None = None,
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
    raw_owner_tracks: dict[int, FixedTrack] = {}
    if cfg.mode in {"bytetrack", "hybrid_bytetrack"}:
        for track in ordered_tracks:
            raw_id = track.top_raw_id()
            if raw_id is not None:
                raw_owner[raw_id] = track.fixed_id
                raw_owner_tracks[raw_id] = track

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
        phase_name: str,
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
                    raw_owner_tracks,
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
            det = detections[det_idx]
            threshold = association_cost_threshold(track, cfg)
            row_costs = costs[row, :]
            finite_costs = row_costs[np.isfinite(row_costs)]
            finite_costs = finite_costs[finite_costs < 1_000_000.0]
            best_cost = float(np.min(finite_costs)) if finite_costs.size else None
            second_cost = None
            if finite_costs.size >= 2:
                second_cost = float(np.partition(finite_costs, 1)[1])
            owner_id = raw_owner.get(det.raw_id) if det.raw_id is not None else None
            owner_track = tracks.get(owner_id) if owner_id is not None else None
            owner_candidate_cost = None
            if owner_track is not None:
                for owner_row, candidate in enumerate(candidate_tracks):
                    if candidate.fixed_id == owner_track.fixed_id:
                        owner_candidate_cost = float(costs[owner_row, col])
                        break
            base_debug_event = {
                "frame": frame_index,
                "phase": phase_name,
                "cost": round(float(costs[row, col]), 6),
                "threshold": round(float(threshold), 6),
                "track_best_cost": (
                    round(best_cost, 6) if best_cost is not None else None
                ),
                "track_second_cost": (
                    round(second_cost, 6) if second_cost is not None else None
                ),
                "det_raw_owner": owner_id,
                "det_raw_owner_cost": (
                    round(owner_candidate_cost, 6)
                    if owner_candidate_cost is not None
                    else None
                ),
                "same_raw_id": track.top_raw_id() == det.raw_id,
                "in_split_recovery": (
                    runtime is not None
                    and track.fixed_id in runtime.current_recovery_track_ids
                ),
                **track_debug_state(track),
                **detection_debug_state(det, det_idx),
            }
            if (
                track.fixed_id in matched_tracks
                or det_idx in matched_detections
                or costs[row, col] > threshold
            ):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": (
                            "assignment_reject_already_matched"
                            if (
                                track.fixed_id in matched_tracks
                                or det_idx in matched_detections
                            )
                            else "assignment_reject_threshold"
                        ),
                    },
                )
                continue
            if raw_owner_conflict_is_ambiguous(
                track,
                owner_track,
                det,
                float(costs[row, col]),
                owner_candidate_cost,
                cfg,
            ):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_reject_ambiguous_raw_owner",
                        "ambiguous": True,
                        "learn_identity": False,
                    },
                )
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
                det,
                det_idx,
                detections,
                occlusion_context,
                width,
                height,
                cfg,
            ):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_area_freeze",
                        "ambiguous": ambiguous,
                        "learn_identity": False,
                    },
                )
                freeze_area_occluded_track(track, width, height, cfg)
                matched_tracks.add(track.fixed_id)
                matched_detections.add(det_idx)
                continue
            if reentry_ambiguous_assignment_should_hold(track, ambiguous, cfg):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_reentry_ambiguous_hold",
                        "ambiguous": ambiguous,
                        "hidden_owner_freeze": False,
                        "learn_identity": False,
                    },
                )
                freeze_area_occluded_track(track, width, height, cfg)
                matched_tracks.add(track.fixed_id)
                matched_detections.add(det_idx)
                continue
            hidden_owner_freeze = hidden_owner_conflict_should_freeze_identity(
                track,
                owner_track,
                det,
                float(costs[row, col]),
                owner_candidate_cost,
                cfg,
            )
            if hidden_owner_freeze and cfg.hidden_owner_guard_hold_assignment:
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_hidden_owner_hold",
                        "ambiguous": True,
                        "hidden_owner_freeze": True,
                        "learn_identity": False,
                    },
                )
                freeze_area_occluded_track(track, width, height, cfg)
                matched_tracks.add(track.fixed_id)
                matched_detections.add(det_idx)
                continue
            learn_identity = not (
                (cfg.freeze_identity_in_occlusion and ambiguous)
                or hidden_owner_freeze
            )
            append_association_debug_event(
                runtime,
                cfg,
                {
                    **base_debug_event,
                    "event": "assignment_accept",
                    "ambiguous": ambiguous,
                    "hidden_owner_freeze": hidden_owner_freeze,
                    "learn_identity": learn_identity,
                },
            )
            track.update_detected(
                det,
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

        if cfg.mode in {"bytetrack", "hybrid_bytetrack"}:
            run_matching_phase(visible_tracks, all_detection_indices, "visible")
            remaining_detection_indices = [
                idx
                for idx in all_detection_indices
                if idx not in matched_detections
            ]
            run_matching_phase(reid_tracks, remaining_detection_indices, "reid")
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

            run_matching_phase(visible_tracks, high_conf_indices, "visible_high_conf")
            remaining_high_conf_indices = [
                idx for idx in high_conf_indices if idx not in matched_detections
            ]
            run_matching_phase(reid_tracks, remaining_high_conf_indices, "reid")

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
                "low_conf_recovery",
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
