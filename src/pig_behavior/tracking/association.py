"""Detection-to-track association for fixed-ID pig tracking."""

from __future__ import annotations

import math

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.detections import hist_distance
from pig_behavior.tracking.geometry import (
    area_log_ratio,
    bbox_center,
    bbox_iom,
    bbox_iou,
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
from pig_behavior.tracking.telemetry import (
    record_association_event,
    record_association_phase,
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
        if cfg.mode == "hybrid_bytetrack":
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
    if cfg.mode == "hybrid_bytetrack" and det.raw_id is not None:
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
    record_association_event(runtime, event.get("event"))
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


def append_detection_candidate_rank_events(
    runtime: TrackingRuntimeState | None,
    cfg: TrackingConfig,
    frame_index: int | None,
    phase_name: str,
    candidate_tracks: list[FixedTrack],
    detection_indices: list[int],
    detections: list[Detection],
    costs: np.ndarray,
    selected_track_by_det: dict[int, int],
    raw_owner: dict[int, int],
) -> None:
    if runtime is None or not cfg.association_debug:
        return
    for col, det_idx in enumerate(detection_indices):
        det = detections[det_idx]
        ranked_rows = [
            row
            for row in np.argsort(costs[:, col])
            if np.isfinite(costs[row, col]) and costs[row, col] < 1_000_000.0
        ]
        owner_id = raw_owner.get(det.raw_id) if det.raw_id is not None else None
        for rank, row in enumerate(ranked_rows[:3], start=1):
            track = candidate_tracks[int(row)]
            append_association_debug_event(
                runtime,
                cfg,
                {
                    "event": "detection_candidate_rank",
                    "frame": frame_index,
                    "phase": phase_name,
                    "candidate_rank": rank,
                    "candidate_selected_by_lap": (
                        selected_track_by_det.get(det_idx) == track.fixed_id
                    ),
                    "candidate_is_raw_owner": owner_id == track.fixed_id,
                    "cost": round(float(costs[row, col]), 6),
                    "threshold": round(
                        float(association_cost_threshold(track, cfg)),
                        6,
                    ),
                    "det_raw_owner": owner_id,
                    "same_raw_id": track.top_raw_id() == det.raw_id,
                    **track_debug_state(track),
                    **detection_debug_state(det, det_idx),
                },
            )


def append_hidden_detection_claim_probe_events(
    runtime: TrackingRuntimeState | None,
    cfg: TrackingConfig,
    frame_index: int | None,
    hidden_tracks: list[FixedTrack],
    detection_indices: list[int],
    detections: list[Detection],
    occlusion_context: OcclusionContext,
    width: int,
    height: int,
    raw_owner: dict[int, int],
    raw_owner_tracks: dict[int, FixedTrack],
) -> None:
    """Record hidden claims before visible matching without changing assignment."""
    if runtime is None or not cfg.association_debug or cfg.mode != "realtime":
        return

    for det_idx in detection_indices:
        det = detections[det_idx]
        claims: list[tuple[float, FixedTrack, float, float, bool]] = []
        for track in hidden_tracks:
            if not track.ever_detected:
                continue
            cost = track_detection_cost(
                track,
                det,
                det_idx,
                occlusion_context,
                width,
                height,
                cfg,
                raw_owner,
                raw_owner_tracks,
            )
            reference = association_reference_box(track, det, width, height, cfg)
            plausible = bool(np.isfinite(cost) and cost < 1_000_000.0)
            claims.append(
                (
                    float(cost),
                    track,
                    bbox_iom(reference, det.box),
                    center_distance_norm(reference, det.box, width, height),
                    plausible,
                )
            )

        owner_id = raw_owner.get(det.raw_id) if det.raw_id is not None else None
        ranked_claims = sorted(
            claims,
            key=lambda item: (item[0], item[1].fixed_id),
        )
        for rank, (cost, track, overlap, center_cost, plausible) in enumerate(
            ranked_claims[:3],
            start=1,
        ):
            append_association_debug_event(
                runtime,
                cfg,
                {
                    "event": "hidden_detection_claim_probe",
                    "frame": frame_index,
                    "phase": "pre_visible_hidden_claim",
                    "claim_rank": rank,
                    "claim_iom": round(float(overlap), 6),
                    "claim_center_distance": round(float(center_cost), 6),
                    "claim_plausible": plausible,
                    "cost": round(cost, 6),
                    "threshold": round(
                        float(association_cost_threshold(track, cfg)),
                        6,
                    ),
                    "det_raw_owner": owner_id,
                    "same_raw_id": track.top_raw_id() == det.raw_id,
                    **track_debug_state(track),
                    **detection_debug_state(det, det_idx),
                },
            )


def apply_causal_hidden_detection_reservation(
    costs: np.ndarray,
    candidate_tracks: list[FixedTrack],
    detection_indices: list[int],
    detections: list[Detection],
    hidden_tracks: list[FixedTrack],
    matched_tracks: set[int],
    occlusion_context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
    raw_owner: dict[int, int],
    raw_owner_tracks: dict[int, FixedTrack],
    runtime: TrackingRuntimeState | None,
    frame_index: int | None,
    phase_name: str,
    reserved_hidden_detection_owners: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve a strong hidden claim before visible realtime matching."""
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(costs)
    if (
        not cfg.causal_hidden_detection_reservation
        or cfg.mode != "realtime"
        or phase_name != "visible_high_conf"
        or not candidate_tracks
        or not hidden_tracks
    ):
        return rows, cols

    attempted_pairs: set[tuple[int, int]] = set()
    reserved_hidden_ids: set[int] = set()
    reserved_detection_indices: set[int] = set()

    while True:
        selected_pairs = list(zip(rows, cols, strict=True))
        candidates: list[
            tuple[float, int, int, int, FixedTrack, float, float, bool]
        ] = []
        for row, col in selected_pairs:
            visible_track = candidate_tracks[row]
            det_idx = detection_indices[col]
            pair_key = (visible_track.fixed_id, det_idx)
            selected_cost = float(costs[row, col])
            if pair_key in attempted_pairs or det_idx in reserved_detection_indices:
                continue
            if visible_track.fixed_id in matched_tracks:
                continue
            if selected_cost >= 1_000_000.0:
                continue
            track_threshold = association_cost_threshold(visible_track, cfg)
            if selected_cost > track_threshold:
                continue

            has_alternative = any(
                alt_col != col
                and np.isfinite(costs[row, alt_col])
                and costs[row, alt_col] < 1_000_000.0
                and (
                    costs[row, alt_col]
                    <= cfg.causal_hidden_detection_reservation_max_alternative_cost
                )
                for alt_col in range(costs.shape[1])
            )
            for hidden_track in hidden_tracks:
                if hidden_track.fixed_id in matched_tracks:
                    continue
                if hidden_track.fixed_id in reserved_hidden_ids:
                    continue
                if not hidden_track.ever_detected:
                    continue
                if not 1 <= hidden_track.missed <= (
                    cfg.causal_hidden_detection_reservation_max_missed
                ):
                    continue
                if hidden_track.last_source not in {"occlusion_hold", "predicted"}:
                    continue
                claim_cost = track_detection_cost(
                    hidden_track,
                    detections[det_idx],
                    det_idx,
                    occlusion_context,
                    width,
                    height,
                    cfg,
                    raw_owner,
                    raw_owner_tracks,
                )
                if not np.isfinite(claim_cost) or claim_cost >= 1_000_000.0:
                    continue
                reference = association_reference_box(
                    hidden_track,
                    detections[det_idx],
                    width,
                    height,
                    cfg,
                )
                claim_iom = bbox_iom(reference, detections[det_idx].box)
                claim_center = center_distance_norm(
                    reference,
                    detections[det_idx].box,
                    width,
                    height,
                )
                if claim_cost > cfg.causal_hidden_detection_reservation_max_claim_cost:
                    continue
                if claim_iom < cfg.causal_hidden_detection_reservation_min_iom:
                    continue
                if (
                    claim_center
                    > cfg.causal_hidden_detection_reservation_max_center_distance
                ):
                    continue
                gain = selected_cost - float(claim_cost)
                if gain < cfg.causal_hidden_detection_reservation_min_gain:
                    continue
                hold_eligible = bool(
                    cfg.causal_hidden_detection_reservation_allow_visible_hold
                    and selected_cost
                    >= cfg.causal_hidden_detection_reservation_hold_min_visible_cost
                    and claim_iom
                    >= cfg.causal_hidden_detection_reservation_hold_min_iom
                    and claim_cost
                    <= cfg.causal_hidden_detection_reservation_hold_max_claim_cost
                    and gain
                    >= cfg.causal_hidden_detection_reservation_hold_min_gain
                )
                if not has_alternative and not hold_eligible:
                    continue
                candidates.append(
                    (
                        -gain,
                        row,
                        col,
                        det_idx,
                        hidden_track,
                        float(claim_cost),
                        float(claim_iom),
                        hold_eligible,
                    )
                )

        if not candidates:
            return rows, cols

        (
            _,
            row,
            col,
            det_idx,
            hidden_track,
            claim_cost,
            claim_iom,
            hold_eligible,
        ) = min(
            candidates,
            key=lambda item: (item[0], item[3], item[4].fixed_id),
        )
        visible_track = candidate_tracks[row]
        selected_cost = float(costs[row, col])
        attempted_pairs.add((visible_track.fixed_id, det_idx))
        trial_costs = costs.copy()
        trial_costs[:, col] = 1_000_000.0
        trial_rows, trial_cols = linear_sum_assignment(trial_costs)
        trial_col_by_track = {
            candidate_tracks[trial_row].fixed_id: trial_col
            for trial_row, trial_col in zip(
                trial_rows,
                trial_cols,
                strict=True,
            )
        }
        replacement_col = trial_col_by_track.get(visible_track.fixed_id)
        visible_track_held = bool(
            replacement_col is None or replacement_col == col
        )
        if visible_track_held and not hold_eligible:
            continue
        replacement_cost: float | None = None
        if not visible_track_held:
            replacement_cost = float(trial_costs[row, replacement_col])
            if (
                not np.isfinite(replacement_cost)
                or replacement_cost >= 1_000_000.0
                or replacement_cost
                > cfg.causal_hidden_detection_reservation_max_alternative_cost
            ):
                continue
        reserved_target_assigned = any(
            trial_col == col and trial_costs[trial_row, trial_col] < 1_000_000.0
            for trial_row, trial_col in zip(
                trial_rows,
                trial_cols,
                strict=True,
            )
        )
        if reserved_target_assigned:
            continue

        costs[:, :] = trial_costs
        rows, cols = trial_rows, trial_cols
        reserved_hidden_ids.add(hidden_track.fixed_id)
        reserved_detection_indices.add(det_idx)
        if reserved_hidden_detection_owners is not None:
            reserved_hidden_detection_owners[det_idx] = hidden_track.fixed_id
        append_association_debug_event(
            runtime,
            cfg,
            {
                "event": "assignment_reserve_hidden_detection",
                "frame": frame_index,
                "phase": phase_name,
                "track_id": visible_track.fixed_id,
                "det_idx": det_idx,
                "cost": round(selected_cost, 6),
                "reserved_for_track_id": hidden_track.fixed_id,
                "hidden_claim_cost": round(claim_cost, 6),
                "hidden_claim_iom": round(claim_iom, 6),
                "reservation_gain": round(selected_cost - claim_cost, 6),
                "replacement_cost": (
                    round(replacement_cost, 6)
                    if replacement_cost is not None
                    else None
                ),
                "hidden_missed": hidden_track.missed,
                "visible_track_held": visible_track_held,
                "learn_identity": False,
            },
        )


def realtime_visible_close_competitor_should_prefer(
    selected_track: FixedTrack,
    competitor_track: FixedTrack,
    det: Detection,
    selected_cost: float,
    competitor_cost: float,
    competitor_selected_cost: float | None,
    width: int,
    cfg: TrackingConfig,
    phase_name: str,
) -> bool:
    """Resolve near-tie visible realtime assignments toward the unserved track."""
    if not cfg.realtime_visible_close_competitor_guard:
        return False
    if cfg.mode != "realtime" or cfg.occlusion_aware_matching:
        return False
    if phase_name != "visible_high_conf":
        return False
    if det.score < cfg.track_high_conf:
        return False
    min_center_x_ratio = (
        cfg.realtime_visible_close_competitor_min_center_x_ratio
    )
    if min_center_x_ratio > 0.0:
        if width <= 0:
            return False
        det_center_x, _ = bbox_center(det.box)
        if det_center_x / float(width) < min_center_x_ratio:
            return False
    if selected_cost > cfg.realtime_visible_close_competitor_max_cost:
        return False
    if competitor_cost > cfg.realtime_visible_close_competitor_max_cost:
        return False
    if competitor_cost - selected_cost < 0.0:
        return False
    if (
        competitor_cost - selected_cost
        > cfg.realtime_visible_close_competitor_margin
    ):
        return False
    if (
        selected_track.hits < cfg.realtime_visible_close_competitor_min_hits
        or competitor_track.hits < cfg.realtime_visible_close_competitor_min_hits
    ):
        return False
    if not track_is_visible_for_association(selected_track):
        return False
    if not track_is_visible_for_association(competitor_track):
        return False
    if competitor_selected_cost is None:
        return True
    return (
        competitor_selected_cost - competitor_cost
        > cfg.realtime_visible_close_competitor_margin
    )


def realtime_visible_better_competitor_should_reject(
    selected_track: FixedTrack,
    competitor_track: FixedTrack,
    selected_cost: float,
    competitor_cost: float,
    cfg: TrackingConfig,
    phase_name: str,
) -> bool:
    """Reject high-cost visible assignments when another track is clearly better."""
    if not cfg.realtime_visible_better_competitor_reject:
        return False
    if cfg.mode != "realtime" or cfg.occlusion_aware_matching:
        return False
    if phase_name != "visible_high_conf":
        return False
    if selected_cost < cfg.realtime_visible_better_competitor_min_cost:
        return False
    if (
        selected_cost - competitor_cost
        < cfg.realtime_visible_better_competitor_min_gain
    ):
        return False
    if not track_is_visible_for_association(selected_track):
        return False
    return track_is_visible_for_association(competitor_track)


def realtime_visible_better_competitor_should_prefer(
    selected_track: FixedTrack,
    competitor_track: FixedTrack,
    selected_cost: float,
    competitor_cost: float,
    competitor_selected_cost: float | None,
    cfg: TrackingConfig,
    phase_name: str,
) -> bool:
    """Move a clearly bad visible realtime assignment to a much better track."""
    if not cfg.realtime_visible_better_competitor_prefer:
        return False
    if cfg.mode != "realtime" or cfg.occlusion_aware_matching:
        return False
    if phase_name != "visible_high_conf":
        return False
    if selected_cost < cfg.realtime_visible_better_competitor_min_cost:
        return False
    if (
        selected_cost - competitor_cost
        < cfg.realtime_visible_better_competitor_min_gain
    ):
        return False
    if not track_is_visible_for_association(selected_track):
        return False
    if not track_is_visible_for_association(competitor_track):
        return False
    if competitor_selected_cost is None:
        return True
    return (
        competitor_selected_cost - competitor_cost
        >= cfg.realtime_visible_better_competitor_min_gain
    )


def realtime_low_conf_recovery_should_reject(
    track: FixedTrack,
    det: Detection,
    cfg: TrackingConfig,
    phase_name: str,
) -> bool:
    """Avoid reviving hidden realtime tracks from very low-confidence detections."""
    if not cfg.realtime_low_conf_recovery_guard:
        return False
    if cfg.mode != "realtime":
        return False
    if phase_name != "low_conf_recovery":
        return False
    if det.score >= cfg.realtime_low_conf_recovery_min_score:
        return False
    if track.missed < cfg.realtime_low_conf_recovery_min_missed:
        return False
    if track.missed > cfg.realtime_low_conf_recovery_max_missed:
        return False
    return not track_is_visible_for_association(track)


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
    frame_index: int | None = None,
    det: Detection | None = None,
    owner_track: FixedTrack | None = None,
    selected_cost: float | None = None,
) -> bool:
    if not cfg.reentry_ambiguous_hold:
        return False
    if not ambiguous:
        return False
    if not video_in_reentry_ambiguous_hold_scope(cfg):
        return False
    if not frame_in_reentry_ambiguous_hold_window(frame_index, cfg):
        return False
    if not reentry_raw_evidence_allows_hold(track, det, owner_track, cfg):
        return False
    if not reentry_assignment_cost_allows_hold(track, selected_cost, cfg):
        return False
    if not track.ever_detected or track.hits < cfg.reentry_ambiguous_hold_min_hits:
        return False
    return (
        track.get_state() in {"OCCLUDED", "LOST"}
        or track.missed >= cfg.reentry_ambiguous_hold_min_missed
        or track.state_reason in {"prediction_only", "occlusion_hold"}
    )


def reentry_raw_evidence_allows_hold(
    track: FixedTrack,
    det: Detection | None,
    owner_track: FixedTrack | None,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.reentry_ambiguous_hold_raw_evidence_only:
        return True
    if det is None or det.raw_id is None:
        return False
    owner_is_other_track = (
        owner_track is not None
        and owner_track.fixed_id != track.fixed_id
    )
    top_raw_id = track.top_raw_id()
    raw_id_mismatch = top_raw_id is not None and det.raw_id != top_raw_id
    return owner_is_other_track or raw_id_mismatch


def reentry_assignment_cost_allows_hold(
    track: FixedTrack,
    selected_cost: float | None,
    cfg: TrackingConfig,
) -> bool:
    if (
        cfg.reentry_ambiguous_hold_max_missed > 0
        and track.missed > cfg.reentry_ambiguous_hold_max_missed
    ):
        return False
    if selected_cost is None:
        return True
    return (
        cfg.reentry_ambiguous_hold_min_cost
        <= selected_cost
        <= cfg.reentry_ambiguous_hold_max_cost
    )


def reentry_unowned_raw_mismatch_should_reject(
    track: FixedTrack,
    det: Detection,
    owner_track: FixedTrack | None,
    ambiguous: bool,
    selected_cost: float,
    cfg: TrackingConfig,
    runtime: TrackingRuntimeState | None = None,
) -> bool:
    if not cfg.reentry_unowned_raw_mismatch_reject:
        return False
    if not ambiguous:
        return False
    if det.raw_id is None or owner_track is not None:
        return False
    top_raw_id = track.top_raw_id()
    if top_raw_id is None or det.raw_id == top_raw_id:
        return False
    raw_is_quarantined = (
        runtime is not None
        and runtime.reentry_unowned_raw_quarantine.get(det.raw_id, 0) > 0
    )
    if not track.ever_detected or track.hits < cfg.reentry_ambiguous_hold_min_hits:
        return False
    if raw_is_quarantined:
        return selected_cost <= cfg.reentry_unowned_raw_mismatch_quarantine_max_cost
    if track.missed < cfg.reentry_unowned_raw_mismatch_min_missed:
        return False
    if (
        cfg.reentry_unowned_raw_mismatch_max_missed > 0
        and track.missed > cfg.reentry_unowned_raw_mismatch_max_missed
    ):
        return False
    if selected_cost > cfg.reentry_unowned_raw_mismatch_max_cost:
        return False
    return (
        track.get_state() in {"OCCLUDED", "LOST"}
        or track.state_reason in {"prediction_only", "occlusion_hold"}
    )


def reentry_unowned_raw_mismatch_episode_should_reject(
    track: FixedTrack,
    det: Detection,
    owner_track: FixedTrack | None,
    ambiguous: bool,
    selected_cost: float,
    cfg: TrackingConfig,
    runtime: TrackingRuntimeState | None,
    frame_index: int | None,
    phase_name: str,
) -> bool:
    if not cfg.reentry_unowned_raw_mismatch_episode_reject:
        return False
    if runtime is None or frame_index is None:
        return False
    if not ambiguous:
        return False
    if not phase_in_reentry_unowned_raw_mismatch_episode_scope(phase_name, cfg):
        return False
    if det.raw_id is None or owner_track is not None:
        return False
    top_raw_id = track.top_raw_id()
    if top_raw_id is None or det.raw_id == top_raw_id:
        return False
    if not track.ever_detected or track.hits < cfg.reentry_ambiguous_hold_min_hits:
        return False
    key = (track.fixed_id, top_raw_id, det.raw_id)
    event_count = update_reentry_unowned_raw_mismatch_episode_history(
        runtime,
        key,
        frame_index,
        cfg,
    )
    if track.missed < cfg.reentry_unowned_raw_mismatch_episode_min_missed:
        return False
    if (
        cfg.reentry_unowned_raw_mismatch_episode_max_missed > 0
        and track.missed > cfg.reentry_unowned_raw_mismatch_episode_max_missed
    ):
        return False
    if not (
        cfg.reentry_unowned_raw_mismatch_episode_min_cost
        <= selected_cost
        <= cfg.reentry_unowned_raw_mismatch_episode_max_cost
    ):
        return False
    if not (
        track.get_state() in {"OCCLUDED", "LOST"}
        or track.state_reason in {"prediction_only", "occlusion_hold"}
    ):
        return False
    if event_count < cfg.reentry_unowned_raw_mismatch_episode_min_events:
        return False
    if (
        cfg.reentry_unowned_raw_mismatch_episode_max_events > 0
        and event_count > cfg.reentry_unowned_raw_mismatch_episode_max_events
    ):
        return False
    return True


def update_reentry_unowned_raw_mismatch_episode_history(
    runtime: TrackingRuntimeState,
    key: tuple[int, int, int],
    frame_index: int,
    cfg: TrackingConfig,
) -> int:
    window_start = frame_index - cfg.reentry_unowned_raw_mismatch_episode_window_frames
    history = [
        prior_frame
        for prior_frame in runtime.reentry_unowned_raw_episode_history.get(key, [])
        if prior_frame >= window_start
    ]
    history.append(frame_index)
    runtime.reentry_unowned_raw_episode_history[key] = history
    return len(history)


def phase_in_reentry_unowned_raw_mismatch_episode_scope(
    phase_name: str,
    cfg: TrackingConfig,
) -> bool:
    phases = {
        item.strip()
        for item in cfg.reentry_unowned_raw_mismatch_episode_phases.split(",")
        if item.strip()
    }
    return not phases or phase_name in phases


def occlusion_reid_bad_match_should_hold(
    track: FixedTrack,
    det: Detection,
    ambiguous: bool,
    selected_cost: float,
    cfg: TrackingConfig,
    phase_name: str,
    runtime: TrackingRuntimeState | None = None,
    owner_track: FixedTrack | None = None,
) -> bool:
    if not cfg.occlusion_reid_prefer_gap_over_bad_match:
        return False
    visible_ambiguous_high_cost = (
        cfg.occlusion_reid_bad_match_include_recent_visible
        and phase_name == "visible"
        and ambiguous
        and selected_cost >= cfg.occlusion_reid_bad_match_visible_min_cost
    )
    if phase_name != "reid" and not visible_ambiguous_high_cost:
        return False
    if not ambiguous:
        return False
    if selected_cost < cfg.occlusion_reid_bad_match_min_cost:
        return False
    if selected_cost > cfg.occlusion_reid_bad_match_max_cost:
        return False
    if (
        cfg.occlusion_reid_bad_match_min_missed > 0
        and track.missed < cfg.occlusion_reid_bad_match_min_missed
    ):
        return False
    if (
        cfg.occlusion_reid_bad_match_max_missed >= 0
        and track.missed > cfg.occlusion_reid_bad_match_max_missed
    ):
        return False
    if cfg.occlusion_reid_bad_match_occlusion_hold_only and track.last_source != "occlusion_hold":
        return False
    if track.get_state() not in {"OCCLUDED", "LOST"} and not visible_ambiguous_high_cost:
        return False
    top_raw_id = track.top_raw_id()
    is_raw_mismatch = top_raw_id is not None and det.raw_id != top_raw_id
    if cfg.occlusion_reid_bad_match_unowned_raw_only and owner_track is not None:
        return False
    if cfg.occlusion_reid_bad_match_raw_mismatch_only:
        should_hold = is_raw_mismatch
    elif cfg.occlusion_reid_bad_match_same_raw_only:
        should_hold = top_raw_id is not None and det.raw_id == top_raw_id
    else:
        should_hold = True
    if not should_hold:
        return False
    if cfg.occlusion_reid_bad_match_once_per_episode:
        if runtime is None:
            return False
        episode_key = (track.fixed_id, det.raw_id)
        if episode_key in runtime.occlusion_reid_bad_match_hold_keys:
            return False
        runtime.occlusion_reid_bad_match_hold_keys.add(episode_key)
    return True


def reid_unowned_competing_candidate_should_hold(
    track: FixedTrack,
    det: Detection,
    owner_track: FixedTrack | None,
    ambiguous: bool,
    selected_cost: float,
    competing_cost: float | None,
    cfg: TrackingConfig,
    phase_name: str,
) -> bool:
    if not cfg.reid_unowned_competing_candidate_hold:
        return False
    if phase_name != "reid" or not ambiguous:
        return False
    if det.raw_id is None or owner_track is not None:
        return False
    top_raw_id = track.top_raw_id()
    if top_raw_id is None or top_raw_id == det.raw_id:
        return False
    if selected_cost < cfg.reid_unowned_competing_candidate_min_cost:
        return False
    if track.missed < cfg.reid_unowned_competing_candidate_min_missed:
        return False
    if cfg.reid_unowned_competing_candidate_occlusion_hold_only and not (
        track.last_source == "occlusion_hold" or track.state_reason == "occlusion_hold"
    ):
        return False
    if competing_cost is None:
        return False
    return (
        selected_cost - competing_cost
        >= cfg.reid_unowned_competing_candidate_min_gap
    )


def advance_reentry_unowned_raw_quarantine(
    runtime: TrackingRuntimeState | None,
) -> None:
    if runtime is None or not runtime.reentry_unowned_raw_quarantine:
        return
    expired: list[int] = []
    for raw_id, remaining in runtime.reentry_unowned_raw_quarantine.items():
        next_remaining = remaining - 1
        if next_remaining <= 0:
            expired.append(raw_id)
        else:
            runtime.reentry_unowned_raw_quarantine[raw_id] = next_remaining
    for raw_id in expired:
        runtime.reentry_unowned_raw_quarantine.pop(raw_id, None)


def seed_reentry_unowned_raw_quarantine(
    runtime: TrackingRuntimeState | None,
    det: Detection,
    cfg: TrackingConfig,
    selected_cost: float,
) -> None:
    if runtime is None or det.raw_id is None:
        return
    if cfg.reentry_unowned_raw_mismatch_quarantine_frames <= 0:
        return
    if selected_cost < cfg.reentry_unowned_raw_mismatch_quarantine_min_seed_cost:
        return
    runtime.reentry_unowned_raw_quarantine[det.raw_id] = max(
        runtime.reentry_unowned_raw_quarantine.get(det.raw_id, 0),
        cfg.reentry_unowned_raw_mismatch_quarantine_frames,
    )


def video_in_reentry_ambiguous_hold_scope(cfg: TrackingConfig) -> bool:
    video_stems = cfg.reentry_ambiguous_hold_video_stems.strip()
    if not video_stems:
        return True
    current_stem = cfg.video_path.stem
    allowed = {
        item.strip()
        for item in video_stems.split(",")
        if item.strip()
    }
    return current_stem in allowed


def frame_in_reentry_ambiguous_hold_window(
    frame_index: int | None,
    cfg: TrackingConfig,
) -> bool:
    windows = cfg.reentry_ambiguous_hold_frame_windows.strip()
    if not windows:
        return True
    if frame_index is None:
        return False
    for item in windows.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
        else:
            start = end = int(item)
        if start > end:
            start, end = end, start
        if start <= frame_index <= end:
            return True
    return False


def apply_realtime_core_unassigned_tiebreak(
    costs: np.ndarray,
    candidate_tracks: list[FixedTrack],
    detection_indices: list[int],
    detections: list[Detection],
    rows: np.ndarray,
    cols: np.ndarray,
    cfg: TrackingConfig,
    phase_name: str,
    runtime: TrackingRuntimeState | None = None,
    frame_index: int | None = None,
    mean_core_cache: dict[int, np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use core appearance only to break a near-tie with an unused detection."""
    if (
        not cfg.realtime_core_unassigned_tiebreak
        or cfg.mode != "realtime"
        or phase_name != "visible_high_conf"
        or len(detection_indices) <= len(cols)
    ):
        return rows, cols

    if mean_core_cache is None:
        mean_core_cache = {}
    selected_cols = {int(col) for col in cols}
    proposals: list[tuple[float, float, int, int, int, float, float]] = []
    for row, selected_col in zip(rows, cols, strict=True):
        track = candidate_tracks[int(row)]
        selected_det = detections[detection_indices[int(selected_col)]]
        if selected_det.core_hist is None:
            continue
        selected_cost = float(costs[int(row), int(selected_col)])
        if selected_cost > cfg.realtime_core_unassigned_max_selected_cost:
            continue
        eligible_alternatives: list[tuple[int, float, Detection]] = []
        for alternative_col in range(costs.shape[1]):
            if alternative_col in selected_cols:
                continue
            alternative_cost = float(costs[int(row), alternative_col])
            if not np.isfinite(alternative_cost) or alternative_cost >= 1_000_000.0:
                continue
            baseline_delta = alternative_cost - selected_cost
            if baseline_delta > cfg.realtime_core_unassigned_max_cost_delta:
                continue
            alternative_det = detections[detection_indices[alternative_col]]
            if alternative_det.core_hist is None:
                continue
            if (
                cfg.realtime_core_unassigned_require_score_nondecrease
                and alternative_det.score < selected_det.score
            ):
                continue
            if bbox_iou(selected_det.box, alternative_det.box) < (
                cfg.realtime_core_unassigned_min_detection_iou
            ):
                continue
            eligible_alternatives.append(
                (alternative_col, baseline_delta, alternative_det)
            )

        if not eligible_alternatives:
            continue
        track_id = track.fixed_id
        if track_id not in mean_core_cache:
            mean_core_cache[track_id] = track.mean_core_hist()
        mean_core_hist = mean_core_cache[track_id]
        if mean_core_hist is None:
            continue
        selected_core_cost = hist_distance(mean_core_hist, selected_det.core_hist)
        for alternative_col, baseline_delta, alternative_det in (
            eligible_alternatives
        ):
            alternative_core_cost = hist_distance(
                mean_core_hist,
                alternative_det.core_hist,
            )
            appearance_gain = selected_core_cost - alternative_core_cost
            if appearance_gain < cfg.realtime_core_unassigned_min_appearance_gain:
                continue
            proposals.append(
                (
                    -appearance_gain,
                    baseline_delta,
                    int(row),
                    int(selected_col),
                    alternative_col,
                    selected_core_cost,
                    alternative_core_cost,
                )
            )

    updated_cols = cols.copy()
    claimed_alternatives: set[int] = set()
    for proposal in sorted(proposals):
        (
            neg_gain,
            baseline_delta,
            row,
            selected_col,
            alternative_col,
            selected_core_cost,
            alternative_core_cost,
        ) = proposal
        if alternative_col in claimed_alternatives:
            continue
        row_position = np.flatnonzero(rows == row)
        if row_position.size != 1:
            continue
        position = int(row_position[0])
        if int(updated_cols[position]) != selected_col:
            continue
        updated_cols[position] = alternative_col
        claimed_alternatives.add(alternative_col)
        append_association_debug_event(
            runtime,
            cfg,
            {
                "event": "core_unassigned_tiebreak",
                "frame": frame_index,
                "phase": phase_name,
                "track_id": candidate_tracks[row].fixed_id,
                "selected_det_idx": detection_indices[selected_col],
                "preferred_det_idx": detection_indices[alternative_col],
                "baseline_cost_delta": round(baseline_delta, 6),
                "core_appearance_gain": round(-neg_gain, 6),
                "selected_core_cost": round(selected_core_cost, 6),
                "preferred_core_cost": round(alternative_core_cost, 6),
            },
        )
    return rows, updated_cols


def apply_realtime_core_pairwise_tiebreak(
    costs: np.ndarray,
    candidate_tracks: list[FixedTrack],
    detection_indices: list[int],
    detections: list[Detection],
    rows: np.ndarray,
    cols: np.ndarray,
    cfg: TrackingConfig,
    phase_name: str,
    runtime: TrackingRuntimeState | None = None,
    frame_index: int | None = None,
    mean_core_cache: dict[int, np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use core appearance to reverse a conservative 2x2 LAP near-tie."""
    if (
        not cfg.realtime_core_pairwise_tiebreak
        or cfg.mode != "realtime"
        or phase_name != "visible_high_conf"
        or len(rows) < 2
    ):
        return rows, cols

    proposals: list[
        tuple[
            float,
            float,
            int,
            int,
            int,
            int,
            int,
            int,
            float,
            float,
            float,
            float,
        ]
    ] = []
    if mean_core_cache is None:
        mean_core_cache = {}
    selected_costs = [
        float(costs[int(row), int(col)])
        for row, col in zip(rows, cols, strict=True)
    ]
    selected_is_row_best = [
        selected_cost <= float(np.min(costs[int(row), :])) + 1e-6
        for selected_cost, row in zip(selected_costs, rows, strict=True)
    ]
    for first_position in range(len(rows) - 1):
        first_row = int(rows[first_position])
        first_col = int(cols[first_position])
        first_track = candidate_tracks[first_row]
        first_det = detections[detection_indices[first_col]]
        if first_det.core_hist is None:
            continue

        for second_position in range(first_position + 1, len(rows)):
            if (
                selected_is_row_best[first_position]
                and selected_is_row_best[second_position]
            ):
                continue
            second_row = int(rows[second_position])
            second_col = int(cols[second_position])
            second_track = candidate_tracks[second_row]
            second_det = detections[detection_indices[second_col]]
            if second_det.core_hist is None:
                continue
            if bbox_iou(first_det.box, second_det.box) < (
                cfg.realtime_core_pairwise_min_detection_iou
            ):
                continue

            first_selected_cost = selected_costs[first_position]
            second_selected_cost = selected_costs[second_position]
            first_swapped_cost = float(costs[first_row, second_col])
            second_swapped_cost = float(costs[second_row, first_col])
            pair_costs = (
                first_selected_cost,
                second_selected_cost,
                first_swapped_cost,
                second_swapped_cost,
            )
            if not all(np.isfinite(value) for value in pair_costs):
                continue
            if any(value >= 1_000_000.0 for value in pair_costs):
                continue
            if first_swapped_cost > association_cost_threshold(first_track, cfg):
                continue
            if second_swapped_cost > association_cost_threshold(second_track, cfg):
                continue

            selected_total = first_selected_cost + second_selected_cost
            swapped_total = first_swapped_cost + second_swapped_cost
            cost_increase = swapped_total - selected_total
            if cost_increase < -1e-6 or cost_increase > (
                cfg.realtime_core_pairwise_max_total_cost_increase
            ):
                continue

            first_track_id = first_track.fixed_id
            second_track_id = second_track.fixed_id
            if first_track_id not in mean_core_cache:
                mean_core_cache[first_track_id] = first_track.mean_core_hist()
            if second_track_id not in mean_core_cache:
                mean_core_cache[second_track_id] = second_track.mean_core_hist()
            first_mean_core = mean_core_cache[first_track_id]
            second_mean_core = mean_core_cache[second_track_id]
            if first_mean_core is None or second_mean_core is None:
                continue
            first_selected_core_cost = hist_distance(
                first_mean_core,
                first_det.core_hist,
            )
            first_swapped_core_cost = hist_distance(
                first_mean_core,
                second_det.core_hist,
            )
            second_selected_core_cost = hist_distance(
                second_mean_core,
                second_det.core_hist,
            )
            second_swapped_core_cost = hist_distance(
                second_mean_core,
                first_det.core_hist,
            )
            first_gain = first_selected_core_cost - first_swapped_core_cost
            second_gain = second_selected_core_cost - second_swapped_core_cost
            append_association_debug_event(
                runtime,
                cfg,
                {
                    "event": "core_pairwise_probe",
                    "frame": frame_index,
                    "phase": phase_name,
                    "first_track_id": first_track.fixed_id,
                    "second_track_id": second_track.fixed_id,
                    "first_selected_det_idx": detection_indices[first_col],
                    "second_selected_det_idx": detection_indices[second_col],
                    "total_cost_increase": round(cost_increase, 6),
                    "first_core_gain": round(first_gain, 6),
                    "second_core_gain": round(second_gain, 6),
                    "total_core_appearance_gain": round(
                        first_gain + second_gain,
                        6,
                    ),
                },
            )
            total_gain = first_gain + second_gain
            if total_gain < (
                cfg.realtime_core_pairwise_min_total_appearance_gain
            ):
                continue
            proposals.append(
                (
                    -total_gain,
                    cost_increase,
                    first_position,
                    second_position,
                    first_row,
                    second_row,
                    first_col,
                    second_col,
                    first_selected_core_cost,
                    first_swapped_core_cost,
                    second_selected_core_cost,
                    second_swapped_core_cost,
                )
            )

    updated_cols = cols.copy()
    used_positions: set[int] = set()
    for proposal in sorted(proposals):
        (
            neg_total_gain,
            cost_increase,
            first_position,
            second_position,
            first_row,
            second_row,
            first_col,
            second_col,
            first_selected_core_cost,
            first_swapped_core_cost,
            second_selected_core_cost,
            second_swapped_core_cost,
        ) = proposal
        if (
            first_position in used_positions
            or second_position in used_positions
            or int(updated_cols[first_position]) != first_col
            or int(updated_cols[second_position]) != second_col
        ):
            continue
        updated_cols[first_position] = second_col
        updated_cols[second_position] = first_col
        used_positions.update((first_position, second_position))
        append_association_debug_event(
            runtime,
            cfg,
            {
                "event": "core_pairwise_tiebreak",
                "frame": frame_index,
                "phase": phase_name,
                "first_track_id": candidate_tracks[first_row].fixed_id,
                "second_track_id": candidate_tracks[second_row].fixed_id,
                "first_selected_det_idx": detection_indices[first_col],
                "second_selected_det_idx": detection_indices[second_col],
                "total_cost_increase": round(cost_increase, 6),
                "total_core_appearance_gain": round(-neg_total_gain, 6),
                "first_selected_core_cost": round(
                    first_selected_core_cost,
                    6,
                ),
                "first_swapped_core_cost": round(first_swapped_core_cost, 6),
                "second_selected_core_cost": round(
                    second_selected_core_cost,
                    6,
                ),
                "second_swapped_core_cost": round(second_swapped_core_cost, 6),
            },
        )
    return rows, updated_cols


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

    if runtime is not None:
        runtime.telemetry["association_calls"] = (
            int(runtime.telemetry.get("association_calls", 0)) + 1
        )
    advance_reentry_unowned_raw_quarantine(runtime)
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
    if cfg.mode == "hybrid_bytetrack":
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
    hidden_tracks_for_reservation = [
        track
        for track in ordered_tracks
        if not track_is_visible_for_association(track)
    ]
    reserved_hidden_detection_owners: dict[int, int] = {}

    def run_matching_phase(
        candidate_tracks: list[FixedTrack],
        detection_indices: list[int],
        phase_name: str,
    ) -> None:
        if not candidate_tracks or not detection_indices:
            return
        record_association_phase(runtime, phase_name)

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
        if phase_name == "visible_high_conf":
            mean_core_cache: dict[int, np.ndarray | None] = {}
            rows, cols = apply_causal_hidden_detection_reservation(
                costs,
                candidate_tracks,
                detection_indices,
                detections,
                hidden_tracks_for_reservation,
                matched_tracks,
                occlusion_context,
                width,
                height,
                cfg,
                raw_owner,
                raw_owner_tracks,
                runtime,
                frame_index,
                phase_name,
                reserved_hidden_detection_owners,
            )
            rows, cols = apply_realtime_core_unassigned_tiebreak(
                costs,
                candidate_tracks,
                detection_indices,
                detections,
                rows,
                cols,
                cfg,
                phase_name,
                runtime,
                frame_index,
                mean_core_cache,
            )
            rows, cols = apply_realtime_core_pairwise_tiebreak(
                costs,
                candidate_tracks,
                detection_indices,
                detections,
                rows,
                cols,
                cfg,
                phase_name,
                runtime,
                frame_index,
                mean_core_cache,
            )
        selected_track_by_det = {
            detection_indices[col]: candidate_tracks[row].fixed_id
            for row, col in zip(rows, cols, strict=True)
        }
        selected_col_by_track = {
            candidate_tracks[row].fixed_id: col
            for row, col in zip(rows, cols, strict=True)
        }
        append_detection_candidate_rank_events(
            runtime,
            cfg,
            frame_index,
            phase_name,
            candidate_tracks,
            detection_indices,
            detections,
            costs,
            selected_track_by_det,
            raw_owner,
        )
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
            competing_costs = [
                float(costs[other_row, col])
                for other_row in range(len(candidate_tracks))
                if other_row != row
                and np.isfinite(costs[other_row, col])
                and costs[other_row, col] < 1_000_000.0
            ]
            competing_cost = min(competing_costs) if competing_costs else None
            competitor_track = None
            competitor_selected_cost = None
            if competing_cost is not None:
                for other_row in range(len(candidate_tracks)):
                    if other_row == row:
                        continue
                    other_cost = float(costs[other_row, col])
                    if not np.isfinite(other_cost) or other_cost >= 1_000_000.0:
                        continue
                    if math.isclose(
                        other_cost,
                        competing_cost,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ):
                        competitor_track = candidate_tracks[other_row]
                        competitor_selected_col = selected_col_by_track.get(
                            competitor_track.fixed_id,
                        )
                        if competitor_selected_col is not None:
                            competitor_selected_cost = float(
                                costs[other_row, competitor_selected_col],
                            )
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
                "det_best_competing_cost": (
                    round(competing_cost, 6)
                    if competing_cost is not None
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
            if realtime_low_conf_recovery_should_reject(
                track,
                det,
                cfg,
                phase_name,
            ):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_reject_low_conf_recovery",
                        "learn_identity": False,
                    },
                )
                continue
            if (
                competitor_track is not None
                and realtime_visible_close_competitor_should_prefer(
                    track,
                    competitor_track,
                    det,
                    float(costs[row, col]),
                    float(competing_cost),
                    competitor_selected_cost,
                    width,
                    cfg,
                    phase_name,
                )
            ):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": (
                            "assignment_prefer_visible_close_competitor"
                        ),
                        "ambiguous": False,
                        "preferred_track_id": competitor_track.fixed_id,
                        "preferred_cost": round(float(competing_cost), 6),
                        "learn_identity": True,
                    },
                )
                competitor_track.update_detected(
                    det,
                    width,
                    height,
                    cfg,
                    learn_identity=True,
                    ambiguous=False,
                )
                matched_tracks.add(competitor_track.fixed_id)
                matched_detections.add(det_idx)
                continue
            if (
                competitor_track is not None
                and realtime_visible_better_competitor_should_prefer(
                    track,
                    competitor_track,
                    float(costs[row, col]),
                    float(competing_cost),
                    competitor_selected_cost,
                    cfg,
                    phase_name,
                )
            ):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_prefer_visible_better_competitor",
                        "ambiguous": False,
                        "preferred_track_id": competitor_track.fixed_id,
                        "preferred_cost": round(float(competing_cost), 6),
                        "competitor_selected_cost": (
                            round(competitor_selected_cost, 6)
                            if competitor_selected_cost is not None
                            else None
                        ),
                        "learn_identity": True,
                    },
                )
                competitor_track.update_detected(
                    det,
                    width,
                    height,
                    cfg,
                    learn_identity=True,
                    ambiguous=False,
                )
                matched_tracks.add(competitor_track.fixed_id)
                matched_detections.add(det_idx)
                continue
            if (
                competitor_track is not None
                and realtime_visible_better_competitor_should_reject(
                    track,
                    competitor_track,
                    float(costs[row, col]),
                    float(competing_cost),
                    cfg,
                    phase_name,
                )
            ):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_reject_visible_better_competitor",
                        "ambiguous": False,
                        "better_competitor_track_id": competitor_track.fixed_id,
                        "better_competitor_cost": round(float(competing_cost), 6),
                        "learn_identity": False,
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
            reserved_owner_id = reserved_hidden_detection_owners.get(det_idx)
            if (
                cfg.causal_hidden_detection_reservation_hold_reserved_reid
                and phase_name == "reid"
                and reserved_owner_id == track.fixed_id
            ):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_hold_reserved_hidden_detection",
                        "ambiguous": True,
                        "reserved_for_track_id": reserved_owner_id,
                        "learn_identity": False,
                    },
                )
                freeze_area_occluded_track(track, width, height, cfg)
                matched_tracks.add(track.fixed_id)
                matched_detections.add(det_idx)
                continue
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
            if reentry_unowned_raw_mismatch_should_reject(
                track,
                det,
                owner_track,
                ambiguous,
                float(costs[row, col]),
                cfg,
                runtime,
            ):
                seed_reentry_unowned_raw_quarantine(
                    runtime,
                    det,
                    cfg,
                    float(costs[row, col]),
                )
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_reject_reentry_unowned_raw_mismatch",
                        "ambiguous": ambiguous,
                        "hidden_owner_freeze": False,
                        "learn_identity": False,
                    },
                )
                continue
            if reentry_unowned_raw_mismatch_episode_should_reject(
                track,
                det,
                owner_track,
                ambiguous,
                float(costs[row, col]),
                cfg,
                runtime,
                frame_index,
                phase_name,
            ):
                episode_action = cfg.reentry_unowned_raw_mismatch_episode_action
                if episode_action == "hold":
                    append_association_debug_event(
                        runtime,
                        cfg,
                        {
                            **base_debug_event,
                            "event": (
                                "assignment_hold_reentry_unowned_raw_mismatch_episode"
                            ),
                            "ambiguous": ambiguous,
                            "hidden_owner_freeze": False,
                            "learn_identity": False,
                        },
                    )
                    freeze_area_occluded_track(track, width, height, cfg)
                    matched_tracks.add(track.fixed_id)
                    matched_detections.add(det_idx)
                    continue
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": (
                            "assignment_reject_reentry_unowned_raw_mismatch_episode"
                        ),
                        "ambiguous": ambiguous,
                        "hidden_owner_freeze": False,
                        "learn_identity": False,
                    },
                )
                continue
            if reid_unowned_competing_candidate_should_hold(
                track,
                det,
                owner_track,
                ambiguous,
                float(costs[row, col]),
                competing_cost,
                cfg,
                phase_name,
            ):
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_hold_reid_unowned_competing_candidate",
                        "ambiguous": ambiguous,
                        "hidden_owner_freeze": False,
                        "learn_identity": False,
                    },
                )
                freeze_area_occluded_track(track, width, height, cfg)
                matched_tracks.add(track.fixed_id)
                matched_detections.add(det_idx)
                continue
            if occlusion_reid_bad_match_should_hold(
                track,
                det,
                ambiguous,
                float(costs[row, col]),
                cfg,
                phase_name,
                runtime,
                owner_track=owner_track,
            ):
                if cfg.occlusion_reid_bad_match_action == "reject":
                    append_association_debug_event(
                        runtime,
                        cfg,
                        {
                            **base_debug_event,
                            "event": "assignment_reject_occlusion_reid_bad_match",
                            "ambiguous": ambiguous,
                            "hidden_owner_freeze": False,
                            "learn_identity": False,
                        },
                    )
                    continue
                append_association_debug_event(
                    runtime,
                    cfg,
                    {
                        **base_debug_event,
                        "event": "assignment_hold_occlusion_reid_bad_match",
                        "ambiguous": ambiguous,
                        "hidden_owner_freeze": False,
                        "learn_identity": False,
                    },
                )
                freeze_area_occluded_track(track, width, height, cfg)
                matched_tracks.add(track.fixed_id)
                matched_detections.add(det_idx)
                continue
            if reentry_ambiguous_assignment_should_hold(
                track,
                ambiguous,
                cfg,
                frame_index,
                det,
                owner_track,
                float(costs[row, col]),
            ):
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

        if cfg.mode == "hybrid_bytetrack":
            run_matching_phase(visible_tracks, all_detection_indices, "visible")
            remaining_detection_indices = [
                idx
                for idx in all_detection_indices
                if idx not in matched_detections
            ]
            run_matching_phase(reid_tracks, remaining_detection_indices, "reid")
        else:
            append_hidden_detection_claim_probe_events(
                runtime,
                cfg,
                frame_index,
                reid_tracks,
                all_detection_indices,
                detections,
                occlusion_context,
                width,
                height,
                raw_owner,
                raw_owner_tracks,
            )
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
