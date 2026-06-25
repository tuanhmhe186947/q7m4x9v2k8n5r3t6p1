"""Occlusion and merged-box handling for fixed-ID tracking."""

from __future__ import annotations

import math

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.constants import (
    SCENE_CLEAR,
    SCENE_HARD_MERGED,
    SCENE_HARD_OCCLUSION_ARMED,
    SCENE_SOFT_PROXIMITY,
    SCENE_SPLIT_RECOVERY,
)
from pig_behavior.tracking.detections import hist_distance
from pig_behavior.tracking.geometry import (
    bbox_area,
    bbox_center,
    bbox_iom,
    bbox_iom_matrix,
    bbox_iou,
    center_distance_norm,
    center_distance_norm_matrix,
)
from pig_behavior.tracking.schemas import (
    ConflictGroup,
    Detection,
    FixedTrack,
    HardSceneDecision,
    OcclusionContext,
    TrackingRuntimeState,
)


def track_speed_norm(track: FixedTrack, width: int, height: int) -> float:
    diag = math.sqrt(width * width + height * height)
    return float(np.linalg.norm(track.velocity_xy) / max(diag, 1e-6))


def track_is_stationary_locked(
    track: FixedTrack,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.occlusion_stationary_lock:
        return False
    if track.motion_state == "moving":
        return False
    return (
        track.stationary_frames >= cfg.hidden_stationary_lock_frames
        and track.reliable_speed_norm(width, height) <= cfg.occlusion_stationary_speed
    )


def build_occlusion_context(
    ordered_tracks: list[FixedTrack],
    detections: list[Detection],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> OcclusionContext:
    predicted_boxes = {
        track.fixed_id: track.predicted_box(width, height)
        for track in ordered_tracks
    }
    appearance_costs = {
        (det_idx, track.fixed_id): hist_distance(track.mean_hist(), det.hist)
        for det_idx, det in enumerate(detections)
        for track in ordered_tracks
    }
    if not cfg.occlusion_aware_matching:
        return OcclusionContext(predicted_boxes, set(), {}, {}, appearance_costs)

    occluded_track_ids: set[int] = set()
    for idx, first in enumerate(ordered_tracks):
        first_box = predicted_boxes[first.fixed_id]
        for second in ordered_tracks[idx + 1 :]:
            second_box = predicted_boxes[second.fixed_id]
            if bbox_iom(first_box, second_box) >= cfg.occlusion_track_iom_threshold:
                occluded_track_ids.update({first.fixed_id, second.fixed_id})

    detection_competitors: dict[int, set[int]] = {}
    for det_idx, det in enumerate(detections):
        competitors = {
            track.fixed_id
            for track in ordered_tracks
            if bbox_iom(predicted_boxes[track.fixed_id], det.box)
            >= cfg.occlusion_detection_iom_threshold
        }
        if len(competitors) > 1:
            detection_competitors[det_idx] = competitors
            occluded_track_ids.update(competitors)

    active_detection_owners: dict[int, set[int]] = {}
    for det_idx, det in enumerate(detections):
        owners = {
            track.fixed_id
            for track in ordered_tracks
            if track.ever_detected
            and track.missed == 0
            and track.last_source == "detected"
            and not track.last_ambiguous
            and (
                bbox_iom(predicted_boxes[track.fixed_id], det.box)
                >= cfg.occlusion_detection_iom_threshold
                or center_distance_norm(
                    predicted_boxes[track.fixed_id],
                    det.box,
                    width,
                    height,
                )
                <= cfg.low_conf_max_center_jump
            )
        }
        if owners:
            active_detection_owners[det_idx] = owners

    return OcclusionContext(
        predicted_boxes=predicted_boxes,
        occluded_track_ids=occluded_track_ids,
        detection_competitors=detection_competitors,
        active_detection_owners=active_detection_owners,
        appearance_costs=appearance_costs,
    )


def assignment_is_occlusion_ambiguous(
    track: FixedTrack,
    det_index: int,
    context: OcclusionContext,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.occlusion_aware_matching:
        return False
    competitors = context.detection_competitors.get(det_index, set())
    return track.fixed_id in context.occluded_track_ids or len(competitors) > 1


def should_hold_occluded_track_box(
    track: FixedTrack,
    detections: list[Detection],
    context: OcclusionContext,
    cfg: TrackingConfig,
) -> bool:
    if not cfg.hold_occluded_box or not track.ever_detected:
        return False
    if track.occlusion_hold_frames >= cfg.occlusion_hold_max_frames:
        return False
    reference = track.reliable_box if track.reliable_box is not None else track.last_box
    if track.fixed_id in context.occluded_track_ids or track.last_ambiguous:
        return True
    if 0 < track.occlusion_hold_frames < cfg.occlusion_hold_max_frames:
        return True
    return any(
        bbox_iom(reference, det.box) >= cfg.occlusion_detection_iom_threshold
        for det in detections
    )


def occlusion_assignment_penalty(
    track: FixedTrack,
    det: Detection,
    det_index: int,
    context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> float:
    if not cfg.occlusion_aware_matching or not track.ever_detected:
        return 0.0

    competitors = context.detection_competitors.get(det_index, set())
    in_occlusion = track.fixed_id in context.occluded_track_ids or len(competitors) > 1
    if not in_occlusion:
        return 0.0

    predicted = context.predicted_boxes.get(
        track.fixed_id,
        track.predicted_box(width, height),
    )
    center_cost = center_distance_norm(predicted, det.box, width, height)
    stationary_allowed = (
        cfg.occlusion_stationary_max_center_jump + min(track.missed, 15) * 0.003
    )
    if track_is_stationary_locked(track, width, height, cfg) and (
        center_cost > stationary_allowed
    ):
        return 1_000_000.0

    penalty = 0.0
    if competitors:
        own_overlap = bbox_iom(predicted, det.box)
        other_overlaps = [
            bbox_iom(context.predicted_boxes[other_id], det.box)
            for other_id in competitors
            if other_id != track.fixed_id and other_id in context.predicted_boxes
        ]
        best_other_overlap = max(other_overlaps, default=0.0)
        if track.fixed_id not in competitors:
            penalty += cfg.occlusion_switch_penalty
        elif best_other_overlap > own_overlap + cfg.occlusion_competitor_margin:
            penalty += cfg.occlusion_switch_penalty
        elif competitors:
            own_app = context.appearance_costs.get((det_index, track.fixed_id), 0.5)
            best_other_app = min(
                (
                    context.appearance_costs.get((det_index, other_id), 0.5)
                    for other_id in competitors
                    if other_id != track.fixed_id
                ),
                default=0.5,
            )
            if own_app > best_other_app + cfg.occlusion_appearance_margin:
                penalty += cfg.occlusion_appearance_penalty

    top_raw_id = track.top_raw_id()
    if det.raw_id is not None and top_raw_id is not None and det.raw_id != top_raw_id:
        penalty += 0.15

    return float(penalty)


def detection_is_reserved_for_active_track(
    track: FixedTrack,
    det: Detection,
    det_index: int,
    context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> bool:
    """Keep reacquiring hidden tracks from stealing an active track's detection."""
    if track.missed == 0 and track.last_source == "detected":
        return False

    owners = context.active_detection_owners.get(det_index, set()) - {track.fixed_id}
    if not owners:
        return False

    top_raw_id = track.top_raw_id()
    if det.raw_id is not None and top_raw_id is not None and det.raw_id == top_raw_id:
        return False

    predicted = context.predicted_boxes.get(
        track.fixed_id,
        track.predicted_box(width, height),
    )
    own_overlap = bbox_iom(predicted, det.box)
    own_center = center_distance_norm(predicted, det.box, width, height)
    owner_overlaps = [
        bbox_iom(context.predicted_boxes[owner_id], det.box)
        for owner_id in owners
        if owner_id in context.predicted_boxes
    ]
    owner_centers = [
        center_distance_norm(context.predicted_boxes[owner_id], det.box, width, height)
        for owner_id in owners
        if owner_id in context.predicted_boxes
    ]
    best_owner_overlap = max(owner_overlaps, default=0.0)
    best_owner_center = min(owner_centers, default=1.0)

    clearly_better_than_owner = (
        own_overlap > best_owner_overlap + cfg.occlusion_competitor_margin
        and own_center < best_owner_center
    )
    return not clearly_better_than_owner


def area_occlusion_should_freeze(
    track: FixedTrack,
    det: Detection,
    det_index: int,
    detections: list[Detection],
    context: OcclusionContext,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> bool:
    """Detect sudden area shrinkage that likely means partial occlusion."""
    if (
        not cfg.USE_AREA_OCCLUSION_FREEZE
        and not cfg.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE
    ):
        return False
    if not track.ever_detected:
        return False
    if track.area_occlusion_frames >= cfg.area_occlusion_freeze_frames:
        return False
    previous_box = (
        track.reliable_box if track.reliable_box is not None else track.last_box
    )
    previous_area = bbox_area(previous_box)
    current_area = bbox_area(det.box)
    is_shrunk = current_area < cfg.area_occlusion_shrink_ratio * previous_area
    if not is_shrunk:
        return False
    if cfg.USE_AREA_OCCLUSION_FREEZE:
        return True

    competitors = context.detection_competitors.get(det_index, set())
    active_tracks_in_scene = sum(
        1 for box in context.predicted_boxes.values() if bbox_iom(box, det.box) > 0.0
    )
    detected_track_count = sum(
        1 for box in context.predicted_boxes.values() if box.size
    )
    detection_deficit = len(detections) < detected_track_count
    hard_scene_hint = track.hard_occlusion_frames > 0 or track.last_ambiguous
    track_in_competition = (
        len(competitors) > 1
        and track.fixed_id in competitors
        and track.fixed_id in context.occluded_track_ids
    )
    nearby_overlap_conflict = (
        track.fixed_id in context.occluded_track_ids
        and active_tracks_in_scene >= 2
        and bbox_iom(previous_box, det.box) >= cfg.occlusion_detection_iom_threshold
    )
    center_jump = center_distance_norm(previous_box, det.box, width, height)
    not_simple_reacquire = center_jump <= cfg.occlusion_stationary_max_center_jump * 2.0
    return bool(
        (track_in_competition or nearby_overlap_conflict or hard_scene_hint)
        and (detection_deficit or len(competitors) > 1 or hard_scene_hint)
        and not_simple_reacquire
    )


def freeze_area_occluded_track(
    track: FixedTrack,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> None:
    """Hold a fixed ID in place for a bounded partial-occlusion window."""
    hold_box = track.reliable_box if track.reliable_box is not None else track.last_box
    track.update_predicted(
        hold_box.copy(),
        width,
        height,
        ambiguous=True,
        hold=True,
    )
    track.is_area_occluded = True
    track.area_occlusion_frames += 1


def apply_iou_fallback(
    tracks: dict[int, FixedTrack],
    detections: list[Detection],
    matched_tracks: set[int],
    matched_detections: set[int],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> None:
    """Reconnect an unmatched fixed ID to a plausible unassigned detection."""
    if not cfg.USE_IOU_FALLBACK:
        return

    remaining_detection_indices = [
        idx for idx in range(len(detections)) if idx not in matched_detections
    ]
    if not remaining_detection_indices:
        return

    ordered_tracks = [
        tracks[idx]
        for idx in range(1, cfg.expected_pigs + 1)
        if idx not in matched_tracks and tracks[idx].ever_detected
    ]
    for track in ordered_tracks:
        predicted = track.predicted_box(width, height)
        best_idx = None
        best_iou = cfg.iou_fallback_threshold
        for det_idx in remaining_detection_indices:
            score = bbox_iou(predicted, detections[det_idx].box)
            if score > best_iou:
                best_iou = score
                best_idx = det_idx
        if best_idx is None:
            continue

        track.update_detected(
            detections[best_idx],
            width,
            height,
            cfg,
            learn_identity=False,
            ambiguous=True,
        )
        matched_tracks.add(track.fixed_id)
        matched_detections.add(best_idx)
        remaining_detection_indices.remove(best_idx)
        if not remaining_detection_indices:
            return


def build_local_conflict_groups(
    ordered_tracks: list[FixedTrack],
    detections: list[Detection],
    predicted_boxes: dict[int, np.ndarray],
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[ConflictGroup]:
    """Build connected local track/detection groups for occlusion actions."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    active_tracks = [track for track in ordered_tracks if track.ever_detected]
    if len(active_tracks) < 2:
        return []

    track_ids = [track.fixed_id for track in active_tracks]
    track_boxes = np.stack(
        [predicted_boxes[track.fixed_id] for track in active_tracks],
        axis=0,
    ).astype(np.float32)
    det_boxes = (
        np.stack([det.box for det in detections], axis=0).astype(np.float32)
        if detections
        else np.zeros((0, 4), dtype=np.float32)
    )
    track_count = len(active_tracks)
    detection_count = len(detections)
    node_count = track_count + detection_count
    adjacency = np.zeros((node_count, node_count), dtype=bool)

    track_iom = bbox_iom_matrix(track_boxes, track_boxes)
    track_distance = center_distance_norm_matrix(
        track_boxes,
        track_boxes,
        width,
        height,
    )
    track_edges = (
        (track_iom >= cfg.occlusion_track_iom_threshold)
        | (track_distance <= cfg.merged_box_neighbor_distance)
    )
    np.fill_diagonal(track_edges, False)
    adjacency[:track_count, :track_count] = track_edges

    if detection_count:
        track_detection_iom = bbox_iom_matrix(track_boxes, det_boxes)
        track_detection_distance = center_distance_norm_matrix(
            track_boxes,
            det_boxes,
            width,
            height,
        )
        track_detection_edges = (
            track_detection_iom >= cfg.occlusion_detection_iom_threshold
        ) | (
            track_detection_distance
            <= max(cfg.low_conf_max_center_jump, cfg.merged_box_neighbor_distance)
        )
        adjacency[:track_count, track_count:] = track_detection_edges
        adjacency[track_count:, :track_count] = track_detection_edges.T

    groups: list[ConflictGroup] = []
    _, labels = connected_components(csr_matrix(adjacency), directed=False)
    for component_id in np.unique(labels):
        component_nodes = np.flatnonzero(labels == component_id)
        component_track_nodes = component_nodes[component_nodes < track_count]
        if len(component_track_nodes) < 2:
            continue
        component_detection_nodes = component_nodes[component_nodes >= track_count]
        groups.append(
            ConflictGroup(
                track_ids={track_ids[idx] for idx in component_track_nodes},
                detection_indices={
                    int(idx - track_count) for idx in component_detection_nodes
                },
            )
        )
    return groups


def tracks_are_moving_toward_each_other(
    first: FixedTrack,
    second: FixedTrack,
) -> bool:
    """Return True when relative motion is closing the distance between tracks."""
    first_center = np.array(bbox_center(first.last_box), dtype=np.float32)
    second_center = np.array(bbox_center(second.last_box), dtype=np.float32)
    relative_center = second_center - first_center
    relative_velocity = second.velocity_xy - first.velocity_xy
    if np.linalg.norm(relative_velocity) <= 1e-6:
        relative_velocity = second.reliable_velocity_xy - first.reliable_velocity_xy
    if np.linalg.norm(relative_velocity) <= 1e-6:
        return False
    return float(np.dot(relative_center, relative_velocity)) < 0.0


def reliable_track_area(track: FixedTrack) -> float:
    reference = track.reliable_box if track.reliable_box is not None else track.last_box
    return bbox_area(reference)


def detection_covers_group_tracks(
    det: Detection,
    tracks: list[FixedTrack],
    predicted_boxes: dict[int, np.ndarray],
    cfg: TrackingConfig,
) -> list[FixedTrack]:
    """Return local tracks substantially covered by a candidate merged detection."""
    covered: list[FixedTrack] = []
    for track in tracks:
        predicted = predicted_boxes[track.fixed_id]
        if (
            bbox_iom(predicted, det.box)
            >= cfg.hard_occlusion_detection_iom_threshold
        ):
            covered.append(track)
    return covered


def conflict_group_key(group: ConflictGroup) -> tuple[int, ...]:
    return tuple(sorted(group.track_ids))


def advance_split_recovery(runtime: TrackingRuntimeState) -> None:
    """Advance active split-recovery windows and expose recovery track IDs."""
    runtime.current_recovery_track_ids.clear()
    next_recovery: dict[tuple[int, ...], int] = {}
    for group_key, remaining in runtime.group_recovery_remaining.items():
        if remaining <= 0:
            continue
        runtime.current_recovery_track_ids.update(group_key)
        runtime.telemetry["recovery_frames_applied"] += 1
        next_recovery[group_key] = remaining - 1
    runtime.group_recovery_remaining = next_recovery


def hard_scene_decision_for_group(
    group: ConflictGroup,
    tracks_by_id: dict[int, FixedTrack],
    detections: list[Detection],
    predicted_boxes: dict[int, np.ndarray],
    cfg: TrackingConfig,
    runtime: TrackingRuntimeState | None = None,
) -> HardSceneDecision:
    """Classify a local group as normal proximity, hard occlusion, or merged."""
    group_tracks = [
        tracks_by_id[track_id]
        for track_id in sorted(group.track_ids)
        if tracks_by_id[track_id].ever_detected
    ]
    if len(group_tracks) < 2:
        return HardSceneDecision(SCENE_CLEAR, False, False, 0.0, False, False)

    group_key = conflict_group_key(group)
    previous_state = (
        runtime.group_states.get(group_key, SCENE_CLEAR)
        if runtime is not None
        else SCENE_CLEAR
    )
    group_boxes = np.stack(
        [predicted_boxes[track.fixed_id] for track in group_tracks],
        axis=0,
    ).astype(np.float32)
    group_iom = bbox_iom_matrix(group_boxes, group_boxes)
    upper = np.triu_indices(len(group_tracks), k=1)
    max_track_iom = float(np.max(group_iom[upper])) if upper[0].size else 0.0

    centers = np.array([bbox_center(track.last_box) for track in group_tracks])
    velocities = np.stack(
        [
            (
                track.velocity_xy
                if np.linalg.norm(track.velocity_xy) > 1e-6
                else track.reliable_velocity_xy
            )
            for track in group_tracks
        ],
        axis=0,
    )
    relative_centers = centers[None, :, :] - centers[:, None, :]
    relative_velocities = velocities[None, :, :] - velocities[:, None, :]
    closing_scores = np.einsum("ijk,ijk->ij", relative_centers, relative_velocities)
    moving_toward = bool(
        upper[0].size
        and np.any(
            (closing_scores[upper] < 0.0)
            & (np.linalg.norm(relative_velocities[upper], axis=1) > 1e-6)
        )
    )

    local_detections = [detections[idx] for idx in sorted(group.detection_indices)]
    has_detection_deficit = len(local_detections) < len(group_tracks)
    has_identity_ambiguity = False
    has_oversized_detection = False
    if local_detections:
        det_boxes = np.stack([det.box for det in local_detections], axis=0).astype(
            np.float32
        )
        coverage = (
            bbox_iom_matrix(group_boxes, det_boxes)
            >= cfg.hard_occlusion_detection_iom_threshold
        )
        covered_counts = coverage.sum(axis=0)
        has_identity_ambiguity = bool(np.any(covered_counts >= 2))
        reliable_areas = np.array(
            [reliable_track_area(track) for track in group_tracks],
            dtype=np.float32,
        )
        detection_areas = np.array([bbox_area(det.box) for det in local_detections])
        for det_col in np.flatnonzero(covered_counts >= 2):
            covered_areas = reliable_areas[coverage[:, det_col]]
            if detection_areas[det_col] > cfg.merged_box_growth_ratio * float(
                np.median(covered_areas)
            ):
                has_oversized_detection = True
                break

    overlap_signal = (
        1.0 if max_track_iom >= cfg.hard_occlusion_track_iom_threshold else 0.0
    )
    previous_hard_frames = (
        runtime.group_hard_frames.get(group_key, 0) if runtime is not None else 0
    )
    soft_proximity = bool(
        overlap_signal
        or max_track_iom >= cfg.occlusion_track_iom_threshold
        or has_identity_ambiguity
    )
    hard_evidence = bool(
        has_identity_ambiguity or has_detection_deficit or moving_toward
    )
    current_hard_frames = previous_hard_frames + 1 if hard_evidence else 0
    duration_signal = min(
        1.0,
        (current_hard_frames + overlap_signal) / cfg.hard_occlusion_min_frames,
    )
    score = (
        0.30 * duration_signal
        + 0.25 * float(has_identity_ambiguity)
        + 0.20 * float(has_detection_deficit)
        + 0.15 * float(has_oversized_detection)
        + 0.10 * float(moving_toward)
    )
    severe_merged_evidence = (
        has_detection_deficit and has_oversized_detection and has_identity_ambiguity
    )
    hard_armed = bool(
        severe_merged_evidence
        or (
            score >= cfg.hard_occlusion_score_threshold
            and current_hard_frames >= cfg.hard_occlusion_min_frames
        )
    )
    if severe_merged_evidence:
        state = SCENE_HARD_MERGED
    elif hard_armed:
        state = SCENE_HARD_OCCLUSION_ARMED
    elif previous_state == SCENE_HARD_MERGED:
        state = SCENE_SPLIT_RECOVERY
    elif soft_proximity:
        state = SCENE_SOFT_PROXIMITY
    else:
        state = SCENE_CLEAR

    if runtime is not None:
        runtime.group_hard_frames[group_key] = current_hard_frames
        if previous_state != SCENE_HARD_MERGED and state == SCENE_HARD_MERGED:
            runtime.telemetry["hard_merges_triggered"] += 1
        if previous_state == SCENE_HARD_MERGED and state == SCENE_SPLIT_RECOVERY:
            runtime.group_recovery_remaining[group_key] = (
                max(0, cfg.hard_occlusion_recovery_frames - 1)
            )
            runtime.current_recovery_track_ids.update(group_key)
            runtime.telemetry["recovery_frames_applied"] += 1
        runtime.group_states[group_key] = state

    is_hard_occlusion = state in {
        SCENE_HARD_OCCLUSION_ARMED,
        SCENE_HARD_MERGED,
    }
    is_merged = state == SCENE_HARD_MERGED
    return HardSceneDecision(
        state=state,
        is_hard_occlusion=is_hard_occlusion,
        is_merged=is_merged,
        score=float(score),
        has_detection_deficit=has_detection_deficit,
        has_oversized_detection=has_oversized_detection,
    )


def update_hard_scene_track_state(
    group: ConflictGroup,
    tracks_by_id: dict[int, FixedTrack],
    decision: HardSceneDecision,
    cfg: TrackingConfig,
) -> None:
    """Keep bounded per-track duration and recovery counters for hard scenes."""
    for track_id in group.track_ids:
        track = tracks_by_id[track_id]
        if decision.is_hard_occlusion:
            track.hard_occlusion_frames += 1
            track.hard_occlusion_recovery_frames = 0
            continue
        track.hard_occlusion_recovery_frames += 1
        if track.hard_occlusion_recovery_frames >= cfg.hard_occlusion_recovery_frames:
            track.hard_occlusion_frames = 0
            track.hard_occlusion_recovery_frames = 0


def nearby_track_pair_for_merged_detection(
    candidate: FixedTrack,
    tracks: list[FixedTrack],
    det: Detection,
    width: int,
    height: int,
    cfg: TrackingConfig,
) -> list[FixedTrack]:
    """Return up to two nearby tracks that are likely covered by one large box."""
    if not candidate.ever_detected:
        return []
    previous_box = (
        candidate.reliable_box
        if candidate.reliable_box is not None
        else candidate.last_box
    )
    previous_area = bbox_area(previous_box)
    if bbox_area(det.box) <= cfg.merged_box_growth_ratio * previous_area:
        return []

    candidate_center = bbox_center(candidate.last_box)
    candidates: list[tuple[float, FixedTrack]] = []
    for other in tracks:
        if other.fixed_id == candidate.fixed_id or not other.ever_detected:
            continue
        if bbox_iom(det.box, other.last_box) <= 0.0:
            continue
        distance = center_distance_norm(
            candidate.last_box,
            other.last_box,
            width,
            height,
        )
        if distance > cfg.merged_box_neighbor_distance:
            continue
        other_center = bbox_center(other.last_box)
        det_center = bbox_center(det.box)
        pair_distance_to_detection = (
            math.dist(candidate_center, det_center)
            + math.dist(other_center, det_center)
        )
        candidates.append((pair_distance_to_detection, other))

    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0])
    return [candidate, candidates[0][1]][: cfg.merged_box_split_max_tracks]


def detect_merged_box_splits(
    tracks: dict[int, FixedTrack],
    detections: list[Detection],
    width: int,
    height: int,
    cfg: TrackingConfig,
    runtime: TrackingRuntimeState | None = None,
) -> tuple[dict[int, np.ndarray], set[int]]:
    """Find oversized detections that should be split by motion prediction."""
    if not cfg.USE_MERGED_BOX_SPLIT:
        return {}, set()

    if runtime is not None:
        advance_split_recovery(runtime)

    ordered_tracks = [tracks[idx] for idx in range(1, cfg.expected_pigs + 1)]
    predicted_boxes = {
        track.fixed_id: track.predicted_box(width, height)
        for track in ordered_tracks
    }
    conflict_groups = build_local_conflict_groups(
        ordered_tracks,
        detections,
        predicted_boxes,
        width,
        height,
        cfg,
    )
    split_boxes: dict[int, np.ndarray] = {}
    ignored_detections: set[int] = set()
    grouped_track_ids: set[int] = set()
    for group in conflict_groups:
        grouped_track_ids.update(group.track_ids)
        decision = hard_scene_decision_for_group(
            group,
            tracks,
            detections,
            predicted_boxes,
            cfg,
            runtime,
        )
        update_hard_scene_track_state(group, tracks, decision, cfg)
        if not decision.is_merged:
            continue

        group_tracks = [
            tracks[track_id]
            for track_id in sorted(group.track_ids)
            if tracks[track_id].ever_detected
        ]
        for det_idx in sorted(
            group.detection_indices,
            key=lambda idx: detections[idx].score,
            reverse=True,
        ):
            if det_idx in ignored_detections:
                continue
            det = detections[det_idx]
            if len(
                detection_covers_group_tracks(
                    det,
                    group_tracks,
                    predicted_boxes,
                    cfg,
                )
            ) < 2:
                continue

            pairs: list[FixedTrack] = []
            for track in group_tracks:
                if track.fixed_id in split_boxes:
                    continue
                if (
                    bbox_iom(det.box, predicted_boxes[track.fixed_id])
                    < cfg.hard_occlusion_detection_iom_threshold
                ):
                    continue
                pairs = nearby_track_pair_for_merged_detection(
                    track,
                    group_tracks,
                    det,
                    width,
                    height,
                    cfg,
                )
                if len(pairs) >= 2:
                    break
            if len(pairs) < 2:
                continue

            ignored_detections.add(det_idx)
            if runtime is not None:
                runtime.telemetry["detections_intentionally_ignored"] += 1
            for track in pairs:
                split_boxes[track.fixed_id] = track.hidden_motion_box(
                    width,
                    height,
                    cfg,
                )
    for track in ordered_tracks:
        if track.fixed_id in grouped_track_ids or track.hard_occlusion_frames == 0:
            continue
        track.hard_occlusion_recovery_frames += 1
        if track.hard_occlusion_recovery_frames >= cfg.hard_occlusion_recovery_frames:
            track.hard_occlusion_frames = 0
            track.hard_occlusion_recovery_frames = 0
    return split_boxes, ignored_detections


def apply_merged_box_splits(
    tracks: dict[int, FixedTrack],
    split_boxes: dict[int, np.ndarray],
    matched_tracks: set[int],
    width: int,
    height: int,
) -> None:
    """Advance split tracks with motion boxes and keep their IDs reserved."""
    for fixed_id, box in split_boxes.items():
        if fixed_id in matched_tracks:
            continue
        track = tracks[fixed_id]
        track.update_predicted(box, width, height, ambiguous=True, hold=True)
        track.last_merged_split = True
        matched_tracks.add(fixed_id)


__all__ = [
    "advance_split_recovery",
    "apply_iou_fallback",
    "apply_merged_box_splits",
    "area_occlusion_should_freeze",
    "assignment_is_occlusion_ambiguous",
    "build_local_conflict_groups",
    "build_occlusion_context",
    "conflict_group_key",
    "detect_merged_box_splits",
    "detection_covers_group_tracks",
    "detection_is_reserved_for_active_track",
    "freeze_area_occluded_track",
    "hard_scene_decision_for_group",
    "nearby_track_pair_for_merged_detection",
    "occlusion_assignment_penalty",
    "reliable_track_area",
    "should_hold_occluded_track_box",
    "track_is_stationary_locked",
    "track_speed_norm",
    "tracks_are_moving_toward_each_other",
    "update_hard_scene_track_state",
]
