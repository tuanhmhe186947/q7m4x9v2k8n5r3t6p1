# ruff: noqa
"""BEV-space data association using Euclidean distance as the primary cost.

# ruff: noqa

The cost function combines:

* BEV Euclidean distance (primary)
* Detection confidence penalty
* Depth ambiguity penalty
* Occlusion penalty
* HSV histogram distance (optional, reused from existing pipeline)

After the Hungarian assignment the function also computes per-track
``best_score``, ``second_best_score``, and ``score_margin`` so the
:mod:`sanity` gate can reject ambiguous matches.
"""

from __future__ import annotations

import logging

import numpy as np

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.kalman import bev_position
from pig_behavior.tracking.rgbd.schemas import (
    AssociationDecision,
    BEVTrackState,
    Detection3D,
)
from pig_behavior.tracking.schemas import FixedTrack

logger = logging.getLogger(__name__)


def _hist_distance_safe(
    track: FixedTrack,
    detection: Detection3D,
) -> float:
    """Reuse the existing HSV histogram distance if data is available."""
    try:
        from pig_behavior.tracking.detections import hist_distance

        hist_det = detection.detection_2d.hist
        if hist_det is None:
            return 0.50
        return hist_distance(track.mean_hist(), hist_det)
    except Exception:
        return 0.50


def _bev_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two BEV points (metres)."""
    return float(np.linalg.norm(a - b))


def _build_cost_matrix(
    candidate_tracks: list[tuple[FixedTrack, BEVTrackState]],
    detections_3d: list[Detection3D],
    predicted_positions: dict[int, np.ndarray],
    occlusion_flags: dict[int, bool],
    cfg: RGBDTrackingConfig,
) -> np.ndarray:
    """Build a ``(T, D)`` cost matrix for the Hungarian solver."""
    n_tracks = len(candidate_tracks)
    n_dets = len(detections_3d)
    costs = np.full((n_tracks, n_dets), 1e6, dtype=np.float64)

    # Build raw_owner mapping for Re-ID consistency check
    raw_owner = {}
    for track, bev in candidate_tracks:
        tid = track.top_raw_id()
        if tid is not None:
            raw_owner[tid] = track.fixed_id

    for row, (track, bev) in enumerate(candidate_tracks):
        pred_bev = predicted_positions.get(bev.fixed_id)
        if pred_bev is None:
            continue
        for col, det3d in enumerate(detections_3d):
            if det3d.bev_xy is None:
                continue

            bev_dist = _bev_euclidean(pred_bev, det3d.bev_xy)
            # Gate: hard reject
            if track.ever_detected and bev_dist > cfg.bev_association_gate_m * 2.0:
                continue

            # Normalise BEV distance by the gate
            norm_bev = min(bev_dist / max(cfg.bev_association_gate_m, 1e-6), 2.0)

            conf_penalty = 1.0 - det3d.detection_2d.confidence
            depth_ambig_penalty = 1.0 if det3d.depth_ambiguous else 0.0
            occ_penalty = 1.0 if occlusion_flags.get(col, False) else 0.0
            hist_dist = _hist_distance_safe(track, det3d)

            # Re-ID raw identity penalty to prevent track identity swaps
            raw_penalty = 0.0
            det_raw_id = det3d.detection_2d.raw_id
            if det_raw_id is not None:
                owner = raw_owner.get(det_raw_id)
                if owner is not None and owner != track.fixed_id:
                    raw_penalty += 0.20
                elif track.top_raw_id() is not None and track.top_raw_id() != det_raw_id:
                    raw_penalty += 0.08

            cost = (
                cfg.w_bev * norm_bev
                + cfg.w_conf * conf_penalty
                + cfg.w_depth_ambiguous * depth_ambig_penalty
                + cfg.w_occlusion * occ_penalty
                + cfg.w_hist * hist_dist
                + raw_penalty
            )
            costs[row, col] = cost

    return costs


def _compute_score_margins(
    costs: np.ndarray,
    row: int,
    col: int,
) -> tuple[float, float | None, float | None]:
    """Return ``(best_cost, second_best_cost, margin)`` for a track row."""
    row_costs = costs[row].copy()
    valid = row_costs < 1e5
    if not np.any(valid):
        return float(costs[row, col]), None, None

    sorted_costs = np.sort(row_costs[valid])
    best = float(sorted_costs[0])
    if len(sorted_costs) >= 2:
        second = float(sorted_costs[1])
        margin = second - best
        return best, second, margin
    return best, None, None


def match_bev_tracks(
    tracks: dict[int, FixedTrack],
    bev_states: dict[int, BEVTrackState],
    detections_3d: list[Detection3D],
    occlusion_flags: dict[int, bool],
    frame_index: int,
    cfg: RGBDTrackingConfig,
) -> tuple[dict[int, int], list[AssociationDecision]]:
    """Match BEV track predictions to 3-D detections.

    Returns
    -------
    assignments:
        ``{track_fixed_id: detection_index}``
    decisions:
        One :class:`AssociationDecision` per candidate track (matched or not).
    """
    decisions: list[AssociationDecision] = []

    # Build candidate lists
    candidate_ids = sorted(bev_states.keys())
    candidate_tracks: list[tuple[FixedTrack, BEVTrackState]] = []
    predicted_positions: dict[int, np.ndarray] = {}

    for fid in candidate_ids:
        bev = bev_states[fid]
        track = tracks.get(fid)
        if track is None:
            continue
        pred = bev_position(bev.kf)
        predicted_positions[fid] = pred
        candidate_tracks.append((track, bev))

    if not candidate_tracks or not detections_3d:
        # All tracks unmatched → predict-only decisions
        for track, bev in candidate_tracks:
            decisions.append(
                AssociationDecision(
                    frame_index=frame_index,
                    track_id=bev.fixed_id,
                    accepted=False,
                    reject_reason="no_detections",
                )
            )
        return {}, decisions

    # Filter detections with valid BEV positions
    valid_det_indices = [
        i for i, d in enumerate(detections_3d) if d.bev_xy is not None
    ]
    valid_dets = [detections_3d[i] for i in valid_det_indices]

    if not valid_dets:
        for track, bev in candidate_tracks:
            decisions.append(
                AssociationDecision(
                    frame_index=frame_index,
                    track_id=bev.fixed_id,
                    accepted=False,
                    reject_reason="no_valid_bev_detections",
                )
            )
        return {}, decisions

    costs = _build_cost_matrix(
        candidate_tracks,
        valid_dets,
        predicted_positions,
        {i: occlusion_flags.get(valid_det_indices[i], False) for i in range(len(valid_dets))},
        cfg,
    )

    # Hungarian assignment
    try:
        from scipy.optimize import linear_sum_assignment

        row_indices, col_indices = linear_sum_assignment(costs)
    except ImportError:
        logger.warning(
            "scipy not available — using greedy BEV matching fallback"
        )
        row_indices, col_indices = _greedy_assignment(costs)

    assignments: dict[int, int] = {}
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()

    for row, col in zip(row_indices, col_indices):
        track, bev = candidate_tracks[row]
        det = valid_dets[col]
        original_det_idx = valid_det_indices[col]
        cost_val = float(costs[row, col])

        pred_bev = predicted_positions[bev.fixed_id]
        bev_dist = _bev_euclidean(pred_bev, det.bev_xy) if det.bev_xy is not None else None

        best, second, margin = _compute_score_margins(costs, row, col)

        is_occ = occlusion_flags.get(original_det_idx, False)

        # Gate check
        if cost_val > 1e5 or (track.ever_detected and bev_dist is not None and bev_dist > cfg.bev_association_gate_m):
            decisions.append(
                AssociationDecision(
                    frame_index=frame_index,
                    track_id=bev.fixed_id,
                    detection_index=original_det_idx,
                    bev_distance_m=bev_dist,
                    cost=cost_val,
                    best_score=best,
                    second_best_score=second,
                    score_margin=margin,
                    accepted=False,
                    reject_reason="bev_distance_too_large",
                    depth_valid=det.depth_valid,
                    depth_ambiguous=det.depth_ambiguous,
                    is_occluded=is_occ,
                )
            )
            continue

        decisions.append(
            AssociationDecision(
                frame_index=frame_index,
                track_id=bev.fixed_id,
                detection_index=original_det_idx,
                bev_distance_m=bev_dist,
                cost=cost_val,
                best_score=best,
                second_best_score=second,
                score_margin=margin,
                accepted=True,  # sanity gate may override later
                depth_valid=det.depth_valid,
                depth_ambiguous=det.depth_ambiguous,
                is_occluded=is_occ,
            )
        )
        assignments[bev.fixed_id] = original_det_idx
        matched_rows.add(row)
        matched_cols.add(col)

    # Unmatched tracks
    for row, (track, bev) in enumerate(candidate_tracks):
        if row not in matched_rows:
            decisions.append(
                AssociationDecision(
                    frame_index=frame_index,
                    track_id=bev.fixed_id,
                    accepted=False,
                    reject_reason="unmatched",
                )
            )

    return assignments, decisions


def _greedy_assignment(
    costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy fallback when scipy is not available."""
    n_rows, n_cols = costs.shape
    rows: list[int] = []
    cols: list[int] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    flat = [(float(costs[r, c]), r, c) for r in range(n_rows) for c in range(n_cols)]
    flat.sort(key=lambda x: x[0])

    for cost_val, r, c in flat:
        if r in used_rows or c in used_cols:
            continue
        if cost_val > 1e5:
            break
        rows.append(r)
        cols.append(c)
        used_rows.add(r)
        used_cols.add(c)

    return np.array(rows, dtype=int), np.array(cols, dtype=int)


__all__ = [
    "match_bev_tracks",
]
