"""Offline tracklet stitching using Hungarian assignment."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from pig_behavior.tracking.detections import hist_distance
from pig_behavior.tracking.stabilization.config import AnnotationStableConfig
from pig_behavior.tracking.stabilization.diagnostics import StitchingReportRow


class StableTrackletRecord:
    """Record of a single continuous tracklet sequence."""

    def __init__(
        self,
        tracklet_id: int,
        fixed_id: int,
        start_frame: int,
        end_frame: int,
        bbox_sequence: list[np.ndarray],
        center_sequence: list[tuple[float, float]],
        area_sequence: list[float],
        confidence_sequence: list[float],
        hist_summary: np.ndarray | None,
        depth_valid_ratio: float,
        bev_valid_ratio: float,
        mean_confidence: float,
        length: int,
        bev_sequence: list[np.ndarray | None] | None = None,
        frames: list[int] | None = None,
    ):
        self.tracklet_id = tracklet_id
        self.fixed_id = fixed_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.bbox_sequence = bbox_sequence
        self.center_sequence = center_sequence
        self.area_sequence = area_sequence
        self.confidence_sequence = confidence_sequence
        self.hist_summary = hist_summary
        self.depth_valid_ratio = depth_valid_ratio
        self.bev_valid_ratio = bev_valid_ratio
        self.mean_confidence = mean_confidence
        self.length = length
        self.bev_sequence = bev_sequence or [None] * length
        self.frames = frames if frames is not None else list(range(start_frame, end_frame + 1))


def compute_stitching_cost(
    a: StableTrackletRecord,
    b: StableTrackletRecord,
    config: AnnotationStableConfig,
    width: int,
    height: int,
) -> tuple[float, float, float, float, float | None]:
    """Computes individual feature costs and final combined cost for stitching.

    Returns:
        (final_cost, cost_center, cost_area, cost_hist, cost_bev)
        or (1e6, ...) if gated.
    """
    gap = b.start_frame - a.end_frame
    if gap <= 0 or gap > config.stitch_max_gap:
        return 1e6, 1.0, 1.0, 1.0, None

    cx_a, cy_a = a.center_sequence[-1]
    cx_b, cy_b = b.center_sequence[0]
    diag = math.sqrt(width * width + height * height)
    dist_norm = math.dist((cx_a, cy_a), (cx_b, cy_b)) / max(diag, 1e-6)

    if dist_norm > config.stitch_max_center_distance_norm:
        return 1e6, dist_norm, 1.0, 1.0, None

    area_a = a.area_sequence[-1]
    area_b = b.area_sequence[0]
    area_ratio = abs(math.log((area_b + 1e-6) / (area_a + 1e-6)))

    if area_ratio > config.stitch_max_area_log_ratio:
        return 1e6, dist_norm, area_ratio, 1.0, None

    hist_dist = 0.5
    if a.hist_summary is not None and b.hist_summary is not None:
        hist_dist = hist_distance(a.hist_summary, b.hist_summary)

    if hist_dist > config.stitch_max_hist_distance:
        return 1e6, dist_norm, area_ratio, hist_dist, None

    # Optional BEV cost
    cost_bev: float | None = None
    if config.stitch_use_bev and a.bev_sequence and b.bev_sequence:
        bev_a = a.bev_sequence[-1]
        bev_b = b.bev_sequence[0]
        if bev_a is not None and bev_b is not None:
            # European distance in BEV space (in meters)
            cost_bev = float(np.linalg.norm(bev_a - bev_b))
            if cost_bev > config.stitch_max_center_distance_norm * 10:  # arbitrary threshold
                return 1e6, dist_norm, area_ratio, hist_dist, cost_bev

    # Calculate normalized cost (0 to 1) for valid pairs
    norm_center = dist_norm / config.stitch_max_center_distance_norm
    norm_area = area_ratio / config.stitch_max_area_log_ratio
    norm_hist = hist_dist / config.stitch_max_hist_distance

    # Combined weight score
    final_cost = 0.4 * norm_center + 0.3 * norm_area + 0.3 * norm_hist
    if cost_bev is not None:
        # Give some weight to BEV
        final_cost = 0.7 * final_cost + 0.3 * min(1.0, cost_bev / 4.0)

    return final_cost, dist_norm, area_ratio, hist_dist, cost_bev


def stitch_tracklets(
    tracklets: list[StableTrackletRecord],
    config: AnnotationStableConfig,
    width: int,
    height: int,
) -> tuple[dict[int, int], list[StitchingReportRow]]:
    """Stitches compatible tracklets into continuous stable track IDs.

    Args:
        tracklets: list of all tracklet records
        config: stable tracking configuration
        width: frame width
        height: frame height

    Returns:
        tracklet_to_stable_id: dict mapping tracklet_id to final stable_track_id
        stitching_report: list of StitchingReportRow details
    """
    tracklet_to_stable_id: dict[int, int] = {}
    stitching_report: list[StitchingReportRow] = []

    if not tracklets:
        return tracklet_to_stable_id, stitching_report

    # Initialize each tracklet to its own stable track id (default)
    for t in tracklets:
        tracklet_to_stable_id[t.tracklet_id] = t.tracklet_id

    # Filter out extremely short tracklets if not configured to keep
    # (But we keep all of them here for full mapping, the filtering is done before)
    n = len(tracklets)
    if n <= 1:
        return tracklet_to_stable_id, stitching_report

    # Bipartite matching setup: potential parents on the left, potential children on the right
    # A tracklet can be a parent if it ends before some other tracklet starts
    # A tracklet can be a child if it starts after some other tracklet ends
    cost_matrix = np.full((n, n), 1e6, dtype=np.float32)
    candidate_details = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t_parent = tracklets[i]
            t_child = tracklets[j]

            # Parent must end before child starts
            if t_parent.end_frame < t_child.start_frame:
                cost, dist_norm, area_ratio, hist_dist, cost_bev = compute_stitching_cost(
                    t_parent, t_child, config, width, height
                )
                if cost < 1.0:  # Accepted cost threshold
                    cost_matrix[i, j] = cost
                    candidate_details[(i, j)] = (
                        dist_norm,
                        area_ratio,
                        hist_dist,
                        cost_bev,
                    )

    # Solve Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Apply assignments if under threshold
    stitched_pairs = []
    for r, c in zip(row_ind, col_ind, strict=False):
        cost = cost_matrix[r, c]
        is_stitched = bool(cost < 1.0)

        t_parent = tracklets[r]
        t_child = tracklets[c]

        # Get detailed metrics
        if (r, c) in candidate_details:
            dist_norm, area_ratio, hist_dist, cost_bev = candidate_details[(r, c)]
        else:
            # Fallback evaluation for the report of rejected assignments
            _, dist_norm, area_ratio, hist_dist, cost_bev = compute_stitching_cost(
                t_parent, t_child, config, width, height
            )

        gap = t_child.start_frame - t_parent.end_frame

        stitching_report.append(
            StitchingReportRow(
                parent_tracklet_id=int(t_parent.tracklet_id),
                child_tracklet_id=int(t_child.tracklet_id),
                gap_frames=int(gap) if gap > 0 else 0,
                cost_iou_2d=float(1.0 - dist_norm),  # high IoU ~ low distance
                cost_center=float(dist_norm),
                cost_area=float(area_ratio),
                cost_hist=float(hist_dist),
                cost_bev=float(cost_bev) if cost_bev is not None else None,
                final_score=float(1.0 - cost) if cost < 1e6 else 0.0,
                is_stitched=is_stitched,
            )
        )

        if is_stitched:
            stitched_pairs.append((t_parent.tracklet_id, t_child.tracklet_id))

    # Resolve tracklet ID chains into final stable track IDs
    # If A stitches to B and B stitches to C, then A, B, C all get the same stable ID (e.g. A's ID)
    parent_to_child = dict(stitched_pairs)

    # We will assign IDs starting from a clean sequence or using the parent's fixed ID/first tracklet ID
    visited = set()
    for t in sorted(tracklets, key=lambda x: x.start_frame):
        tid = t.tracklet_id
        if tid in visited:
            continue

        # Traverse the chain to the end
        chain = [tid]
        curr = tid
        while curr in parent_to_child:
            curr = parent_to_child[curr]
            chain.append(curr)

        # Determine stable ID for this chain
        # Try to use the fixed ID of the first tracklet in the chain if valid, otherwise make new
        stable_id = tracklets[0].fixed_id  # dummy fallback
        for tracklet in tracklets:
            if tracklet.tracklet_id == tid:
                stable_id = tracklet.fixed_id
                break

        for chain_tid in chain:
            tracklet_to_stable_id[chain_tid] = stable_id
            visited.add(chain_tid)

    return tracklet_to_stable_id, stitching_report
