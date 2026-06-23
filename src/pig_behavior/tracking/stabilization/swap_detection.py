"""Cross-trajectory ID swap detection and correction for pig tracking."""

from __future__ import annotations

import math

import numpy as np

from pig_behavior.tracking.detections import hist_distance
from pig_behavior.tracking.geometry import bbox_center, bbox_iou
from pig_behavior.tracking.stabilization.config import AnnotationStableConfig
from pig_behavior.tracking.stabilization.diagnostics import SwapCandidateRow


def detect_and_optionally_fix_swaps(
    tracks: dict[int, dict[int, tuple[np.ndarray, np.ndarray | None]]],
    config: AnnotationStableConfig,
    width: int,
    height: int,
) -> tuple[dict[int, dict[int, tuple[np.ndarray, np.ndarray | None]]], list[SwapCandidateRow]]:
    """Scans all active trajectories for crossing patterns and appearance mismatch.

    If config.auto_fix_high_confidence_swaps is True, automatically swaps the
    identities of the tracks from the crossing frame onwards.

    Args:
        tracks: dict of {track_id: {frame: (bbox_xyxy, hist_vector)}}
        config: stable tracking configuration
        width: frame width
        height: frame height

    Returns:
        updated_tracks: copy of tracks with fixed IDs if auto-fix is enabled
        swap_candidates: list of detected ID swap candidate events
    """
    if not config.detect_candidate_swaps:
        return tracks, []

    # Make a copy of tracks to potentially edit
    updated_tracks = {tid: dict(tdata) for tid, tdata in tracks.items()}
    swap_candidates: list[SwapCandidateRow] = []

    track_ids = sorted(list(tracks.keys()))
    diag = math.sqrt(width * width + height * height)

    proximity = config.swap_proximity_frames

    # We will process pairs of tracks
    for idx_a in range(len(track_ids)):
        for idx_b in range(idx_a + 1, len(track_ids)):
            tid_a = track_ids[idx_a]
            tid_b = track_ids[idx_b]

            frames_a = set(tracks[tid_a].keys())
            frames_b = set(tracks[tid_b].keys())
            common_frames = sorted(list(frames_a.intersection(frames_b)))

            if len(common_frames) < proximity * 2:
                continue

            # Slide window of size 2 * proximity + 1
            i = proximity
            while i < len(common_frames) - proximity:
                t_c = common_frames[i]

                # Before window: [t_c - proximity, t_c - 1]
                before_frames = [f for f in common_frames if t_c - proximity <= f < t_c]
                # After window: [t_c + 1, t_c + proximity]
                after_frames = [f for f in common_frames if t_c < f <= t_c + proximity]

                if len(before_frames) < proximity // 2 or len(after_frames) < proximity // 2:
                    i += 1
                    continue

                # 1. Compute positions
                centers_a_pre = np.array([bbox_center(tracks[tid_a][f][0]) for f in before_frames])
                centers_b_pre = np.array([bbox_center(tracks[tid_b][f][0]) for f in before_frames])
                centers_a_post = np.array([bbox_center(tracks[tid_a][f][0]) for f in after_frames])
                centers_b_post = np.array([bbox_center(tracks[tid_b][f][0]) for f in after_frames])

                pos_a_pre = np.mean(centers_a_pre, axis=0)
                pos_b_pre = np.mean(centers_b_pre, axis=0)
                pos_a_post = np.mean(centers_a_post, axis=0)
                pos_b_post = np.mean(centers_b_post, axis=0)

                # Crossover check
                d_aa = np.linalg.norm(pos_a_pre - pos_a_post)
                d_bb = np.linalg.norm(pos_b_pre - pos_b_post)
                d_ab = np.linalg.norm(pos_a_pre - pos_b_post)
                d_ba = np.linalg.norm(pos_b_pre - pos_a_post)

                # Distance at crossing frame
                box_a_c = tracks[tid_a][t_c][0]
                box_b_c = tracks[tid_b][t_c][0]
                dist_c_norm = np.linalg.norm(np.array(bbox_center(box_a_c)) - np.array(bbox_center(box_b_c))) / diag

                # Did they cross geometrically?
                # Total distance if staying same ID: d_aa + d_bb
                # Total distance if swapping ID: d_ab + d_ba
                crossed_geometrically = (d_ab + d_ba) < (d_aa + d_bb) * 0.90

                # Check overlap (IoU) at crossing frame or in vicinity
                max_iou = 0.0
                for f in before_frames + [t_c] + after_frames:
                    max_iou = max(max_iou, bbox_iou(tracks[tid_a][f][0], tracks[tid_b][f][0]))

                # 2. Compute appearance consistency
                # Find mean histograms
                hists_a_pre = [tracks[tid_a][f][1] for f in before_frames if tracks[tid_a][f][1] is not None]
                hists_b_pre = [tracks[tid_b][f][1] for f in before_frames if tracks[tid_b][f][1] is not None]
                hists_a_post = [tracks[tid_a][f][1] for f in after_frames if tracks[tid_a][f][1] is not None]
                hists_b_post = [tracks[tid_b][f][1] for f in after_frames if tracks[tid_b][f][1] is not None]

                appearance_swap_confidence = 0.0
                if hists_a_pre and hists_b_pre and hists_a_post and hists_b_post:
                    h_a_pre = np.mean(hists_a_pre, axis=0)
                    h_b_pre = np.mean(hists_b_pre, axis=0)
                    h_a_post = np.mean(hists_a_post, axis=0)
                    h_b_post = np.mean(hists_b_post, axis=0)

                    sim_aa = 1.0 - hist_distance(h_a_pre, h_a_post)
                    sim_ab = 1.0 - hist_distance(h_a_pre, h_b_post)
                    sim_bb = 1.0 - hist_distance(h_b_pre, h_b_post)
                    sim_ba = 1.0 - hist_distance(h_b_pre, h_a_post)

                    # Does A match B better and B match A better?
                    if sim_ab > sim_aa and sim_ba > sim_bb:
                        appearance_swap_confidence = float((sim_ab + sim_ba) / 2.0)

                # 3. Overall confidence score
                # High score if they crossed geometrically, were close/overlapped, and appearance matches the swap.
                geo_factor = 1.0 if crossed_geometrically else 0.1
                proximity_factor = 1.0 - min(1.0, dist_c_norm / config.swap_min_overlap_iou)

                # Combine factors
                swap_confidence = 0.0
                if crossed_geometrically or max_iou > config.swap_min_overlap_iou:
                    if appearance_swap_confidence > 0.0:
                        swap_confidence = 0.6 * appearance_swap_confidence + 0.4 * geo_factor
                    else:
                        # Fallback to geometry and proximity alone
                        swap_confidence = 0.5 * geo_factor + 0.5 * proximity_factor

                if swap_confidence > config.swap_confidence_threshold:
                    is_fixed = False
                    if config.auto_fix_high_confidence_swaps:
                        # Swap all frames after t_c
                        all_frames_a = sorted([f for f in updated_tracks[tid_a].keys() if f > t_c])
                        all_frames_b = sorted([f for f in updated_tracks[tid_b].keys() if f > t_c])

                        # Temporal swap
                        temp_a_data = {f: updated_tracks[tid_a][f] for f in all_frames_a}
                        temp_b_data = {f: updated_tracks[tid_b][f] for f in all_frames_b}

                        # Clear post-crossover data
                        for f in all_frames_a:
                            del updated_tracks[tid_a][f]
                        for f in all_frames_b:
                            del updated_tracks[tid_b][f]

                        # Reassign swapped data
                        for f, val in temp_a_data.items():
                            updated_tracks[tid_b][f] = val
                        for f, val in temp_b_data.items():
                            updated_tracks[tid_a][f] = val

                        is_fixed = True

                    # Record swap event
                    swap_candidates.append(
                        SwapCandidateRow(
                            track_id_a=tid_a,
                            track_id_b=tid_b,
                            frame_start=before_frames[0],
                            frame_end=after_frames[-1],
                            crossing_frame=t_c,
                            swap_confidence=swap_confidence,
                            is_fixed=is_fixed,
                            distance_norm=dist_c_norm,
                        )
                    )
                    # Skip window to prevent duplicate swap triggers for the same event
                    i += proximity
                    continue

                i += 1

    return updated_tracks, swap_candidates
