"""Tracking metrics calculation and aggregation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cvat_io import TrackingObject
from .matching import match_frame


@dataclass(slots=True)
class TrackingMetrics:
    """Tracking metrics for one video or aggregate."""

    video_stem: str
    gt_detections: int
    pred_detections: int
    matches: int
    fp: int
    fn: int
    idsw: int
    fragments: int
    tracklets: int
    avg_tracklet_length_frames: float
    gap_tolerance_frames: int
    gap_tolerant_fragments: int
    gap_tolerant_tracklets: int
    gap_tolerant_avg_tracklet_length_frames: float
    gap_tolerant_suppressed_fragments: int
    mota: float
    motp_iou: float
    precision: float
    recall: float
    idf1: float
    idtp: int
    idfp: int
    idfn: int
    deta: float
    assa: float
    hota: float
    evaluated_frames: int
    gt_ids: int
    pred_ids: int
    gt_xml: str = ""
    pred_xml: str = ""
    video_path: str = ""
    remapped_idsw: int = 0
    remapped_fragments: int = 0
    remapped_tracklets: int = 0
    remapped_avg_tracklet_length_frames: float = 0.0
    remapped_gap_tolerant_fragments: int = 0
    remapped_gap_tolerant_tracklets: int = 0
    remapped_gap_tolerant_avg_tracklet_length_frames: float = 0.0
    remapped_gap_tolerant_suppressed_fragments: int = 0
    remapped_mota: float = 0.0
    remapped_idf1: float = 0.0
    remapped_assa: float = 0.0
    remapped_hota: float = 0.0
    remapped_idtp: int = 0
    remapped_idfp: int = 0
    remapped_idfn: int = 0
    idmap_matched_detections: int = 0
    idmap_coverage: float = 0.0


def compute_id_metrics(
    pair_counts: Counter[tuple[str, str]],
    gt_id_counts: Counter[str],
    pred_id_counts: Counter[str],
) -> tuple[int, int, int, float]:
    """Compute IDTP/IDFP/IDFN/IDF1 via global identity assignment."""
    gt_ids = sorted(gt_id_counts)
    pred_ids = sorted(pred_id_counts)
    if not gt_ids or not pred_ids:
        idtp = 0
    else:
        counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=int)
        gt_index = {obj_id: idx for idx, obj_id in enumerate(gt_ids)}
        pred_index = {obj_id: idx for idx, obj_id in enumerate(pred_ids)}
        for (gt_id, pred_id), count in pair_counts.items():
            counts[gt_index[gt_id], pred_index[pred_id]] = count
        row_ind, col_ind = linear_sum_assignment(-counts)
        idtp = int(counts[row_ind, col_ind].sum())

    total_gt = int(sum(gt_id_counts.values()))
    total_pred = int(sum(pred_id_counts.values()))
    idfn = total_gt - idtp
    idfp = total_pred - idtp
    denom = (2 * idtp) + idfp + idfn
    idf1 = (2 * idtp / denom) if denom else 0.0
    return idtp, idfp, idfn, idf1


def compute_association_accuracy(
    pair_counts: Counter[tuple[str, str]],
    gt_id_counts: Counter[str],
    pred_id_counts: Counter[str],
) -> float:
    """Compute HOTA-style association accuracy over matched detections."""
    total_matches = sum(pair_counts.values())
    if not total_matches:
        return 0.0
    weighted_sum = 0.0
    for (gt_id, pred_id), count in pair_counts.items():
        union = gt_id_counts[gt_id] + pred_id_counts[pred_id] - count
        if union > 0:
            weighted_sum += count * (count / union)
    return float(weighted_sum / total_matches)


def matched_identity_counts(
    gt_by_frame: dict[int, list[TrackingObject]],
    pred_by_frame: dict[int, list[TrackingObject]],
    *,
    iou_threshold: float,
) -> Counter[tuple[str, str]]:
    """Count matched GT/prediction ID pairs over the whole video."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for frame in sorted(set(gt_by_frame).union(pred_by_frame)):
        gt_objects = gt_by_frame.get(frame, [])
        pred_objects = pred_by_frame.get(frame, [])
        for gt_idx, pred_idx, _iou in match_frame(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold,
        ):
            pair_counts[
                (gt_objects[gt_idx].obj_id, pred_objects[pred_idx].obj_id)
            ] += 1
    return pair_counts


def continuity_stats_from_matches(
    matched_frames_by_gt: dict[str, list[int]],
    *,
    gap_tolerance_frames: int,
) -> tuple[int, int, float, int]:
    """Summarize matched track continuity with short-gap tolerance."""
    tolerance = max(0, int(gap_tolerance_frames))
    tracklet_lengths: list[int] = []
    fragments = 0
    suppressed_fragments = 0

    for frames in matched_frames_by_gt.values():
        ordered_frames = sorted(set(frames))
        if not ordered_frames:
            continue

        current_length = 1
        previous_frame = ordered_frames[0]
        for frame in ordered_frames[1:]:
            gap = frame - previous_frame - 1
            if gap <= tolerance:
                if gap > 0:
                    suppressed_fragments += 1
                current_length += 1
            else:
                tracklet_lengths.append(current_length)
                fragments += 1
                current_length = 1
            previous_frame = frame
        tracklet_lengths.append(current_length)

    tracklets = len(tracklet_lengths)
    avg_length = float(sum(tracklet_lengths) / tracklets) if tracklets else 0.0
    return fragments, tracklets, avg_length, suppressed_fragments


def best_id_mapping(
    pair_counts: Counter[tuple[str, str]],
) -> tuple[dict[str, str], int, int]:
    """Map prediction IDs to GT IDs once, maximizing matched detections."""
    gt_ids = sorted({gt_id for gt_id, _pred_id in pair_counts})
    pred_ids = sorted({pred_id for _gt_id, pred_id in pair_counts})
    if not gt_ids or not pred_ids:
        return {}, 0, 0

    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=int)
    gt_index = {obj_id: idx for idx, obj_id in enumerate(gt_ids)}
    pred_index = {obj_id: idx for idx, obj_id in enumerate(pred_ids)}
    for (gt_id, pred_id), count in pair_counts.items():
        counts[gt_index[gt_id], pred_index[pred_id]] = count

    row_ind, col_ind = linear_sum_assignment(-counts)
    mapping: dict[str, str] = {}
    matched = 0
    for row, col in zip(row_ind, col_ind, strict=False):
        count = int(counts[row, col])
        if count <= 0:
            continue
        mapping[pred_ids[col]] = gt_ids[row]
        matched += count
    return mapping, matched, int(sum(pair_counts.values()))


def remap_prediction_ids(
    gt_by_frame: dict[int, list[TrackingObject]],
    pred_by_frame: dict[int, list[TrackingObject]],
    *,
    iou_threshold: float,
) -> tuple[dict[int, list[TrackingObject]], dict[str, str], int, float]:
    """Apply one fixed prediction->GT ID mapping for permutation-safe scoring."""
    pair_counts = matched_identity_counts(
        gt_by_frame,
        pred_by_frame,
        iou_threshold=iou_threshold,
    )
    mapping, mapped_matches, total_matches = best_id_mapping(pair_counts)
    remapped: dict[int, list[TrackingObject]] = {}
    for frame, objects in pred_by_frame.items():
        remapped[frame] = [
            TrackingObject(
                frame=obj.frame,
                obj_id=mapping.get(obj.obj_id, obj.obj_id),
                bbox=obj.bbox,
                hidden=obj.hidden,
                source_track_id=obj.source_track_id,
                label=obj.label,
            )
            for obj in objects
        ]
    coverage = mapped_matches / total_matches if total_matches else 0.0
    return remapped, mapping, mapped_matches, coverage


def attach_remapped_metrics(
    metrics: TrackingMetrics,
    remapped: TrackingMetrics,
    *,
    mapped_matches: int,
    coverage: float,
) -> TrackingMetrics:
    """Copy permutation-safe identity fields onto the raw metric row."""
    metrics.remapped_idsw = remapped.idsw
    metrics.remapped_fragments = remapped.fragments
    metrics.remapped_tracklets = remapped.tracklets
    metrics.remapped_avg_tracklet_length_frames = (
        remapped.avg_tracklet_length_frames
    )
    metrics.remapped_gap_tolerant_fragments = remapped.gap_tolerant_fragments
    metrics.remapped_gap_tolerant_tracklets = remapped.gap_tolerant_tracklets
    metrics.remapped_gap_tolerant_avg_tracklet_length_frames = (
        remapped.gap_tolerant_avg_tracklet_length_frames
    )
    metrics.remapped_gap_tolerant_suppressed_fragments = (
        remapped.gap_tolerant_suppressed_fragments
    )
    metrics.remapped_mota = remapped.mota
    metrics.remapped_idf1 = remapped.idf1
    metrics.remapped_assa = remapped.assa
    metrics.remapped_hota = remapped.hota
    metrics.remapped_idtp = remapped.idtp
    metrics.remapped_idfp = remapped.idfp
    metrics.remapped_idfn = remapped.idfn
    metrics.idmap_matched_detections = mapped_matches
    metrics.idmap_coverage = coverage
    return metrics


def aggregate_metrics(metrics: list[TrackingMetrics]) -> TrackingMetrics:
    """Aggregate metric rows by summing counts and recomputing ratios."""
    if not metrics:
        return TrackingMetrics(
            video_stem="ALL",
            gt_detections=0,
            pred_detections=0,
            matches=0,
            fp=0,
            fn=0,
            idsw=0,
            fragments=0,
            mota=0.0,
            motp_iou=0.0,
            precision=0.0,
            recall=0.0,
            idf1=0.0,
            idtp=0,
            idfp=0,
            idfn=0,
            deta=0.0,
            assa=0.0,
            hota=0.0,
            evaluated_frames=0,
            gt_ids=0,
            pred_ids=0,
            tracklets=0,
            avg_tracklet_length_frames=0.0,
            gap_tolerance_frames=0,
            gap_tolerant_fragments=0,
            gap_tolerant_tracklets=0,
            gap_tolerant_avg_tracklet_length_frames=0.0,
            gap_tolerant_suppressed_fragments=0,
        )

    gt_total = sum(m.gt_detections for m in metrics)
    pred_total = sum(m.pred_detections for m in metrics)
    matches_total = sum(m.matches for m in metrics)
    fp_total = sum(m.fp for m in metrics)
    fn_total = sum(m.fn for m in metrics)
    idsw_total = sum(m.idsw for m in metrics)
    remapped_idsw_total = sum(m.remapped_idsw for m in metrics)
    remapped_fragments_total = sum(m.remapped_fragments for m in metrics)
    tracklets_total = sum(m.tracklets for m in metrics)
    remapped_tracklets_total = sum(m.remapped_tracklets for m in metrics)
    gap_tolerance_frames = max(m.gap_tolerance_frames for m in metrics)
    gap_tolerant_fragments_total = sum(m.gap_tolerant_fragments for m in metrics)
    gap_tolerant_tracklets_total = sum(m.gap_tolerant_tracklets for m in metrics)
    gap_tolerant_suppressed_total = sum(
        m.gap_tolerant_suppressed_fragments for m in metrics
    )
    remapped_gap_tolerant_fragments_total = sum(
        m.remapped_gap_tolerant_fragments for m in metrics
    )
    remapped_gap_tolerant_tracklets_total = sum(
        m.remapped_gap_tolerant_tracklets for m in metrics
    )
    remapped_gap_tolerant_suppressed_total = sum(
        m.remapped_gap_tolerant_suppressed_fragments for m in metrics
    )
    idtp_total = sum(m.idtp for m in metrics)
    idfp_total = sum(m.idfp for m in metrics)
    idfn_total = sum(m.idfn for m in metrics)
    remapped_idtp_total = sum(m.remapped_idtp for m in metrics)
    remapped_idfp_total = sum(m.remapped_idfp for m in metrics)
    remapped_idfn_total = sum(m.remapped_idfn for m in metrics)
    motp_num = sum(m.motp_iou * m.matches for m in metrics)
    deta = (
        matches_total / (matches_total + fp_total + fn_total)
        if matches_total + fp_total + fn_total
        else 0.0
    )
    idf1_denom = (2 * idtp_total) + idfp_total + idfn_total
    idf1 = (2 * idtp_total / idf1_denom) if idf1_denom else 0.0
    assa_num = sum(m.assa * m.matches for m in metrics)
    assa = assa_num / matches_total if matches_total else 0.0
    remapped_idf1_denom = (
        (2 * remapped_idtp_total) + remapped_idfp_total + remapped_idfn_total
    )
    remapped_idf1 = (
        (2 * remapped_idtp_total / remapped_idf1_denom)
        if remapped_idf1_denom
        else 0.0
    )
    remapped_mota = (
        1.0 - ((fn_total + fp_total + remapped_idsw_total) / gt_total)
        if gt_total
        else 0.0
    )
    remapped_assa_num = sum(m.remapped_assa * m.matches for m in metrics)
    remapped_assa = remapped_assa_num / matches_total if matches_total else 0.0
    remapped_hota = (
        math.sqrt(deta * remapped_assa) if deta > 0 and remapped_assa > 0 else 0.0
    )
    idmap_matched_detections = sum(m.idmap_matched_detections for m in metrics)
    idmap_coverage = (
        idmap_matched_detections / matches_total if matches_total else 0.0
    )
    avg_tracklet_length = matches_total / tracklets_total if tracklets_total else 0.0
    remapped_avg_tracklet_length = (
        matches_total / remapped_tracklets_total if remapped_tracklets_total else 0.0
    )
    gap_tolerant_avg_tracklet_length = (
        matches_total / gap_tolerant_tracklets_total
        if gap_tolerant_tracklets_total
        else 0.0
    )
    remapped_gap_tolerant_avg_tracklet_length = (
        matches_total / remapped_gap_tolerant_tracklets_total
        if remapped_gap_tolerant_tracklets_total
        else 0.0
    )

    return TrackingMetrics(
        video_stem="ALL",
        gt_detections=gt_total,
        pred_detections=pred_total,
        matches=matches_total,
        fp=fp_total,
        fn=fn_total,
        idsw=idsw_total,
        fragments=sum(m.fragments for m in metrics),
        mota=1.0 - ((fn_total + fp_total + idsw_total) / gt_total)
        if gt_total
        else 0.0,
        motp_iou=motp_num / matches_total if matches_total else 0.0,
        precision=matches_total / pred_total if pred_total else 0.0,
        recall=matches_total / gt_total if gt_total else 0.0,
        idf1=idf1,
        idtp=idtp_total,
        idfp=idfp_total,
        idfn=idfn_total,
        deta=deta,
        assa=assa,
        hota=math.sqrt(deta * assa) if deta > 0 and assa > 0 else 0.0,
        evaluated_frames=sum(m.evaluated_frames for m in metrics),
        gt_ids=sum(m.gt_ids for m in metrics),
        pred_ids=sum(m.pred_ids for m in metrics),
        tracklets=tracklets_total,
        avg_tracklet_length_frames=avg_tracklet_length,
        gap_tolerance_frames=gap_tolerance_frames,
        gap_tolerant_fragments=gap_tolerant_fragments_total,
        gap_tolerant_tracklets=gap_tolerant_tracklets_total,
        gap_tolerant_avg_tracklet_length_frames=gap_tolerant_avg_tracklet_length,
        gap_tolerant_suppressed_fragments=gap_tolerant_suppressed_total,
        remapped_idsw=remapped_idsw_total,
        remapped_fragments=remapped_fragments_total,
        remapped_tracklets=remapped_tracklets_total,
        remapped_avg_tracklet_length_frames=remapped_avg_tracklet_length,
        remapped_gap_tolerant_fragments=remapped_gap_tolerant_fragments_total,
        remapped_gap_tolerant_tracklets=remapped_gap_tolerant_tracklets_total,
        remapped_gap_tolerant_avg_tracklet_length_frames=(
            remapped_gap_tolerant_avg_tracklet_length
        ),
        remapped_gap_tolerant_suppressed_fragments=(
            remapped_gap_tolerant_suppressed_total
        ),
        remapped_mota=remapped_mota,
        remapped_idf1=remapped_idf1,
        remapped_assa=remapped_assa,
        remapped_hota=remapped_hota,
        remapped_idtp=remapped_idtp_total,
        remapped_idfp=remapped_idfp_total,
        remapped_idfn=remapped_idfn_total,
        idmap_matched_detections=idmap_matched_detections,
        idmap_coverage=idmap_coverage,
    )
