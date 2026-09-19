"""Versioned standard identity metrics for tracking evaluator V2.

The global identity assignment follows TrackEval's Identity metric. CLEAR
matching preserves an eligible pairing from the immediately preceding frame
before matching the remaining detections. Identity-switch memory is separate
and persists across unmatched gaps until the sequence boundary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cvat_io import TrackingObject
from .matching import iou_xyxy
from .matching_standard_v2 import match_frame_eligible

FrameObjects = Mapping[int, Sequence[TrackingObject]]


@dataclass(frozen=True, slots=True)
class IdentityStandardV2Metrics:
    """Identity and CLEAR continuity metrics for one sequence or an aggregate."""

    sequence_id: str
    sequence_count: int
    gt_detections: int
    pred_detections: int
    idtp: int
    idfp: int
    idfn: int
    idf1: float
    idp: float
    idr: float
    idsw_standard: int
    fragments: int
    clear_tp: int
    clear_fp: int
    clear_fn: int


def _validate_threshold(iou_threshold: float) -> float:
    threshold = float(iou_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and between 0 and 1")
    return threshold


def _object_key(obj: TrackingObject) -> tuple[object, ...]:
    return (
        obj.obj_id,
        obj.source_track_id,
        obj.label,
        tuple(float(value) for value in obj.bbox),
        bool(obj.hidden),
    )


def _canonical_frames(
    by_frame: FrameObjects,
    *,
    population_name: str,
) -> dict[int, list[TrackingObject]]:
    canonical: dict[int, list[TrackingObject]] = {}
    for frame, objects in by_frame.items():
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise ValueError(f"{population_name} frame keys must be integers")
        items = list(objects)
        for obj in items:
            if obj.frame != frame:
                raise ValueError(
                    f"{population_name} object frame {obj.frame} does not match key {frame}"
                )
        ids = [obj.obj_id for obj in items]
        duplicates = sorted(
            obj_id for obj_id, count in Counter(ids).items() if count > 1
        )
        if duplicates:
            joined = ", ".join(duplicates)
            raise ValueError(
                f"{population_name} frame {frame} has duplicate IDs: {joined}"
            )
        canonical[frame] = sorted(items, key=_object_key)
    return canonical


def _identity_counts(
    gt_by_frame: Mapping[int, Sequence[TrackingObject]],
    pred_by_frame: Mapping[int, Sequence[TrackingObject]],
    *,
    iou_threshold: float,
) -> tuple[int, int, int]:
    gt_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    potential_matches: Counter[tuple[str, str]] = Counter()

    for frame in sorted(set(gt_by_frame).union(pred_by_frame)):
        gt_objects = gt_by_frame.get(frame, ())
        pred_objects = pred_by_frame.get(frame, ())
        gt_counts.update(obj.obj_id for obj in gt_objects)
        pred_counts.update(obj.obj_id for obj in pred_objects)
        for gt in gt_objects:
            for pred in pred_objects:
                if iou_xyxy(gt.bbox, pred.bbox) >= iou_threshold:
                    potential_matches[(gt.obj_id, pred.obj_id)] += 1

    total_gt = int(sum(gt_counts.values()))
    total_pred = int(sum(pred_counts.values()))
    if not gt_counts:
        return 0, total_pred, 0
    if not pred_counts:
        return 0, 0, total_gt

    gt_ids = sorted(gt_counts)
    pred_ids = sorted(pred_counts)
    num_gt_ids = len(gt_ids)
    num_pred_ids = len(pred_ids)
    matrix_size = num_gt_ids + num_pred_ids
    gt_index = {obj_id: index for index, obj_id in enumerate(gt_ids)}
    pred_index = {obj_id: index for index, obj_id in enumerate(pred_ids)}

    false_positive_cost = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    false_negative_cost = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    forbidden_cost = total_gt + total_pred + 1
    false_positive_cost[num_gt_ids:, :num_pred_ids] = forbidden_cost
    false_negative_cost[:num_gt_ids, num_pred_ids:] = forbidden_cost

    for gt_id, count in gt_counts.items():
        row = gt_index[gt_id]
        false_negative_cost[row, :num_pred_ids] = count
        false_negative_cost[row, num_pred_ids + row] = count
    for pred_id, count in pred_counts.items():
        column = pred_index[pred_id]
        false_positive_cost[:num_gt_ids, column] = count
        false_positive_cost[num_gt_ids + column, column] = count
    for (gt_id, pred_id), count in potential_matches.items():
        row = gt_index[gt_id]
        column = pred_index[pred_id]
        false_negative_cost[row, column] -= count
        false_positive_cost[row, column] -= count

    rows, columns = linear_sum_assignment(
        false_positive_cost + false_negative_cost
    )
    idfn = int(false_negative_cost[rows, columns].sum())
    idfp = int(false_positive_cost[rows, columns].sum())
    idtp = total_gt - idfn
    if idtp + idfn != total_gt or idtp + idfp != total_pred:
        raise RuntimeError("Identity assignment violated detection conservation")
    return idtp, idfp, idfn


def _clear_matches(
    gt_objects: Sequence[TrackingObject],
    pred_objects: Sequence[TrackingObject],
    *,
    iou_threshold: float,
    previous_timestep_pairs: Mapping[str, str],
) -> list[tuple[int, int, float]]:
    """Preserve eligible prior pairs, then match remaining objects with V2."""
    pred_index_by_id = {
        pred.obj_id: index for index, pred in enumerate(pred_objects)
    }
    preserved: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    for gt_index, gt in enumerate(gt_objects):
        previous_pred_id = previous_timestep_pairs.get(gt.obj_id)
        pred_index = pred_index_by_id.get(previous_pred_id)
        if pred_index is None or pred_index in used_pred:
            continue
        candidate = match_frame_eligible(
            [gt],
            [pred_objects[pred_index]],
            iou_threshold=iou_threshold,
        )
        if not candidate:
            continue
        preserved.append((gt_index, pred_index, candidate[0][2]))
        used_gt.add(gt_index)
        used_pred.add(pred_index)

    remaining_gt_indices = [
        index for index in range(len(gt_objects)) if index not in used_gt
    ]
    remaining_pred_indices = [
        index for index in range(len(pred_objects)) if index not in used_pred
    ]
    remaining_matches = match_frame_eligible(
        [gt_objects[index] for index in remaining_gt_indices],
        [pred_objects[index] for index in remaining_pred_indices],
        iou_threshold=iou_threshold,
    )
    for gt_index, pred_index, iou in remaining_matches:
        preserved.append(
            (
                remaining_gt_indices[gt_index],
                remaining_pred_indices[pred_index],
                iou,
            )
        )
    return sorted(preserved, key=lambda item: (item[0], item[1]))


def _clear_identity_counts(
    gt_by_frame: Mapping[int, Sequence[TrackingObject]],
    pred_by_frame: Mapping[int, Sequence[TrackingObject]],
    *,
    iou_threshold: float,
) -> tuple[int, int, int, int, int]:
    frames = sorted(set(gt_by_frame).union(pred_by_frame))
    last_match_for_gt: dict[str, str] = {}
    previous_timestep_pairs: dict[str, str] = {}
    previous_matched_gt: set[str] = set()
    ever_matched_gt: set[str] = set()
    previous_frame: int | None = None
    clear_tp = 0
    idsw_standard = 0
    fragments = 0

    for frame in frames:
        if previous_frame is None or frame != previous_frame + 1:
            previous_timestep_pairs = {}
            previous_matched_gt = set()
        gt_objects = gt_by_frame.get(frame, ())
        pred_objects = pred_by_frame.get(frame, ())
        matches = _clear_matches(
            gt_objects,
            pred_objects,
            iou_threshold=iou_threshold,
            previous_timestep_pairs=previous_timestep_pairs,
        )
        current_pairs: dict[str, str] = {}
        for gt_index, pred_index, _iou in matches:
            gt_id = gt_objects[gt_index].obj_id
            pred_id = pred_objects[pred_index].obj_id
            prior_pred_id = last_match_for_gt.get(gt_id)
            if prior_pred_id is not None and prior_pred_id != pred_id:
                idsw_standard += 1
            if gt_id in ever_matched_gt and gt_id not in previous_matched_gt:
                fragments += 1
            last_match_for_gt[gt_id] = pred_id
            ever_matched_gt.add(gt_id)
            current_pairs[gt_id] = pred_id

        clear_tp += len(matches)
        previous_timestep_pairs = current_pairs
        previous_matched_gt = set(current_pairs)
        previous_frame = frame

    total_gt = sum(len(objects) for objects in gt_by_frame.values())
    total_pred = sum(len(objects) for objects in pred_by_frame.values())
    clear_fp = total_pred - clear_tp
    clear_fn = total_gt - clear_tp
    return clear_tp, clear_fp, clear_fn, idsw_standard, fragments


def _ratios(idtp: int, idfp: int, idfn: int) -> tuple[float, float, float]:
    idr = idtp / max(1.0, idtp + idfn)
    idp = idtp / max(1.0, idtp + idfp)
    idf1 = idtp / max(1.0, idtp + (0.5 * idfp) + (0.5 * idfn))
    return float(idf1), float(idp), float(idr)


def evaluate_identity_standard_v2(
    gt_by_frame: FrameObjects,
    pred_by_frame: FrameObjects,
    *,
    iou_threshold: float = 0.5,
    sequence_id: str = "",
) -> IdentityStandardV2Metrics:
    """Evaluate one authoritative sequence under the V2 identity contract."""
    threshold = _validate_threshold(iou_threshold)
    gt = _canonical_frames(gt_by_frame, population_name="GT")
    pred = _canonical_frames(pred_by_frame, population_name="prediction")
    idtp, idfp, idfn = _identity_counts(
        gt,
        pred,
        iou_threshold=threshold,
    )
    clear_tp, clear_fp, clear_fn, idsw_standard, fragments = (
        _clear_identity_counts(
            gt,
            pred,
            iou_threshold=threshold,
        )
    )
    idf1, idp, idr = _ratios(idtp, idfp, idfn)
    return IdentityStandardV2Metrics(
        sequence_id=sequence_id,
        sequence_count=1,
        gt_detections=sum(len(objects) for objects in gt.values()),
        pred_detections=sum(len(objects) for objects in pred.values()),
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        idf1=idf1,
        idp=idp,
        idr=idr,
        idsw_standard=idsw_standard,
        fragments=fragments,
        clear_tp=clear_tp,
        clear_fp=clear_fp,
        clear_fn=clear_fn,
    )


def aggregate_identity_standard_v2(
    sequence_metrics: Sequence[IdentityStandardV2Metrics],
    *,
    aggregate_id: str = "ALL",
) -> IdentityStandardV2Metrics:
    """Combine sequence-local counts without assigning identities across videos."""
    metrics = list(sequence_metrics)
    idtp = sum(metric.idtp for metric in metrics)
    idfp = sum(metric.idfp for metric in metrics)
    idfn = sum(metric.idfn for metric in metrics)
    idf1, idp, idr = _ratios(idtp, idfp, idfn)
    return IdentityStandardV2Metrics(
        sequence_id=aggregate_id,
        sequence_count=sum(metric.sequence_count for metric in metrics),
        gt_detections=sum(metric.gt_detections for metric in metrics),
        pred_detections=sum(metric.pred_detections for metric in metrics),
        idtp=idtp,
        idfp=idfp,
        idfn=idfn,
        idf1=idf1,
        idp=idp,
        idr=idr,
        idsw_standard=sum(metric.idsw_standard for metric in metrics),
        fragments=sum(metric.fragments for metric in metrics),
        clear_tp=sum(metric.clear_tp for metric in metrics),
        clear_fp=sum(metric.clear_fp for metric in metrics),
        clear_fn=sum(metric.clear_fn for metric in metrics),
    )
