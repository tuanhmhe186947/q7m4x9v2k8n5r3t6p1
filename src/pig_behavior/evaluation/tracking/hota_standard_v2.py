"""Standard threshold-averaged HOTA for the evaluator V2 contract.

The implementation follows the pinned official TrackEval HOTA sequence and
sequence-combination semantics without making TrackEval a runtime dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contracts import HOTA_ALPHAS
from .cvat_io import TrackingObject
from .matching import iou_xyxy
from .matching_standard_v2 import tracking_object_sort_key

_FLOAT_EPSILON = np.finfo(float).eps
_LOCA_FLOOR = 1e-10


@dataclass(frozen=True, slots=True)
class HOTAStandardV2Result:
    """Immutable HOTA values and sufficient statistics for one sequence."""

    sequence_key: str
    alphas: tuple[float, ...]
    hota: tuple[float, ...]
    deta: tuple[float, ...]
    assa: tuple[float, ...]
    loca: tuple[float, ...]
    detection_recall: tuple[float, ...]
    detection_precision: tuple[float, ...]
    association_recall: tuple[float, ...]
    association_precision: tuple[float, ...]
    tp: tuple[int, ...]
    fp: tuple[int, ...]
    fn: tuple[int, ...]
    association_weighted_sum: tuple[float, ...]
    localization_iou_sum: tuple[float, ...]

    @property
    def hota_mean(self) -> float:
        """Return standard HOTA averaged over the canonical alpha set."""
        return float(np.mean(self.hota))

    @property
    def deta_mean(self) -> float:
        """Return standard DetA averaged over the canonical alpha set."""
        return float(np.mean(self.deta))

    @property
    def assa_mean(self) -> float:
        """Return standard AssA averaged over the canonical alpha set."""
        return float(np.mean(self.assa))

    @property
    def loca_mean(self) -> float:
        """Return standard LocA averaged over the canonical alpha set."""
        return float(np.mean(self.loca))


def _as_float_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _as_int_tuple(values: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _validate_and_order_frame(
    frame: int,
    objects: Sequence[TrackingObject],
    *,
    role: str,
) -> list[TrackingObject]:
    ordered = sorted(objects, key=tracking_object_sort_key)
    identifiers = [str(obj.obj_id) for obj in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate {role} identity in frame {frame}")
    if any(int(obj.frame) != frame for obj in ordered):
        raise ValueError(f"{role} observation has a mismatched frame index")
    return ordered


def _similarity_matrix(
    gt_objects: Sequence[TrackingObject],
    pred_objects: Sequence[TrackingObject],
) -> np.ndarray:
    similarities = np.zeros((len(gt_objects), len(pred_objects)), dtype=float)
    for gt_index, gt in enumerate(gt_objects):
        for pred_index, pred in enumerate(pred_objects):
            similarities[gt_index, pred_index] = iou_xyxy(gt.bbox, pred.bbox)
    return similarities


def _finalize_result(
    *,
    sequence_key: str,
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    association_weighted_sum: np.ndarray,
    association_recall_weighted_sum: np.ndarray,
    association_precision_weighted_sum: np.ndarray,
    localization_iou_sum: np.ndarray,
) -> HOTAStandardV2Result:
    association_denominator = np.maximum(1.0, tp)
    assa = association_weighted_sum / association_denominator
    association_recall = (
        association_recall_weighted_sum / association_denominator
    )
    association_precision = (
        association_precision_weighted_sum / association_denominator
    )
    loca = np.maximum(_LOCA_FLOOR, localization_iou_sum) / np.maximum(
        _LOCA_FLOOR,
        tp,
    )
    detection_recall = tp / np.maximum(1.0, tp + fn)
    detection_precision = tp / np.maximum(1.0, tp + fp)
    deta = tp / np.maximum(1.0, tp + fn + fp)
    hota = np.sqrt(deta * assa)
    return HOTAStandardV2Result(
        sequence_key=sequence_key,
        alphas=HOTA_ALPHAS,
        hota=_as_float_tuple(hota),
        deta=_as_float_tuple(deta),
        assa=_as_float_tuple(assa),
        loca=_as_float_tuple(loca),
        detection_recall=_as_float_tuple(detection_recall),
        detection_precision=_as_float_tuple(detection_precision),
        association_recall=_as_float_tuple(association_recall),
        association_precision=_as_float_tuple(association_precision),
        tp=_as_int_tuple(tp),
        fp=_as_int_tuple(fp),
        fn=_as_int_tuple(fn),
        association_weighted_sum=_as_float_tuple(association_weighted_sum),
        localization_iou_sum=_as_float_tuple(localization_iou_sum),
    )


def evaluate_hota_sequence(
    gt_by_frame: Mapping[int, Sequence[TrackingObject]],
    pred_by_frame: Mapping[int, Sequence[TrackingObject]],
    *,
    sequence_key: str = "",
) -> HOTAStandardV2Result:
    """Evaluate one independent sequence with official TrackEval HOTA semantics."""
    frames = sorted(set(gt_by_frame).union(pred_by_frame))
    gt_frames = [
        _validate_and_order_frame(
            frame,
            gt_by_frame.get(frame, ()),
            role="GT",
        )
        for frame in frames
    ]
    pred_frames = [
        _validate_and_order_frame(
            frame,
            pred_by_frame.get(frame, ()),
            role="prediction",
        )
        for frame in frames
    ]
    gt_identifiers = sorted(
        {str(obj.obj_id) for objects in gt_frames for obj in objects}
    )
    pred_identifiers = sorted(
        {str(obj.obj_id) for objects in pred_frames for obj in objects}
    )
    gt_index = {identifier: index for index, identifier in enumerate(gt_identifiers)}
    pred_index = {
        identifier: index for index, identifier in enumerate(pred_identifiers)
    }

    gt_ids_by_frame = [
        np.asarray([gt_index[str(obj.obj_id)] for obj in objects], dtype=int)
        for objects in gt_frames
    ]
    pred_ids_by_frame = [
        np.asarray([pred_index[str(obj.obj_id)] for obj in objects], dtype=int)
        for objects in pred_frames
    ]
    similarities = [
        _similarity_matrix(gt_objects, pred_objects)
        for gt_objects, pred_objects in zip(gt_frames, pred_frames, strict=True)
    ]

    potential_matches = np.zeros(
        (len(gt_identifiers), len(pred_identifiers)),
        dtype=float,
    )
    gt_id_count = np.zeros((len(gt_identifiers), 1), dtype=float)
    pred_id_count = np.zeros((1, len(pred_identifiers)), dtype=float)
    for gt_ids, pred_ids, similarity in zip(
        gt_ids_by_frame,
        pred_ids_by_frame,
        similarities,
        strict=True,
    ):
        denominator = (
            similarity.sum(axis=0)[np.newaxis, :]
            + similarity.sum(axis=1)[:, np.newaxis]
            - similarity
        )
        normalized = np.zeros_like(similarity)
        valid = denominator > _FLOAT_EPSILON
        normalized[valid] = similarity[valid] / denominator[valid]
        potential_matches[np.ix_(gt_ids, pred_ids)] += normalized
        gt_id_count[gt_ids] += 1
        pred_id_count[0, pred_ids] += 1

    alignment_denominator = gt_id_count + pred_id_count - potential_matches
    global_alignment = np.zeros_like(potential_matches)
    valid_alignment = alignment_denominator > _FLOAT_EPSILON
    global_alignment[valid_alignment] = (
        potential_matches[valid_alignment] / alignment_denominator[valid_alignment]
    )

    alpha_count = len(HOTA_ALPHAS)
    tp = np.zeros(alpha_count, dtype=int)
    fp = np.zeros(alpha_count, dtype=int)
    fn = np.zeros(alpha_count, dtype=int)
    localization_iou_sum = np.zeros(alpha_count, dtype=float)
    matches_by_alpha = [
        np.zeros_like(potential_matches) for _alpha in HOTA_ALPHAS
    ]

    for gt_ids, pred_ids, similarity in zip(
        gt_ids_by_frame,
        pred_ids_by_frame,
        similarities,
        strict=True,
    ):
        if len(gt_ids) == 0:
            fp += len(pred_ids)
            continue
        if len(pred_ids) == 0:
            fn += len(gt_ids)
            continue

        score = global_alignment[np.ix_(gt_ids, pred_ids)] * similarity
        matched_rows, matched_cols = linear_sum_assignment(-score)
        for alpha_index, alpha in enumerate(HOTA_ALPHAS):
            accepted = (
                similarity[matched_rows, matched_cols]
                >= alpha - _FLOAT_EPSILON
            )
            alpha_rows = matched_rows[accepted]
            alpha_cols = matched_cols[accepted]
            match_count = len(alpha_rows)
            tp[alpha_index] += match_count
            fn[alpha_index] += len(gt_ids) - match_count
            fp[alpha_index] += len(pred_ids) - match_count
            if match_count:
                localization_iou_sum[alpha_index] += float(
                    similarity[alpha_rows, alpha_cols].sum()
                )
                matches_by_alpha[alpha_index][
                    gt_ids[alpha_rows],
                    pred_ids[alpha_cols],
                ] += 1

    association_weighted_sum = np.zeros(alpha_count, dtype=float)
    association_recall_weighted_sum = np.zeros(alpha_count, dtype=float)
    association_precision_weighted_sum = np.zeros(alpha_count, dtype=float)
    for alpha_index, match_counts in enumerate(matches_by_alpha):
        association_accuracy = match_counts / np.maximum(
            1.0,
            gt_id_count + pred_id_count - match_counts,
        )
        association_recall = match_counts / np.maximum(1.0, gt_id_count)
        association_precision = match_counts / np.maximum(1.0, pred_id_count)
        association_weighted_sum[alpha_index] = float(
            np.sum(match_counts * association_accuracy)
        )
        association_recall_weighted_sum[alpha_index] = float(
            np.sum(match_counts * association_recall)
        )
        association_precision_weighted_sum[alpha_index] = float(
            np.sum(match_counts * association_precision)
        )

    return _finalize_result(
        sequence_key=sequence_key,
        tp=tp,
        fp=fp,
        fn=fn,
        association_weighted_sum=association_weighted_sum,
        association_recall_weighted_sum=association_recall_weighted_sum,
        association_precision_weighted_sum=association_precision_weighted_sum,
        localization_iou_sum=localization_iou_sum,
    )


def combine_hota_sequences(
    results: Sequence[HOTAStandardV2Result],
    *,
    sequence_key: str = "ALL",
) -> HOTAStandardV2Result:
    """Combine independent sequences using official TrackEval sufficient stats."""
    if not results:
        raise ValueError("At least one HOTA sequence result is required")
    if any(result.alphas != HOTA_ALPHAS for result in results):
        raise ValueError("All HOTA results must use the canonical alpha set")

    tp = np.sum([result.tp for result in results], axis=0, dtype=int)
    fp = np.sum([result.fp for result in results], axis=0, dtype=int)
    fn = np.sum([result.fn for result in results], axis=0, dtype=int)
    association_weighted_sum = np.sum(
        [result.association_weighted_sum for result in results],
        axis=0,
    )
    association_recall_weighted_sum = np.sum(
        [
            np.asarray(result.association_recall) * np.asarray(result.tp)
            for result in results
        ],
        axis=0,
    )
    association_precision_weighted_sum = np.sum(
        [
            np.asarray(result.association_precision) * np.asarray(result.tp)
            for result in results
        ],
        axis=0,
    )
    localization_iou_sum = np.sum(
        [result.localization_iou_sum for result in results],
        axis=0,
    )
    return _finalize_result(
        sequence_key=sequence_key,
        tp=np.asarray(tp),
        fp=np.asarray(fp),
        fn=np.asarray(fn),
        association_weighted_sum=np.asarray(association_weighted_sum),
        association_recall_weighted_sum=np.asarray(
            association_recall_weighted_sum
        ),
        association_precision_weighted_sum=np.asarray(
            association_precision_weighted_sum
        ),
        localization_iou_sum=np.asarray(localization_iou_sum),
    )


def headline_hota_metrics(result: HOTAStandardV2Result) -> dict[str, float]:
    """Return the four standard threshold-averaged headline values."""
    return {
        "hota": result.hota_mean,
        "deta": result.deta_mean,
        "assa": result.assa_mean,
        "loca": result.loca_mean,
    }


def hota_at_alpha(
    result: HOTAStandardV2Result,
    alpha: float,
) -> dict[str, float | int]:
    """Return one explicitly alpha-qualified diagnostic row."""
    try:
        index = HOTA_ALPHAS.index(float(alpha))
    except ValueError as exc:
        raise ValueError(f"Alpha is not in the canonical set: {alpha}") from exc
    return {
        "alpha": HOTA_ALPHAS[index],
        "hota": result.hota[index],
        "deta": result.deta[index],
        "assa": result.assa[index],
        "loca": result.loca[index],
        "tp": result.tp[index],
        "fp": result.fp[index],
        "fn": result.fn[index],
    }
