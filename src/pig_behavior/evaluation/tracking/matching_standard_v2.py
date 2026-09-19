"""Versioned matching helpers for the tracking evaluator V2 contract."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .cvat_io import TrackingObject
from .matching import iou_xyxy


def tracking_object_sort_key(obj: TrackingObject) -> tuple[object, ...]:
    """Return a stable semantic ordering key for one tracking observation."""
    return (
        str(obj.obj_id),
        str(obj.source_track_id),
        int(obj.frame),
        tuple(float(value) for value in obj.bbox),
        bool(obj.hidden),
        str(obj.label),
    )


def _ordered_indices(keys: Sequence[object] | None, length: int) -> list[int]:
    if keys is None:
        return list(range(length))
    if len(keys) != length:
        raise ValueError("Assignment keys must match the matrix dimensions")
    tokens = [repr(key) for key in keys]
    if len(set(tokens)) != len(tokens):
        raise ValueError("Assignment keys must be unique")
    return sorted(range(length), key=tokens.__getitem__)


def eligible_iou_assignment(
    similarity: np.ndarray,
    *,
    threshold: float,
    row_keys: Sequence[object] | None = None,
    col_keys: Sequence[object] | None = None,
) -> list[tuple[int, int, float]]:
    """Match eligible edges, maximizing cardinality before total similarity.

    Edges below ``threshold`` are forbidden before assignment. Stable semantic
    keys make tied inputs independent of their incoming container order.
    """
    matrix = np.asarray(similarity, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Similarity must be a two-dimensional matrix")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be within [0, 1]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Similarity values must be finite")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("Similarity values must be within [0, 1]")

    row_count, col_count = matrix.shape
    if row_count == 0 or col_count == 0:
        return []

    row_order = _ordered_indices(row_keys, row_count)
    col_order = _ordered_indices(col_keys, col_count)
    return _solve_eligible_assignment(
        matrix,
        threshold=threshold,
        row_order=row_order,
        col_order=col_order,
    )


def _solve_eligible_assignment(
    matrix: np.ndarray,
    *,
    threshold: float,
    row_order: Sequence[int],
    col_order: Sequence[int],
    forbidden_edges: frozenset[tuple[int, int]] = frozenset(),
) -> list[tuple[int, int, float]]:
    row_count, col_count = matrix.shape
    ordered = matrix[np.ix_(row_order, col_order)]
    eligible = ordered >= threshold
    row_rank = {original: rank for rank, original in enumerate(row_order)}
    col_rank = {original: rank for rank, original in enumerate(col_order)}
    for original_row, original_col in forbidden_edges:
        eligible[row_rank[original_row], col_rank[original_col]] = False

    max_cardinality = min(row_count, col_count)
    cardinality_bonus = float(max_cardinality + 1)
    forbidden_score = -cardinality_bonus * float(max_cardinality + 1)
    size = row_count + col_count
    score = np.zeros((size, size), dtype=float)
    score[:row_count, :col_count] = forbidden_score
    eligible_scores = cardinality_bonus + ordered
    score[:row_count, :col_count][eligible] = eligible_scores[eligible]

    assigned_rows, assigned_cols = linear_sum_assignment(-score)
    matches: list[tuple[int, int, float]] = []
    for ordered_row, ordered_col in zip(
        assigned_rows,
        assigned_cols,
        strict=True,
    ):
        if ordered_row >= row_count or ordered_col >= col_count:
            continue
        if not eligible[ordered_row, ordered_col]:
            continue
        original_row = row_order[int(ordered_row)]
        original_col = col_order[int(ordered_col)]
        matches.append(
            (
                original_row,
                original_col,
                float(matrix[original_row, original_col]),
            )
        )

    matches.sort(key=lambda item: (row_rank[item[0]], col_rank[item[1]]))
    return matches


def ambiguous_optimal_assignment_edges(
    similarity: np.ndarray,
    *,
    threshold: float,
    matches: Sequence[tuple[int, int, float]],
    row_keys: Sequence[object] | None = None,
    col_keys: Sequence[object] | None = None,
    tolerance: float = 1e-12,
) -> frozenset[tuple[int, int]]:
    """Return chosen edges that are replaceable by an equal optimal assignment."""
    matrix = np.asarray(similarity, dtype=float)
    row_count, col_count = matrix.shape
    row_order = _ordered_indices(row_keys, row_count)
    col_order = _ordered_indices(col_keys, col_count)
    target_cardinality = len(matches)
    target_similarity = sum(match[2] for match in matches)
    ambiguous: set[tuple[int, int]] = set()
    for row_index, col_index, _similarity in matches:
        alternative = _solve_eligible_assignment(
            matrix,
            threshold=threshold,
            row_order=row_order,
            col_order=col_order,
            forbidden_edges=frozenset({(row_index, col_index)}),
        )
        if len(alternative) != target_cardinality:
            continue
        alternative_similarity = sum(match[2] for match in alternative)
        if np.isclose(
            alternative_similarity,
            target_similarity,
            rtol=0.0,
            atol=tolerance,
        ):
            ambiguous.add((row_index, col_index))
    return frozenset(ambiguous)


def match_frame_eligible(
    gt_objects: Sequence[TrackingObject],
    pred_objects: Sequence[TrackingObject],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Match a frame under the pre-eligible V2 detection/CLEAR contract."""
    gt_keys = [tracking_object_sort_key(obj) for obj in gt_objects]
    pred_keys = [tracking_object_sort_key(obj) for obj in pred_objects]
    similarities = np.zeros((len(gt_objects), len(pred_objects)), dtype=float)
    for gt_index, gt in enumerate(gt_objects):
        for pred_index, pred in enumerate(pred_objects):
            similarities[gt_index, pred_index] = iou_xyxy(gt.bbox, pred.bbox)
    return eligible_iou_assignment(
        similarities,
        threshold=iou_threshold,
        row_keys=gt_keys,
        col_keys=pred_keys,
    )


def match_frame_eligible_with_ambiguity(
    gt_objects: Sequence[TrackingObject],
    pred_objects: Sequence[TrackingObject],
    *,
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], frozenset[tuple[int, int]]]:
    """Match one frame and identify equal-optimum authority ambiguity."""
    gt_keys = [tracking_object_sort_key(obj) for obj in gt_objects]
    pred_keys = [tracking_object_sort_key(obj) for obj in pred_objects]
    similarities = np.zeros((len(gt_objects), len(pred_objects)), dtype=float)
    for gt_index, gt in enumerate(gt_objects):
        for pred_index, pred in enumerate(pred_objects):
            similarities[gt_index, pred_index] = iou_xyxy(gt.bbox, pred.bbox)
    matches = eligible_iou_assignment(
        similarities,
        threshold=iou_threshold,
        row_keys=gt_keys,
        col_keys=pred_keys,
    )
    ambiguous = ambiguous_optimal_assignment_edges(
        similarities,
        threshold=iou_threshold,
        matches=matches,
        row_keys=gt_keys,
        col_keys=pred_keys,
    )
    return matches, ambiguous
