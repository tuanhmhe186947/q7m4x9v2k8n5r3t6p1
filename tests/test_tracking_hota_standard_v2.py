from __future__ import annotations

import math

import numpy as np
import pytest

from pig_behavior.evaluation.tracking.contracts import HOTA_ALPHAS
from pig_behavior.evaluation.tracking.cvat_io import TrackingObject
from pig_behavior.evaluation.tracking.hota_standard_v2 import (
    combine_hota_sequences,
    evaluate_hota_sequence,
    headline_hota_metrics,
    hota_at_alpha,
)
from pig_behavior.evaluation.tracking.matching_standard_v2 import (
    eligible_iou_assignment,
    match_frame_eligible,
    match_frame_eligible_with_ambiguity,
)


def _object(
    frame: int,
    obj_id: str,
    bbox: tuple[float, float, float, float],
) -> TrackingObject:
    return TrackingObject(
        frame=frame,
        obj_id=obj_id,
        bbox=bbox,
        source_track_id=obj_id,
    )


def test_detection_assignment_forbids_ineligible_edges_before_matching() -> None:
    similarities = np.asarray(
        [
            [0.99, 0.74],
            [0.73, 0.49],
        ]
    )
    matches = eligible_iou_assignment(similarities, threshold=0.5)
    assert matches == [(0, 1, 0.74), (1, 0, 0.73)]


def test_detection_matching_is_independent_of_input_order() -> None:
    gt = [
        _object(0, "gt_a", (0.0, 0.0, 1.0, 1.0)),
        _object(0, "gt_b", (10.0, 0.0, 11.0, 1.0)),
    ]
    pred = [
        _object(0, "pred_a", (0.0, 0.0, 1.0, 1.0)),
        _object(0, "pred_b", (10.0, 0.0, 11.0, 1.0)),
    ]

    def identity_pairs(
        gt_items: list[TrackingObject],
        pred_items: list[TrackingObject],
    ) -> set[tuple[str, str]]:
        return {
            (gt_items[gt_index].obj_id, pred_items[pred_index].obj_id)
            for gt_index, pred_index, _iou in match_frame_eligible(
                gt_items,
                pred_items,
                iou_threshold=0.5,
            )
        }

    expected = {("gt_a", "pred_a"), ("gt_b", "pred_b")}
    assert identity_pairs(gt, pred) == expected
    assert identity_pairs(list(reversed(gt)), list(reversed(pred))) == expected


def test_equal_optimum_assignment_is_marked_ambiguous() -> None:
    box = (0.0, 0.0, 1.0, 1.0)
    gt = [_object(0, "gt_a", box), _object(0, "gt_b", box)]
    pred = [_object(0, "pred_a", box), _object(0, "pred_b", box)]

    matches, ambiguous = match_frame_eligible_with_ambiguity(
        gt,
        pred,
        iou_threshold=0.5,
    )

    assert len(matches) == 2
    assert ambiguous == frozenset(
        (gt_index, pred_index)
        for gt_index, pred_index, _iou in matches
    )


def test_unique_optimum_assignment_has_no_ambiguity() -> None:
    gt = [
        _object(0, "gt_a", (0.0, 0.0, 1.0, 1.0)),
        _object(0, "gt_b", (10.0, 0.0, 11.0, 1.0)),
    ]
    pred = [
        _object(0, "pred_a", (0.0, 0.0, 1.0, 1.0)),
        _object(0, "pred_b", (10.0, 0.0, 11.0, 1.0)),
    ]

    matches, ambiguous = match_frame_eligible_with_ambiguity(
        gt,
        pred,
        iou_threshold=0.5,
    )

    assert len(matches) == 2
    assert ambiguous == frozenset()


def test_perfect_hota_uses_all_nineteen_thresholds() -> None:
    gt = {
        frame: [_object(frame, "gt", (0.0, 0.0, 1.0, 1.0))]
        for frame in range(3)
    }
    pred = {
        frame: [_object(frame, "pred", (0.0, 0.0, 1.0, 1.0))]
        for frame in range(3)
    }
    result = evaluate_hota_sequence(gt, pred, sequence_key="perfect")

    assert result.alphas == HOTA_ALPHAS
    assert len(result.alphas) == 19
    assert result.tp == (3,) * 19
    assert result.fp == (0,) * 19
    assert result.fn == (0,) * 19
    assert result.hota == (1.0,) * 19
    assert headline_hota_metrics(result) == {
        "hota": 1.0,
        "deta": 1.0,
        "assa": 1.0,
        "loca": 1.0,
    }


def test_hota_threshold_acceptance_and_qualified_diagnostic() -> None:
    gt = {0: [_object(0, "gt", (0.0, 0.0, 1.0, 1.0))]}
    pred = {0: [_object(0, "pred", (0.0, 0.0, 0.6, 1.0))]}
    result = evaluate_hota_sequence(gt, pred)

    at_060 = hota_at_alpha(result, 0.60)
    at_065 = hota_at_alpha(result, 0.65)
    assert at_060["tp"] == 1
    assert at_060["fp"] == 0
    assert at_060["fn"] == 0
    assert at_060["hota"] == pytest.approx(1.0)
    assert at_060["loca"] == pytest.approx(0.6)
    assert at_065 == {
        "alpha": 0.65,
        "hota": 0.0,
        "deta": 0.0,
        "assa": 0.0,
        "loca": 1.0,
        "tp": 0,
        "fp": 1,
        "fn": 1,
    }


def test_hota_association_is_detection_weighted() -> None:
    gt = {
        frame: [_object(frame, "gt", (0.0, 0.0, 1.0, 1.0))]
        for frame in range(2)
    }
    pred = {
        0: [_object(0, "pred_a", (0.0, 0.0, 1.0, 1.0))],
        1: [_object(1, "pred_b", (0.0, 0.0, 1.0, 1.0))],
    }
    result = evaluate_hota_sequence(gt, pred)

    assert result.deta == (1.0,) * 19
    assert result.assa == (0.5,) * 19
    assert result.hota == pytest.approx((math.sqrt(0.5),) * 19)


def test_sequence_aggregation_uses_sufficient_counts_not_video_mean() -> None:
    first_gt = {
        frame: [_object(frame, "gt", (0.0, 0.0, 1.0, 1.0))]
        for frame in range(2)
    }
    first_pred = {
        frame: [_object(frame, "pred", (0.0, 0.0, 1.0, 1.0))]
        for frame in range(2)
    }
    second_gt = {0: [_object(0, "gt", (0.0, 0.0, 1.0, 1.0))]}
    first = evaluate_hota_sequence(first_gt, first_pred, sequence_key="one")
    second = evaluate_hota_sequence(second_gt, {}, sequence_key="two")
    combined = combine_hota_sequences([first, second])

    assert combined.tp == (2,) * 19
    assert combined.fn == (1,) * 19
    assert combined.fp == (0,) * 19
    assert combined.deta == pytest.approx((2.0 / 3.0,) * 19)
    assert combined.assa == (1.0,) * 19
    assert combined.hota == pytest.approx((math.sqrt(2.0 / 3.0),) * 19)
    assert combined.hota_mean != pytest.approx(
        (first.hota_mean + second.hota_mean) / 2.0
    )


def test_sequence_combination_does_not_link_identity_state_across_videos() -> None:
    box = (0.0, 0.0, 1.0, 1.0)
    first = evaluate_hota_sequence(
        {0: [_object(0, "same_gt", box)]},
        {0: [_object(0, "pred_a", box)]},
        sequence_key="one",
    )
    second = evaluate_hota_sequence(
        {0: [_object(0, "same_gt", box)]},
        {0: [_object(0, "pred_b", box)]},
        sequence_key="two",
    )
    combined = combine_hota_sequences([first, second])

    assert combined.assa == (1.0,) * 19
    assert combined.hota == (1.0,) * 19


def test_hota_conserves_gt_and_prediction_detections_at_every_alpha() -> None:
    box = (0.0, 0.0, 1.0, 1.0)
    gt = {
        0: [_object(0, "gt", box)],
        1: [_object(1, "gt", box)],
    }
    pred = {
        0: [_object(0, "pred", box)],
        2: [_object(2, "false_positive", box)],
    }
    result = evaluate_hota_sequence(gt, pred)

    for tp, fp, fn in zip(result.tp, result.fp, result.fn, strict=True):
        assert tp + fn == 2
        assert tp + fp == 2
        assert (tp, fp, fn) == (1, 1, 1)


def test_hota_result_is_invariant_to_per_frame_input_permutation() -> None:
    gt_a = _object(0, "gt_a", (0.0, 0.0, 1.0, 1.0))
    gt_b = _object(0, "gt_b", (10.0, 0.0, 11.0, 1.0))
    pred_a = _object(0, "pred_a", (0.0, 0.0, 1.0, 1.0))
    pred_b = _object(0, "pred_b", (10.0, 0.0, 11.0, 1.0))
    ordered = evaluate_hota_sequence(
        {0: [gt_a, gt_b]},
        {0: [pred_a, pred_b]},
    )
    permuted = evaluate_hota_sequence(
        {0: [gt_b, gt_a]},
        {0: [pred_b, pred_a]},
    )
    assert ordered == permuted


def test_empty_sequence_matches_trackeval_zero_denominator_policy() -> None:
    result = evaluate_hota_sequence({}, {})
    assert result.hota == (0.0,) * 19
    assert result.deta == (0.0,) * 19
    assert result.assa == (0.0,) * 19
    assert result.loca == (1.0,) * 19


def test_duplicate_identity_in_one_frame_fails_closed() -> None:
    duplicate = [
        _object(0, "gt", (0.0, 0.0, 1.0, 1.0)),
        _object(0, "gt", (2.0, 0.0, 3.0, 1.0)),
    ]
    with pytest.raises(ValueError, match="Duplicate GT identity"):
        evaluate_hota_sequence({0: duplicate}, {})
