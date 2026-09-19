from __future__ import annotations

import pytest

from pig_behavior.evaluation.tracking.cvat_io import TrackingObject
from pig_behavior.evaluation.tracking.identity_standard_v2 import (
    aggregate_identity_standard_v2,
    evaluate_identity_standard_v2,
)

BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (10.0, 0.0, 20.0, 10.0)


def _object(
    frame: int,
    obj_id: str,
    bbox: tuple[float, float, float, float] = BOX_A,
) -> TrackingObject:
    return TrackingObject(
        frame=frame,
        obj_id=obj_id,
        bbox=bbox,
        source_track_id=obj_id,
        label=obj_id,
    )


def _single_identity(
    pred_ids: list[str | None],
) -> tuple[dict[int, list[TrackingObject]], dict[int, list[TrackingObject]]]:
    gt = {
        frame: [_object(frame, "A")]
        for frame in range(len(pred_ids))
    }
    pred = {
        frame: [_object(frame, pred_id)]
        for frame, pred_id in enumerate(pred_ids)
        if pred_id is not None
    }
    return gt, pred


def test_perfect_sequence_has_trackeval_identity_scores() -> None:
    gt, pred = _single_identity(["X", "X", "X"])

    result = evaluate_identity_standard_v2(gt, pred, sequence_id="perfect")

    assert (result.idtp, result.idfp, result.idfn) == (3, 0, 0)
    assert (result.idf1, result.idp, result.idr) == (1.0, 1.0, 1.0)
    assert (result.idsw_standard, result.fragments) == (0, 0)
    assert (result.clear_tp, result.clear_fp, result.clear_fn) == (3, 0, 0)


def test_switch_and_recovery_are_two_standard_switch_events() -> None:
    gt, pred = _single_identity(["X", "Y", "X"])

    result = evaluate_identity_standard_v2(gt, pred)

    assert (result.idtp, result.idfp, result.idfn) == (2, 1, 1)
    assert result.idf1 == pytest.approx(2 / 3)
    assert result.idp == pytest.approx(2 / 3)
    assert result.idr == pytest.approx(2 / 3)
    assert result.idsw_standard == 2
    assert result.fragments == 0


def test_global_identity_assignment_handles_two_crossing_trajectories() -> None:
    gt: dict[int, list[TrackingObject]] = {}
    pred: dict[int, list[TrackingObject]] = {}
    for frame in range(4):
        gt[frame] = [
            _object(frame, "A", BOX_A),
            _object(frame, "B", BOX_B),
        ]
        pred_ids = ("X", "Y") if frame < 2 else ("Y", "X")
        pred[frame] = [
            _object(frame, pred_ids[0], BOX_A),
            _object(frame, pred_ids[1], BOX_B),
        ]

    result = evaluate_identity_standard_v2(gt, pred)

    assert (result.idtp, result.idfp, result.idfn) == (4, 4, 4)
    assert result.idf1 == pytest.approx(0.5)
    assert result.idsw_standard == 2


def test_gap_with_same_identity_is_fragment_not_switch() -> None:
    gt, pred = _single_identity(["X", None, "X"])

    result = evaluate_identity_standard_v2(gt, pred)

    assert (result.idtp, result.idfp, result.idfn) == (2, 0, 1)
    assert result.idf1 == pytest.approx(0.8)
    assert result.idsw_standard == 0
    assert result.fragments == 1
    assert (result.clear_tp, result.clear_fp, result.clear_fn) == (2, 0, 1)


def test_switch_memory_persists_across_an_unmatched_gap() -> None:
    gt, pred = _single_identity(["X", None, "Y"])

    result = evaluate_identity_standard_v2(gt, pred)

    assert (result.idtp, result.idfp, result.idfn) == (1, 1, 2)
    assert result.idf1 == pytest.approx(0.4)
    assert result.idsw_standard == 1
    assert result.fragments == 1


def test_clear_matching_preserves_eligible_previous_timestep_pairs() -> None:
    gt = {
        0: [_object(0, "A", BOX_A), _object(0, "B", BOX_B)],
        1: [_object(1, "A", BOX_A), _object(1, "B", BOX_B)],
    }
    pred = {
        0: [_object(0, "X", BOX_A), _object(0, "Y", BOX_B)],
        1: [
            _object(1, "X", (8.0, 0.0, 18.0, 10.0)),
            _object(1, "Y", (2.0, 0.0, 12.0, 10.0)),
        ],
    }

    result = evaluate_identity_standard_v2(
        gt,
        pred,
        iou_threshold=0.1,
    )

    assert result.clear_tp == 4
    assert result.idsw_standard == 0
    assert result.fragments == 0


def test_ineligible_previous_pair_is_not_preserved() -> None:
    gt = {
        0: [_object(0, "A", BOX_A), _object(0, "B", BOX_B)],
        1: [_object(1, "A", BOX_A), _object(1, "B", BOX_B)],
    }
    pred = {
        0: [_object(0, "X", BOX_A), _object(0, "Y", BOX_B)],
        1: [_object(1, "X", BOX_B), _object(1, "Y", BOX_A)],
    }

    result = evaluate_identity_standard_v2(gt, pred)

    assert result.clear_tp == 4
    assert result.idsw_standard == 2


def test_aggregation_respects_sequence_boundaries() -> None:
    first_gt, first_pred = _single_identity(["X"])
    second_gt, second_pred = _single_identity(["X"])
    first = evaluate_identity_standard_v2(
        first_gt,
        first_pred,
        sequence_id="first",
    )
    second = evaluate_identity_standard_v2(
        second_gt,
        second_pred,
        sequence_id="second",
    )

    aggregate = aggregate_identity_standard_v2([first, second])

    assert aggregate.sequence_count == 2
    assert (aggregate.idtp, aggregate.idfp, aggregate.idfn) == (2, 0, 0)
    assert aggregate.idf1 == 1.0
    assert aggregate.idsw_standard == 0
    assert aggregate.fragments == 0


def test_aggregation_recomputes_identity_ratios_from_summed_counts() -> None:
    first_gt, first_pred = _single_identity(["X", None])
    second_gt, second_pred = _single_identity(["Y"])
    second_pred[0].append(_object(0, "Z", BOX_B))
    first = evaluate_identity_standard_v2(first_gt, first_pred)
    second = evaluate_identity_standard_v2(second_gt, second_pred)

    aggregate = aggregate_identity_standard_v2([first, second])

    assert (aggregate.idtp, aggregate.idfp, aggregate.idfn) == (2, 1, 1)
    assert aggregate.idf1 == pytest.approx(2 / 3)
    assert aggregate.idp == pytest.approx(2 / 3)
    assert aggregate.idr == pytest.approx(2 / 3)


def test_input_order_permutation_does_not_change_results() -> None:
    gt = {
        0: [_object(0, "A", BOX_A), _object(0, "B", BOX_B)],
        1: [_object(1, "A", BOX_A), _object(1, "B", BOX_B)],
    }
    pred = {
        0: [_object(0, "X", BOX_A), _object(0, "Y", BOX_B)],
        1: [_object(1, "X", BOX_A), _object(1, "Y", BOX_B)],
    }

    ordered = evaluate_identity_standard_v2(gt, pred, sequence_id="order")
    reversed_result = evaluate_identity_standard_v2(
        {frame: list(reversed(objects)) for frame, objects in gt.items()},
        {frame: list(reversed(objects)) for frame, objects in pred.items()},
        sequence_id="order",
    )

    assert ordered == reversed_result


@pytest.mark.parametrize(
    ("gt", "pred", "expected"),
    [
        ({}, {}, (0, 0, 0, 0, 0)),
        ({0: [_object(0, "A")]}, {}, (0, 0, 1, 0, 1)),
        ({}, {0: [_object(0, "X")]}, (0, 1, 0, 1, 0)),
    ],
)
def test_empty_population_policies(
    gt: dict[int, list[TrackingObject]],
    pred: dict[int, list[TrackingObject]],
    expected: tuple[int, int, int, int, int],
) -> None:
    result = evaluate_identity_standard_v2(gt, pred)

    assert (
        result.idtp,
        result.idfp,
        result.idfn,
        result.clear_fp,
        result.clear_fn,
    ) == expected
    assert (result.idf1, result.idp, result.idr) == (0.0, 0.0, 0.0)


def test_duplicate_identity_in_one_frame_fails_closed() -> None:
    gt = {
        0: [
            _object(0, "A", BOX_A),
            _object(0, "A", BOX_B),
        ]
    }

    with pytest.raises(ValueError, match="GT frame 0 has duplicate IDs: A"):
        evaluate_identity_standard_v2(gt, {})


def test_aggregate_of_no_sequences_is_zero_not_perfect() -> None:
    result = aggregate_identity_standard_v2([])

    assert result.sequence_count == 0
    assert (result.idtp, result.idfp, result.idfn) == (0, 0, 0)
    assert (result.idf1, result.idp, result.idr) == (0.0, 0.0, 0.0)
    assert (result.idsw_standard, result.fragments) == (0, 0)
