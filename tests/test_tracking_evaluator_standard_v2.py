from __future__ import annotations

from copy import deepcopy
from dataclasses import fields

import pytest

from pig_behavior.evaluation.tracking.contracts import EVALUATOR_CONTRACT_ID
from pig_behavior.evaluation.tracking.cvat_io import TrackingObject
from pig_behavior.evaluation.tracking.evaluator_standard_v2 import (
    aggregate_tracking_standard_v2,
    evaluate_tracking_standard_v2,
    metrics_to_dataframe_standard_v2,
)
from pig_behavior.evaluation.tracking.identity_episodes_v2 import (
    IdentityAuthority,
    IdentityErrorEpisode,
    MatchedIdentityRow,
    PairwiseIdentitySwapEvent,
)
from pig_behavior.evaluation.tracking.reporting_standard_v2 import (
    identity_ambiguity_dataframe,
    identity_authority_dataframe,
    identity_episode_dataframe,
    pairwise_swap_dataframe,
)


def _object(
    frame: int,
    obj_id: str,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0),
    *,
    hidden: bool = False,
) -> TrackingObject:
    return TrackingObject(
        frame=frame,
        obj_id=obj_id,
        bbox=bbox,
        hidden=hidden,
    )


def _evaluate(
    gt: dict[int, list[TrackingObject]],
    pred: dict[int, list[TrackingObject]],
    *,
    video_stem: str = "synthetic",
    include_hidden: bool = True,
    fps: float | None = 1.0,
    authority: dict[tuple[str, str], str] | None = None,
):
    return evaluate_tracking_standard_v2(
        gt,
        pred,
        video_stem=video_stem,
        include_hidden=include_hidden,
        frames_per_second=fps,
        evaluator_code_sha="a" * 40,
        explicit_identity_authority=authority,
    )


def test_perfect_trajectory_has_standard_unit_scores() -> None:
    gt = {frame: [_object(frame, "A")] for frame in range(3)}
    pred = {frame: [_object(frame, "pA")] for frame in range(3)}
    result = _evaluate(gt, pred, authority={("synthetic", "A"): "pA"})
    metrics = result.metrics

    assert metrics.evaluator_contract_id == EVALUATOR_CONTRACT_ID
    assert metrics.hota == metrics.deta == metrics.assa == metrics.loca == 1.0
    assert metrics.idf1 == metrics.id_precision == metrics.id_recall == 1.0
    assert (metrics.tp, metrics.fp, metrics.fn) == (3, 0, 0)
    assert metrics.idsw_standard == 0
    assert metrics.identity_error_episode_count == 0


def test_hota_is_threshold_averaged_not_single_threshold() -> None:
    gt = {0: [_object(0, "A", (0.0, 0.0, 10.0, 10.0))]}
    pred = {0: [_object(0, "pA", (0.0, 0.0, 6.0, 10.0))]}
    result = _evaluate(gt, pred, authority={("synthetic", "A"): "pA"})

    assert result.metrics.hota_at_alpha_050_diagnostic == 1.0
    assert result.metrics.hota == pytest.approx(12 / 19)
    assert result.metrics.deta == pytest.approx(12 / 19)
    assert result.metrics.assa == pytest.approx(12 / 19)


def test_one_frame_wrong_identity_recovers_and_counts_two_switches() -> None:
    gt = {frame: [_object(frame, "A")] for frame in range(3)}
    pred_ids = ["pA", "pB", "pA"]
    pred = {
        frame: [_object(frame, pred_id)]
        for frame, pred_id in enumerate(pred_ids)
    }
    result = _evaluate(gt, pred, authority={("synthetic", "A"): "pA"})

    assert result.metrics.idsw_standard == 2
    assert result.metrics.wrong_id_matched_frames == 1
    assert result.metrics.wrong_id_matched_seconds == 1.0
    assert result.metrics.identity_error_episode_count == 1
    assert result.metrics.recovered_identity_error_episode_count == 1
    assert result.metrics.terminal_identity_error_episode_count == 0


def test_terminal_identity_error_is_not_a_pairwise_swap() -> None:
    gt = {frame: [_object(frame, "A")] for frame in range(4)}
    pred = {
        0: [_object(0, "pA")],
        1: [_object(1, "pB")],
        2: [_object(2, "pB")],
        3: [_object(3, "pB")],
    }
    result = _evaluate(gt, pred, authority={("synthetic", "A"): "pA"})

    assert result.metrics.idsw_standard == 1
    assert result.metrics.wrong_id_matched_frames == 3
    assert result.metrics.terminal_identity_error_episode_count == 1
    assert result.metrics.persistent_pairwise_identity_swap_count == 0


def test_hidden_policy_changes_only_the_declared_population() -> None:
    gt = {
        0: [_object(0, "A")],
        1: [_object(1, "A", hidden=True)],
        2: [_object(2, "A")],
    }
    pred = {
        0: [_object(0, "pA")],
        1: [_object(1, "pB", hidden=True)],
        2: [_object(2, "pA")],
    }
    authority = {("synthetic", "A"): "pA"}
    visible = _evaluate(
        gt,
        pred,
        include_hidden=False,
        authority=authority,
    )
    hidden = _evaluate(gt, pred, include_hidden=True, authority=authority)

    assert visible.metrics.tp == 2
    assert visible.metrics.wrong_id_matched_frames == 0
    assert visible.metrics.evaluated_hidden_rows == 0
    assert hidden.metrics.tp == 3
    assert hidden.metrics.wrong_id_matched_frames == 1
    assert hidden.metrics.evaluated_hidden_rows == 1


def test_tied_first_identity_assignment_stays_audit_ambiguous() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    gt = {0: [_object(0, "A", box), _object(0, "B", box)]}
    pred = {0: [_object(0, "pA", box), _object(0, "pB", box)]}

    result = _evaluate(gt, pred)

    assert result.metrics.tp == 2
    assert result.metrics.ambiguous_identity_rows == 2
    assert result.episode_result.authorities == ()
    assert result.metrics.identity_error_episode_count == 0


def test_aggregate_uses_sufficient_statistics_and_boundaries() -> None:
    first = _evaluate(
        {0: [_object(0, "A")]},
        {0: [_object(0, "pA")]},
        video_stem="one",
        authority={("one", "A"): "pA"},
    )
    second = _evaluate(
        {0: [_object(0, "A")]},
        {},
        video_stem="two",
        authority={("two", "A"): "pA"},
    )
    aggregate = aggregate_tracking_standard_v2([first, second])

    assert aggregate.metrics.sequence_count == 2
    assert (aggregate.metrics.tp, aggregate.metrics.fn) == (1, 1)
    assert aggregate.metrics.idsw_standard == 0
    assert aggregate.metrics.identity_error_episode_count == 0
    assert aggregate.metrics.detection_recall == 0.5


def test_metric_dataframe_carries_contract_on_every_row() -> None:
    result = _evaluate(
        {0: [_object(0, "A")]},
        {0: [_object(0, "pA")]},
        authority={("synthetic", "A"): "pA"},
    )
    dataframe = metrics_to_dataframe_standard_v2([result])

    assert dataframe.loc[0, "evaluator_contract_id"] == EVALUATOR_CONTRACT_ID
    assert dataframe.loc[0, "hota_pct"] == 100.0
    assert dataframe.loc[0, "metric_config_sha256"]


def test_input_order_and_raw_objects_are_immutable() -> None:
    gt = {
        0: [
            _object(0, "B", (20.0, 0.0, 30.0, 10.0)),
            _object(0, "A"),
        ]
    }
    pred = {
        0: [
            _object(0, "pB", (20.0, 0.0, 30.0, 10.0)),
            _object(0, "pA"),
        ]
    }
    original_gt = deepcopy(gt)
    original_pred = deepcopy(pred)
    authority = {
        ("synthetic", "A"): "pA",
        ("synthetic", "B"): "pB",
    }
    ordered = _evaluate(gt, pred, authority=authority)
    reversed_result = _evaluate(
        {0: list(reversed(gt[0]))},
        {0: list(reversed(pred[0]))},
        authority=authority,
    )

    assert ordered.metrics == reversed_result.metrics
    assert gt == original_gt
    assert pred == original_pred


def test_duplicate_frame_identity_fails_closed() -> None:
    duplicate = {0: [_object(0, "A"), _object(0, "A")]}
    with pytest.raises(ValueError, match="Duplicate GT identity"):
        _evaluate(duplicate, {})


def test_noncanonical_detection_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="fixed at 0.5"):
        evaluate_tracking_standard_v2(
            {},
            {},
            video_stem="synthetic",
            detection_iou_threshold=0.6,
            evaluator_code_sha="a" * 40,
        )


def test_empty_identity_audit_tables_keep_their_versioned_columns() -> None:
    assert identity_authority_dataframe([]).columns.tolist() == [
        field.name for field in fields(IdentityAuthority)
    ]
    assert identity_episode_dataframe([]).columns.tolist() == [
        field.name for field in fields(IdentityErrorEpisode)
    ]
    assert pairwise_swap_dataframe([]).columns.tolist() == [
        field.name for field in fields(PairwiseIdentitySwapEvent)
    ]
    assert identity_ambiguity_dataframe([]).columns.tolist() == [
        *(field.name for field in fields(MatchedIdentityRow)),
        "ambiguity_reason",
    ]
