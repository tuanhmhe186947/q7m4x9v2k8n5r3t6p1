from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.evaluation.tracking.cvat_io import (
    TrackingObject,
    parse_cvat_video_xml,
)
from pig_behavior.evaluation.tracking.evaluator import evaluate_tracking
from pig_behavior.evaluation.tracking.matching import iou_xyxy, match_frame
from pig_behavior.evaluation.tracking.metrics import aggregate_metrics

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "tracking"
    / "build_rf_acc23_error_taxonomy.py"
)
SPEC = importlib.util.spec_from_file_location("metric_audit_taxonomy", SCRIPT)
assert SPEC and SPEC.loader
TAXONOMY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TAXONOMY
SPEC.loader.exec_module(TAXONOMY)

BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (20.0, 0.0, 30.0, 10.0)


def _object(frame: int, obj_id: str, bbox: tuple[float, ...]) -> TrackingObject:
    return TrackingObject(
        frame=frame,
        obj_id=obj_id,
        bbox=bbox,
        hidden=False,
        source_track_id=obj_id,
        label=obj_id,
    )


def _single_identity_metrics(
    pred_ids: list[str | None],
) -> tuple[object, int]:
    gt = {
        frame: [_object(frame, "A", BOX_A)]
        for frame in range(len(pred_ids))
    }
    pred = {
        frame: [_object(frame, pred_id, BOX_A)]
        for frame, pred_id in enumerate(pred_ids)
        if pred_id is not None
    }
    wrong_rows = sum(
        1
        for frame, pred_id in enumerate(pred_ids)
        if pred_id is not None and pred_id != gt[frame][0].obj_id
    )
    return evaluate_tracking(gt, pred, iou_threshold=0.5), wrong_rows


def _two_identity_metrics(
    states: list[str],
) -> tuple[object, int]:
    gt: dict[int, list[TrackingObject]] = {}
    pred: dict[int, list[TrackingObject]] = {}
    wrong_rows = 0
    for frame, state in enumerate(states):
        gt[frame] = [
            _object(frame, "A", BOX_A),
            _object(frame, "B", BOX_B),
        ]
        if state == "correct":
            pred[frame] = [
                _object(frame, "A", BOX_A),
                _object(frame, "B", BOX_B),
            ]
        else:
            pred[frame] = [
                _object(frame, "B", BOX_A),
                _object(frame, "A", BOX_B),
            ]
            wrong_rows += 2
    return evaluate_tracking(gt, pred, iou_threshold=0.5), wrong_rows


def test_identity_golden_a_perfect_continuity() -> None:
    metrics, wrong_rows = _single_identity_metrics(["A"] * 5)

    assert metrics.idsw == 0
    assert wrong_rows == 0


def test_identity_golden_b_one_frame_wrong_then_recovery() -> None:
    metrics, wrong_rows = _single_identity_metrics(["A", "B", "A"])

    assert metrics.idsw == 2
    assert wrong_rows == 1


def test_identity_golden_c_multi_frame_wrong_then_recovery() -> None:
    metrics, wrong_rows = _single_identity_metrics(
        ["A", "B", "B", "B", "A"]
    )

    assert metrics.idsw == 2
    assert wrong_rows == 3


def test_identity_golden_d_wrong_assignment_persists_to_end() -> None:
    metrics, wrong_rows = _single_identity_metrics(["A", *(["B"] * 9)])

    assert metrics.idsw == 1
    assert wrong_rows == 9


def test_identity_golden_e_two_animals_exchange_and_remain() -> None:
    metrics, wrong_rows = _two_identity_metrics(
        ["correct", *(["swapped"] * 5)]
    )

    assert metrics.idsw == 2
    assert wrong_rows == 10


def test_identity_golden_f_two_animals_exchange_and_recover() -> None:
    metrics, wrong_rows = _two_identity_metrics(
        ["correct", "swapped", "swapped", "swapped", "correct"]
    )

    assert metrics.idsw == 4
    assert wrong_rows == 6


def test_identity_golden_g_lost_then_different_id() -> None:
    metrics, wrong_rows = _single_identity_metrics(
        ["A", None, None, "B", "A"]
    )

    assert metrics.idsw == 2
    assert metrics.fragments == 1
    assert wrong_rows == 1


def test_identity_golden_h_fragmentation_without_wrong_identity() -> None:
    metrics, wrong_rows = _single_identity_metrics(["A", None, "A"])

    assert metrics.idsw == 0
    assert metrics.fragments == 1
    assert wrong_rows == 0


def test_identity_golden_i_unmatched_gap_then_same_id() -> None:
    metrics, wrong_rows = _single_identity_metrics(
        ["A", None, None, None, "A"]
    )

    assert metrics.idsw == 0
    assert metrics.fragments == 1
    assert wrong_rows == 0


def test_identity_golden_j_video_boundary_resets_switch_state() -> None:
    first, _ = _single_identity_metrics(["A"])
    second_gt = {0: [_object(0, "B", BOX_A)]}
    second_pred = {0: [_object(0, "B", BOX_A)]}
    second = evaluate_tracking(second_gt, second_pred, iou_threshold=0.5)

    aggregate = aggregate_metrics([first, second])

    assert first.idsw == 0
    assert second.idsw == 0
    assert aggregate.idsw == 0


def test_identity_golden_k_hidden_population_changes_explicitly(
    tmp_path: Path,
) -> None:
    xml = """\
<annotations>
  <track id="0" label="Pig_1">
    <box frame="0" outside="0" xtl="0" ytl="0" xbr="10" ybr="10">
      <attribute name="ID">ID_1</attribute>
      <attribute name="Hidden">No</attribute>
    </box>
    <box frame="1" outside="0" xtl="0" ytl="0" xbr="10" ybr="10">
      <attribute name="ID">ID_1</attribute>
      <attribute name="Hidden">Yes</attribute>
    </box>
    <box frame="2" outside="0" xtl="0" ytl="0" xbr="10" ybr="10">
      <attribute name="ID">ID_1</attribute>
      <attribute name="Hidden">No</attribute>
    </box>
  </track>
</annotations>
"""
    pred_xml = xml.replace(
        '<attribute name="ID">ID_1</attribute>\n'
        '      <attribute name="Hidden">Yes</attribute>',
        '<attribute name="ID">ID_2</attribute>\n'
        '      <attribute name="Hidden">Yes</attribute>',
        1,
    )
    gt_path = tmp_path / "hidden_gt.xml"
    pred_path = tmp_path / "hidden_pred.xml"
    gt_path.write_text(xml, encoding="utf-8")
    pred_path.write_text(pred_xml, encoding="utf-8")

    gt_visible = parse_cvat_video_xml(gt_path, include_hidden=False)
    pred_visible = parse_cvat_video_xml(pred_path, include_hidden=False)
    gt_including_hidden = parse_cvat_video_xml(gt_path, include_hidden=True)
    pred_including_hidden = parse_cvat_video_xml(
        pred_path,
        include_hidden=True,
    )
    visible_metrics = evaluate_tracking(gt_visible, pred_visible)
    hidden_metrics = evaluate_tracking(
        gt_including_hidden,
        pred_including_hidden,
    )

    assert sum(map(len, gt_visible.values())) == 2
    assert sum(map(len, gt_including_hidden.values())) == 3
    assert 1 not in gt_visible
    assert gt_including_hidden[1][0].hidden is True
    assert visible_metrics.idsw == 0
    assert hidden_metrics.idsw == 2


def test_identity_golden_l_unresolved_gt_is_not_authoritative() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "resolved",
                "gt_authority": "SUFFICIENT_FOR_EVENT_TAXONOMY",
            },
            {
                "event_id": "unresolved",
                "gt_authority": "UNRESOLVED_SOURCE_AUTHORITY",
            },
        ]
    )

    authoritative = events[
        events["gt_authority"] == "SUFFICIENT_FOR_EVENT_TAXONOMY"
    ]

    assert authoritative["event_id"].tolist() == ["resolved"]
    assert set(events["event_id"]) == {"resolved", "unresolved"}


def test_metric_reference_hota_is_single_threshold_not_standard_average() -> None:
    gt = {0: [_object(0, "A", (0.0, 0.0, 10.0, 10.0))]}
    pred = {0: [_object(0, "A", (2.5, 0.0, 12.5, 10.0))]}

    current = evaluate_tracking(gt, pred, iou_threshold=0.5)
    above_overlap = evaluate_tracking(gt, pred, iou_threshold=0.65)
    manually_expected_standard_hota = 12 / 19

    assert current.hota == 1.0
    assert above_overlap.hota == 0.0
    assert manually_expected_standard_hota == pytest.approx(0.6315789474)
    assert current.hota != pytest.approx(manually_expected_standard_hota)


def test_metric_reference_post_threshold_assignment_loses_valid_match() -> None:
    gt = [
        _object(0, "A", (0.0, 0.0, 3.0, 1.0)),
        _object(0, "B", (0.0, 0.0, 5.0, 1.0)),
    ]
    pred = [
        _object(0, "A", (0.0, 0.0, 3.0, 1.0)),
        _object(0, "B", (1.0, 0.0, 3.0, 1.0)),
    ]

    current_matches = match_frame(gt, pred, iou_threshold=0.5)
    manually_valid_gated_assignment = {(0, 1), (1, 0)}

    assert len(current_matches) == 1
    assert len(manually_valid_gated_assignment) == 2
    assert all(
        iou_xyxy(gt[gt_index].bbox, pred[pred_index].bbox) >= 0.5
        for gt_index, pred_index in manually_valid_gated_assignment
    )


def test_metric_reference_fp_fn_tp_conservation() -> None:
    gt = {
        0: [_object(0, "A", BOX_A), _object(0, "B", BOX_B)],
        1: [_object(1, "A", BOX_A), _object(1, "B", BOX_B)],
    }
    pred = {
        0: [_object(0, "A", BOX_A)],
        1: [
            _object(1, "A", BOX_A),
            _object(1, "B", BOX_B),
            _object(1, "C", (40.0, 0.0, 50.0, 10.0)),
        ],
    }

    metrics = evaluate_tracking(gt, pred, iou_threshold=0.5)

    assert metrics.matches + metrics.fn == metrics.gt_detections
    assert metrics.matches + metrics.fp == metrics.pred_detections


def test_metric_reference_aggregate_is_micro_not_unweighted_mean() -> None:
    large, _ = _single_identity_metrics(["A"] * 100)
    missed, _ = _single_identity_metrics([None])

    aggregate = aggregate_metrics([large, missed])

    assert aggregate.recall == pytest.approx(100 / 101)
    assert aggregate.recall != pytest.approx(
        (large.recall + missed.recall) / 2
    )


def test_metric_reference_idf1_respects_sequence_boundaries() -> None:
    first_gt = {0: [_object(0, "A", BOX_A)]}
    first_pred = {0: [_object(0, "X", BOX_A)]}
    second_gt = {0: [_object(0, "B", BOX_A)]}
    second_pred = {0: [_object(0, "X", BOX_A)]}
    first = evaluate_tracking(first_gt, first_pred, iou_threshold=0.5)
    second = evaluate_tracking(second_gt, second_pred, iou_threshold=0.5)

    aggregate = aggregate_metrics([first, second])

    assert first.idf1 == 1.0
    assert second.idf1 == 1.0
    assert aggregate.idf1 == 1.0


def test_metric_reference_empty_sequence_is_not_perfect() -> None:
    metrics = evaluate_tracking({}, {}, iou_threshold=0.5)

    assert metrics.hota == 0.0
    assert metrics.deta == 0.0
    assert metrics.assa == 0.0
    assert metrics.idf1 == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_metric_reference_input_order_is_deterministic() -> None:
    gt_ordered = {
        0: [_object(0, "A", BOX_A), _object(0, "B", BOX_B)]
    }
    pred_ordered = {
        0: [_object(0, "A", BOX_A), _object(0, "B", BOX_B)]
    }
    gt_reversed = {0: list(reversed(gt_ordered[0]))}
    pred_reversed = {0: list(reversed(pred_ordered[0]))}

    ordered = evaluate_tracking(
        gt_ordered,
        pred_ordered,
        iou_threshold=0.5,
    )
    reversed_result = evaluate_tracking(
        gt_reversed,
        pred_reversed,
        iou_threshold=0.5,
    )

    assert ordered.matches == reversed_result.matches
    assert ordered.idsw == reversed_result.idsw
    assert ordered.idf1 == reversed_result.idf1
    assert ordered.hota == reversed_result.hota


def test_metric_reference_episode_span_is_not_wrong_row_duration() -> None:
    rows = pd.DataFrame(
        [
            {
                "video_stem": "v",
                "frame": 10,
                "gt_id": "A",
                "pred_id": "B",
                "event": "id_mismatch",
            },
            {
                "video_stem": "v",
                "frame": 25,
                "gt_id": "A",
                "pred_id": "B",
                "event": "id_mismatch",
            },
        ]
    )

    events = TAXONOMY.group_error_events(rows)

    assert len(events) == 1
    assert events[0]["duration_frames"] == 16
    assert events[0]["wrong_id_matched_frames"] == 2


def test_metric_reference_episode_never_crosses_video_boundary() -> None:
    rows = pd.DataFrame(
        [
            {
                "video_stem": "first",
                "frame": 10,
                "gt_id": "A",
                "pred_id": "B",
                "event": "id_mismatch",
            },
            {
                "video_stem": "second",
                "frame": 10,
                "gt_id": "A",
                "pred_id": "B",
                "event": "id_mismatch",
            },
        ]
    )

    events = TAXONOMY.group_error_events(rows)

    assert len(events) == 2
    assert {event["video_key"] for event in events} == {"first", "second"}
