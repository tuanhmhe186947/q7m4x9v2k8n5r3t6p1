from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pig_behavior.evaluation.tracking.cvat_io import TrackingObject
from pig_behavior.evaluation.tracking.hota_standard_v2 import (
    combine_hota_sequences,
    evaluate_hota_sequence,
)
from pig_behavior.evaluation.tracking.identity_standard_v2 import (
    aggregate_identity_standard_v2,
    evaluate_identity_standard_v2,
)

EXPECTED_TRACK_EVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (20.0, 0.0, 30.0, 10.0)
TOLERANCE = 1e-12


@dataclass(frozen=True)
class ParityCase:
    name: str
    sequences: tuple[
        tuple[
            dict[int, list[TrackingObject]],
            dict[int, list[TrackingObject]],
        ],
        ...,
    ]


def _object(
    frame: int,
    obj_id: str,
    bbox: tuple[float, float, float, float] = BOX_A,
    *,
    hidden: bool = False,
) -> TrackingObject:
    return TrackingObject(
        frame=frame,
        obj_id=obj_id,
        bbox=bbox,
        hidden=hidden,
        source_track_id=obj_id,
        label=obj_id,
    )


def _single(
    pred_ids: list[str | None],
) -> tuple[dict[int, list[TrackingObject]], dict[int, list[TrackingObject]]]:
    gt = {frame: [_object(frame, "A")] for frame in range(len(pred_ids))}
    pred = {
        frame: [_object(frame, pred_id)]
        for frame, pred_id in enumerate(pred_ids)
        if pred_id is not None
    }
    return gt, pred


def _cases() -> tuple[ParityCase, ...]:
    perfect = _single(["X", "X", "X"])
    miss = _single(["X", None, "X"])
    false_positive = _single(["X"])
    false_positive[1][0].append(_object(0, "Y", BOX_B))
    identity_switch = _single(["X", "Y", "Y"])
    fragmentation = _single(["X", None, "X"])
    gap_new_identity = _single(["X", None, "Y"])
    crossing_gt: dict[int, list[TrackingObject]] = {}
    crossing_pred: dict[int, list[TrackingObject]] = {}
    for frame in range(4):
        crossing_gt[frame] = [
            _object(frame, "A", BOX_A),
            _object(frame, "B", BOX_B),
        ]
        pred_ids = ("X", "Y") if frame < 2 else ("Y", "X")
        crossing_pred[frame] = [
            _object(frame, pred_ids[0], BOX_A),
            _object(frame, pred_ids[1], BOX_B),
        ]
    all_hidden_gt = {0: [_object(0, "A", hidden=True)]}
    all_hidden_pred = {0: [_object(0, "X", hidden=True)]}
    second_video = _single(["Z", "Z"])
    return (
        ParityCase("perfect", (perfect,)),
        ParityCase("detection_miss", (miss,)),
        ParityCase("false_positive", (false_positive,)),
        ParityCase("identity_switch", (identity_switch,)),
        ParityCase("fragmentation", (fragmentation,)),
        ParityCase("crossing_identities", ((crossing_gt, crossing_pred),)),
        ParityCase("gap_reappearance", (gap_new_identity,)),
        ParityCase("empty_sequence", (({}, {}),)),
        ParityCase("all_hidden_excluded", ((all_hidden_gt, all_hidden_pred),)),
        ParityCase("multiple_videos", (perfect, second_video)),
    )


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _reference_data(
    gt_by_frame: dict[int, list[TrackingObject]],
    pred_by_frame: dict[int, list[TrackingObject]],
) -> dict[str, object]:
    frames = sorted(set(gt_by_frame).union(pred_by_frame))
    gt_names = sorted(
        {obj.obj_id for objects in gt_by_frame.values() for obj in objects}
    )
    pred_names = sorted(
        {obj.obj_id for objects in pred_by_frame.values() for obj in objects}
    )
    gt_lookup = {name: index for index, name in enumerate(gt_names)}
    pred_lookup = {name: index for index, name in enumerate(pred_names)}
    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    similarities: list[np.ndarray] = []
    for frame in frames:
        gt_objects = sorted(
            gt_by_frame.get(frame, []),
            key=lambda obj: obj.obj_id,
        )
        pred_objects = sorted(
            pred_by_frame.get(frame, []),
            key=lambda obj: obj.obj_id,
        )
        gt_ids.append(
            np.asarray([gt_lookup[obj.obj_id] for obj in gt_objects], dtype=int)
        )
        tracker_ids.append(
            np.asarray(
                [pred_lookup[obj.obj_id] for obj in pred_objects],
                dtype=int,
            )
        )
        similarities.append(
            np.asarray(
                [
                    [_iou(gt_obj.bbox, pred_obj.bbox) for pred_obj in pred_objects]
                    for gt_obj in gt_objects
                ],
                dtype=float,
            ).reshape(len(gt_objects), len(pred_objects))
        )
    return {
        "num_timesteps": len(frames),
        "num_gt_ids": len(gt_names),
        "num_tracker_ids": len(pred_names),
        "num_gt_dets": sum(len(ids) for ids in gt_ids),
        "num_tracker_dets": sum(len(ids) for ids in tracker_ids),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
    }


def _reference_metrics() -> tuple[object, object, object]:
    reference_text = os.environ.get("TRACK_EVAL_REFERENCE_DIR")
    if not reference_text:
        pytest.skip("TRACK_EVAL_REFERENCE_DIR is required for reference parity")
    reference_dir = Path(reference_text).resolve()
    expected_suffix = f"TrackEval_{EXPECTED_TRACK_EVAL_COMMIT}"
    if reference_dir.name != expected_suffix:
        pytest.fail(f"TrackEval reference is not the pinned commit: {reference_dir}")
    sys.path.insert(0, str(reference_dir))
    np.float = float  # type: ignore[attr-defined]
    np.int = int  # type: ignore[attr-defined]
    from trackeval.metrics.clear import CLEAR
    from trackeval.metrics.hota import HOTA
    from trackeval.metrics.identity import Identity

    return (
        HOTA(),
        Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
    )


def _filter_hidden(
    by_frame: dict[int, list[TrackingObject]],
) -> dict[int, list[TrackingObject]]:
    return {
        frame: [obj for obj in objects if not obj.hidden]
        for frame, objects in by_frame.items()
        if any(not obj.hidden for obj in objects)
    }


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case.name)
def test_standard_v2_matches_pinned_trackeval(case: ParityCase) -> None:
    hota_reference, identity_reference, clear_reference = _reference_metrics()
    production_hota = []
    production_identity = []
    reference_hota: dict[str, dict[str, object]] = {}
    reference_identity: dict[str, dict[str, object]] = {}
    reference_clear: dict[str, dict[str, object]] = {}
    for index, (gt_source, pred_source) in enumerate(case.sequences):
        gt = _filter_hidden(gt_source)
        pred = _filter_hidden(pred_source)
        sequence_key = f"{case.name}_{index}"
        data = _reference_data(gt, pred)
        production_hota.append(
            evaluate_hota_sequence(gt, pred, sequence_key=sequence_key)
        )
        production_identity.append(
            evaluate_identity_standard_v2(
                gt,
                pred,
                sequence_id=sequence_key,
            )
        )
        reference_hota[sequence_key] = hota_reference.eval_sequence(data)
        reference_identity[sequence_key] = identity_reference.eval_sequence(data)
        reference_clear[sequence_key] = clear_reference.eval_sequence(data)

    actual_hota = combine_hota_sequences(production_hota)
    actual_identity = aggregate_identity_standard_v2(production_identity)
    expected_hota = hota_reference.combine_sequences(reference_hota)
    expected_identity = identity_reference.combine_sequences(reference_identity)
    expected_clear = clear_reference.combine_sequences(reference_clear)

    np.testing.assert_allclose(
        actual_hota.hota,
        expected_hota["HOTA"],
        rtol=0.0,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        actual_hota.deta,
        expected_hota["DetA"],
        rtol=0.0,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        actual_hota.assa,
        expected_hota["AssA"],
        rtol=0.0,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        actual_hota.loca,
        expected_hota["LocA"],
        rtol=0.0,
        atol=TOLERANCE,
    )
    assert actual_hota.tp == tuple(int(value) for value in expected_hota["HOTA_TP"])
    assert actual_hota.fp == tuple(int(value) for value in expected_hota["HOTA_FP"])
    assert actual_hota.fn == tuple(int(value) for value in expected_hota["HOTA_FN"])
    assert actual_identity.idtp == int(expected_identity["IDTP"])
    assert actual_identity.idfp == int(expected_identity["IDFP"])
    assert actual_identity.idfn == int(expected_identity["IDFN"])
    assert actual_identity.idf1 == pytest.approx(
        float(expected_identity["IDF1"]),
        abs=TOLERANCE,
    )
    assert actual_identity.idp == pytest.approx(
        float(expected_identity["IDP"]),
        abs=TOLERANCE,
    )
    assert actual_identity.idr == pytest.approx(
        float(expected_identity["IDR"]),
        abs=TOLERANCE,
    )
    assert actual_identity.clear_tp == int(expected_clear["CLR_TP"])
    assert actual_identity.clear_fp == int(expected_clear["CLR_FP"])
    assert actual_identity.clear_fn == int(expected_clear["CLR_FN"])
    assert actual_identity.idsw_standard == int(expected_clear["IDSW"])
