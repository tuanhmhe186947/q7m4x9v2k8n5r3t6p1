from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import pig_behavior.tracking.refinement as tracking_refinement
from pig_behavior.evaluation.tracking.assets import TrackingPair
from pig_behavior.evaluation.tracking.pipeline import (
    runtime_telemetry_to_dataframe,
)
from pig_behavior.tracking import (
    Detection,
    TrackingConfig,
    TrackingRuntimeState,
    get_telemetry_summary,
    initialize_tracks,
    match_and_update_tracks,
    validate_config,
)
from pig_behavior.tracking.exporters.quality import build_quality_report
from pig_behavior.tracking.profiles.realtime import REALTIME_FAST_CONFIG
from pig_behavior.tracking.refinement import stabilize_realtime_motion_pairs
from pig_behavior.tracking.runner import _model_to_device
from pig_behavior.tracking.telemetry import (
    record_timing_sample,
    resolve_output_timing_contract,
)


def _detection(fixed_id: int, x1: float) -> Detection:
    return Detection(
        box=np.asarray([x1, 40.0, x1 + 24.0, 64.0], dtype=np.float32),
        score=0.95,
        raw_id=fixed_id,
        class_id=0,
        hist=np.full((16,), fixed_id / 8.0, dtype=np.float32),
    )


def _track_snapshot(tracks: dict[int, object]) -> list[tuple[object, ...]]:
    return [
        (
            fixed_id,
            tuple(float(value) for value in track.last_box),
            track.state,
            track.state_reason,
            track.missed,
            track.hits,
            tuple(sorted(track.raw_id_counts.items())),
        )
        for fixed_id, track in sorted(tracks.items())
    ]


def _shape(frame: int, fixed_id: int, center_x: float) -> dict[str, object]:
    return {
        "type": "rectangle",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "points": [
            center_x - 10.0,
            40.0,
            center_x + 10.0,
            60.0,
        ],
        "frame": frame,
        "label": f"Pig_{fixed_id}",
        "group": 0,
        "source": "auto",
        "attributes": [
            {"name": "ID", "value": f"ID_{fixed_id}"},
            {"name": "Behavior", "value": "lying"},
            {"name": "Hidden", "value": "No"},
        ],
        "score": 0.95,
        "elements": [],
        "_track_source": "detected",
        "_needs_review": False,
    }


def _shape_id(shape: dict[str, object]) -> str:
    attributes = shape["attributes"]
    assert isinstance(attributes, list)
    return str(
        next(
            attribute["value"]
            for attribute in attributes
            if attribute["name"] == "ID"
        )
    )


def _frame_ids_by_center(
    shapes: list[dict[str, object]],
    frame: int,
) -> dict[float, str]:
    result = {}
    for shape in shapes:
        if int(shape["frame"]) != frame:
            continue
        points = shape["points"]
        assert isinstance(points, list)
        center_x = (float(points[0]) + float(points[2])) / 2.0
        result[center_x] = _shape_id(shape)
    return result


def _fixed_lag_motion_pair_config(fixed_lag_frames: int) -> TrackingConfig:
    return TrackingConfig(
        mode="realtime",
        realtime_motion_pair_stabilizer=True,
        realtime_motion_pair_fixed_lag_frames=fixed_lag_frames,
        realtime_motion_pair_max_jump=0.50,
        realtime_motion_pair_min_gain=0.05,
        realtime_motion_pair_memory_frames=30,
        realtime_motion_pair_max_component_size=2,
        realtime_motion_pair_max_component_edges=2,
        realtime_motion_pair_dense_fallback_max_edges=0,
        realtime_motion_pair_dense_fallback_max_support_ratio=0.0,
        realtime_motion_pair_simple_min_gain=0.0,
    )


def test_motion_pair_shape_clone_preserves_input_independence() -> None:
    shape = _shape(0, 1, 10.0)
    shape["elements"] = [{"points": [1.0, 2.0]}]

    cloned = tracking_refinement._clone_motion_pair_shape(shape)

    assert cloned == shape
    cloned["points"][0] = -1.0
    cloned["attributes"][0]["value"] = "ID_8"
    cloned["elements"][0]["points"][0] = -2.0
    assert shape["points"][0] != -1.0
    assert shape["attributes"][0]["value"] == "ID_1"
    assert shape["elements"][0]["points"][0] == 1.0


@pytest.mark.parametrize(
    ("fixed_lag_frames", "memory_frames"),
    [(0, 30), (15, 10), (15, 15), (15, 20), (15, 30)],
)
def test_motion_pair_schema_clone_matches_deepcopy_output(
    monkeypatch: pytest.MonkeyPatch,
    fixed_lag_frames: int,
    memory_frames: int,
) -> None:
    cfg = _fixed_lag_motion_pair_config(fixed_lag_frames)
    cfg.realtime_motion_pair_memory_frames = memory_frames
    shapes = [
        _shape(0, 1, 10.0),
        _shape(0, 2, 110.0),
        _shape(0, 3, 210.0),
        _shape(1, 1, 110.0),
        _shape(1, 2, 10.0),
        _shape(1, 3, 210.0),
    ]
    for frame in range(2, 20):
        shapes.extend(
            [
                _shape(frame, 1, 10.0),
                _shape(frame, 2, 110.0),
                _shape(frame, 3, 210.0),
            ]
        )

    with monkeypatch.context() as reference_patch:
        reference_patch.setattr(
            tracking_refinement,
            "_clone_motion_pair_shape",
            deepcopy,
        )
        reference = stabilize_realtime_motion_pairs(
            shapes,
            300,
            100,
            cfg,
        )
    candidate = stabilize_realtime_motion_pairs(
        shapes,
        300,
        100,
        cfg,
    )

    assert candidate == reference


def test_runtime_telemetry_summary_has_stable_timing_and_delay() -> None:
    runtime = TrackingRuntimeState()
    runtime.telemetry.update(
        {
            "frames_processed": 2,
            "detection_frames": 2,
            "source_fps": 25.0,
            "declared_delay_frames": 2,
            "output_timing_contract": "fixed_delay",
            "postprocess_time_ms_total": 10.0,
            "peak_process_rss_bytes": 123,
        }
    )
    for value in (0.010, 0.030):
        record_timing_sample(runtime, "frame", value)
    for value in (0.006, 0.014):
        record_timing_sample(runtime, "detector", value)
    for value in (0.002, 0.004):
        record_timing_sample(runtime, "association", value)

    telemetry = get_telemetry_summary(runtime)

    assert telemetry["frame_time_ms_p50"] == pytest.approx(20.0)
    assert telemetry["frame_time_ms_p95"] == pytest.approx(29.0)
    assert telemetry["detector_time_ms_total"] == pytest.approx(20.0)
    assert telemetry["association_time_ms_mean"] == pytest.approx(3.0)
    assert telemetry["tracking_loop_effective_fps"] == pytest.approx(50.0)
    assert telemetry["effective_fps"] == pytest.approx(40.0)
    assert telemetry["realtime_factor"] == pytest.approx(1.6)
    assert telemetry["backlog_growth_frames_per_second"] == 0.0
    assert telemetry["frame_deadline_ms"] == pytest.approx(40.0)
    assert telemetry["frame_deadline_miss_count"] == 0
    assert telemetry["frame_deadline_miss_rate"] == 0.0
    assert telemetry["max_backlog_frames"] == 0
    assert telemetry["final_backlog_frames"] == 0
    assert telemetry["output_age_ms_p50"] == pytest.approx(100.0)
    assert telemetry["output_age_ms_p95"] == pytest.approx(109.0)
    assert telemetry["output_age_ms_max"] == pytest.approx(110.0)
    assert telemetry["output_age_ms_final"] == pytest.approx(110.0)
    assert telemetry["output_age_deadline_ms"] == pytest.approx(120.0)
    assert telemetry["output_age_deadline_miss_count"] == 0
    assert telemetry["output_age_deadline_miss_rate"] == 0.0
    assert telemetry["declared_delay_ms"] == pytest.approx(80.0)
    assert telemetry["peak_process_rss_bytes"] == 123


def test_runtime_telemetry_reports_stream_backlog_and_deadline_misses() -> None:
    runtime = TrackingRuntimeState()
    runtime.telemetry.update(
        {
            "frames_processed": 3,
            "source_fps": 30.0,
            "declared_delay_frames": 0,
            "output_timing_contract": "causal_framewise",
        }
    )
    for value in (0.020, 0.050, 0.050):
        record_timing_sample(runtime, "frame", value)

    telemetry = get_telemetry_summary(runtime)

    assert telemetry["effective_fps"] == pytest.approx(25.0)
    assert telemetry["realtime_factor"] == pytest.approx(5.0 / 6.0)
    assert telemetry["backlog_growth_frames_per_second"] == pytest.approx(5.0)
    assert telemetry["frame_deadline_miss_count"] == 2
    assert telemetry["frame_deadline_miss_rate"] == pytest.approx(2.0 / 3.0)
    assert telemetry["max_backlog_frames"] == 1
    assert telemetry["final_backlog_frames"] == 1
    assert telemetry["output_age_ms_max"] == pytest.approx(200.0 / 3.0)
    assert telemetry["output_age_ms_final"] == pytest.approx(200.0 / 3.0)
    assert telemetry["output_age_deadline_miss_count"] == 2
    assert telemetry["output_age_deadline_miss_rate"] == pytest.approx(2.0 / 3.0)


def test_post_video_runtime_telemetry_marks_output_age_unavailable() -> None:
    runtime = TrackingRuntimeState()
    runtime.telemetry.update(
        {
            "frames_processed": 1,
            "source_fps": 30.0,
            "declared_delay_frames": -1,
            "output_timing_contract": "post_video_global_graph",
        }
    )
    record_timing_sample(runtime, "frame", 0.020)

    telemetry = get_telemetry_summary(runtime)

    assert telemetry["frame_deadline_miss_count"] == 0
    assert telemetry["output_age_deadline_ms"] == -1.0
    assert telemetry["output_age_deadline_miss_count"] == -1
    assert telemetry["output_age_deadline_miss_rate"] == -1.0
    assert telemetry["output_age_ms_max"] == -1.0
    assert telemetry["output_age_ms_final"] == -1.0


def test_numeric_cuda_device_is_normalized_for_torch_model() -> None:
    assert _model_to_device("0") == "cuda:0"
    assert _model_to_device("cuda:1") == "cuda:1"
    assert _model_to_device("cpu") == "cpu"


def test_quality_report_keeps_runtime_telemetry_without_rule_counters() -> None:
    cfg = TrackingConfig(mode="realtime")
    telemetry = {
        "output_timing_contract": "causal_framewise",
        "frame_time_ms_p95": 12.5,
        "hard_merges_triggered": 0,
    }

    report = build_quality_report(
        [_shape(0, 1, 10.0)],
        cfg,
        Path("video.mp4"),
        source_fps=30.0,
        source_frame_count=1,
        telemetry=telemetry,
    )

    assert report["telemetry"]["output_timing_contract"] == "causal_framewise"
    assert report["telemetry"]["frame_time_ms_p95"] == 12.5
    assert "hard_merges_triggered" not in report["telemetry"]


def test_pipeline_collects_and_marks_per_video_runtime_telemetry(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "pig.mp4"
    gt_xml = tmp_path / "pig_gt.xml"
    pred_xml = tmp_path / "annotations_cvat_video_1_1.xml"
    telemetry_path = pred_xml.with_name("tracking_quality_report.json")
    video_path.write_bytes(b"video")
    gt_xml.write_text("<annotations />", encoding="utf-8")
    pred_xml.write_text("<annotations />", encoding="utf-8")
    telemetry_path.write_text(
        json.dumps(
            {
                "telemetry": {
                    "output_timing_contract": "causal_framewise",
                    "effective_fps": 31.25,
                }
            }
        ),
        encoding="utf-8",
    )
    pair = TrackingPair(
        video_stem="pig",
        video_path=video_path,
        gt_xml=gt_xml,
        pred_xml=pred_xml,
    )

    telemetry_df = runtime_telemetry_to_dataframe([pair])

    assert telemetry_df.loc[0, "telemetry_available"]
    assert telemetry_df.loc[0, "effective_fps"] == pytest.approx(31.25)
    assert (
        telemetry_df.loc[0, "output_timing_contract"]
        == "causal_framewise"
    )


def test_association_telemetry_does_not_change_track_predictions() -> None:
    cfg = TrackingConfig(
        mode="realtime",
        occlusion_aware_matching=False,
        smooth_boxes=False,
    )
    detections = [_detection(fixed_id, 35.0 * fixed_id) for fixed_id in range(1, 9)]
    base_tracks = initialize_tracks(detections, None, 400, 120, cfg)
    next_detections = [
        _detection(fixed_id, 35.0 * fixed_id + 1.0)
        for fixed_id in range(1, 9)
    ]
    frame = np.zeros((120, 400, 3), dtype=np.uint8)
    without_telemetry = deepcopy(base_tracks)
    with_telemetry = deepcopy(base_tracks)
    runtime = TrackingRuntimeState()

    match_and_update_tracks(
        without_telemetry,
        deepcopy(next_detections),
        frame,
        None,
        cfg,
        runtime=None,
        frame_index=1,
    )
    match_and_update_tracks(
        with_telemetry,
        deepcopy(next_detections),
        frame,
        None,
        cfg,
        runtime=runtime,
        frame_index=1,
    )

    assert _track_snapshot(with_telemetry) == _track_snapshot(without_telemetry)
    telemetry = get_telemetry_summary(runtime)
    assert telemetry["association_calls"] == 1
    assert telemetry["association_phase_visible_high_conf_calls"] == 1
    assert telemetry["association_assignments_accepted"] == 8
    assert runtime.association_debug_events == []


def test_hidden_claim_probe_is_prediction_invariant() -> None:
    base_cfg = TrackingConfig(
        mode="realtime",
        occlusion_aware_matching=False,
        smooth_boxes=False,
    )
    debug_cfg = TrackingConfig(
        mode="realtime",
        occlusion_aware_matching=False,
        smooth_boxes=False,
        association_debug=True,
    )
    detections = [_detection(fixed_id, 35.0 * fixed_id) for fixed_id in range(1, 9)]
    base_tracks = initialize_tracks(detections, None, 400, 120, base_cfg)
    assert base_tracks[1].ever_detected
    base_tracks[1].update_predicted(
        base_tracks[1].last_box.copy(),
        400,
        120,
        ambiguous=True,
        hold=True,
        cfg=base_cfg,
    )
    next_detections = [
        _detection(fixed_id, 35.0 * fixed_id + 1.0)
        for fixed_id in range(1, 9)
    ]
    frame = np.zeros((120, 400, 3), dtype=np.uint8)
    without_debug = deepcopy(base_tracks)
    with_debug = deepcopy(base_tracks)
    runtime = TrackingRuntimeState()

    match_and_update_tracks(
        without_debug,
        deepcopy(next_detections),
        frame,
        None,
        base_cfg,
        runtime=None,
        frame_index=1,
    )
    match_and_update_tracks(
        with_debug,
        deepcopy(next_detections),
        frame,
        None,
        debug_cfg,
        runtime=runtime,
        frame_index=1,
    )

    assert _track_snapshot(with_debug) == _track_snapshot(without_debug)
    claim_events = [
        event
        for event in runtime.association_debug_events
        if event.get("event") == "hidden_detection_claim_probe"
        and event.get("track_id") == 1
        and event.get("same_raw_id") is True
    ]
    assert len(claim_events) == 1
    assert claim_events[0]["phase"] == "pre_visible_hidden_claim"
    assert claim_events[0]["claim_rank"] == 1
    assert claim_events[0]["claim_plausible"] is True
    assert float(claim_events[0]["claim_iom"]) > 0.90
    assert float(claim_events[0]["claim_center_distance"]) < 0.01
    telemetry = get_telemetry_summary(runtime)
    assert telemetry["association_assignments_accepted"] == 8


def test_realtime_fast_declares_truthful_causality_contract() -> None:
    fast = TrackingConfig(mode="realtime", **REALTIME_FAST_CONFIG)
    global_graph = TrackingConfig(
        mode="realtime",
        realtime_motion_pair_stabilizer=True,
    )

    assert resolve_output_timing_contract(fast) == ("causal_framewise", 0)
    assert resolve_output_timing_contract(global_graph) == (
        "post_video_global_graph",
        -1,
    )


@pytest.mark.parametrize("fixed_lag_frames", [12, 15, 30])
def test_fixed_lag_quality_declares_exact_delay(fixed_lag_frames: int) -> None:
    cfg = _fixed_lag_motion_pair_config(fixed_lag_frames)

    assert resolve_output_timing_contract(cfg) == (
        "fixed_lag_framewise",
        fixed_lag_frames,
    )


def test_negative_motion_pair_fixed_lag_is_rejected() -> None:
    cfg = _fixed_lag_motion_pair_config(-1)

    with pytest.raises(
        ValueError,
        match="realtime_motion_pair_fixed_lag_frames must be >= 0",
    ):
        validate_config(cfg)


def test_realtime_fast_keeps_prefix_immutable_with_future_frames() -> None:
    cfg = TrackingConfig(mode="realtime", **REALTIME_FAST_CONFIG)
    prefix = [
        _shape(0, 1, 10.0),
        _shape(0, 2, 110.0),
        _shape(1, 1, 12.0),
        _shape(1, 2, 108.0),
    ]
    future = [
        _shape(2, 1, 110.0),
        _shape(2, 2, 10.0),
    ]

    prefix_output = stabilize_realtime_motion_pairs(
        deepcopy(prefix),
        200,
        100,
        cfg,
    )
    extended_output = stabilize_realtime_motion_pairs(
        deepcopy([*prefix, *future]),
        200,
        100,
        cfg,
    )

    for frame in (0, 1):
        assert _frame_ids_by_center(prefix_output, frame) == (
            _frame_ids_by_center(extended_output, frame)
        )
    assert resolve_output_timing_contract(cfg) == ("causal_framewise", 0)


def test_global_graph_profile_can_change_past_output_with_future_frames() -> None:
    cfg = TrackingConfig(
        mode="realtime",
        realtime_motion_pair_stabilizer=True,
        realtime_motion_pair_max_jump=0.50,
        realtime_motion_pair_min_gain=0.05,
        realtime_motion_pair_memory_frames=30,
        realtime_motion_pair_max_component_size=2,
        realtime_motion_pair_max_component_edges=2,
        realtime_motion_pair_dense_fallback_max_edges=0,
        realtime_motion_pair_dense_fallback_max_support_ratio=0.0,
        realtime_motion_pair_simple_min_gain=0.0,
    )
    prefix = [
        _shape(0, 1, 10.0),
        _shape(0, 2, 110.0),
        _shape(0, 3, 210.0),
        _shape(1, 1, 110.0),
        _shape(1, 2, 10.0),
        _shape(1, 3, 210.0),
    ]
    future = [
        _shape(2, 1, 10.0),
        _shape(2, 2, 210.0),
        _shape(2, 3, 110.0),
    ]

    prefix_output = stabilize_realtime_motion_pairs(
        deepcopy(prefix),
        300,
        100,
        cfg,
    )
    extended_output = stabilize_realtime_motion_pairs(
        deepcopy([*prefix, *future]),
        300,
        100,
        cfg,
    )

    assert _frame_ids_by_center(prefix_output, 1) == {
        10.0: "ID_1",
        110.0: "ID_2",
        210.0: "ID_3",
    }
    assert _frame_ids_by_center(extended_output, 1) == {
        10.0: "ID_2",
        110.0: "ID_1",
        210.0: "ID_3",
    }
    assert resolve_output_timing_contract(cfg) == (
        "post_video_global_graph",
        -1,
    )


@pytest.mark.parametrize(
    (
        "fixed_lag_frames",
        "min_gain",
        "max_component_edges",
        "memory_frames",
    ),
    [
        pytest.param(12, 0.05, 2, 30, id="lag-d12"),
        pytest.param(15, 0.05, 2, 30, id="lag-d15"),
        pytest.param(30, 0.05, 2, 30, id="lag-d30"),
        pytest.param(15, 0.02, 1, 30, id="rq2-s1"),
        pytest.param(15, 0.04, 1, 30, id="rq2-s2"),
        pytest.param(15, 0.06, 1, 30, id="rq2-s3"),
        pytest.param(15, 0.02, 1, 10, id="rq3-m10"),
        pytest.param(15, 0.02, 1, 15, id="rq3-m15"),
        pytest.param(15, 0.02, 1, 20, id="rq3-m20"),
    ],
)
def test_fixed_lag_motion_pair_output_is_prefix_invariant(
    fixed_lag_frames: int,
    min_gain: float,
    max_component_edges: int,
    memory_frames: int,
) -> None:
    cfg = _fixed_lag_motion_pair_config(fixed_lag_frames)
    cfg.realtime_motion_pair_min_gain = min_gain
    cfg.realtime_motion_pair_max_component_edges = max_component_edges
    cfg.realtime_motion_pair_memory_frames = memory_frames
    prefix = [
        _shape(0, 1, 10.0),
        _shape(0, 2, 110.0),
        _shape(0, 3, 210.0),
        _shape(1, 1, 110.0),
        _shape(1, 2, 10.0),
        _shape(1, 3, 210.0),
    ]
    for frame in range(2, fixed_lag_frames + 2):
        prefix.extend(
            [
                _shape(frame, 1, 10.0),
                _shape(frame, 2, 110.0),
                _shape(frame, 3, 210.0),
            ]
        )
    future_frame = fixed_lag_frames + 2
    future = [
        _shape(future_frame, 1, 10.0),
        _shape(future_frame, 2, 210.0),
        _shape(future_frame, 3, 110.0),
    ]

    prefix_output = stabilize_realtime_motion_pairs(
        deepcopy(prefix),
        300,
        100,
        cfg,
    )
    extended_output = stabilize_realtime_motion_pairs(
        deepcopy([*prefix, *future]),
        300,
        100,
        cfg,
    )

    expected_committed_ids = {
        10.0: "ID_1",
        110.0: "ID_2",
        210.0: "ID_3",
    }
    assert _frame_ids_by_center(prefix_output, 1) == expected_committed_ids
    assert _frame_ids_by_center(extended_output, 1) == expected_committed_ids
    assert [
        shape for shape in prefix_output if int(shape["frame"]) <= 1
    ] == [
        shape for shape in extended_output if int(shape["frame"]) <= 1
    ]
    assert resolve_output_timing_contract(cfg) == (
        "fixed_lag_framewise",
        fixed_lag_frames,
    )
