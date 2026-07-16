from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

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
)
from pig_behavior.tracking.exporters.quality import build_quality_report
from pig_behavior.tracking.profiles.realtime import (
    REALTIME_BALANCED_CONFIG,
    REALTIME_FAST_CONFIG,
    REALTIME_QUALITY_DELAYED_CONFIG,
)
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
    assert telemetry["declared_delay_ms"] == pytest.approx(80.0)
    assert telemetry["peak_process_rss_bytes"] == 123


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


def test_realtime_profiles_declare_truthful_causality_contracts() -> None:
    fast = TrackingConfig(mode="realtime", **REALTIME_FAST_CONFIG)
    balanced = TrackingConfig(mode="realtime", **REALTIME_BALANCED_CONFIG)
    quality = TrackingConfig(mode="realtime", **REALTIME_QUALITY_DELAYED_CONFIG)

    assert resolve_output_timing_contract(fast) == ("causal_framewise", 0)
    assert resolve_output_timing_contract(balanced) == ("causal_framewise", 0)
    assert resolve_output_timing_contract(quality) == (
        "post_video_global_graph",
        -1,
    )


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
