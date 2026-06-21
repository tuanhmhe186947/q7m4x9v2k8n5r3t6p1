from pathlib import Path

import numpy as np

from pig_behavior.evaluation.tracking_metrics import (
    TrackingObject,
    TrackingPair,
    continuity_gaps_for_pair,
    evaluate_pair,
    evaluate_tracking,
    identity_events_for_pair,
    identity_mapping_for_pair,
    parse_cvat_video_xml,
)
from pig_behavior.tracking import (
    Detection,
    FixedTrack,
    TrackingConfig,
    TrackingRuntimeState,
    apply_identity_swap_guard,
    center_distance_norm,
    get_telemetry_summary,
    initialize_tracks,
    match_and_update_tracks,
    track_detection_overlap_score,
)


def _shape(frame: int, fixed_id: int, points: list[float]) -> dict:
    return {
        "type": "rectangle",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": points,
        "group": 0,
        "source": "file",
        "frame": frame,
        "attributes": [
            {"value": f"ID_{fixed_id}", "name": "ID"},
            {"value": "lying", "name": "Behavior"},
            {"value": "No", "name": "Hidden"},
        ],
        "score": 0.9,
        "elements": [],
        "label": f"Pig_{fixed_id}",
        "_track_source": "detected",
        "_needs_review": False,
        "_ambiguous_occlusion": frame == 1,
        "_occlusion_hold": False,
    }


def test_identity_swap_guard_swaps_geometry_without_relabeling() -> None:
    cfg = TrackingConfig(identity_swap_min_gain=0.01)
    shapes = [
        _shape(0, 1, [0, 0, 20, 20]),
        _shape(0, 2, [100, 0, 120, 20]),
        _shape(1, 1, [102, 0, 122, 20]),
        _shape(1, 2, [2, 0, 22, 20]),
    ]

    guarded = apply_identity_swap_guard(shapes, width=200, height=100, cfg=cfg)
    frame_one = {
        shape["label"]: shape
        for shape in guarded
        if int(shape["frame"]) == 1
    }

    assert frame_one["Pig_1"]["points"] == [2, 0, 22, 20]
    assert frame_one["Pig_2"]["points"] == [102, 0, 122, 20]
    assert frame_one["Pig_1"]["attributes"][0]["value"] == "ID_1"
    assert frame_one["Pig_2"]["attributes"][0]["value"] == "ID_2"
    assert frame_one["Pig_1"]["_identity_swap_guard"] is True
    assert frame_one["Pig_2"]["_identity_swap_with"] == 1


def test_initialize_tracks_uses_spatial_anchor_ids() -> None:
    cfg = TrackingConfig(expected_pigs=2, initial_track_conf=0.5)
    hist = np.full((16 * 16 * 4,), 1.0 / (16 * 16 * 4), dtype=np.float32)
    detections = [
        Detection(
            box=np.array([150, 20, 190, 60], dtype=np.float32),
            score=0.95,
            raw_id=20,
            class_id=0,
            hist=hist,
        ),
        Detection(
            box=np.array([10, 20, 50, 60], dtype=np.float32),
            score=0.90,
            raw_id=10,
            class_id=0,
            hist=hist,
        ),
    ]

    tracks = initialize_tracks(
        detections,
        mask=None,
        width=200,
        height=100,
        cfg=cfg,
    )

    assert tracks[1].top_raw_id() == 10
    assert tracks[2].top_raw_id() == 20


def _binary_mask(
    height: int,
    width: int,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def _hist_at(index: int) -> np.ndarray:
    hist = np.zeros((16 * 16 * 4,), dtype=np.float32)
    hist[index] = 1.0
    return hist


def test_track_detection_overlap_prefers_mask_iou_when_available() -> None:
    cfg = TrackingConfig(use_mask_iou=True)
    hist = np.full((16 * 16 * 4,), 1.0 / (16 * 16 * 4), dtype=np.float32)
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 30, 30], dtype=np.float32),
        last_mask=_binary_mask(80, 80, (0, 0, 12, 30)),
        ever_detected=True,
    )
    det_same_shape = Detection(
        box=np.array([0, 0, 30, 30], dtype=np.float32),
        score=0.9,
        raw_id=1,
        class_id=0,
        hist=hist,
        mask=_binary_mask(80, 80, (0, 0, 12, 30)),
    )
    det_box_only_match = Detection(
        box=np.array([0, 0, 30, 30], dtype=np.float32),
        score=0.9,
        raw_id=2,
        class_id=0,
        hist=hist,
        mask=_binary_mask(80, 80, (18, 0, 30, 30)),
    )

    predicted = track.predicted_box(width=80, height=80)

    assert (
        track_detection_overlap_score(track, predicted, det_same_shape, cfg)
        > track_detection_overlap_score(track, predicted, det_box_only_match, cfg)
    )


def test_track_detection_overlap_falls_back_to_bbox_iou_without_masks() -> None:
    cfg = TrackingConfig(use_mask_iou=True)
    hist = np.full((16 * 16 * 4,), 1.0 / (16 * 16 * 4), dtype=np.float32)
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        ever_detected=True,
    )
    det = Detection(
        box=np.array([10, 0, 30, 20], dtype=np.float32),
        score=0.9,
        raw_id=1,
        class_id=0,
        hist=hist,
    )

    predicted = track.predicted_box(width=80, height=80)

    assert np.isclose(track_detection_overlap_score(track, predicted, det, cfg), 1 / 3)


def test_hidden_track_does_not_steal_active_track_detection() -> None:
    cfg = TrackingConfig(
        expected_pigs=2,
        motion_gate_confidence=0.5,
        low_conf_max_center_jump=0.08,
    )
    hist = np.full((16 * 16 * 4,), 1.0 / (16 * 16 * 4), dtype=np.float32)
    hidden_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([92, 0, 112, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        missed=6,
        last_score=0.1,
        last_source="predicted",
        ever_detected=True,
    )
    hidden_track.hist_bank.append(hist)
    active_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([100, 0, 120, 20], dtype=np.float32),
        reliable_box=np.array([100, 0, 120, 20], dtype=np.float32),
        missed=0,
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    active_track.hist_bank.append(hist)
    tracks = {1: hidden_track, 2: active_track}
    detections = [
        Detection(
            box=np.array([100, 0, 120, 20], dtype=np.float32),
            score=0.9,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
        Detection(
            box=np.array([0, 0, 20, 20], dtype=np.float32),
            score=0.3,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
    ]
    frame = np.zeros((40, 140, 3), dtype=np.uint8)

    match_and_update_tracks(tracks, detections, frame, prev_frame=None, cfg=cfg)

    assert center_distance_norm(tracks[1].last_box, detections[1].box, 140, 40) < 0.08
    assert np.allclose(tracks[2].last_box, detections[0].box)


def test_lost_track_reid_uses_remaining_detection_after_active_match() -> None:
    cfg = TrackingConfig(
        expected_pigs=2,
        motion_gate_confidence=0.5,
        low_conf_max_center_jump=0.05,
        use_mask_iou=False,
    )
    hist_lost = _hist_at(0)
    hist_active = _hist_at(1)
    lost_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        missed=8,
        last_score=0.1,
        last_source="predicted",
        ever_detected=True,
    )
    lost_track.hist_bank.append(hist_lost)
    active_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([90, 0, 110, 20], dtype=np.float32),
        reliable_box=np.array([90, 0, 110, 20], dtype=np.float32),
        missed=0,
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    active_track.hist_bank.append(hist_active)
    tracks = {1: lost_track, 2: active_track}
    detections = [
        Detection(
            box=np.array([90, 0, 110, 20], dtype=np.float32),
            score=0.92,
            raw_id=None,
            class_id=0,
            hist=hist_active,
        ),
        Detection(
            box=np.array([160, 0, 180, 20], dtype=np.float32),
            score=0.3,
            raw_id=None,
            class_id=0,
            hist=hist_lost,
        ),
    ]
    frame = np.zeros((50, 220, 3), dtype=np.uint8)

    match_and_update_tracks(tracks, detections, frame, prev_frame=None, cfg=cfg)

    lost_center_x = float(tracks[1].last_box[[0, 2]].mean())
    assert tracks[1].last_source == "detected"
    assert tracks[1].missed == 0
    assert lost_center_x > 120
    assert np.allclose(tracks[2].last_box, detections[0].box)


def test_iou_fallback_reconnects_unmatched_track_by_predicted_box() -> None:
    cfg = TrackingConfig(
        expected_pigs=1,
        USE_IOU_FALLBACK=True,
        lost_track_cost_threshold=0.01,
        use_mask_iou=False,
    )
    track_hist = _hist_at(0)
    det_hist = _hist_at(1)
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([10, 10, 30, 30], dtype=np.float32),
        reliable_box=np.array([10, 10, 30, 30], dtype=np.float32),
        missed=3,
        last_score=0.2,
        last_source="predicted",
        ever_detected=True,
    )
    track.hist_bank.append(track_hist)
    tracks = {1: track}
    detections = [
        Detection(
            box=np.array([10, 10, 30, 30], dtype=np.float32),
            score=0.9,
            raw_id=None,
            class_id=0,
            hist=det_hist,
        )
    ]
    frame = np.zeros((60, 60, 3), dtype=np.uint8)

    match_and_update_tracks(tracks, detections, frame, prev_frame=None, cfg=cfg)

    assert tracks[1].last_source == "detected"
    assert tracks[1].missed == 0
    assert np.allclose(tracks[1].last_box, detections[0].box)


def test_area_occlusion_freeze_consumes_shrunk_detection() -> None:
    cfg = TrackingConfig(
        expected_pigs=2,
        USE_AREA_OCCLUSION_FREEZE=True,
        area_occlusion_shrink_ratio=0.6,
        smooth_boxes=False,
    )
    hist = _hist_at(0)
    active_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 100, 100], dtype=np.float32),
        reliable_box=np.array([0, 0, 100, 100], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    active_track.hist_bank.append(hist)
    placeholder = FixedTrack(
        fixed_id=2,
        last_box=np.array([120, 0, 150, 30], dtype=np.float32),
        ever_detected=False,
    )
    tracks = {1: active_track, 2: placeholder}
    detections = [
        Detection(
            box=np.array([0, 0, 40, 40], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        )
    ]
    frame = np.zeros((160, 180, 3), dtype=np.uint8)

    match_and_update_tracks(tracks, detections, frame, prev_frame=None, cfg=cfg)

    assert tracks[1].is_area_occluded is True
    assert tracks[1].last_source == "occlusion_hold"
    assert np.allclose(tracks[1].last_box, active_track.reliable_box)
    assert tracks[2].ever_detected is False


def test_conditional_area_occlusion_freeze_ignores_isolated_shrink() -> None:
    cfg = TrackingConfig(
        expected_pigs=2,
        USE_AREA_OCCLUSION_FREEZE=False,
        USE_CONDITIONAL_AREA_OCCLUSION_FREEZE=True,
        area_occlusion_shrink_ratio=0.6,
        smooth_boxes=False,
    )
    hist = _hist_at(0)
    active_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 100, 100], dtype=np.float32),
        reliable_box=np.array([0, 0, 100, 100], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    active_track.hist_bank.append(hist)
    separate_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([130, 0, 160, 30], dtype=np.float32),
        reliable_box=np.array([130, 0, 160, 30], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    separate_track.hist_bank.append(hist)
    tracks = {1: active_track, 2: separate_track}
    detections = [
        Detection(
            box=np.array([0, 0, 40, 40], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
        Detection(
            box=np.array([130, 0, 160, 30], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
    ]
    frame = np.zeros((180, 200, 3), dtype=np.uint8)

    match_and_update_tracks(tracks, detections, frame, prev_frame=None, cfg=cfg)

    assert tracks[1].is_area_occluded is False
    assert tracks[1].last_source == "detected"
    assert np.allclose(tracks[1].last_box, detections[0].box)


def test_conditional_area_occlusion_freeze_triggers_in_heavy_overlap() -> None:
    cfg = TrackingConfig(
        expected_pigs=2,
        USE_AREA_OCCLUSION_FREEZE=False,
        USE_CONDITIONAL_AREA_OCCLUSION_FREEZE=True,
        area_occlusion_shrink_ratio=0.6,
        occlusion_detection_iom_threshold=0.2,
        smooth_boxes=False,
    )
    hist = _hist_at(0)
    front_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 100, 100], dtype=np.float32),
        reliable_box=np.array([0, 0, 100, 100], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    hidden_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([20, 20, 120, 120], dtype=np.float32),
        reliable_box=np.array([20, 20, 120, 120], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    front_track.hist_bank.append(hist)
    hidden_track.hist_bank.append(hist)
    tracks = {1: front_track, 2: hidden_track}
    detections = [
        Detection(
            box=np.array([0, 0, 40, 40], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        )
    ]
    frame = np.zeros((180, 200, 3), dtype=np.uint8)

    match_and_update_tracks(tracks, detections, frame, prev_frame=None, cfg=cfg)

    assert tracks[1].is_area_occluded is True
    assert tracks[1].last_source == "occlusion_hold"
    assert np.allclose(tracks[1].last_box, front_track.reliable_box)


def test_merged_box_split_ignores_oversized_detection_for_nearby_tracks() -> None:
    cfg = TrackingConfig(
        expected_pigs=2,
        USE_MERGED_BOX_SPLIT=True,
        merged_box_growth_ratio=1.5,
        merged_box_neighbor_distance=0.4,
    )
    hist = _hist_at(0)
    first_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    second_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([22, 0, 42, 20], dtype=np.float32),
        reliable_box=np.array([22, 0, 42, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    first_track.hist_bank.append(hist)
    second_track.hist_bank.append(hist)
    tracks = {1: first_track, 2: second_track}
    detections = [
        Detection(
            box=np.array([0, 0, 42, 20], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        )
    ]
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    match_and_update_tracks(tracks, detections, frame, prev_frame=None, cfg=cfg)

    assert tracks[1].last_merged_split is True
    assert tracks[2].last_merged_split is True
    assert tracks[1].last_source == "occlusion_hold"
    assert tracks[2].last_source == "occlusion_hold"
    assert np.allclose(tracks[1].last_box, np.array([0, 0, 20, 20], dtype=np.float32))
    assert np.allclose(tracks[2].last_box, np.array([22, 0, 42, 20], dtype=np.float32))


def test_merged_box_split_does_not_trigger_for_normal_close_detections() -> None:
    cfg = TrackingConfig(
        expected_pigs=2,
        USE_MERGED_BOX_SPLIT=True,
        merged_box_growth_ratio=1.5,
        merged_box_neighbor_distance=0.4,
        use_mask_iou=False,
        smooth_boxes=False,
    )
    hist = _hist_at(0)
    first_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    second_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([22, 0, 42, 20], dtype=np.float32),
        reliable_box=np.array([22, 0, 42, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    first_track.hist_bank.append(hist)
    second_track.hist_bank.append(hist)
    tracks = {1: first_track, 2: second_track}
    detections = [
        Detection(
            box=np.array([0, 0, 20, 20], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
        Detection(
            box=np.array([22, 0, 42, 20], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
    ]
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    match_and_update_tracks(tracks, detections, frame, prev_frame=None, cfg=cfg)

    assert tracks[1].last_merged_split is False
    assert tracks[2].last_merged_split is False
    assert tracks[1].last_source == "detected"
    assert tracks[2].last_source == "detected"
    assert np.allclose(tracks[1].last_box, detections[0].box)
    assert np.allclose(tracks[2].last_box, detections[1].box)


def test_merged_box_split_is_local_to_conflict_group() -> None:
    cfg = TrackingConfig(
        expected_pigs=3,
        USE_MERGED_BOX_SPLIT=True,
        merged_box_growth_ratio=1.5,
        merged_box_neighbor_distance=0.4,
        use_mask_iou=False,
        smooth_boxes=False,
    )
    hist = _hist_at(0)
    first_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    second_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([22, 0, 42, 20], dtype=np.float32),
        reliable_box=np.array([22, 0, 42, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    third_track = FixedTrack(
        fixed_id=3,
        last_box=np.array([80, 0, 100, 20], dtype=np.float32),
        reliable_box=np.array([80, 0, 100, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    first_track.hist_bank.append(hist)
    second_track.hist_bank.append(hist)
    third_track.hist_bank.append(hist)
    tracks = {1: first_track, 2: second_track, 3: third_track}
    detections = [
        Detection(
            box=np.array([0, 0, 42, 20], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
        Detection(
            box=np.array([80, 0, 100, 20], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
    ]
    frame = np.zeros((80, 140, 3), dtype=np.uint8)

    match_and_update_tracks(tracks, detections, frame, prev_frame=None, cfg=cfg)

    assert tracks[1].last_merged_split is True
    assert tracks[2].last_merged_split is True
    assert tracks[3].last_merged_split is False
    assert tracks[3].last_source == "detected"
    assert np.allclose(tracks[3].last_box, detections[1].box)


def test_merged_box_split_updates_runtime_telemetry() -> None:
    cfg = TrackingConfig(
        expected_pigs=2,
        USE_MERGED_BOX_SPLIT=True,
        merged_box_growth_ratio=1.5,
        merged_box_neighbor_distance=0.4,
    )
    hist = _hist_at(0)
    first_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    second_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([22, 0, 42, 20], dtype=np.float32),
        reliable_box=np.array([22, 0, 42, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    first_track.hist_bank.append(hist)
    second_track.hist_bank.append(hist)
    tracks = {1: first_track, 2: second_track}
    detections = [
        Detection(
            box=np.array([0, 0, 42, 20], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        )
    ]
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    runtime = TrackingRuntimeState()

    match_and_update_tracks(
        tracks,
        detections,
        frame,
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
    )

    telemetry = get_telemetry_summary(runtime)
    assert telemetry["hard_merges_triggered"] == 1
    assert telemetry["detections_intentionally_ignored"] == 1
    assert telemetry["recovery_frames_applied"] == 0

    recovered_detections = [
        Detection(
            box=np.array([0, 0, 20, 20], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
        Detection(
            box=np.array([22, 0, 42, 20], dtype=np.float32),
            score=0.95,
            raw_id=None,
            class_id=0,
            hist=hist,
        ),
    ]

    match_and_update_tracks(
        tracks,
        recovered_detections,
        frame,
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
    )

    telemetry = get_telemetry_summary(runtime)
    assert telemetry["hard_merges_triggered"] == 1
    assert telemetry["detections_intentionally_ignored"] == 1
    assert telemetry["recovery_frames_applied"] == 1
    assert tracks[1].last_ambiguous is True
    assert tracks[2].last_ambiguous is True


def _write_cvat_xml(path: Path, frame_one_swapped: bool) -> None:
    boxes = {
        1: {
            0: [0, 0, 20, 20],
            1: [100, 0, 120, 20] if frame_one_swapped else [0, 0, 20, 20],
        },
        2: {
            0: [100, 0, 120, 20],
            1: [0, 0, 20, 20] if frame_one_swapped else [100, 0, 120, 20],
        },
    }
    tracks = []
    for fixed_id, by_frame in boxes.items():
        box_xml = []
        for frame, box in by_frame.items():
            x1, y1, x2, y2 = box
            box_xml.append(
                f'<box frame="{frame}" xtl="{x1}" ytl="{y1}" '
                f'xbr="{x2}" ybr="{y2}" outside="0">'
                f'<attribute name="ID">ID_{fixed_id}</attribute>'
                '<attribute name="Hidden">No</attribute>'
                "</box>"
            )
        tracks.append(
            f'<track id="{fixed_id}" label="Pig_{fixed_id}">'
            + "".join(box_xml)
            + "</track>"
        )
    path.write_text(
        "<annotations>" + "".join(tracks) + "</annotations>",
        encoding="utf-8",
    )


def test_identity_events_flag_swapped_prediction_ids(tmp_path: Path) -> None:
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    video_path = tmp_path / "video.mp4"
    _write_cvat_xml(gt_xml, frame_one_swapped=False)
    _write_cvat_xml(pred_xml, frame_one_swapped=True)

    events = identity_events_for_pair(
        TrackingPair(
            video_stem="video",
            video_path=video_path,
            gt_xml=gt_xml,
            pred_xml=pred_xml,
        ),
    )

    pairs = {(event["gt_id"], event["pred_id"]) for event in events}
    assert ("ID_1", "ID_2") in pairs
    assert ("ID_2", "ID_1") in pairs
    assert all(event["frame"] == 1 for event in events)


def test_cvat_parser_prefers_id_attribute_over_pig_label(tmp_path: Path) -> None:
    xml_path = tmp_path / "annotation.xml"
    xml_path.write_text(
        """
        <annotations>
          <track id="1" label="Pig_1">
            <box frame="0" xtl="0" ytl="0" xbr="10" ybr="10" outside="0">
              <attribute name="ID">ID_7</attribute>
              <attribute name="Hidden">No</attribute>
            </box>
          </track>
        </annotations>
        """,
        encoding="utf-8",
    )

    parsed = parse_cvat_video_xml(xml_path)

    assert parsed[0][0].label == "Pig_1"
    assert parsed[0][0].obj_id == "ID_7"


def _write_stable_two_id_xml(
    path: Path,
    *,
    left_id: str,
    right_id: str,
) -> None:
    tracks = []
    specs = [
        ("1", "Pig_1", left_id, [0, 0, 20, 20]),
        ("2", "Pig_2", right_id, [100, 0, 120, 20]),
    ]
    for track_id, label, id_value, box in specs:
        x1, y1, x2, y2 = box
        box_xml = "".join(
            f'<box frame="{frame}" xtl="{x1}" ytl="{y1}" '
            f'xbr="{x2}" ybr="{y2}" outside="0">'
            f'<attribute name="ID">{id_value}</attribute>'
            '<attribute name="Hidden">No</attribute>'
            "</box>"
            for frame in range(2)
        )
        tracks.append(f'<track id="{track_id}" label="{label}">{box_xml}</track>')
    path.write_text(
        "<annotations>" + "".join(tracks) + "</annotations>",
        encoding="utf-8",
    )


def test_remapped_metrics_ignore_arbitrary_initial_id_numbering(
    tmp_path: Path,
) -> None:
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    video_path = tmp_path / "video.mp4"
    _write_stable_two_id_xml(gt_xml, left_id="ID_1", right_id="ID_2")
    _write_stable_two_id_xml(pred_xml, left_id="ID_2", right_id="ID_1")
    pair = TrackingPair(
        video_stem="video",
        video_path=video_path,
        gt_xml=gt_xml,
        pred_xml=pred_xml,
    )

    raw_events = identity_events_for_pair(pair)
    remapped_events = identity_events_for_pair(pair, remap_ids=True)
    mapping_rows = identity_mapping_for_pair(pair)
    metrics = evaluate_pair(pair)

    assert raw_events
    assert remapped_events == []
    assert {(row["pred_id"], row["mapped_gt_id"]) for row in mapping_rows} == {
        ("ID_1", "ID_2"),
        ("ID_2", "ID_1"),
    }
    assert metrics is not None
    assert metrics.tracklets == 2
    assert metrics.avg_tracklet_length_frames == 2.0
    assert metrics.remapped_tracklets == 2
    assert metrics.remapped_avg_tracklet_length_frames == 2.0
    assert metrics.remapped_gap_tolerant_tracklets == 2
    assert metrics.remapped_gap_tolerant_avg_tracklet_length_frames == 2.0
    assert metrics.remapped_idsw == 0
    assert metrics.remapped_idf1 == 1.0
    assert metrics.idmap_coverage == 1.0


def test_tracklets_split_when_gt_identity_is_absent() -> None:
    first = TrackingObject(frame=0, obj_id="ID_1", bbox=(0.0, 0.0, 10.0, 10.0))
    second = TrackingObject(frame=2, obj_id="ID_1", bbox=(0.0, 0.0, 10.0, 10.0))
    metrics = evaluate_tracking(
        gt_by_frame={0: [first], 1: [], 2: [second]},
        pred_by_frame={0: [first], 1: [], 2: [second]},
        video_stem="gap",
    )

    assert metrics.tracklets == 2
    assert metrics.avg_tracklet_length_frames == 1.0
    assert metrics.fragments == 1
    assert metrics.gap_tolerant_tracklets == 1
    assert metrics.gap_tolerant_avg_tracklet_length_frames == 2.0
    assert metrics.gap_tolerant_fragments == 0
    assert metrics.gap_tolerant_suppressed_fragments == 1

    strict_metrics = evaluate_tracking(
        gt_by_frame={0: [first], 1: [], 2: [second]},
        pred_by_frame={0: [first], 1: [], 2: [second]},
        video_stem="gap",
        gap_tolerance_frames=0,
    )

    assert strict_metrics.gap_tolerant_tracklets == 2
    assert strict_metrics.gap_tolerant_fragments == 1
    assert strict_metrics.gap_tolerant_suppressed_fragments == 0


def test_continuity_gaps_explain_tolerated_interruptions(tmp_path: Path) -> None:
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    video_path = tmp_path / "video.mp4"
    xml = """
    <annotations>
      <track id="1" label="Pig_1">
        <box frame="0" xtl="0" ytl="0" xbr="10" ybr="10" outside="0">
          <attribute name="ID">ID_1</attribute>
          <attribute name="Hidden">No</attribute>
        </box>
        <box frame="2" xtl="0" ytl="0" xbr="10" ybr="10" outside="0">
          <attribute name="ID">ID_1</attribute>
          <attribute name="Hidden">No</attribute>
        </box>
      </track>
    </annotations>
    """
    gt_xml.write_text(xml, encoding="utf-8")
    pred_xml.write_text(xml, encoding="utf-8")
    pair = TrackingPair(
        video_stem="video",
        video_path=video_path,
        gt_xml=gt_xml,
        pred_xml=pred_xml,
    )

    gaps = continuity_gaps_for_pair(pair, gap_tolerance_frames=1)

    assert len(gaps) == 1
    assert gaps[0]["gap_frames"] == 1
    assert gaps[0]["tolerated"] is True
    assert gaps[0]["event"] == "tolerated_gap"
