from pathlib import Path

import numpy as np

from pig_behavior.evaluation.tracking.api import (
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
    OcclusionContext,
    TrackingConfig,
    TrackingRuntimeState,
    apply_identity_swap_guard,
    center_distance_norm,
    get_telemetry_summary,
    initialize_tracks,
    match_and_update_tracks,
    repair_episode_pair_swaps,
    repair_hidden_suffix_id_swaps,
    repair_local_pair_swaps,
    repair_long_pair_swaps,
    repair_suffix_pair_swaps,
    shape_fixed_id,
    shape_hidden_value,
    suppress_overlapped_small_low_confidence_boxes,
    track_detection_overlap_score,
)
from pig_behavior.tracking.association import (
    apply_causal_hidden_detection_reservation,
    frame_in_reentry_ambiguous_hold_window,
    hidden_owner_conflict_should_freeze_identity,
    occlusion_reid_bad_match_should_hold,
    raw_owner_conflict_is_ambiguous,
    realtime_visible_close_competitor_should_prefer,
    reentry_ambiguous_assignment_should_hold,
    reentry_assignment_cost_allows_hold,
    reentry_raw_evidence_allows_hold,
    reentry_unowned_raw_mismatch_episode_should_reject,
    reentry_unowned_raw_mismatch_should_reject,
    reid_unowned_competing_candidate_should_hold,
    seed_reentry_unowned_raw_quarantine,
    video_in_reentry_ambiguous_hold_scope,
)
from pig_behavior.tracking.config import validate_config
from pig_behavior.tracking.refinement import (
    nearby_anchor_indices,
    refine_far_camera_hidden_geometry,
    refine_near_wall_hidden_geometry,
    stabilize_overlap_hidden_islands,
)
from pig_behavior.tracking.tracks import shape_for_track


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


def _set_hidden(shape: dict, hidden: bool) -> dict:
    for attribute in shape["attributes"]:
        if attribute["name"] == "Hidden":
            attribute["value"] = "Yes" if hidden else "No"
    shape["occluded"] = hidden
    return shape


def _config_with_existing_video(
    tmp_path: Path,
    **kwargs: object,
) -> TrackingConfig:
    video_path = tmp_path / "tracking_fixture.mp4"
    weights_path = tmp_path / "tracking_fixture.pt"
    mask_path = tmp_path / "tracking_fixture.png"
    video_path.write_bytes(b"fixture")
    weights_path.write_bytes(b"fixture")
    mask_path.write_bytes(b"fixture")
    return TrackingConfig(
        video_path=video_path,
        weights_path=weights_path,
        mask_path=mask_path,
        **kwargs,
    )


def test_nearby_anchor_indices_default_remains_symmetric() -> None:
    shapes = [{"frame": frame} for frame in (0, 20, 40)]
    cfg = TrackingConfig(refine_max_gap_frames=30)

    assert nearby_anchor_indices(shapes, [0, 2], 1, cfg) == (0, 2)


def test_nearby_anchor_indices_supports_asymmetric_gap_limits() -> None:
    shapes = [{"frame": frame} for frame in (0, 20, 40)]
    cfg = TrackingConfig(
        refine_max_gap_frames=30,
        refine_max_previous_gap_frames=15,
    )

    assert nearby_anchor_indices(shapes, [0, 2], 1, cfg) == (0, 2)
    assert nearby_anchor_indices(shapes[:2], [0], 1, cfg) == (None, None)
    assert nearby_anchor_indices(shapes[1:], [1], 0, cfg) == (None, 1)


def test_negative_refine_max_previous_gap_is_rejected() -> None:
    cfg = TrackingConfig(refine_max_previous_gap_frames=-1)

    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "refine_max_previous_gap_frames" in str(exc)
    else:
        raise AssertionError("negative previous refinement gap should fail")


def test_realtime_close_competitor_can_be_limited_to_far_right() -> None:
    cfg = TrackingConfig(
        mode="realtime",
        occlusion_aware_matching=False,
        realtime_visible_close_competitor_guard=True,
        realtime_visible_close_competitor_margin=0.08,
        realtime_visible_close_competitor_max_cost=0.40,
        realtime_visible_close_competitor_min_center_x_ratio=0.67,
    )
    selected_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([850, 400, 1000, 550], dtype=np.float32),
        hits=20,
        ever_detected=True,
        last_source="detected",
        state="VISIBLE",
    )
    competitor_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([900, 400, 1050, 550], dtype=np.float32),
        hits=20,
        ever_detected=True,
        last_source="detected",
        state="VISIBLE",
    )
    hist = np.zeros((16 * 16 * 4,), dtype=np.float32)
    far_detection = Detection(
        box=np.array([895.783, 465.226, 1074.067, 567.833]),
        score=0.90,
        raw_id=4,
        class_id=0,
        hist=hist,
    )
    left_detection = Detection(
        box=np.array([270.109, 156.079, 528.822, 411.506]),
        score=0.90,
        raw_id=6,
        class_id=0,
        hist=hist,
    )

    common = {
        "selected_track": selected_track,
        "competitor_track": competitor_track,
        "selected_cost": 0.21,
        "competitor_cost": 0.28,
        "competitor_selected_cost": None,
        "width": 1280,
        "cfg": cfg,
        "phase_name": "visible_high_conf",
    }
    assert realtime_visible_close_competitor_should_prefer(
        det=far_detection,
        **common,
    )
    assert not realtime_visible_close_competitor_should_prefer(
        det=left_detection,
        **common,
    )

    cfg.realtime_visible_close_competitor_min_center_x_ratio = 0.0
    assert realtime_visible_close_competitor_should_prefer(
        det=left_detection,
        **common,
    )


def test_realtime_close_competitor_center_ratio_is_validated() -> None:
    for value in (-0.01, 1.01):
        cfg = TrackingConfig(
            realtime_visible_close_competitor_min_center_x_ratio=value,
        )

        try:
            validate_config(cfg)
        except ValueError as exc:
            assert "min_center_x_ratio" in str(exc)
        else:
            raise AssertionError("invalid center-x ratio should fail")


def test_near_wall_hidden_geometry_requires_mask_config() -> None:
    cfg = TrackingConfig(
        near_wall_hidden_geometry_refine=True,
        use_mask=False,
        mask_path=None,
    )

    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "near_wall_hidden_geometry_refine requires" in str(exc)
    else:
        raise AssertionError("near-wall geometry without a mask should fail")


def test_near_wall_hidden_geometry_refines_only_bbox_payload() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    shapes = [
        _shape(0, 1, [10.0, 40.0, 30.0, 60.0]),
        _set_hidden(_shape(1, 1, [8.0, 40.0, 34.0, 60.0]), True),
        _shape(2, 1, [10.0, 40.0, 30.0, 60.0]),
    ]
    hidden_before = {
        key: value
        for key, value in shapes[1].items()
        if key != "points" and not key.startswith("_")
    }
    cfg = TrackingConfig(
        near_wall_hidden_geometry_refine=True,
        near_wall_hidden_geometry_original_weight=0.50,
    )

    refined = refine_near_wall_hidden_geometry(
        shapes,
        width=100,
        height=100,
        mask=mask,
        cfg=cfg,
    )

    assert refined[0]["points"] == shapes[0]["points"]
    assert refined[1]["points"] == [9.0, 40.0, 32.0, 60.0]
    assert refined[2]["points"] == shapes[2]["points"]
    assert refined[1]["_near_wall_hidden_geometry_refined"] is True
    assert {
        key: value
        for key, value in refined[1].items()
        if key != "points" and not key.startswith("_")
    } == hidden_before
    assert shapes[1]["points"] == [8.0, 40.0, 34.0, 60.0]


def test_near_wall_hidden_geometry_uses_final_id_not_label_slot() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    hidden = _set_hidden(
        _shape(1, 1, [8.0, 40.0, 34.0, 60.0]),
        True,
    )
    hidden["attributes"][0]["value"] = "ID_2"
    shapes = [
        _shape(0, 2, [10.0, 40.0, 30.0, 60.0]),
        hidden,
        _shape(2, 2, [10.0, 40.0, 30.0, 60.0]),
    ]
    cfg = TrackingConfig(near_wall_hidden_geometry_refine=True)

    refined = refine_near_wall_hidden_geometry(
        shapes,
        width=100,
        height=100,
        mask=mask,
        cfg=cfg,
    )

    assert refined[1]["points"] == [9.0, 40.0, 32.0, 60.0]
    assert refined[1]["label"] == "Pig_1"
    assert refined[1]["attributes"][0]["value"] == "ID_2"


def test_near_wall_hidden_geometry_ignores_far_box() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    shapes = [
        _shape(0, 1, [40.0, 40.0, 60.0, 60.0]),
        _set_hidden(_shape(1, 1, [38.0, 40.0, 64.0, 60.0]), True),
        _shape(2, 1, [40.0, 40.0, 60.0, 60.0]),
    ]
    cfg = TrackingConfig(near_wall_hidden_geometry_refine=True)

    refined = refine_near_wall_hidden_geometry(
        shapes,
        width=100,
        height=100,
        mask=mask,
        cfg=cfg,
    )

    assert refined[1]["points"] == shapes[1]["points"]
    assert "_near_wall_hidden_geometry_refined" not in refined[1]


def test_far_camera_hidden_geometry_rejects_invalid_future_gap() -> None:
    cfg = TrackingConfig(
        far_camera_hidden_geometry_refine=True,
        far_camera_hidden_geometry_max_future_gap_frames=0,
    )

    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "far_camera_hidden_geometry_max_future_gap_frames" in str(exc)
    else:
        raise AssertionError("zero far-camera future gap should fail")


def test_far_camera_hidden_geometry_refines_only_bbox_payload() -> None:
    hidden = _set_hidden(
        _shape(1, 7, [68.0, 20.0, 98.0, 80.0]),
        True,
    )
    shapes = [
        hidden,
        _shape(1, 5, [68.0, 30.0, 98.0, 80.0]),
        _shape(3, 7, [72.0, 24.0, 96.0, 60.0]),
    ]
    hidden_before = {
        key: value
        for key, value in hidden.items()
        if key != "points" and not key.startswith("_")
    }
    cfg = TrackingConfig(far_camera_hidden_geometry_refine=True)

    refined = refine_far_camera_hidden_geometry(
        shapes,
        width=100,
        height=100,
        cfg=cfg,
    )

    assert refined[0]["points"] == [71.6, 23.6, 96.2, 62.0]
    assert refined[1]["points"] == shapes[1]["points"]
    assert refined[2]["points"] == shapes[2]["points"]
    assert refined[0]["_far_camera_hidden_geometry_refined"] is True
    assert refined[0]["_far_camera_hidden_geometry_anchor_frame"] == 3
    assert refined[0]["_far_camera_hidden_geometry_overlap_identity"] == "ID_5"
    assert {
        key: value
        for key, value in refined[0].items()
        if key != "points" and not key.startswith("_")
    } == hidden_before
    assert shapes[0]["points"] == [68.0, 20.0, 98.0, 80.0]


def test_far_camera_hidden_geometry_requires_visible_identity_conflict() -> None:
    shapes = [
        _set_hidden(
            _shape(1, 7, [68.0, 20.0, 98.0, 80.0]),
            True,
        ),
        _set_hidden(
            _shape(1, 5, [68.0, 30.0, 98.0, 80.0]),
            True,
        ),
        _shape(3, 7, [72.0, 24.0, 96.0, 60.0]),
    ]
    cfg = TrackingConfig(far_camera_hidden_geometry_refine=True)

    refined = refine_far_camera_hidden_geometry(
        shapes,
        width=100,
        height=100,
        cfg=cfg,
    )

    assert refined[0]["points"] == shapes[0]["points"]
    assert "_far_camera_hidden_geometry_refined" not in refined[0]


def test_far_camera_hidden_geometry_ignores_near_camera_box() -> None:
    shapes = [
        _set_hidden(
            _shape(1, 7, [8.0, 20.0, 38.0, 80.0]),
            True,
        ),
        _shape(1, 5, [8.0, 30.0, 38.0, 80.0]),
        _shape(3, 7, [12.0, 24.0, 36.0, 60.0]),
    ]
    cfg = TrackingConfig(far_camera_hidden_geometry_refine=True)

    refined = refine_far_camera_hidden_geometry(
        shapes,
        width=100,
        height=100,
        cfg=cfg,
    )

    assert refined[0]["points"] == shapes[0]["points"]
    assert "_far_camera_hidden_geometry_refined" not in refined[0]


def test_overlap_hidden_stabilization_restores_hidden_owner() -> None:
    cfg = TrackingConfig()
    id1_box = [100.0, 100.0, 260.0, 220.0]
    id8_box = [108.0, 102.0, 268.0, 222.0]
    shapes = [
        _shape(frame, 1, id1_box)
        for frame in range(5)
    ] + [
        _set_hidden(_shape(frame, 8, id8_box), frame != 2)
        for frame in range(5)
    ]
    _set_hidden(shapes[2], True)

    stabilized = stabilize_overlap_hidden_islands(shapes, cfg)
    frame_2 = {
        shape["label"]: shape
        for shape in stabilized
        if int(shape["frame"]) == 2
    }

    assert frame_2["Pig_1"]["attributes"][2]["value"] == "No"
    assert frame_2["Pig_8"]["attributes"][2]["value"] == "Yes"
    assert frame_2["Pig_8"]["occluded"] is True


def test_realtime_keeps_explicit_causal_box_smoothing(
    tmp_path: Path,
) -> None:
    cfg = _config_with_existing_video(
        tmp_path,
        mode="realtime",
        enable_offline_smoothing=False,
        smooth_boxes=True,
        refine_boxes=False,
        overrides={
            "enable_offline_smoothing",
            "smooth_boxes",
            "refine_boxes",
        },
    )

    validate_config(cfg)

    assert cfg.enable_offline_smoothing is False
    assert cfg.smooth_boxes is True
    assert cfg.refine_boxes is False


def test_causal_box_smoothing_keeps_prefix_immutable() -> None:
    cfg = TrackingConfig(
        mode="realtime",
        enable_offline_smoothing=False,
        smooth_boxes=True,
        refine_boxes=False,
    )
    prefix = [
        [10.0, 20.0, 110.0, 100.0],
        [18.0, 20.0, 140.0, 100.0],
        [26.0, 20.0, 120.0, 100.0],
    ]
    future = [
        [180.0, 20.0, 300.0, 100.0],
        [40.0, 20.0, 130.0, 100.0],
    ]

    def run_sequence(boxes: list[list[float]]) -> list[np.ndarray]:
        track = FixedTrack(
            fixed_id=1,
            last_box=np.asarray(boxes[0], dtype=np.float32),
            ever_detected=True,
            hits=1,
        )
        outputs = [track.last_box.copy()]
        for box in boxes[1:]:
            detection = Detection(
                box=np.asarray(box, dtype=np.float32),
                score=0.90,
                raw_id=1,
                class_id=0,
                hist=np.ones(4, dtype=np.float32),
            )
            track.update_detected(detection, 400, 200, cfg)
            outputs.append(track.last_box.copy())
        return outputs

    prefix_outputs = run_sequence(prefix)
    extended_outputs = run_sequence([*prefix, *future])

    np.testing.assert_allclose(
        np.stack(prefix_outputs),
        np.stack(extended_outputs[: len(prefix_outputs)]),
    )
    assert not np.allclose(prefix_outputs[1], np.asarray(prefix[1]))


def test_hybrid_bytetrack_uses_hybrid_defaults_without_forced_postprocessing(
    tmp_path: Path,
) -> None:
    cfg = _config_with_existing_video(
        tmp_path,
        mode="hybrid_bytetrack",
        detect_every_n_frames=3,
    )

    validate_config(cfg)

    assert cfg.det_conf == 0.25
    assert cfg.track_high_conf == 0.50
    assert cfg.review_conf == 0.75
    assert cfg.nms_iou == 0.80
    assert cfg.track_match_iou == 0.80
    assert cfg.dup_iou_threshold == 0.80
    assert cfg.initial_track_conf == 0.50
    assert cfg.motion_gate_confidence == 0.50
    assert cfg.USE_IOU_FALLBACK is False
    assert cfg.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE is False
    assert cfg.max_missing_frames == 90
    assert cfg.detect_every_n_frames == 1
    assert cfg.enable_offline_smoothing is False


def test_hybrid_bytetrack_keeps_rule_flag_defaults(tmp_path: Path) -> None:
    cfg = _config_with_existing_video(
        tmp_path,
        mode="hybrid_bytetrack",
    )
    validate_config(cfg)

    assert cfg.USE_IOU_FALLBACK is False
    assert cfg.USE_AREA_OCCLUSION_FREEZE is False
    assert cfg.USE_CONDITIONAL_AREA_OCCLUSION_FREEZE is False
    assert cfg.enable_offline_smoothing is False


def test_hybrid_bytetrack_keeps_explicit_threshold_overrides(
    tmp_path: Path,
) -> None:
    cfg = _config_with_existing_video(
        tmp_path,
        mode="hybrid_bytetrack",
        det_conf=0.30,
        nms_iou=0.70,
        track_match_iou=0.65,
        overrides={"det_conf", "nms_iou", "track_match_iou"},
    )

    validate_config(cfg)

    assert cfg.det_conf == 0.30
    assert cfg.nms_iou == 0.70
    assert cfg.track_match_iou == 0.65
    assert cfg.track_high_conf == 0.50


def test_hybrid_bytetrack_keeps_explicit_legacy_iou_alias(
    tmp_path: Path,
) -> None:
    cfg = _config_with_existing_video(
        tmp_path,
        mode="hybrid_bytetrack",
        iou=0.72,
        overrides={"iou"},
    )

    validate_config(cfg)

    assert cfg.nms_iou == 0.72
    assert cfg.iou == 0.72


def test_legacy_bytetrack_mode_is_rejected() -> None:
    cfg = TrackingConfig(mode="bytetrack")

    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "hybrid_bytetrack" in str(exc)
    else:
        raise AssertionError("legacy mode=bytetrack should be rejected")


def test_missing_prediction_remains_evaluable_until_track_is_lost() -> None:
    cfg = TrackingConfig(mode="realtime", max_missing_frames=30)
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.8,
        last_source="predicted",
        ever_detected=True,
        state="MISSING",
    )

    missing_shape = shape_for_track(track, frame_index=1, cfg=cfg)
    assert missing_shape["outside"] is False
    assert missing_shape["occluded"] is False

    track.missed = 31
    track.state = "LOST"
    lost_shape = shape_for_track(track, frame_index=2, cfg=cfg)
    assert lost_shape["outside"] is True


def test_uninitialized_placeholder_is_outside_evaluation() -> None:
    cfg = TrackingConfig(mode="realtime")
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
    )

    shape = shape_for_track(track, frame_index=0, cfg=cfg)
    assert shape["outside"] is True


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


def test_identity_swap_guard_mixed_hold_veto_is_opt_in() -> None:
    shapes = [
        _shape(0, 1, [0, 0, 20, 20]),
        _shape(0, 2, [100, 0, 120, 20]),
        _shape(1, 1, [102, 0, 122, 20]),
        _shape(1, 2, [2, 0, 22, 20]),
    ]
    shapes[-1]["_track_source"] = "predicted"
    shapes[-1]["_occlusion_hold"] = True

    default_guarded = apply_identity_swap_guard(
        shapes,
        width=200,
        height=100,
        cfg=TrackingConfig(identity_swap_min_gain=0.01),
    )
    explicit_default_guarded = apply_identity_swap_guard(
        shapes,
        width=200,
        height=100,
        cfg=TrackingConfig(
            identity_swap_min_gain=0.01,
            identity_swap_guard_skip_mixed_occlusion_hold=False,
        ),
    )
    vetoed = apply_identity_swap_guard(
        shapes,
        width=200,
        height=100,
        cfg=TrackingConfig(
            identity_swap_min_gain=0.01,
            identity_swap_guard_skip_mixed_occlusion_hold=True,
        ),
    )

    assert default_guarded == explicit_default_guarded
    assert default_guarded[2]["points"] == [2, 0, 22, 20]
    assert vetoed[2]["points"] == [102, 0, 122, 20]
    assert vetoed[3]["points"] == [2, 0, 22, 20]
    assert "_identity_swap_guard" not in vetoed[2]
    assert "_identity_swap_guard" not in vetoed[3]


def test_identity_swap_guard_far_only_keeps_wall_mixed_hold_behavior() -> None:
    shapes = [
        _shape(0, 1, [100, 0, 120, 20]),
        _shape(0, 2, [160, 0, 180, 20]),
        _shape(1, 1, [162, 0, 182, 20]),
        _shape(1, 2, [102, 0, 122, 20]),
    ]
    shapes[-1]["_track_source"] = "predicted"
    shapes[-1]["_occlusion_hold"] = True

    far_only = apply_identity_swap_guard(
        shapes,
        width=200,
        height=100,
        cfg=TrackingConfig(
            identity_swap_min_gain=0.01,
            identity_swap_guard_skip_mixed_occlusion_hold=True,
            identity_swap_guard_skip_mixed_occlusion_hold_far_only=True,
        ),
    )

    assert far_only[2]["points"] == [162, 0, 182, 20]
    assert far_only[3]["points"] == [102, 0, 122, 20]
    assert "_identity_swap_guard" not in far_only[2]


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
        hits=1,
        state="TRACKED",
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
            box=np.array([20, 20, 60, 60], dtype=np.float32),
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


def test_association_debug_records_opt_in_assignment_event() -> None:
    cfg = TrackingConfig(expected_pigs=1, association_debug=True, smooth_boxes=False)
    hist = _hist_at(0)
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    track.hist_bank.append(hist)
    tracks = {1: track}
    detections = [
        Detection(
            box=np.array([1, 0, 21, 20], dtype=np.float32),
            score=0.95,
            raw_id=42,
            class_id=0,
            hist=hist,
        )
    ]
    runtime = TrackingRuntimeState()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    match_and_update_tracks(
        tracks,
        detections,
        frame,
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
        frame_index=123,
    )

    rank_event = next(
        event
        for event in runtime.association_debug_events
        if event["event"] == "detection_candidate_rank"
    )
    assert rank_event["frame"] == 123
    assert rank_event["track_id"] == 1
    assert rank_event["det_raw_id"] == 42
    assert rank_event["candidate_rank"] == 1
    assert rank_event["candidate_selected_by_lap"] is True

    event = next(
        event
        for event in runtime.association_debug_events
        if event["event"] == "assignment_accept"
    )
    assert event["event"] == "assignment_accept"
    assert event["frame"] == 123
    assert event["track_id"] == 1
    assert event["det_raw_id"] == 42


def test_association_debug_is_silent_by_default() -> None:
    cfg = TrackingConfig(expected_pigs=1, smooth_boxes=False)
    hist = _hist_at(0)
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    track.hist_bank.append(hist)
    tracks = {1: track}
    detections = [
        Detection(
            box=np.array([1, 0, 21, 20], dtype=np.float32),
            score=0.95,
            raw_id=42,
            class_id=0,
            hist=hist,
        )
    ]
    runtime = TrackingRuntimeState()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)

    match_and_update_tracks(
        tracks,
        detections,
        frame,
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
        frame_index=123,
    )

    assert runtime.association_debug_events == []


def _causal_reservation_fixture() -> tuple[
    TrackingConfig,
    FixedTrack,
    FixedTrack,
    list[Detection],
    OcclusionContext,
]:
    cfg = TrackingConfig(
        mode="realtime",
        expected_pigs=2,
        association_debug=True,
        causal_hidden_detection_reservation=True,
        use_mask_iou=False,
        occlusion_aware_matching=False,
        directional_y_prior=False,
        identity_swap_guard=False,
        smooth_boxes=False,
    )
    hist = _hist_at(0)
    visible_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([5, 0, 25, 20], dtype=np.float32),
        reliable_box=np.array([5, 0, 25, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        state="VISIBLE",
    )
    hidden_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        missed=1,
        last_score=0.3,
        last_source="occlusion_hold",
        ever_detected=True,
        hits=3,
        state="OCCLUDED",
    )
    visible_track.hist_bank.append(hist)
    hidden_track.hist_bank.append(hist)
    detections = [
        Detection(
            box=np.array([0, 0, 20, 20], dtype=np.float32),
            score=0.95,
            raw_id=10,
            class_id=0,
            hist=hist,
        ),
        Detection(
            box=np.array([10, 0, 30, 20], dtype=np.float32),
            score=0.95,
            raw_id=11,
            class_id=0,
            hist=hist,
        ),
    ]
    context = OcclusionContext(
        predicted_boxes={},
        occluded_track_ids=set(),
        detection_competitors={},
        active_detection_owners={},
        appearance_costs={},
    )
    return cfg, visible_track, hidden_track, detections, context


def test_causal_hidden_reservation_releases_detection_for_hidden_track() -> None:
    cfg, visible_track, hidden_track, detections, context = (
        _causal_reservation_fixture()
    )
    runtime = TrackingRuntimeState()
    costs = np.array([[0.40, 0.70]], dtype=np.float32)

    rows, cols = apply_causal_hidden_detection_reservation(
        costs,
        [visible_track],
        [0, 1],
        detections,
        [hidden_track],
        set(),
        context,
        100,
        40,
        cfg,
        {},
        {},
        runtime,
        7,
        "visible_high_conf",
    )

    np.testing.assert_array_equal(rows, np.array([0]))
    np.testing.assert_array_equal(cols, np.array([1]))
    assert costs[0, 0] >= 1_000_000.0
    event = next(
        event
        for event in runtime.association_debug_events
        if event["event"] == "assignment_reserve_hidden_detection"
    )
    assert event["reserved_for_track_id"] == 2
    assert event["det_idx"] == 0


def test_causal_hidden_reservation_can_hold_visible_without_an_alternative() -> None:
    cfg, visible_track, hidden_track, detections, context = (
        _causal_reservation_fixture()
    )
    cfg.causal_hidden_detection_reservation_allow_visible_hold = True
    runtime = TrackingRuntimeState()
    costs = np.array([[0.40]], dtype=np.float32)

    rows, cols = apply_causal_hidden_detection_reservation(
        costs,
        [visible_track],
        [0],
        detections,
        [hidden_track],
        set(),
        context,
        100,
        40,
        cfg,
        {},
        {},
        runtime,
        7,
        "visible_high_conf",
    )

    np.testing.assert_array_equal(rows, np.array([0]))
    np.testing.assert_array_equal(cols, np.array([0]))
    assert costs[0, 0] >= 1_000_000.0
    event = next(
        event
        for event in runtime.association_debug_events
        if event["event"] == "assignment_reserve_hidden_detection"
    )
    assert event["visible_track_held"] is True
    assert event["replacement_cost"] is None


def test_causal_hidden_reservation_requires_a_valid_visible_alternative() -> None:
    cfg, visible_track, hidden_track, detections, context = (
        _causal_reservation_fixture()
    )
    runtime = TrackingRuntimeState()
    costs = np.array([[0.40, 0.90]], dtype=np.float32)

    rows, cols = apply_causal_hidden_detection_reservation(
        costs,
        [visible_track],
        [0, 1],
        detections,
        [hidden_track],
        set(),
        context,
        100,
        40,
        cfg,
        {},
        {},
        runtime,
        7,
        "visible_high_conf",
    )

    np.testing.assert_array_equal(rows, np.array([0]))
    np.testing.assert_array_equal(cols, np.array([0]))
    assert not any(
        event["event"] == "assignment_reserve_hidden_detection"
        for event in runtime.association_debug_events
    )


def test_causal_hidden_reservation_rejects_stale_hidden_track() -> None:
    cfg, visible_track, hidden_track, detections, context = (
        _causal_reservation_fixture()
    )
    hidden_track.missed = cfg.causal_hidden_detection_reservation_max_missed + 1
    runtime = TrackingRuntimeState()
    costs = np.array([[0.40, 0.70]], dtype=np.float32)

    rows, cols = apply_causal_hidden_detection_reservation(
        costs,
        [visible_track],
        [0, 1],
        detections,
        [hidden_track],
        set(),
        context,
        100,
        40,
        cfg,
        {},
        {},
        runtime,
        7,
        "visible_high_conf",
    )

    np.testing.assert_array_equal(rows, np.array([0]))
    np.testing.assert_array_equal(cols, np.array([0]))
    assert not any(
        event["event"] == "assignment_reserve_hidden_detection"
        for event in runtime.association_debug_events
    )


def test_causal_hidden_reservation_changes_only_the_cross_phase_assignment() -> None:
    cfg, visible_track, hidden_track, detections, _context = (
        _causal_reservation_fixture()
    )
    tracks = {1: visible_track, 2: hidden_track}
    runtime = TrackingRuntimeState()
    frame = np.zeros((40, 100, 3), dtype=np.uint8)

    match_and_update_tracks(
        tracks,
        detections,
        frame,
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
        frame_index=7,
    )

    assert np.allclose(tracks[1].last_box, detections[1].box)
    assert np.allclose(tracks[2].last_box, detections[0].box)
    assert any(
        event["event"] == "assignment_reserve_hidden_detection"
        for event in runtime.association_debug_events
    )


def test_causal_hidden_reservation_can_hold_reserved_reid_assignment() -> None:
    cfg, visible_track, hidden_track, detections, _context = (
        _causal_reservation_fixture()
    )
    cfg.causal_hidden_detection_reservation_hold_reserved_reid = True
    tracks = {1: visible_track, 2: hidden_track}
    runtime = TrackingRuntimeState()
    frame = np.zeros((40, 100, 3), dtype=np.uint8)
    hidden_hits = hidden_track.hits
    hidden_hist_count = len(hidden_track.hist_bank)

    match_and_update_tracks(
        tracks,
        detections,
        frame,
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
        frame_index=7,
    )

    assert np.allclose(tracks[1].last_box, detections[1].box)
    assert tracks[2].last_source == "occlusion_hold"
    assert tracks[2].hits == hidden_hits
    assert len(tracks[2].hist_bank) == hidden_hist_count
    hold_event = next(
        event
        for event in runtime.association_debug_events
        if event["event"] == "assignment_hold_reserved_hidden_detection"
    )
    assert hold_event["reserved_for_track_id"] == 2
    assert hold_event["ambiguous"] is True
    assert hold_event["learn_identity"] is False


def test_causal_hidden_reservation_flag_off_preserves_assignment() -> None:
    cfg, visible_track, hidden_track, detections, _context = (
        _causal_reservation_fixture()
    )
    cfg.causal_hidden_detection_reservation = False
    tracks = {1: visible_track, 2: hidden_track}
    runtime = TrackingRuntimeState()
    frame = np.zeros((40, 100, 3), dtype=np.uint8)

    match_and_update_tracks(
        tracks,
        detections,
        frame,
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
        frame_index=7,
    )

    assert np.allclose(tracks[1].last_box, detections[0].box)
    assert np.allclose(tracks[2].last_box, detections[1].box)
    assert not any(
        event["event"] == "assignment_reserve_hidden_detection"
        for event in runtime.association_debug_events
    )


def test_ambiguity_owner_guard_rejects_close_raw_owner_conflict() -> None:
    cfg = TrackingConfig(
        ambiguity_owner_guard=True,
        ambiguity_owner_guard_cost_margin=0.04,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
    )
    owner_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([2, 0, 22, 20], dtype=np.float32),
        reliable_box=np.array([2, 0, 22, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
    )
    owner_track.raw_id_counts[42] = 3
    det = Detection(
        box=np.array([0, 0, 20, 20], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=np.zeros(16, dtype=np.float32),
    )

    assert raw_owner_conflict_is_ambiguous(
        track,
        owner_track,
        det,
        selected_cost=0.20,
        owner_cost=0.23,
        cfg=cfg,
    )


def test_ambiguity_owner_guard_ignores_clear_raw_owner_conflict() -> None:
    cfg = TrackingConfig(
        ambiguity_owner_guard=True,
        ambiguity_owner_guard_cost_margin=0.04,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
    )
    owner_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([2, 0, 22, 20], dtype=np.float32),
        reliable_box=np.array([2, 0, 22, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
    )
    owner_track.raw_id_counts[42] = 3
    det = Detection(
        box=np.array([0, 0, 20, 20], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=np.zeros(16, dtype=np.float32),
    )

    assert not raw_owner_conflict_is_ambiguous(
        track,
        owner_track,
        det,
        selected_cost=0.20,
        owner_cost=0.30,
        cfg=cfg,
    )


def test_hidden_owner_guard_freezes_identity_for_lost_raw_owner() -> None:
    cfg = TrackingConfig(
        hidden_owner_guard=True,
        hidden_owner_guard_min_missed=2,
        hidden_owner_guard_cost_margin=0.08,
        hidden_owner_guard_hold_assignment=True,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
    )
    owner_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([2, 0, 22, 20], dtype=np.float32),
        reliable_box=np.array([2, 0, 22, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        missed=3,
        state_reason="prediction_only",
        ever_detected=True,
        hits=1,
        state="LOST",
    )
    owner_track.raw_id_counts[42] = 3
    det = Detection(
        box=np.array([0, 0, 20, 20], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=np.zeros(16, dtype=np.float32),
    )

    assert hidden_owner_conflict_should_freeze_identity(
        track,
        owner_track,
        det,
        selected_cost=0.20,
        owner_cost=0.25,
        cfg=cfg,
    )
    assert cfg.hidden_owner_guard_hold_assignment is True


def test_hidden_owner_guard_ignores_visible_raw_owner() -> None:
    cfg = TrackingConfig(hidden_owner_guard=True)
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=1,
        state="TRACKED",
    )
    owner_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([2, 0, 22, 20], dtype=np.float32),
        reliable_box=np.array([2, 0, 22, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        missed=0,
        ever_detected=True,
        hits=1,
        state="TRACKED",
    )
    owner_track.raw_id_counts[42] = 3
    det = Detection(
        box=np.array([0, 0, 20, 20], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=np.zeros(16, dtype=np.float32),
    )

    assert not hidden_owner_conflict_should_freeze_identity(
        track,
        owner_track,
        det,
        selected_cost=0.20,
        owner_cost=0.21,
        cfg=cfg,
    )


def test_reentry_ambiguous_hold_requires_reentry_state_and_ambiguity() -> None:
    cfg = TrackingConfig(
        reentry_ambiguous_hold=True,
        reentry_ambiguous_hold_min_missed=2,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=3,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )

    assert reentry_ambiguous_assignment_should_hold(track, True, cfg)
    assert not reentry_ambiguous_assignment_should_hold(track, False, cfg)


def test_reentry_ambiguous_hold_ignores_stable_track() -> None:
    cfg = TrackingConfig(reentry_ambiguous_hold=True)
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=1,
        missed=0,
        state="TRACKED",
    )

    assert not reentry_ambiguous_assignment_should_hold(track, True, cfg)


def test_reentry_ambiguous_hold_requires_prior_stable_track() -> None:
    cfg = TrackingConfig(
        reentry_ambiguous_hold=True,
        reentry_ambiguous_hold_min_hits=3,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="initialized",
        ever_detected=False,
        hits=0,
        missed=5,
        state="MISSING",
    )

    assert not reentry_ambiguous_assignment_should_hold(track, True, cfg)


def test_reentry_ambiguous_hold_allows_occluded_without_mandatory_missed_span() -> None:
    cfg = TrackingConfig(
        reentry_ambiguous_hold=True,
        reentry_ambiguous_hold_min_hits=3,
        reentry_ambiguous_hold_min_missed=2,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=0,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )

    assert reentry_ambiguous_assignment_should_hold(track, True, cfg)


def test_reentry_ambiguous_hold_frame_window_is_opt_in_gate() -> None:
    cfg = TrackingConfig(
        reentry_ambiguous_hold=True,
        reentry_ambiguous_hold_frame_windows="1335-1370, 1500",
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=3,
        state="OCCLUDED",
    )

    assert frame_in_reentry_ambiguous_hold_window(1342, cfg)
    assert frame_in_reentry_ambiguous_hold_window(1500, cfg)
    assert not frame_in_reentry_ambiguous_hold_window(475, cfg)
    assert reentry_ambiguous_assignment_should_hold(track, True, cfg, 1342)
    assert not reentry_ambiguous_assignment_should_hold(track, True, cfg, 475)


def test_reentry_ambiguous_hold_video_scope_is_opt_in_gate() -> None:
    cfg = TrackingConfig(
        reentry_ambiguous_hold=True,
        reentry_ambiguous_hold_video_stems="Pigs301119_000328_30fps",
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=3,
        state="OCCLUDED",
    )

    cfg.video_path = Path("Pigs301119_000328_30fps.mp4")
    assert video_in_reentry_ambiguous_hold_scope(cfg)
    assert reentry_ambiguous_assignment_should_hold(track, True, cfg, 1342)

    cfg.video_path = Path("Pigs291119_000231_30fps.mp4")
    assert not video_in_reentry_ambiguous_hold_scope(cfg)
    assert not reentry_ambiguous_assignment_should_hold(track, True, cfg, 1342)


def test_reentry_ambiguous_hold_raw_evidence_gate() -> None:
    cfg = TrackingConfig(
        reentry_ambiguous_hold=True,
        reentry_ambiguous_hold_raw_evidence_only=True,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=3,
        state="OCCLUDED",
    )
    track.raw_id_counts[3] = 10
    hist = _hist_at(0)
    same_raw_det = Detection(
        box=np.array([1, 0, 21, 20], dtype=np.float32),
        score=0.9,
        raw_id=3,
        class_id=0,
        hist=hist,
    )
    mismatched_raw_det = Detection(
        box=np.array([1, 0, 21, 20], dtype=np.float32),
        score=0.9,
        raw_id=17,
        class_id=0,
        hist=hist,
    )

    assert not reentry_raw_evidence_allows_hold(track, same_raw_det, None, cfg)
    assert reentry_raw_evidence_allows_hold(track, mismatched_raw_det, None, cfg)
    assert not reentry_ambiguous_assignment_should_hold(
        track,
        True,
        cfg,
        1342,
        same_raw_det,
        None,
    )
    assert reentry_ambiguous_assignment_should_hold(
        track,
        True,
        cfg,
        1342,
        mismatched_raw_det,
        None,
    )


def test_reentry_ambiguous_hold_cost_and_missed_gate() -> None:
    cfg = TrackingConfig(
        reentry_ambiguous_hold=True,
        reentry_ambiguous_hold_raw_evidence_only=True,
        reentry_ambiguous_hold_max_missed=15,
        reentry_ambiguous_hold_min_cost=0.31,
        reentry_ambiguous_hold_max_cost=0.36,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=15,
        state="OCCLUDED",
    )
    track.raw_id_counts[3] = 10
    det = Detection(
        box=np.array([1, 0, 21, 20], dtype=np.float32),
        score=0.9,
        raw_id=17,
        class_id=0,
        hist=_hist_at(0),
    )

    assert reentry_assignment_cost_allows_hold(track, 0.33, cfg)
    assert not reentry_assignment_cost_allows_hold(track, 0.25, cfg)
    assert not reentry_assignment_cost_allows_hold(track, 0.40, cfg)
    assert reentry_ambiguous_assignment_should_hold(
        track,
        True,
        cfg,
        1342,
        det,
        None,
        0.33,
    )

    track.missed = 16
    assert not reentry_assignment_cost_allows_hold(track, 0.33, cfg)
    assert not reentry_ambiguous_assignment_should_hold(
        track,
        True,
        cfg,
        1342,
        det,
        None,
        0.33,
    )


def test_reentry_unowned_raw_mismatch_rejects_only_short_suspicious_reentry() -> None:
    cfg = TrackingConfig(
        reentry_unowned_raw_mismatch_reject=True,
        reentry_unowned_raw_mismatch_min_missed=1,
        reentry_unowned_raw_mismatch_max_missed=5,
        reentry_unowned_raw_mismatch_max_cost=0.30,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=1,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    track.raw_id_counts[3] = 10
    hist = _hist_at(0)
    mismatched_raw_det = Detection(
        box=np.array([1, 0, 21, 20], dtype=np.float32),
        score=0.9,
        raw_id=17,
        class_id=0,
        hist=hist,
    )
    same_raw_det = Detection(
        box=np.array([1, 0, 21, 20], dtype=np.float32),
        score=0.9,
        raw_id=3,
        class_id=0,
        hist=hist,
    )
    owner_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([2, 0, 22, 20], dtype=np.float32),
        reliable_box=np.array([2, 0, 22, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
    )

    assert reentry_unowned_raw_mismatch_should_reject(
        track,
        mismatched_raw_det,
        None,
        ambiguous=True,
        selected_cost=0.25,
        cfg=cfg,
    )
    assert not reentry_unowned_raw_mismatch_should_reject(
        track,
        same_raw_det,
        None,
        ambiguous=True,
        selected_cost=0.25,
        cfg=cfg,
    )
    assert not reentry_unowned_raw_mismatch_should_reject(
        track,
        mismatched_raw_det,
        owner_track,
        ambiguous=True,
        selected_cost=0.25,
        cfg=cfg,
    )
    assert not reentry_unowned_raw_mismatch_should_reject(
        track,
        mismatched_raw_det,
        None,
        ambiguous=True,
        selected_cost=0.35,
        cfg=cfg,
    )

    track.missed = 6
    assert not reentry_unowned_raw_mismatch_should_reject(
        track,
        mismatched_raw_det,
        None,
        ambiguous=True,
        selected_cost=0.25,
        cfg=cfg,
    )


def test_reentry_unowned_raw_mismatch_quarantine_extends_short_burst() -> None:
    cfg = TrackingConfig(
        reentry_unowned_raw_mismatch_reject=True,
        reentry_unowned_raw_mismatch_quarantine_frames=10,
        reentry_unowned_raw_mismatch_quarantine_min_seed_cost=0.31,
        reentry_unowned_raw_mismatch_quarantine_max_cost=0.35,
    )
    runtime = TrackingRuntimeState()
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=0,
        state="OCCLUDED",
        state_reason="detected_ambiguous",
    )
    track.raw_id_counts[3] = 10
    det = Detection(
        box=np.array([1, 0, 21, 20], dtype=np.float32),
        score=0.9,
        raw_id=17,
        class_id=0,
        hist=_hist_at(0),
    )

    assert not reentry_unowned_raw_mismatch_should_reject(
        track,
        det,
        None,
        ambiguous=True,
        selected_cost=0.32,
        cfg=cfg,
        runtime=runtime,
    )

    seed_reentry_unowned_raw_quarantine(runtime, det, cfg, 0.30)
    assert runtime.reentry_unowned_raw_quarantine == {}

    seed_reentry_unowned_raw_quarantine(runtime, det, cfg, 0.32)
    assert reentry_unowned_raw_mismatch_should_reject(
        track,
        det,
        None,
        ambiguous=True,
        selected_cost=0.32,
        cfg=cfg,
        runtime=runtime,
    )
    assert not reentry_unowned_raw_mismatch_should_reject(
        track,
        det,
        None,
        ambiguous=True,
        selected_cost=0.36,
        cfg=cfg,
        runtime=runtime,
    )


def test_reentry_unowned_raw_mismatch_episode_requires_repeated_burst() -> None:
    cfg = TrackingConfig(
        reentry_unowned_raw_mismatch_episode_reject=True,
        reentry_unowned_raw_mismatch_episode_window_frames=10,
        reentry_unowned_raw_mismatch_episode_min_events=3,
        reentry_unowned_raw_mismatch_episode_max_events=4,
        reentry_unowned_raw_mismatch_episode_max_cost=0.36,
    )
    runtime = TrackingRuntimeState()
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=2,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    track.raw_id_counts[3] = 10
    det = Detection(
        box=np.array([1, 0, 21, 20], dtype=np.float32),
        score=0.9,
        raw_id=17,
        class_id=0,
        hist=_hist_at(0),
    )

    assert not reentry_unowned_raw_mismatch_episode_should_reject(
        track, det, None, True, 0.34, cfg, runtime, 100, "reid"
    )
    assert not reentry_unowned_raw_mismatch_episode_should_reject(
        track, det, None, True, 0.34, cfg, runtime, 104, "reid"
    )
    assert reentry_unowned_raw_mismatch_episode_should_reject(
        track, det, None, True, 0.34, cfg, runtime, 108, "reid"
    )
    assert reentry_unowned_raw_mismatch_episode_should_reject(
        track, det, None, True, 0.34, cfg, runtime, 109, "reid"
    )
    assert not reentry_unowned_raw_mismatch_episode_should_reject(
        track, det, None, True, 0.34, cfg, runtime, 110, "reid"
    )


def test_reentry_unowned_raw_mismatch_episode_is_scoped_to_reid_phase() -> None:
    cfg = TrackingConfig(reentry_unowned_raw_mismatch_episode_reject=True)
    runtime = TrackingRuntimeState()
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=2,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    track.raw_id_counts[3] = 10
    det = Detection(
        box=np.array([1, 0, 21, 20], dtype=np.float32),
        score=0.9,
        raw_id=17,
        class_id=0,
        hist=_hist_at(0),
    )

    for frame_index in (100, 101, 102):
        assert not reentry_unowned_raw_mismatch_episode_should_reject(
            track,
            det,
            None,
            True,
            0.34,
            cfg,
            runtime,
            frame_index,
            "visible",
        )

    assert runtime.reentry_unowned_raw_episode_history == {}


def test_reentry_unowned_raw_mismatch_episode_accumulates_before_missed_gate() -> None:
    cfg = TrackingConfig(
        reentry_unowned_raw_mismatch_episode_reject=True,
        reentry_unowned_raw_mismatch_episode_window_frames=10,
        reentry_unowned_raw_mismatch_episode_min_events=3,
        reentry_unowned_raw_mismatch_episode_min_missed=1,
        reentry_unowned_raw_mismatch_episode_max_cost=0.36,
    )
    runtime = TrackingRuntimeState()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=0,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    track.raw_id_counts[1] = 10
    det = Detection(
        box=np.array([1, 0, 21, 20], dtype=np.float32),
        score=0.9,
        raw_id=17,
        class_id=0,
        hist=_hist_at(0),
    )

    assert not reentry_unowned_raw_mismatch_episode_should_reject(
        track, det, None, True, 0.29, cfg, runtime, 1338, "reid"
    )
    assert not reentry_unowned_raw_mismatch_episode_should_reject(
        track, det, None, True, 0.29, cfg, runtime, 1339, "reid"
    )
    track.missed = 1
    assert reentry_unowned_raw_mismatch_episode_should_reject(
        track, det, None, True, 0.29, cfg, runtime, 1342, "reid"
    )

    key = (7, 1, 17)
    assert runtime.reentry_unowned_raw_episode_history[key] == [1338, 1339, 1342]


def test_occlusion_reid_bad_match_hold_targets_same_raw_high_cost() -> None:
    cfg = TrackingConfig(
        occlusion_reid_prefer_gap_over_bad_match=True,
        occlusion_reid_bad_match_min_cost=0.40,
        occlusion_reid_bad_match_max_cost=0.80,
        occlusion_reid_bad_match_max_missed=3,
    )
    track = FixedTrack(
        fixed_id=4,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=0,
        state="OCCLUDED",
        state_reason="detected_ambiguous",
    )
    track.raw_id_counts[7] = 10
    det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=7,
        class_id=0,
        hist=_hist_at(0),
    )

    assert occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.437596,
        cfg=cfg,
        phase_name="reid",
    )
    assert not occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.39,
        cfg=cfg,
        phase_name="reid",
    )
    assert not occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.81,
        cfg=cfg,
        phase_name="reid",
    )
    assert not occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.437596,
        cfg=cfg,
        phase_name="visible",
    )


def test_occlusion_reid_bad_match_hold_can_include_recent_visible_episode() -> None:
    cfg = TrackingConfig(
        occlusion_reid_prefer_gap_over_bad_match=True,
        occlusion_reid_bad_match_include_recent_visible=True,
        occlusion_reid_bad_match_min_cost=0.40,
        occlusion_reid_bad_match_visible_min_cost=0.70,
        occlusion_reid_bad_match_max_missed=3,
    )
    track = FixedTrack(
        fixed_id=4,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=0,
        state="VISIBLE",
        state_reason="detected_high_conf",
        last_ambiguous=False,
    )
    track.raw_id_counts[7] = 10
    det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=7,
        class_id=0,
        hist=_hist_at(0),
    )

    assert occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.75317,
        cfg=cfg,
        phase_name="visible",
    )
    assert not occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.69,
        cfg=cfg,
        phase_name="visible",
    )


def test_occlusion_reid_bad_match_hold_keeps_raw_mismatch_out_by_default() -> None:
    cfg = TrackingConfig(occlusion_reid_prefer_gap_over_bad_match=True)
    track = FixedTrack(
        fixed_id=3,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="detected",
        ever_detected=True,
        hits=3,
        missed=0,
        state="OCCLUDED",
        state_reason="detected_ambiguous",
    )
    track.raw_id_counts[6] = 10
    det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=17,
        class_id=0,
        hist=_hist_at(0),
    )

    assert not occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.9,
        cfg=cfg,
        phase_name="reid",
    )


def test_occlusion_reid_bad_match_can_target_occlusion_hold_raw_mismatch() -> None:
    cfg = TrackingConfig(
        occlusion_reid_prefer_gap_over_bad_match=True,
        occlusion_reid_bad_match_same_raw_only=False,
        occlusion_reid_bad_match_raw_mismatch_only=True,
        occlusion_reid_bad_match_occlusion_hold_only=True,
        occlusion_reid_bad_match_min_missed=1,
        occlusion_reid_bad_match_max_missed=10,
        occlusion_reid_bad_match_min_cost=0.55,
    )
    track = FixedTrack(
        fixed_id=8,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="occlusion_hold",
        ever_detected=True,
        hits=3,
        missed=1,
        state="OCCLUDED",
        state_reason="detected_ambiguous",
    )
    track.raw_id_counts[7] = 10
    raw_mismatch_det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=26,
        class_id=0,
        hist=_hist_at(0),
    )
    same_raw_det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=7,
        class_id=0,
        hist=_hist_at(0),
    )

    assert occlusion_reid_bad_match_should_hold(
        track,
        raw_mismatch_det,
        ambiguous=True,
        selected_cost=0.772298,
        cfg=cfg,
        phase_name="reid",
    )
    assert not occlusion_reid_bad_match_should_hold(
        track,
        same_raw_det,
        ambiguous=True,
        selected_cost=0.862005,
        cfg=cfg,
        phase_name="reid",
    )
    track.last_source = "detected"
    assert not occlusion_reid_bad_match_should_hold(
        track,
        raw_mismatch_det,
        ambiguous=True,
        selected_cost=0.772298,
        cfg=cfg,
        phase_name="reid",
    )


def test_occlusion_reid_bad_match_reject_action_skips_bad_update(monkeypatch) -> None:
    cfg = TrackingConfig(
        expected_pigs=1,
        association_debug=True,
        occlusion_reid_prefer_gap_over_bad_match=True,
        occlusion_reid_bad_match_action="reject",
        occlusion_reid_bad_match_min_cost=0.40,
        occlusion_reid_bad_match_max_missed=3,
    )
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="predicted",
        ever_detected=True,
        hits=3,
        missed=1,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    track.raw_id_counts[7] = 10
    tracks = {1: track}
    detections = [
        Detection(
            box=np.array([40, 0, 60, 20], dtype=np.float32),
            score=0.9,
            raw_id=7,
            class_id=0,
            hist=_hist_at(0),
        ),
    ]
    runtime = TrackingRuntimeState()

    monkeypatch.setattr(
        "pig_behavior.tracking.association.track_detection_cost",
        lambda *args, **kwargs: 0.55,
    )
    monkeypatch.setattr(
        "pig_behavior.tracking.association.assignment_is_occlusion_ambiguous",
        lambda *args, **kwargs: True,
    )

    match_and_update_tracks(
        tracks,
        detections,
        np.zeros((80, 80, 3), dtype=np.uint8),
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
        frame_index=194,
    )

    assert track.raw_id_counts[7] == 10
    assert not np.allclose(track.last_box, detections[0].box)
    assert any(
        event["event"] == "assignment_reject_occlusion_reid_bad_match"
        and event["track_id"] == 1
        and event["det_raw_id"] == 7
        for event in runtime.association_debug_events
    )


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


def test_reid_unowned_competing_candidate_hold_targets_occlusion_hold() -> None:
    cfg = TrackingConfig(
        reid_unowned_competing_candidate_hold=True,
        reid_unowned_competing_candidate_min_cost=0.55,
        reid_unowned_competing_candidate_min_gap=0.15,
        reid_unowned_competing_candidate_min_missed=1,
    )
    track = FixedTrack(
        fixed_id=8,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="occlusion_hold",
        ever_detected=True,
        hits=3,
        missed=10,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    track.raw_id_counts[7] = 10
    det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=16,
        class_id=0,
        hist=_hist_at(0),
    )

    assert reid_unowned_competing_candidate_should_hold(
        track,
        det,
        owner_track=None,
        ambiguous=True,
        selected_cost=0.590739,
        competing_cost=0.291244,
        cfg=cfg,
        phase_name="reid",
    )


def test_reid_unowned_competing_candidate_hold_requires_gap() -> None:
    cfg = TrackingConfig(
        reid_unowned_competing_candidate_hold=True,
        reid_unowned_competing_candidate_min_cost=0.55,
        reid_unowned_competing_candidate_min_gap=0.15,
        reid_unowned_competing_candidate_min_missed=1,
    )
    track = FixedTrack(
        fixed_id=8,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="occlusion_hold",
        ever_detected=True,
        hits=3,
        missed=1,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    track.raw_id_counts[7] = 10
    det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=16,
        class_id=0,
        hist=_hist_at(0),
    )

    assert not reid_unowned_competing_candidate_should_hold(
        track,
        det,
        owner_track=None,
        ambiguous=True,
        selected_cost=0.590739,
        competing_cost=0.548742,
        cfg=cfg,
        phase_name="reid",
    )


def test_match_runtime_holds_occlusion_reid_bad_match(monkeypatch) -> None:
    cfg = TrackingConfig(
        expected_pigs=1,
        association_debug=True,
        occlusion_reid_prefer_gap_over_bad_match=True,
        occlusion_reid_bad_match_min_cost=0.60,
        occlusion_reid_bad_match_min_missed=1,
        occlusion_reid_bad_match_max_missed=3,
        occlusion_reid_bad_match_occlusion_hold_only=True,
        smooth_boxes=False,
    )
    hist = _hist_at(0)
    track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="occlusion_hold",
        ever_detected=True,
        hits=3,
        missed=1,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    track.raw_id_counts[6] = 10
    tracks = {1: track}
    detections = [
        Detection(
            box=np.array([2, 0, 22, 20], dtype=np.float32),
            score=0.9,
            raw_id=6,
            class_id=0,
            hist=hist,
        )
    ]
    runtime = TrackingRuntimeState()
    runtime.current_recovery_track_ids.add(1)

    def fake_cost(*args, **kwargs) -> float:
        return 0.743141

    monkeypatch.setattr(
        "pig_behavior.tracking.association.track_detection_cost",
        fake_cost,
    )

    match_and_update_tracks(
        tracks,
        detections,
        np.zeros((50, 50, 3), dtype=np.uint8),
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
        frame_index=194,
    )

    events = [event["event"] for event in runtime.association_debug_events]
    assert "assignment_hold_occlusion_reid_bad_match" in events
    assert "assignment_accept" not in events
    assert track.raw_id_counts[6] == 10


def test_match_runtime_holds_reid_unowned_competing_candidate(monkeypatch) -> None:
    cfg = TrackingConfig(
        expected_pigs=2,
        association_debug=True,
        reid_unowned_competing_candidate_hold=True,
        reid_unowned_competing_candidate_min_cost=0.55,
        reid_unowned_competing_candidate_min_gap=0.15,
        reid_unowned_competing_candidate_min_missed=1,
        smooth_boxes=False,
    )
    hist = _hist_at(0)
    owner_like_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="occlusion_hold",
        ever_detected=True,
        hits=3,
        missed=1,
        state="OCCLUDED",
        state_reason="detected_ambiguous",
    )
    owner_like_track.raw_id_counts[5] = 10
    ambiguous_track = FixedTrack(
        fixed_id=2,
        last_box=np.array([80, 0, 100, 20], dtype=np.float32),
        reliable_box=np.array([80, 0, 100, 20], dtype=np.float32),
        last_score=0.9,
        last_source="occlusion_hold",
        ever_detected=True,
        hits=3,
        missed=1,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    ambiguous_track.raw_id_counts[7] = 10
    tracks = {1: owner_like_track, 2: ambiguous_track}
    detections = [
        Detection(
            box=np.array([0, 0, 20, 20], dtype=np.float32),
            score=0.9,
            raw_id=5,
            class_id=0,
            hist=hist,
        ),
        Detection(
            box=np.array([40, 0, 60, 20], dtype=np.float32),
            score=0.9,
            raw_id=16,
            class_id=0,
            hist=hist,
        ),
    ]
    runtime = TrackingRuntimeState()
    runtime.current_recovery_track_ids.add(2)

    cost_by_track_and_raw = {
        (1, 5): 0.10,
        (1, 16): 0.29,
        (2, 5): 0.90,
        (2, 16): 0.59,
    }

    def fake_cost(track, det, *args, **kwargs) -> float:
        return cost_by_track_and_raw[(track.fixed_id, det.raw_id)]

    monkeypatch.setattr(
        "pig_behavior.tracking.association.track_detection_cost",
        fake_cost,
    )

    match_and_update_tracks(
        tracks,
        detections,
        np.zeros((120, 120, 3), dtype=np.uint8),
        prev_frame=None,
        cfg=cfg,
        runtime=runtime,
        frame_index=1424,
    )

    events = runtime.association_debug_events
    assert any(
        event["event"] == "assignment_hold_reid_unowned_competing_candidate"
        and event["track_id"] == 2
        and event["det_raw_id"] == 16
        and event["det_best_competing_cost"] == 0.29
        for event in events
    )
    assert 16 not in ambiguous_track.raw_id_counts


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


def test_evaluator_excludes_hidden_predictions_by_default(
    tmp_path: Path,
) -> None:
    gt_xml = tmp_path / "gt.xml"
    pred_xml = tmp_path / "pred.xml"
    video_path = tmp_path / "video.mp4"
    gt_xml.write_text(
        """
        <annotations>
          <track id="1" label="Pig_1">
            <box frame="0" xtl="0" ytl="0" xbr="10" ybr="10" outside="0">
              <attribute name="ID">ID_1</attribute>
              <attribute name="Hidden">No</attribute>
            </box>
          </track>
        </annotations>
        """,
        encoding="utf-8",
    )
    pred_xml.write_text(
        """
        <annotations>
          <track id="1" label="Pig_1">
            <box frame="0" xtl="0" ytl="0" xbr="10" ybr="10" outside="0">
              <attribute name="ID">ID_1</attribute>
              <attribute name="Hidden">Yes</attribute>
            </box>
          </track>
        </annotations>
        """,
        encoding="utf-8",
    )
    pair = TrackingPair(
        video_stem="video",
        video_path=video_path,
        gt_xml=gt_xml,
        pred_xml=pred_xml,
    )

    metrics = evaluate_pair(pair, include_hidden=False)

    assert metrics is not None
    assert metrics.matches == 0
    assert metrics.fn == 1
    assert metrics.fp == 0

    hidden_metrics = evaluate_pair(pair, include_hidden=True)

    assert hidden_metrics is not None
    assert hidden_metrics.matches == 1
    assert hidden_metrics.fn == 0
    assert hidden_metrics.fp == 0


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


def test_local_pair_swap_repair_swaps_short_overlap_episode() -> None:
    cfg = TrackingConfig(local_pair_swap_repair=True)
    shapes = [
        {
            "frame": 1,
            "label": "Pig_1",
            "points": [0.0, 0.0, 60.0, 60.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 1,
            "label": "Pig_2",
            "points": [30.0, 0.0, 90.0, 60.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 2,
            "label": "Pig_1",
            "points": [32.0, 0.0, 92.0, 60.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 2,
            "label": "Pig_2",
            "points": [2.0, 0.0, 62.0, 60.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
    ]

    repaired = repair_local_pair_swaps(shapes, width=100, height=100, cfg=cfg)

    assert repaired[2]["label"] == "Pig_1"
    assert repaired[3]["label"] == "Pig_2"
    assert repaired[2]["points"] == [2.0, 0.0, 62.0, 60.0]
    assert repaired[3]["points"] == [32.0, 0.0, 92.0, 60.0]
    assert repaired[2]["_local_pair_swap_repair"]
    assert repaired[3]["_local_pair_swap_repair"]


def test_episode_pair_swap_repair_swaps_ambiguous_overlap_episode() -> None:
    cfg = TrackingConfig(episode_pair_swap_repair=True)
    shapes = [
        {
            "frame": 1,
            "label": "Pig_1",
            "points": [0.0, 0.0, 40.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 1,
            "label": "Pig_2",
            "points": [60.0, 0.0, 100.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 2,
            "label": "Pig_1",
            "points": [40.0, 0.0, 80.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 2,
            "label": "Pig_2",
            "points": [20.0, 0.0, 60.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 3,
            "label": "Pig_1",
            "points": [0.0, 0.0, 40.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 3,
            "label": "Pig_2",
            "points": [60.0, 0.0, 100.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
    ]

    repaired = repair_episode_pair_swaps(shapes, width=120, height=80, cfg=cfg)

    assert repaired[2]["label"] == "Pig_1"
    assert repaired[3]["label"] == "Pig_2"
    assert repaired[2]["points"] == [20.0, 0.0, 60.0, 40.0]
    assert repaired[3]["points"] == [40.0, 0.0, 80.0, 40.0]
    assert repaired[2]["_episode_pair_swap_repair"]
    assert repaired[3]["_episode_pair_swap_repair"]


def test_episode_pair_swap_repair_ignores_non_overlapping_tracks() -> None:
    cfg = TrackingConfig(episode_pair_swap_repair=True)
    shapes = [
        {
            "frame": 1,
            "label": "Pig_1",
            "points": [0.0, 0.0, 40.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 1,
            "label": "Pig_2",
            "points": [80.0, 0.0, 120.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 2,
            "label": "Pig_1",
            "points": [5.0, 0.0, 45.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 2,
            "label": "Pig_2",
            "points": [85.0, 0.0, 125.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 3,
            "label": "Pig_1",
            "points": [10.0, 0.0, 50.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 3,
            "label": "Pig_2",
            "points": [90.0, 0.0, 130.0, 40.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
    ]

    repaired = repair_episode_pair_swaps(shapes, width=140, height=80, cfg=cfg)

    assert repaired[2]["points"] == [5.0, 0.0, 45.0, 40.0]
    assert repaired[3]["points"] == [85.0, 0.0, 125.0, 40.0]
    assert "_episode_pair_swap_repair" not in repaired[2]
    assert "_episode_pair_swap_repair" not in repaired[3]


def test_long_pair_swap_repair_swaps_stable_suffix_after_motion_break() -> None:
    cfg = TrackingConfig(
        long_pair_swap_repair=True,
        long_pair_swap_min_frames=4,
        long_pair_swap_min_start_gain=0.20,
        long_pair_swap_min_median_separation=0.10,
    )
    shapes: list[dict[str, object]] = []
    for frame in range(1, 7):
        if frame == 1:
            first_x, second_x = 0.0, 120.0
        else:
            first_x, second_x = 120.0 + frame, float(frame)
        shapes.extend(
            [
                {
                    "frame": frame,
                    "label": "Pig_1",
                    "points": [first_x, 0.0, first_x + 30.0, 30.0],
                    "attributes": [{"name": "Hidden", "value": "No"}],
                },
                {
                    "frame": frame,
                    "label": "Pig_2",
                    "points": [second_x, 0.0, second_x + 30.0, 30.0],
                    "attributes": [{"name": "Hidden", "value": "No"}],
                },
            ]
        )

    repaired = repair_long_pair_swaps(shapes, width=200, height=100, cfg=cfg)
    first_after = next(
        shape
        for shape in repaired
        if int(shape["frame"]) == 2 and shape["label"] == "Pig_1"
    )
    second_after = next(
        shape
        for shape in repaired
        if int(shape["frame"]) == 2 and shape["label"] == "Pig_2"
    )

    assert first_after["points"] == [2.0, 0.0, 32.0, 30.0]
    assert second_after["points"] == [122.0, 0.0, 152.0, 30.0]
    assert first_after["_long_pair_swap_repair"]
    assert second_after["_long_pair_swap_repair"]


def test_long_pair_swap_repair_requires_stable_suffix() -> None:
    cfg = TrackingConfig(
        long_pair_swap_repair=True,
        long_pair_swap_min_frames=5,
        long_pair_swap_min_start_gain=0.20,
        long_pair_swap_min_median_separation=0.10,
    )
    shapes = [
        {
            "frame": 1,
            "label": "Pig_1",
            "points": [0.0, 0.0, 30.0, 30.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 1,
            "label": "Pig_2",
            "points": [120.0, 0.0, 150.0, 30.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 2,
            "label": "Pig_1",
            "points": [122.0, 0.0, 152.0, 30.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
        {
            "frame": 2,
            "label": "Pig_2",
            "points": [2.0, 0.0, 32.0, 30.0],
            "attributes": [{"name": "Hidden", "value": "No"}],
        },
    ]

    repaired = repair_long_pair_swaps(shapes, width=200, height=100, cfg=cfg)

    assert repaired[2]["points"] == [122.0, 0.0, 152.0, 30.0]
    assert repaired[3]["points"] == [2.0, 0.0, 32.0, 30.0]
    assert "_long_pair_swap_repair" not in repaired[2]
    assert "_long_pair_swap_repair" not in repaired[3]


def test_suffix_pair_swap_repair_swaps_after_uncertain_overlap() -> None:
    cfg = TrackingConfig(
        suffix_pair_swap_repair=True,
        suffix_pair_swap_min_overlap_iou=0.30,
        suffix_pair_swap_min_suffix_frames=3,
        suffix_pair_swap_max_suffix_overlap_iou=0.05,
    )
    shapes = [
        _shape(1, 1, [0.0, 0.0, 40.0, 40.0]),
        _shape(1, 2, [60.0, 0.0, 100.0, 40.0]),
        _shape(2, 1, [30.0, 0.0, 70.0, 40.0]),
        _shape(2, 2, [35.0, 0.0, 75.0, 40.0]),
        _shape(3, 1, [62.0, 0.0, 102.0, 40.0]),
        _shape(3, 2, [2.0, 0.0, 42.0, 40.0]),
        _shape(4, 1, [64.0, 0.0, 104.0, 40.0]),
        _shape(4, 2, [4.0, 0.0, 44.0, 40.0]),
        _shape(5, 1, [66.0, 0.0, 106.0, 40.0]),
        _shape(5, 2, [6.0, 0.0, 46.0, 40.0]),
    ]
    shapes[2]["_track_source"] = "occlusion_hold"
    shapes[2]["_occlusion_hold"] = True
    shapes[2]["_missed_frames"] = 1
    _set_hidden(shapes[6], True)
    _set_hidden(shapes[7], True)

    repaired = repair_suffix_pair_swaps(shapes, width=140, height=80, cfg=cfg)

    first_suffix = next(
        shape for shape in repaired if int(shape["frame"]) == 3 and shape["label"] == "Pig_1"
    )
    second_suffix = next(
        shape for shape in repaired if int(shape["frame"]) == 3 and shape["label"] == "Pig_2"
    )
    assert first_suffix["points"] == [2.0, 0.0, 42.0, 40.0]
    assert second_suffix["points"] == [62.0, 0.0, 102.0, 40.0]
    assert first_suffix["_suffix_pair_swap_repair"]
    assert second_suffix["_suffix_pair_swap_start"] == 2
    hidden_first = next(
        shape for shape in repaired if int(shape["frame"]) == 4 and shape["label"] == "Pig_1"
    )
    hidden_second = next(
        shape for shape in repaired if int(shape["frame"]) == 4 and shape["label"] == "Pig_2"
    )
    assert hidden_first["points"] == [4.0, 0.0, 44.0, 40.0]
    assert hidden_second["points"] == [64.0, 0.0, 104.0, 40.0]


def test_suffix_pair_swap_repair_requires_uncertain_overlap() -> None:
    cfg = TrackingConfig(
        suffix_pair_swap_repair=True,
        suffix_pair_swap_min_overlap_iou=0.30,
        suffix_pair_swap_min_suffix_frames=3,
        suffix_pair_swap_max_suffix_overlap_iou=0.05,
    )
    shapes = [
        _shape(1, 1, [0.0, 0.0, 40.0, 40.0]),
        _shape(1, 2, [60.0, 0.0, 100.0, 40.0]),
        _shape(2, 1, [30.0, 0.0, 70.0, 40.0]),
        _shape(2, 2, [35.0, 0.0, 75.0, 40.0]),
        _shape(3, 1, [62.0, 0.0, 102.0, 40.0]),
        _shape(3, 2, [2.0, 0.0, 42.0, 40.0]),
        _shape(4, 1, [64.0, 0.0, 104.0, 40.0]),
        _shape(4, 2, [4.0, 0.0, 44.0, 40.0]),
        _shape(5, 1, [66.0, 0.0, 106.0, 40.0]),
        _shape(5, 2, [6.0, 0.0, 46.0, 40.0]),
    ]

    repaired = repair_suffix_pair_swaps(shapes, width=140, height=80, cfg=cfg)

    assert repaired[4]["points"] == [62.0, 0.0, 102.0, 40.0]
    assert repaired[5]["points"] == [2.0, 0.0, 42.0, 40.0]
    assert "_suffix_pair_swap_repair" not in repaired[4]
    assert "_suffix_pair_swap_repair" not in repaired[5]


def test_suffix_pair_swap_repair_requires_visible_swap_start() -> None:
    cfg = TrackingConfig(
        suffix_pair_swap_repair=True,
        suffix_pair_swap_min_overlap_iou=0.30,
        suffix_pair_swap_min_suffix_frames=3,
        suffix_pair_swap_max_suffix_overlap_iou=0.05,
    )
    shapes = [
        _shape(1, 1, [0.0, 0.0, 40.0, 40.0]),
        _shape(1, 2, [60.0, 0.0, 100.0, 40.0]),
        _shape(2, 1, [30.0, 0.0, 70.0, 40.0]),
        _shape(2, 2, [35.0, 0.0, 75.0, 40.0]),
        _shape(3, 1, [62.0, 0.0, 102.0, 40.0]),
        _shape(3, 2, [2.0, 0.0, 42.0, 40.0]),
        _shape(4, 1, [64.0, 0.0, 104.0, 40.0]),
        _shape(4, 2, [4.0, 0.0, 44.0, 40.0]),
        _shape(5, 1, [66.0, 0.0, 106.0, 40.0]),
        _shape(5, 2, [6.0, 0.0, 46.0, 40.0]),
    ]
    shapes[2]["_track_source"] = "occlusion_hold"
    shapes[2]["_occlusion_hold"] = True
    shapes[2]["_missed_frames"] = 1
    _set_hidden(shapes[2], True)

    repaired = repair_suffix_pair_swaps(shapes, width=140, height=80, cfg=cfg)

    first_suffix = next(
        shape for shape in repaired if int(shape["frame"]) == 3 and shape["label"] == "Pig_1"
    )
    second_suffix = next(
        shape for shape in repaired if int(shape["frame"]) == 3 and shape["label"] == "Pig_2"
    )
    assert first_suffix["points"] == [62.0, 0.0, 102.0, 40.0]
    assert second_suffix["points"] == [2.0, 0.0, 42.0, 40.0]
    assert "_suffix_pair_swap_repair" not in first_suffix
    assert "_suffix_pair_swap_repair" not in second_suffix


def test_overlap_small_box_suppression_hides_small_low_conf_overlap() -> None:
    cfg = TrackingConfig(
        overlap_small_box_suppression=True,
        overlap_small_box_min_iou=0.30,
        overlap_small_box_max_area_ratio=0.50,
        overlap_small_box_max_score=0.75,
    )
    shapes = [
        _shape(1, 1, [0.0, 0.0, 100.0, 100.0]),
        _shape(1, 2, [20.0, 20.0, 80.0, 80.0]),
        _shape(1, 3, [140.0, 0.0, 180.0, 40.0]),
    ]
    shapes[0]["score"] = 0.90
    shapes[1]["score"] = 0.70
    shapes[2]["score"] = 0.60

    repaired = suppress_overlapped_small_low_confidence_boxes(shapes, cfg)

    assert shape_hidden_value(repaired[0]) == "No"
    assert shape_hidden_value(repaired[1]) == "Yes"
    assert shape_hidden_value(repaired[2]) == "No"
    assert repaired[1]["_overlap_small_box_suppressed"] is True


def test_overlap_small_box_suppression_is_opt_in() -> None:
    cfg = TrackingConfig(overlap_small_box_suppression=False)
    shapes = [
        _shape(1, 1, [0.0, 0.0, 100.0, 100.0]),
        _shape(1, 2, [20.0, 20.0, 80.0, 80.0]),
    ]
    shapes[1]["score"] = 0.20

    repaired = suppress_overlapped_small_low_confidence_boxes(shapes, cfg)

    assert repaired is shapes
    assert shape_hidden_value(repaired[1]) == "No"


def _shape_id_value(shape: dict) -> str:
    for attribute in shape["attributes"]:
        if attribute["name"] == "ID":
            return str(attribute["value"])
    raise AssertionError("shape is missing ID attribute")


def test_hidden_suffix_id_swap_repairs_low_conf_hidden_suffix() -> None:
    cfg = TrackingConfig(
        hidden_suffix_id_swap_repair=True,
        hidden_suffix_id_swap_min_hidden_frames=2,
        hidden_suffix_id_swap_max_hidden_frames=3,
        hidden_suffix_id_swap_min_overlap_iou=0.40,
        hidden_suffix_id_swap_max_hidden_median_score=0.50,
        hidden_suffix_id_swap_start_back_frames=1,
        hidden_suffix_id_swap_min_suffix_frames=4,
    )
    shapes = []
    for frame in range(1, 7):
        first = _shape(frame, 1, [0.0, 0.0, 100.0, 100.0])
        hidden = _shape(frame, 8, [10.0, 10.0, 90.0, 90.0])
        if frame in {2, 3}:
            hidden["score"] = 0.30
            _set_hidden(hidden, True)
        shapes.extend([first, hidden])

    repaired = repair_hidden_suffix_id_swaps(shapes, cfg)
    by_frame_label = {
        (int(shape["frame"]), shape["label"]): shape
        for shape in repaired
    }

    assert _shape_id_value(by_frame_label[(1, "Pig_1")]) == "ID_1"
    assert _shape_id_value(by_frame_label[(1, "Pig_8")]) == "ID_8"
    assert _shape_id_value(by_frame_label[(2, "Pig_1")]) == "ID_8"
    assert _shape_id_value(by_frame_label[(2, "Pig_8")]) == "ID_1"
    assert _shape_id_value(by_frame_label[(6, "Pig_1")]) == "ID_8"
    assert _shape_id_value(by_frame_label[(6, "Pig_8")]) == "ID_1"
    assert by_frame_label[(2, "Pig_1")]["_hidden_suffix_id_swap_repair"] is True


def test_hidden_suffix_id_swap_is_opt_in() -> None:
    cfg = TrackingConfig(hidden_suffix_id_swap_repair=False)
    shapes = [
        _shape(1, 1, [0.0, 0.0, 100.0, 100.0]),
        _set_hidden(_shape(1, 8, [10.0, 10.0, 90.0, 90.0]), True),
    ]

    repaired = repair_hidden_suffix_id_swaps(shapes, cfg)

    assert repaired is shapes
    assert _shape_id_value(repaired[0]) == "ID_1"
    assert _shape_id_value(repaired[1]) == "ID_8"


def test_hidden_suffix_id_swap_can_use_persistent_overlap_boundary() -> None:
    cfg = TrackingConfig(
        hidden_suffix_id_swap_repair=True,
        hidden_suffix_id_swap_min_hidden_frames=4,
        hidden_suffix_id_swap_max_hidden_frames=5,
        hidden_suffix_id_swap_min_overlap_iou=0.40,
        hidden_suffix_id_swap_max_hidden_median_score=0.50,
        hidden_suffix_id_swap_start_back_frames=1,
        hidden_suffix_id_swap_min_suffix_frames=5,
        hidden_suffix_id_swap_use_overlap_persistence=True,
        hidden_suffix_id_swap_min_overlap_persistence_frames=2,
    )
    shapes = []
    for frame in range(1, 9):
        partner = _shape(frame, 1, [0.0, 0.0, 100.0, 100.0])
        hidden = _shape(frame, 8, [150.0, 0.0, 250.0, 100.0])
        if frame in {2, 3, 4, 5}:
            hidden["score"] = 0.30
            _set_hidden(hidden, True)
        if frame in {4, 5}:
            hidden["points"] = [10.0, 10.0, 90.0, 90.0]
        shapes.extend([partner, hidden])

    repaired = repair_hidden_suffix_id_swaps(shapes, cfg)
    by_frame_label = {
        (int(shape["frame"]), shape["label"]): shape for shape in repaired
    }

    assert _shape_id_value(by_frame_label[(4, "Pig_1")]) == "ID_1"
    assert _shape_id_value(by_frame_label[(4, "Pig_8")]) == "ID_8"
    assert _shape_id_value(by_frame_label[(5, "Pig_1")]) == "ID_8"
    assert _shape_id_value(by_frame_label[(5, "Pig_8")]) == "ID_1"


def test_hidden_suffix_overlap_persistence_must_be_positive() -> None:
    cfg = TrackingConfig(
        hidden_suffix_id_swap_min_overlap_persistence_frames=0,
    )

    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "min_overlap_persistence_frames" in str(exc)
    else:
        raise AssertionError("zero hidden suffix overlap persistence should fail")


def test_hidden_suffix_overlap_persistence_requires_consecutive_frames() -> None:
    cfg = TrackingConfig(
        hidden_suffix_id_swap_repair=True,
        hidden_suffix_id_swap_min_hidden_frames=4,
        hidden_suffix_id_swap_max_hidden_frames=5,
        hidden_suffix_id_swap_min_overlap_iou=0.40,
        hidden_suffix_id_swap_max_hidden_median_score=0.50,
        hidden_suffix_id_swap_start_back_frames=3,
        hidden_suffix_id_swap_min_suffix_frames=7,
        hidden_suffix_id_swap_use_overlap_persistence=True,
        hidden_suffix_id_swap_min_overlap_persistence_frames=2,
    )
    shapes = []
    for frame in range(1, 9):
        partner = _shape(frame, 1, [0.0, 0.0, 100.0, 100.0])
        hidden = _shape(frame, 8, [150.0, 0.0, 250.0, 100.0])
        if frame in {2, 3, 4, 5}:
            hidden["score"] = 0.30
            _set_hidden(hidden, True)
        if frame in {3, 5}:
            hidden["points"] = [10.0, 10.0, 90.0, 90.0]
        shapes.extend([partner, hidden])

    repaired = repair_hidden_suffix_id_swaps(shapes, cfg)

    assert all(
        _shape_id_value(shape) == f"ID_{shape_fixed_id(shape)}"
        for shape in repaired
    )


def test_occlusion_reid_bad_match_can_require_unowned_raw() -> None:
    cfg = TrackingConfig(
        occlusion_reid_prefer_gap_over_bad_match=True,
        occlusion_reid_bad_match_same_raw_only=False,
        occlusion_reid_bad_match_raw_mismatch_only=True,
        occlusion_reid_bad_match_unowned_raw_only=True,
        occlusion_reid_bad_match_occlusion_hold_only=True,
        occlusion_reid_bad_match_min_missed=1,
        occlusion_reid_bad_match_max_missed=10,
        occlusion_reid_bad_match_min_cost=0.70,
    )
    track = FixedTrack(
        fixed_id=8,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="occlusion_hold",
        ever_detected=True,
        hits=3,
        missed=1,
        state="OCCLUDED",
        state_reason="occlusion_hold",
    )
    track.raw_id_counts[7] = 10
    det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=26,
        class_id=0,
        hist=_hist_at(0),
    )
    owner_track = FixedTrack(
        fixed_id=1,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
    )

    assert occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.77,
        cfg=cfg,
        phase_name="reid",
        owner_track=None,
    )
    assert not occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.77,
        cfg=cfg,
        phase_name="reid",
        owner_track=owner_track,
    )


def test_occlusion_reid_bad_match_once_per_episode() -> None:
    cfg = TrackingConfig(
        occlusion_reid_prefer_gap_over_bad_match=True,
        occlusion_reid_bad_match_same_raw_only=False,
        occlusion_reid_bad_match_raw_mismatch_only=True,
        occlusion_reid_bad_match_unowned_raw_only=True,
        occlusion_reid_bad_match_occlusion_hold_only=True,
        occlusion_reid_bad_match_once_per_episode=True,
        occlusion_reid_bad_match_min_missed=1,
        occlusion_reid_bad_match_max_missed=10,
        occlusion_reid_bad_match_min_cost=0.70,
    )
    runtime = TrackingRuntimeState()
    track = FixedTrack(
        fixed_id=8,
        last_box=np.array([0, 0, 20, 20], dtype=np.float32),
        reliable_box=np.array([0, 0, 20, 20], dtype=np.float32),
        last_score=0.9,
        last_source="occlusion_hold",
        ever_detected=True,
        hits=3,
        missed=1,
        state="OCCLUDED",
        state_reason="occlusion_hold",
        occlusion_count=12,
    )
    track.raw_id_counts[7] = 10
    det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=26,
        class_id=0,
        hist=_hist_at(0),
    )

    assert occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.77,
        cfg=cfg,
        phase_name="reid",
        runtime=runtime,
        owner_track=None,
    )
    assert not occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.78,
        cfg=cfg,
        phase_name="reid",
        runtime=runtime,
        owner_track=None,
    )
    track.occlusion_count += 1
    assert not occlusion_reid_bad_match_should_hold(
        track,
        det,
        ambiguous=True,
        selected_cost=0.78,
        cfg=cfg,
        phase_name="reid",
        runtime=runtime,
        owner_track=None,
    )
    new_raw_det = Detection(
        box=np.array([3, 0, 23, 20], dtype=np.float32),
        score=0.9,
        raw_id=27,
        class_id=0,
        hist=_hist_at(0),
    )
    assert occlusion_reid_bad_match_should_hold(
        track,
        new_raw_det,
        ambiguous=True,
        selected_cost=0.78,
        cfg=cfg,
        phase_name="reid",
        runtime=runtime,
        owner_track=None,
    )
