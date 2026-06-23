"""Tests for the sanity gate that validates track updates."""

from __future__ import annotations

import numpy as np
import pytest

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.kalman import create_bev_kalman
from pig_behavior.tracking.rgbd.sanity import (
    REJECT_AREA_RATIO,
    REJECT_CENTER_JUMP,
    REJECT_INVALID_DEPTH,
    validate_rgbd_update_with_frame_size,
)
from pig_behavior.tracking.rgbd.schemas import (
    AssociationDecision,
    BEVTrackState,
    Detection2D,
    Detection3D,
)
from pig_behavior.tracking.schemas import FixedTrack


def _make_cfg(**overrides) -> RGBDTrackingConfig:
    from pathlib import Path

    from pig_behavior.tracking.config import TrackingConfig

    defaults = dict(
        tracking_config=TrackingConfig(),
        depth_video_path=Path("d.mp4"),
        depth_scale_path=Path("s.npy"),
        inverse_intrinsic_path=Path("k.npy"),
        rotation_path=Path("r.npy"),
        bev_association_gate_m=0.50,
        max_center_jump_norm=0.06,
        min_area_ratio=0.60,
        max_area_ratio=1.50,
    )
    defaults.update(overrides)
    return RGBDTrackingConfig(**defaults)


def _make_track(box: list[float]) -> FixedTrack:
    t = FixedTrack(fixed_id=1, last_box=np.array(box, dtype=np.float32))
    t.ever_detected = True
    t.hits = 5
    return t


def _make_bev_state(pos: list[float], cfg: RGBDTrackingConfig) -> BEVTrackState:
    kf = create_bev_kalman(np.array(pos, dtype=np.float64), cfg)
    return BEVTrackState(
        fixed_id=1, kf=kf,
        bev_position=np.array(pos, dtype=np.float64),
        bev_velocity=np.zeros(2, dtype=np.float64),
        state="confirmed",
    )


def _make_decision(**kwargs) -> AssociationDecision:
    defaults = dict(
        frame_index=0, track_id=1, detection_index=0,
        bev_distance_m=0.1, cost=0.2, best_score=0.2,
        second_best_score=0.8, score_margin=0.6,
        accepted=True, depth_valid=True, is_occluded=False,
    )
    defaults.update(kwargs)
    return AssociationDecision(**defaults)


class TestSanityRejectsCenterJump:
    """Detections that jump too far from the track center should be rejected."""

    def test_large_jump_rejected(self):
        cfg = _make_cfg(max_center_jump_norm=0.02)
        track = _make_track([100, 100, 200, 200])
        bev = _make_bev_state([0.5, 0.5], cfg)
        # Detection far away: center at (900, 900) vs track center (150, 150)
        det = Detection3D(
            detection_2d=Detection2D(bbox=(850, 850, 950, 950), confidence=0.9),
            depth_m=1.5, depth_valid=True,
        )
        decision = _make_decision()
        ok, reason = validate_rgbd_update_with_frame_size(
            track, bev, det, decision, cfg, 1920, 1080
        )
        assert not ok
        assert reason == REJECT_CENTER_JUMP

    def test_small_jump_accepted(self):
        cfg = _make_cfg(max_center_jump_norm=0.10)
        track = _make_track([100, 100, 200, 200])
        bev = _make_bev_state([0.5, 0.5], cfg)
        det = Detection3D(
            detection_2d=Detection2D(bbox=(110, 110, 210, 210), confidence=0.9),
            depth_m=1.5, depth_valid=True,
        )
        decision = _make_decision()
        ok, reason = validate_rgbd_update_with_frame_size(
            track, bev, det, decision, cfg, 1920, 1080
        )
        assert ok
        assert reason is None


class TestSanityRejectsAreaSpike:
    """Sudden bbox area changes should be rejected."""

    def test_area_too_large(self):
        cfg = _make_cfg(max_area_ratio=1.50)
        track = _make_track([100, 100, 200, 200])  # area = 10000
        bev = _make_bev_state([0.5, 0.5], cfg)
        # Detection 3x the area
        det = Detection3D(
            detection_2d=Detection2D(bbox=(50, 50, 350, 350), confidence=0.9),
            depth_m=1.5, depth_valid=True,
        )
        decision = _make_decision()
        ok, reason = validate_rgbd_update_with_frame_size(
            track, bev, det, decision, cfg, 1920, 1080
        )
        assert not ok
        assert reason == REJECT_AREA_RATIO

    def test_area_too_small(self):
        cfg = _make_cfg(min_area_ratio=0.60)
        track = _make_track([100, 100, 200, 200])
        bev = _make_bev_state([0.5, 0.5], cfg)
        # Detection 10% of the area
        det = Detection3D(
            detection_2d=Detection2D(bbox=(140, 140, 160, 160), confidence=0.9),
            depth_m=1.5, depth_valid=True,
        )
        decision = _make_decision()
        ok, reason = validate_rgbd_update_with_frame_size(
            track, bev, det, decision, cfg, 1920, 1080
        )
        assert not ok
        assert reason == REJECT_AREA_RATIO


class TestRejectedUpdateDoesNotChangeVelocity:
    """If sanity rejects, the caller should not update velocity.

    This test verifies the gate returns False — the runner is responsible
    for not calling update.
    """

    def test_invalid_depth_rejects(self):
        cfg = _make_cfg()
        track = _make_track([100, 100, 200, 200])
        bev = _make_bev_state([0.5, 0.5], cfg)
        det = Detection3D(
            detection_2d=Detection2D(bbox=(110, 110, 210, 210), confidence=0.9),
            depth_valid=False,
            invalid_reason="too_few_valid_pixels",
        )
        decision = _make_decision()
        ok, reason = validate_rgbd_update_with_frame_size(
            track, bev, det, decision, cfg, 1920, 1080
        )
        assert not ok
        assert reason == REJECT_INVALID_DEPTH


class TestExisting2DCLIHelpStillWorks:
    """Regression: CLI --help should work without error."""

    def test_cli_help_parses(self):
        from pig_behavior.tracking.cli import parse_args

        # --help calls sys.exit(0) which raises SystemExit
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--help"])
        assert exc_info.value.code == 0
