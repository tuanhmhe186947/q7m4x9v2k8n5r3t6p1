"""Tests for BEV association matching."""

from __future__ import annotations

import numpy as np

from pig_behavior.tracking.rgbd.association_bev import match_bev_tracks
from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.kalman import create_bev_kalman
from pig_behavior.tracking.rgbd.schemas import (
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
    )
    defaults.update(overrides)
    return RGBDTrackingConfig(**defaults)


def _make_track(fid: int, box: list[float]) -> FixedTrack:
    track = FixedTrack(
        fixed_id=fid,
        last_box=np.array(box, dtype=np.float32),
    )
    track.ever_detected = True
    return track


def _make_bev_state(fid: int, pos: list[float], cfg: RGBDTrackingConfig) -> BEVTrackState:
    kf = create_bev_kalman(np.array(pos, dtype=np.float64), cfg)
    return BEVTrackState(
        fixed_id=fid,
        kf=kf,
        bev_position=np.array(pos, dtype=np.float64),
        bev_velocity=np.zeros(2, dtype=np.float64),
        state="confirmed",
    )


def _make_det3d(bev: list[float], conf: float = 0.9) -> Detection3D:
    return Detection3D(
        detection_2d=Detection2D(bbox=(0, 0, 50, 50), confidence=conf),
        bev_xy=np.array(bev, dtype=np.float64),
        depth_m=1.5,
        depth_valid=True,
    )


class TestBEVAssociationPrefersNearestTrack:
    """The association should match each track to the nearest BEV detection."""

    def test_nearest_wins(self):
        cfg = _make_cfg()
        tracks = {
            1: _make_track(1, [10, 10, 60, 60]),
            2: _make_track(2, [100, 100, 150, 150]),
        }
        bev_states = {
            1: _make_bev_state(1, [0.0, 0.0], cfg),
            2: _make_bev_state(2, [1.0, 1.0], cfg),
        }
        dets = [
            _make_det3d([0.05, 0.05]),  # near track 1
            _make_det3d([0.95, 0.95]),  # near track 2
        ]
        occ_flags = {0: False, 1: False}
        assignments, decisions = match_bev_tracks(
            tracks, bev_states, dets, occ_flags, 0, cfg
        )
        assert assignments.get(1) == 0  # track 1 → det 0
        assert assignments.get(2) == 1  # track 2 → det 1


class TestBEVAssociationRejectsFarDetection:
    """Detections beyond the BEV gate should be rejected."""

    def test_far_rejected(self):
        cfg = _make_cfg(bev_association_gate_m=0.10)
        tracks = {1: _make_track(1, [10, 10, 60, 60])}
        bev_states = {1: _make_bev_state(1, [0.0, 0.0], cfg)}
        dets = [_make_det3d([5.0, 5.0])]  # 7.07m away, gate is 0.10m
        occ_flags = {0: False}
        assignments, decisions = match_bev_tracks(
            tracks, bev_states, dets, occ_flags, 0, cfg
        )
        assert 1 not in assignments
        # Check there's a rejection decision
        rejected = [d for d in decisions if not d.accepted]
        assert len(rejected) > 0


class TestAmbiguousAssignment:
    """When score margin is tiny, the decision should still be produced."""

    def test_two_equidistant_detections(self):
        cfg = _make_cfg(min_score_margin=0.05, bev_association_gate_m=1.0)
        tracks = {1: _make_track(1, [10, 10, 60, 60])}
        bev_states = {1: _make_bev_state(1, [0.5, 0.5], cfg)}
        dets = [
            _make_det3d([0.55, 0.55]),  # very close
            _make_det3d([0.56, 0.56]),  # almost identical distance
        ]
        occ_flags = {0: False, 1: False}
        assignments, decisions = match_bev_tracks(
            tracks, bev_states, dets, occ_flags, 0, cfg
        )
        # Should match to one; the decision should record score_margin
        matched = [d for d in decisions if d.accepted]
        assert len(matched) >= 1
        if matched[0].score_margin is not None:
            assert matched[0].score_margin >= 0
