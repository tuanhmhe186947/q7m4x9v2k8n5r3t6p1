"""Tests for BEV Kalman Filter predict/update cycle."""

from __future__ import annotations

import numpy as np

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.kalman import (
    bev_position,
    bev_velocity,
    create_bev_kalman,
    predict_bev,
    update_bev,
)


def _make_cfg(**overrides) -> RGBDTrackingConfig:
    from pathlib import Path

    from pig_behavior.tracking.config import TrackingConfig

    defaults = dict(
        tracking_config=TrackingConfig(),
        depth_video_path=Path("d.mp4"),
        depth_scale_path=Path("s.npy"),
        inverse_intrinsic_path=Path("k.npy"),
        rotation_path=Path("r.npy"),
        kf_process_std=0.10,
        kf_measurement_std=0.05,
    )
    defaults.update(overrides)
    return RGBDTrackingConfig(**defaults)


class TestBEVKalmanPredictUpdate:
    """Core Kalman filter predict/update cycle."""

    def test_initial_position(self):
        cfg = _make_cfg()
        kf = create_bev_kalman(np.array([1.0, 2.0]), cfg)
        pos = bev_position(kf)
        np.testing.assert_allclose(pos, [1.0, 2.0])

    def test_predict_moves_forward(self):
        cfg = _make_cfg()
        kf = create_bev_kalman(np.array([1.0, 2.0]), cfg)
        # Give it some velocity via update
        update_bev(kf, np.array([1.1, 2.1]))
        predict_bev(kf)
        pos = bev_position(kf)
        # Should have moved slightly from (1.1, 2.1)
        assert pos[0] > 1.0
        assert pos[1] > 2.0

    def test_update_corrects(self):
        cfg = _make_cfg()
        kf = create_bev_kalman(np.array([0.0, 0.0]), cfg)
        predict_bev(kf)
        pos_after_update = update_bev(kf, np.array([1.0, 1.0]))
        # After update, position should move towards measurement
        assert pos_after_update[0] > 0.5
        assert pos_after_update[1] > 0.5

    def test_velocity_after_updates(self):
        cfg = _make_cfg()
        kf = create_bev_kalman(np.array([0.0, 0.0]), cfg)
        for i in range(10):
            predict_bev(kf)
            update_bev(kf, np.array([i * 0.1, i * 0.2]))
        vel = bev_velocity(kf)
        # Should have learned a positive velocity
        assert vel[0] > 0
        assert vel[1] > 0

    def test_predict_only_does_not_update_covariance_small(self):
        cfg = _make_cfg()
        kf = create_bev_kalman(np.array([1.0, 1.0]), cfg)
        p_before = kf.P.copy()
        predict_bev(kf)
        # Prediction increases uncertainty
        assert np.trace(kf.P) > np.trace(p_before)

    def test_multiple_predicts_increase_uncertainty(self):
        cfg = _make_cfg()
        kf = create_bev_kalman(np.array([0.0, 0.0]), cfg)
        traces = []
        for _ in range(5):
            predict_bev(kf)
            traces.append(np.trace(kf.P))
        # Each predict should increase covariance trace
        for i in range(1, len(traces)):
            assert traces[i] > traces[i - 1]
