"""Tests for 2D→3D projection via RGBDProjector."""

from __future__ import annotations

import numpy as np
import pytest

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.depth import RGBDCalibration
from pig_behavior.tracking.rgbd.projector import RGBDProjector
from pig_behavior.tracking.rgbd.schemas import Detection2D


def _make_cfg(**overrides) -> RGBDTrackingConfig:
    from pathlib import Path

    from pig_behavior.tracking.config import TrackingConfig

    defaults = dict(
        tracking_config=TrackingConfig(),
        depth_video_path=Path("d.mp4"),
        depth_scale_path=Path("s.npy"),
        inverse_intrinsic_path=Path("k.npy"),
        rotation_path=Path("r.npy"),
    )
    defaults.update(overrides)
    return RGBDTrackingConfig(**defaults)


def _identity_calibration() -> RGBDCalibration:
    """K_inv = I, R = I → world = camera = depth * [u, v, 1]."""
    return RGBDCalibration(
        depth_scale=1.0,
        inverse_intrinsic=np.eye(3, dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
        background_depth_m=None,
    )


class TestProjectionIdentityMatrices:
    """With identity matrices, projection should be transparent."""

    def test_single_point(self):
        cfg = _make_cfg()
        proj = RGBDProjector(_identity_calibration(), cfg)
        cam, world = proj.project_single_point(100.0, 200.0, 2.0)
        # camera = 2.0 * [100, 200, 1] = [200, 400, 2]
        np.testing.assert_allclose(cam, [200.0, 400.0, 2.0])
        # R = I → world == camera
        np.testing.assert_allclose(world, cam)

    def test_bev_axes_default(self):
        cfg = _make_cfg(bev_axes=(0, 1))
        proj = RGBDProjector(_identity_calibration(), cfg)
        cam, world = proj.project_single_point(10.0, 20.0, 1.0)
        # world = [10, 20, 1], bev_axes=(0,1) → bev = [10, 20]
        assert world[0] == pytest.approx(10.0)
        assert world[1] == pytest.approx(20.0)

    def test_bev_axes_custom(self):
        cfg = _make_cfg(bev_axes=(0, 2))
        proj = RGBDProjector(_identity_calibration(), cfg)
        cam, world = proj.project_single_point(5.0, 10.0, 3.0)
        # world = 3 * [5, 10, 1] = [15, 30, 3]
        # bev_axes=(0,2) → bev = [15, 3]
        assert world[0] == pytest.approx(15.0)
        assert world[2] == pytest.approx(3.0)


class TestForegroundPointsProjectionMedian:
    """Multi-point projection should take component-wise median."""

    def test_batch_projection(self):
        cfg = _make_cfg()
        proj = RGBDProjector(_identity_calibration(), cfg)
        pixel_uv = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float64)
        depths = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        cam, world = proj.project_foreground_pixels(pixel_uv, depths)
        assert cam.shape == (3, 3)
        assert world.shape == (3, 3)
        # Check one point: depth=2, u=30, v=40 → cam = [60, 80, 2]
        np.testing.assert_allclose(cam[1], [60.0, 80.0, 2.0])

    def test_project_detection_foreground(self):
        cfg = _make_cfg(
            depth_strategy="foreground_points_median",
            min_valid_depth_pixels=2,
            min_depth_m=0.01,
            max_depth_m=100.0,
        )
        proj = RGBDProjector(_identity_calibration(), cfg)
        det = Detection2D(bbox=(0, 0, 10, 10), confidence=0.9)
        depth_frame = np.full((20, 20), 2.0, dtype=np.float64)
        d3 = proj.project_detection(det, depth_frame)
        assert d3.depth_valid
        assert d3.bev_xy is not None
        assert d3.world_xyz is not None


class TestProjectionWithRealMatrices:
    """Non-identity matrices should transform correctly."""

    def test_rotation_90(self):
        """90° rotation around Z → x'=y, y'=-x."""
        cfg = _make_cfg()
        k_inv = np.eye(3, dtype=np.float64)
        rot = np.array(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64
        )
        cal = RGBDCalibration(
            depth_scale=1.0,
            inverse_intrinsic=k_inv,
            rotation=rot,
            background_depth_m=None,
        )
        proj = RGBDProjector(cal, cfg)
        cam, world = proj.project_single_point(1.0, 0.0, 1.0)
        # cam = 1 * [1, 0, 1] = [1, 0, 1]
        # world = rot @ cam = [0, 1, 1]
        np.testing.assert_allclose(world, [0.0, 1.0, 1.0], atol=1e-10)
