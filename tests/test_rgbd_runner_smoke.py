"""Smoke test for the RGB-D runner (no real video/depth data)."""

from __future__ import annotations

import pytest


class TestRunnerImportable:
    """Verify the runner module can be imported without side effects."""

    def test_import(self):
        from pig_behavior.tracking.rgbd.runner_rgbd import run_rgbd_tracking
        assert callable(run_rgbd_tracking)


class TestSchemaImports:
    """All RGB-D schemas should be importable."""

    def test_all_schemas(self):
        from pig_behavior.tracking.rgbd.schemas import (
            BEVTrackState,
            Detection2D,
            Detection3D,
        )
        # Smoke: instantiate with defaults
        d = Detection2D(bbox=(0, 0, 1, 1), confidence=0.5)
        assert d.confidence == 0.5

        d3 = Detection3D(detection_2d=d)
        assert d3.depth_valid is False

        bev = BEVTrackState(
            fixed_id=1,
            kf=None,
            bev_position=__import__("numpy").zeros(2),
            bev_velocity=__import__("numpy").zeros(2),
        )
        assert bev.state == "tentative"


class TestConfigValidation:
    """RGBDTrackingConfig validation catches bad values."""

    def test_bad_bev_axes(self):
        import tempfile
        from pathlib import Path

        import numpy as np

        from pig_behavior.tracking.config import TrackingConfig
        from pig_behavior.tracking.rgbd.config import (
            RGBDTrackingConfig,
            validate_rgbd_config,
        )

        # Create temporary files for required paths
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "depth.mp4").touch()
            np.save(tmp_path / "depth_scale.npy", np.array([0.001]))
            np.save(tmp_path / "k_inv.npy", np.eye(3))
            np.save(tmp_path / "rot.npy", np.eye(3))

            cfg = RGBDTrackingConfig(
                tracking_config=TrackingConfig(),
                depth_video_path=tmp_path / "depth.mp4",
                depth_scale_path=tmp_path / "depth_scale.npy",
                inverse_intrinsic_path=tmp_path / "k_inv.npy",
                rotation_path=tmp_path / "rot.npy",
                bev_axes=(0, 0),  # invalid: same axis
            )
            with pytest.raises(ValueError, match="bev_axes"):
                validate_rgbd_config(cfg)


class TestCLIRGBDRequiresFiles:
    """If --rgbd is set but required files are missing, error should be clear."""

    def test_missing_depth_video(self):
        from pig_behavior.tracking.cli import parse_args

        args = parse_args(["--rgbd"])
        assert args.rgbd is True
        assert args.depth_video is None
