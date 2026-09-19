"""Tests for depth extraction, calibration, and ambiguity detection."""

from __future__ import annotations

import numpy as np

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.depth import (
    DepthExtractionResult,
    compute_depth_confidence,
    depth_frame_to_meters,
    extract_depth_for_bbox,
)


def _make_cfg(**overrides) -> RGBDTrackingConfig:
    """Create a minimal RGBDTrackingConfig for testing."""
    from pathlib import Path

    from pig_behavior.tracking.config import TrackingConfig

    defaults = dict(
        tracking_config=TrackingConfig(),
        depth_video_path=Path("dummy_depth.mp4"),
        depth_scale_path=Path("dummy_scale.npy"),
        inverse_intrinsic_path=Path("dummy_kinv.npy"),
        rotation_path=Path("dummy_rot.npy"),
    )
    defaults.update(overrides)
    return RGBDTrackingConfig(**defaults)


class TestDepthScaleUnits:
    """Ensure depth_frame_to_meters converts correctly."""

    def test_scale_factor(self):
        raw = np.array([[1000, 2000], [3000, 0]], dtype=np.uint16)
        scale = 0.001  # millimetres → metres
        result = depth_frame_to_meters(raw, scale)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result[0, 0], 1.0)
        np.testing.assert_allclose(result[0, 1], 2.0)
        np.testing.assert_allclose(result[1, 0], 3.0)
        np.testing.assert_allclose(result[1, 1], 0.0)


class TestBackgroundFilterRemovesFloor:
    """Background filter should exclude floor pixels."""

    def test_floor_removed(self):
        cfg = _make_cfg(
            background_filter_m=0.10,
            min_valid_depth_pixels=1,
            depth_strategy="foreground_median",
        )
        # Depth frame: 2 metres everywhere
        depth = np.full((100, 100), 2.0, dtype=np.float64)
        # Background at 2 metres → should be filtered out
        bg = np.full((100, 100), 2.0, dtype=np.float64)
        # Place a pig at depth 1.5m in a region
        depth[40:60, 40:60] = 1.5

        bbox = (30.0, 30.0, 70.0, 70.0)
        result = extract_depth_for_bbox(depth, bbox, cfg, background_depth_m=bg)
        # Should pick up the 1.5m pixels, not the 2.0m floor
        assert result.depth_valid
        assert result.median_depth_m is not None
        assert abs(result.median_depth_m - 1.5) < 0.2


class TestInvalidDepthRejected:
    """NaN, inf, 0, out-of-range depths should be excluded."""

    def test_nan_excluded(self):
        cfg = _make_cfg(min_valid_depth_pixels=1, depth_strategy="foreground_median")
        depth = np.full((50, 50), np.nan, dtype=np.float64)
        depth[20:30, 20:30] = 1.5
        result = extract_depth_for_bbox(depth, (10, 10, 40, 40), cfg)
        assert result.depth_valid
        assert result.median_depth_m is not None
        assert abs(result.median_depth_m - 1.5) < 0.1

    def test_zero_excluded(self):
        cfg = _make_cfg(min_valid_depth_pixels=1, depth_strategy="foreground_median")
        depth = np.zeros((50, 50), dtype=np.float64)
        result = extract_depth_for_bbox(depth, (0, 0, 50, 50), cfg)
        assert not result.depth_valid

    def test_out_of_range_excluded(self):
        cfg = _make_cfg(
            min_depth_m=0.1, max_depth_m=5.0,
            min_valid_depth_pixels=1, depth_strategy="foreground_median",
        )
        depth = np.full((50, 50), 10.0, dtype=np.float64)
        result = extract_depth_for_bbox(depth, (0, 0, 50, 50), cfg)
        assert not result.depth_valid


class TestDepthAmbiguityDetectedByIQR:
    """High IQR should flag depth_ambiguous=True."""

    def test_ambiguous_large_iqr(self):
        cfg = _make_cfg(
            depth_ambiguity_iqr_m=0.10,
            min_valid_depth_pixels=5,
            depth_strategy="foreground_median",
        )
        # Create depth with large spread
        depth = np.ones((50, 50), dtype=np.float64)
        depth[:25, :] = 1.0
        depth[25:, :] = 2.5  # IQR ≈ 1.5m >> 0.10m threshold
        result = extract_depth_for_bbox(depth, (0, 0, 50, 50), cfg)
        assert result.depth_valid
        assert result.depth_ambiguous

    def test_not_ambiguous_small_iqr(self):
        cfg = _make_cfg(
            depth_ambiguity_iqr_m=0.50,
            min_valid_depth_pixels=5,
            depth_strategy="foreground_median",
        )
        depth = np.full((50, 50), 1.5, dtype=np.float64)
        depth += np.random.default_rng(42).normal(0, 0.01, (50, 50))
        result = extract_depth_for_bbox(depth, (0, 0, 50, 50), cfg)
        assert result.depth_valid
        assert not result.depth_ambiguous


class TestDepthConfidence:
    """compute_depth_confidence should return 0-1 range."""

    def test_invalid_zero(self):
        r = DepthExtractionResult(None, None, 0, False, False, "empty")
        assert compute_depth_confidence(r) == 0.0

    def test_valid_high(self):
        r = DepthExtractionResult(1.5, 0.02, 200, True, False, None)
        score = compute_depth_confidence(r)
        assert 0.5 < score <= 1.0
