"""Tests for depth-based occlusion inference."""

from __future__ import annotations

from pig_behavior.tracking.rgbd.config import RGBDTrackingConfig
from pig_behavior.tracking.rgbd.occlusion import infer_occlusions
from pig_behavior.tracking.rgbd.schemas import Detection2D, Detection3D


def _make_cfg(**overrides) -> RGBDTrackingConfig:
    from pathlib import Path

    from pig_behavior.tracking.config import TrackingConfig

    defaults = dict(
        tracking_config=TrackingConfig(),
        depth_video_path=Path("d.mp4"),
        depth_scale_path=Path("s.npy"),
        inverse_intrinsic_path=Path("k.npy"),
        rotation_path=Path("r.npy"),
        occlusion_iou_threshold=0.30,
        larger_depth_is_farther=True,
    )
    defaults.update(overrides)
    return RGBDTrackingConfig(**defaults)


def _make_det3d(
    bbox: tuple[float, ...], depth: float, valid: bool = True
) -> Detection3D:
    return Detection3D(
        detection_2d=Detection2D(bbox=bbox, confidence=0.9),
        depth_m=depth,
        depth_valid=valid,
    )


class TestOcclusionMarksFartherDepth:
    """The farther detection in an overlapping pair should be occluded."""

    def test_overlap_deeper_is_occluded(self):
        cfg = _make_cfg()
        dets = [
            _make_det3d((0, 0, 100, 100), 1.5),   # closer
            _make_det3d((30, 30, 130, 130), 2.5),  # farther, high overlap
        ]
        flags = infer_occlusions(dets, cfg)
        assert flags[0] is False  # closer is not occluded
        assert flags[1] is True   # farther is occluded

    def test_no_overlap_no_occlusion(self):
        cfg = _make_cfg()
        dets = [
            _make_det3d((0, 0, 50, 50), 1.5),
            _make_det3d((200, 200, 300, 300), 2.5),
        ]
        flags = infer_occlusions(dets, cfg)
        assert all(not v for v in flags.values())

    def test_invalid_depth_excluded(self):
        cfg = _make_cfg()
        dets = [
            _make_det3d((0, 0, 100, 100), 1.5, valid=True),
            _make_det3d((50, 50, 150, 150), 2.5, valid=False),
        ]
        flags = infer_occlusions(dets, cfg)
        # Can't determine ordering without valid depth → no occlusion
        assert flags[0] is False
        assert flags[1] is False

    def test_smaller_depth_is_farther(self):
        """Reversed depth convention: smaller = farther."""
        cfg = _make_cfg(larger_depth_is_farther=False)
        dets = [
            _make_det3d((0, 0, 100, 100), 0.5),   # farther (smaller depth)
            _make_det3d((30, 30, 130, 130), 2.0),  # closer, high overlap
        ]
        flags = infer_occlusions(dets, cfg)
        assert flags[0] is True   # smaller depth → farther → occluded
        assert flags[1] is False
