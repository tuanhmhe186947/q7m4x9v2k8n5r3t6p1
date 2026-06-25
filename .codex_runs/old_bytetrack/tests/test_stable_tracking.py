"""Tests for the Stable CVAT Tracking Pipeline and its subcomponents."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.stabilization.bbox_smoothing import smooth_trajectory_boxes
from pig_behavior.tracking.stabilization.config import AnnotationStableConfig
from pig_behavior.tracking.stabilization.cvat_export import write_stable_cvat_xml
from pig_behavior.tracking.stabilization.swap_detection import (
    detect_and_optionally_fix_swaps,
)
from pig_behavior.tracking.stabilization.tracklet_stitching import (
    StableTrackletRecord,
    stitch_tracklets,
)


def _make_config() -> AnnotationStableConfig:
    return AnnotationStableConfig(
        tracking_config=TrackingConfig(),
        rgbd_config=None,
    )


class TestStableTrackingComponents:
    """Tests the core components of the stable tracking pipeline."""

    def test_bbox_smoothing_median(self):
        config = _make_config()
        config.smooth_bbox = True
        config.smooth_method = "median"
        config.smooth_bbox_window = 3
        config.max_smoothing_shift_px = 100

        # Create a trajectory with a sudden single-frame jitter
        boxes = np.array(
            [
                [100, 100, 200, 200],
                [101, 101, 201, 201],
                [150, 150, 250, 250],  # Jitter frame
                [102, 102, 202, 202],
                [103, 103, 203, 203],
            ],
            dtype=np.float32,
        )

        smoothed = smooth_trajectory_boxes(boxes, config, 1920, 1080)

        # The jitter frame (index 2) should be smoothed out by the median filter
        assert abs(smoothed[2][0] - 102) < 10
        assert abs(smoothed[2][1] - 102) < 10

    def test_bbox_smoothing_clamp_shift(self):
        config = _make_config()
        config.smooth_bbox = True
        config.smooth_method = "median"
        config.max_smoothing_shift_px = 5  # very small shift allowed

        # Big difference
        boxes = np.array(
            [
                [100, 100, 200, 200],
                [150, 150, 250, 250],
                [150, 150, 250, 250],
                [150, 150, 250, 250],
            ],
            dtype=np.float32,
        )

        smoothed = smooth_trajectory_boxes(boxes, config, 1920, 1080)

        # Center shift from original at index 0 (which has median box [150,150,250,250])
        # should be clamped to max_smoothing_shift_px
        dx = (smoothed[0][0] + smoothed[0][2]) / 2.0 - 150
        dy = (smoothed[0][1] + smoothed[0][3]) / 2.0 - 150
        shift = np.hypot(dx, dy)
        assert shift <= 5.01

    def test_tracklet_stitching(self):
        config = _make_config()
        config.stitch_max_gap = 10
        config.stitch_max_center_distance_norm = 0.2

        # Create two tracklets that end and start near each other
        t1 = StableTrackletRecord(
            tracklet_id=1,
            fixed_id=1,
            start_frame=0,
            end_frame=5,
            bbox_sequence=[np.array([100, 100, 200, 200])] * 6,
            center_sequence=[(150, 150)] * 6,
            area_sequence=[10000.0] * 6,
            confidence_sequence=[0.9] * 6,
            hist_summary=np.ones(1024),
            depth_valid_ratio=0.0,
            bev_valid_ratio=0.0,
            mean_confidence=0.9,
            length=6,
        )

        t2 = StableTrackletRecord(
            tracklet_id=2,
            fixed_id=2,
            start_frame=8,  # Gap of 2 frames
            end_frame=15,
            bbox_sequence=[np.array([102, 102, 202, 202])] * 8,
            center_sequence=[(152, 152)] * 8,
            area_sequence=[10000.0] * 8,
            confidence_sequence=[0.9] * 8,
            hist_summary=np.ones(1024),
            depth_valid_ratio=0.0,
            bev_valid_ratio=0.0,
            mean_confidence=0.9,
            length=8,
        )

        tracklet_to_stable_id, report = stitch_tracklets([t1, t2], config, 1920, 1080)

        # They should be stitched together (have same stable track ID)
        assert tracklet_to_stable_id[1] == tracklet_to_stable_id[2]
        assert len(report) > 0
        assert report[0].is_stitched is True

    def test_tracklet_stitching_reject_incompatible(self):
        config = _make_config()
        config.stitch_max_center_distance_norm = 0.05  # strict distance gate

        t1 = StableTrackletRecord(
            tracklet_id=1,
            fixed_id=1,
            start_frame=0,
            end_frame=5,
            bbox_sequence=[np.array([100, 100, 200, 200])],
            center_sequence=[(150, 150)],
            area_sequence=[10000.0],
            confidence_sequence=[0.9],
            hist_summary=np.ones(1024),
            depth_valid_ratio=0.0,
            bev_valid_ratio=0.0,
            mean_confidence=0.9,
            length=1,
        )

        t2 = StableTrackletRecord(
            tracklet_id=2,
            fixed_id=2,
            start_frame=8,
            end_frame=15,
            bbox_sequence=[np.array([500, 500, 600, 600])],  # far away
            center_sequence=[(550, 550)],
            area_sequence=[10000.0],
            confidence_sequence=[0.9],
            hist_summary=np.ones(1024),
            depth_valid_ratio=0.0,
            bev_valid_ratio=0.0,
            mean_confidence=0.9,
            length=1,
        )

        tracklet_to_stable_id, report = stitch_tracklets([t1, t2], config, 1920, 1080)

        # They should NOT be stitched together
        assert tracklet_to_stable_id[1] != tracklet_to_stable_id[2]
        assert len(report) > 0
        assert report[0].is_stitched is False

    def test_swap_detection_crossover(self):
        config = _make_config()
        config.detect_candidate_swaps = True
        config.swap_confidence_threshold = 0.5
        config.auto_fix_high_confidence_swaps = True
        config.swap_proximity_frames = 3

        # Create two tracks that cross and swap identities (by swapping appearances/locations)
        # Track 1: starts at 100, goes to 200, then goes to 100 (after crossing at frame 5)
        # Track 2: starts at 200, goes to 100, then goes to 200
        # If they swapped identities, after frame 5:
        # Track 1 should have stayed at 200, Track 2 should have stayed at 100.
        # Let's represent this.
        h1 = np.array([1.0, 0.0])
        h2 = np.array([0.0, 1.0])

        tracks = {
            1: {
                0: (np.array([100, 100, 200, 200]), h1),
                1: (np.array([120, 120, 220, 220]), h1),
                2: (np.array([140, 140, 240, 240]), h1),
                3: (np.array([150, 150, 250, 250]), h1),  # intersection
                4: (
                    np.array([140, 140, 240, 240]),
                    h2,
                ),  # swapped histograms and locations!
                5: (np.array([120, 120, 220, 220]), h2),
                6: (np.array([100, 100, 200, 200]), h2),
            },
            2: {
                0: (np.array([200, 200, 300, 300]), h2),
                1: (np.array([180, 180, 280, 280]), h2),
                2: (np.array([160, 160, 260, 260]), h2),
                3: (np.array([150, 150, 250, 250]), h2),  # intersection
                4: (np.array([160, 160, 260, 260]), h1),  # swapped
                5: (np.array([180, 180, 280, 280]), h1),
                6: (np.array([200, 200, 300, 300]), h1),
            },
        }

        updated_tracks, swap_candidates = detect_and_optionally_fix_swaps(tracks, config, 1920, 1080)

        assert len(swap_candidates) > 0
        # Post crossover, histograms should be fixed
        # At frame 6, updated Track 1 should have h1 histogram (which was originally on Track 2)
        np.testing.assert_allclose(updated_tracks[1][6][1], h1)
        np.testing.assert_allclose(updated_tracks[2][6][1], h2)

    def test_cvat_export_gap_handling(self, tmp_path):
        # Test that gap frames correctly write outside="1" in CVAT XML
        output_file = tmp_path / "cvat_test.xml"

        h = np.ones(1024)
        stable_tracks = {
            1: {
                0: (np.array([100, 100, 200, 200]), h, "lying", False),
                # Frame 1 is missing (gap)
                2: (np.array([105, 105, 205, 205]), h, "lying", False),
            }
        }

        write_stable_cvat_xml(
            output_file,
            stable_tracks,
            Path("dummy.mp4"),
            1920,
            1080,
            frame_count=3,
            expected_pigs=1,
        )

        assert output_file.exists()
        xml_content = output_file.read_text(encoding="utf-8")

        # Verify that frame 1 has outside="1"
        assert 'frame="1"' in xml_content
        assert 'outside="1"' in xml_content
        # Verify that frame 0 and 2 have outside="0"
        assert 'frame="0"' in xml_content
        assert 'frame="2"' in xml_content
        assert 'outside="0"' in xml_content
