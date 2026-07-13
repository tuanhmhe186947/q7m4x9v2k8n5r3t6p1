from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.spatial_sequence_export import (
    export_spatial_sequences,
)


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["track-a|win=2|0-1"],
            "object_track_key": ["track-a"],
            "window_start_frame": [0],
            "window_end_frame": [1],
            "window_length_frames": [2],
        }
    )


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "object_track_key": ["track-a", "track-a"],
            "frame_index": [0, 1],
            "cx_n": [0.25, 0.30],
            "cy_n": [0.40, 0.42],
            "bbox_valid": [True, True],
        }
    )


def test_spatial_export_audits_complete_alignment_without_row_loss() -> None:
    result = export_spatial_sequences(_windows(), _frames())

    assert result.audit["input_window_rows"] == 1
    assert result.audit["aligned_window_rows"] == 1
    assert result.audit["input_frame_rows"] == 2
    assert result.audit["aligned_frame_rows"] == 2
    assert result.audit["invalid_frame_alignment_rows"] == 0
    assert result.audit["duplicate_frame_alignment_rows"] == 0
    assert result.audit["observed_frame_slots"] == 2


@pytest.mark.parametrize("invalid_frame_index", [None, "bad", 0.5])
def test_spatial_export_rejects_invalid_frame_alignment_rows(
    invalid_frame_index: object,
) -> None:
    frames = _frames()
    frames["frame_index"] = frames["frame_index"].astype(object)
    frames.loc[1, "frame_index"] = invalid_frame_index

    with pytest.raises(
        ValueError,
        match=r"Frame alignment contract failed: invalid_rows=1",
    ):
        export_spatial_sequences(_windows(), frames)


def test_spatial_export_rejects_missing_track_key_instead_of_dropping_row() -> None:
    frames = _frames()
    frames.loc[1, "object_track_key"] = pd.NA

    with pytest.raises(
        ValueError,
        match=r"Frame alignment contract failed: invalid_rows=1",
    ):
        export_spatial_sequences(_windows(), frames)


def test_spatial_export_rejects_duplicate_track_frame_alignment() -> None:
    frames = _frames()
    frames.loc[1, "frame_index"] = 0

    with pytest.raises(
        ValueError,
        match=r"duplicate_frame_alignment_rows=2",
    ):
        export_spatial_sequences(_windows(), frames)


def test_spatial_export_rejects_inconsistent_window_span() -> None:
    windows = _windows()
    windows.loc[0, "window_length_frames"] = 3

    with pytest.raises(
        ValueError,
        match=r"Window alignment contract failed: invalid_rows=1",
    ):
        export_spatial_sequences(windows, _frames())


def test_spatial_motion_is_rebased_inside_each_window() -> None:
    frames = pd.DataFrame(
        {
            "object_track_key": ["track-a"] * 3,
            "frame_index": [0, 1, 2],
            "cx_n": [0.0, 0.5, 0.6],
            "cy_n": [0.0, 0.0, 0.0],
            "bw_n": [0.2, 0.2, 0.2],
            "bh_n": [0.1, 0.1, 0.1],
            "area_n": [0.02, 0.02, 0.02],
            "aspect_ratio": [2.0, 2.0, 2.0],
            "delta_cx_n": [0.0, 0.5, 0.1],
            "delta_cy_n": [0.0, 0.0, 0.0],
            "speed_n_per_frame": [0.0, 0.5, 0.1],
            "speed_n_per_sec": [0.0, 15.0, 3.0],
            "abs_accel_n_per_frame2": [0.0, 0.5, 0.4],
            "abs_direction_change_rad": [0.0, 1.0, 1.0],
            "bbox_valid": [True, True, True],
        }
    )
    windows = _windows()
    windows["window_id"] = "track-a|win=2|1-2"
    windows["window_start_frame"] = 1
    windows["window_end_frame"] = 2

    result = export_spatial_sequences(windows, frames)
    names = result.feature_names["motion_delta"]
    motion = result.arrays["motion_delta"][0]

    assert motion[0, names.index("delta_cx_n")] == 0.0
    assert motion[0, names.index("speed_n_per_frame")] == 0.0
    assert motion[0, names.index("speed_n_per_sec")] == 0.0
    assert motion[1, names.index("delta_cx_n")] == pytest.approx(0.1)
    assert motion[1, names.index("speed_n_per_frame")] == pytest.approx(0.1)
    assert motion[1, names.index("speed_n_per_sec")] == 3.0
    assert motion[1, names.index("abs_accel_n_per_frame2")] == 0.0
    assert motion[1, names.index("abs_direction_change_rad")] == 0.0
    assert result.audit["motion_rebased_windows"] == 1


def test_spatial_social_motion_is_rebased_inside_each_window() -> None:
    frames = pd.DataFrame(
        {
            "object_track_key": ["track-a"] * 3,
            "frame_index": [0, 1, 2],
            "cx_n": [0.0, 0.5, 0.6],
            "cy_n": [0.0, 0.0, 0.0],
            "bw_n": [0.2, 0.2, 0.2],
            "bh_n": [0.1, 0.1, 0.1],
            "nearest_pig_id": ["ID_2"] * 3,
            "roi_feeder_available": [True, True, True],
            "roi_drinker_available": [False, False, False],
            "roi_toy_available": [True, True, True],
            "nearest_dist_n": [0.5, 0.2, 0.1],
            "nearest_dist_delta": [0.0, -0.3, -0.1],
            "approach_speed_n_per_frame": [0.0, 0.3, 0.1],
            "separation_speed_n_per_frame": [0.0, 0.0, 0.0],
            "pair_contact_with_nearest": [True, True, True],
            "social_density_near_count": [0.0, 0.0, 0.0],
            "aggression_score_proxy": [0.0, 99.0, 99.0],
            "speed_n_per_frame": [0.0, 0.5, 0.1],
            "bbox_valid": [True, True, True],
        }
    )
    windows = _windows()
    windows["window_id"] = "track-a|win=2|1-2"
    windows["window_start_frame"] = 1
    windows["window_end_frame"] = 2

    result = export_spatial_sequences(windows, frames)
    names = result.feature_names["social_relation"]
    social = result.arrays["social_relation"][0]
    quality_names = result.feature_names["quality_mask"]
    quality = result.arrays["quality_mask"][0]

    assert social[0, names.index("nearest_dist_delta")] == 0.0
    assert social[0, names.index("approach_speed_n_per_frame")] == 0.0
    assert social[0, names.index("aggression_score_proxy")] == 0.0
    assert social[1, names.index("nearest_dist_delta")] == pytest.approx(-0.1)
    assert social[1, names.index("approach_speed_n_per_frame")] == pytest.approx(
        0.1
    )
    assert social[1, names.index("aggression_score_proxy")] == pytest.approx(
        0.2
    )
    assert quality[0, quality_names.index("roi_feeder_available")] == 1.0
    assert quality[0, quality_names.index("roi_drinker_available")] == 0.0
    assert quality[0, quality_names.index("roi_toy_available")] == 1.0
    assert quality[0, quality_names.index("social_neighbor_available")] == 1.0
    selected = {
        feature
        for group_features in result.feature_names.values()
        for feature in group_features
    }
    assert "nearest_pig_id" not in selected
    assert result.audit["social_rebased_windows"] == 1
