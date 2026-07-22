from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.spatial_sequence_export import (
    export_spatial_sequences,
)


def _windows(*, start: int = 0, end: int = 1) -> pd.DataFrame:
    indices = list(range(start, end + 1))
    offsets = [value - start for value in indices]
    window_id = f"track-a|win={len(indices)}|{start}-{end}"
    return pd.DataFrame(
        {
            "window_id": [window_id],
            "object_track_key": ["track-a"],
            "window_start_frame": [start],
            "window_end_frame": [end],
            "window_length_frames": [len(indices)],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": [window_id],
            "view_type": [f"T{len(indices)}_contiguous"],
            "sampling_pattern": ["contiguous"],
            "selected_frame_offsets": [str(offsets).replace(" ", "")],
            "selected_frame_indices": [str(indices).replace(" ", "")],
            "selected_timestamps_seconds": [
                str([value / 30.0 for value in indices]).replace(" ", "")
            ],
            "pair_delta_frames": [
                str([1] * max(0, len(indices) - 1)).replace(" ", "")
            ],
            "pair_delta_seconds": [
                str([1.0 / 30.0] * max(0, len(indices) - 1)).replace(
                    " ",
                    "",
                )
            ],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "object_track_key": ["track-a", "track-a"],
            "frame_index": [0, 1],
            "timestamp_sec": [0.0, 1.0 / 30.0],
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
        match=r"Final-view selected-slot count mismatch",
    ):
        export_spatial_sequences(windows, _frames())


def test_spatial_motion_is_rebased_inside_each_window() -> None:
    frames = pd.DataFrame(
        {
            "object_track_key": ["track-a"] * 3,
            "frame_index": [0, 1, 2],
            "timestamp_sec": [0.0, 1.0 / 30.0, 2.0 / 30.0],
            "cx_n": [0.0, 0.5, 0.6],
            "cy_n": [0.0, 0.0, 0.0],
            "bw_n": [0.2, 0.2, 0.2],
            "bh_n": [0.1, 0.1, 0.1],
            "area_n": [0.02, 0.02, 0.02],
            "aspect_ratio": [2.0, 2.0, 2.0],
            "vx_n_per_second": [0.0, 15.0, 99.0],
            "vy_n_per_second": [0.0, 0.0, 99.0],
            "speed_n_per_second": [0.0, 15.0, 99.0],
            "abs_acceleration_n_per_second2": [0.0, 450.0, 99.0],
            "abs_direction_change_rad": [0.0, 1.0, 1.0],
            "bbox_valid": [True, True, True],
        }
    )
    windows = _windows(start=1, end=2)

    result = export_spatial_sequences(windows, frames)
    names = result.feature_names["motion_delta"]
    motion = result.arrays["motion_delta"][0]

    assert motion[0, names.index("vx_n_per_second")] == 0.0
    assert motion[0, names.index("speed_n_per_second")] == 0.0
    assert motion[1, names.index("vx_n_per_second")] == pytest.approx(3.0)
    assert motion[1, names.index("speed_n_per_second")] == pytest.approx(3.0)
    assert motion[1, names.index("abs_acceleration_n_per_second2")] == 0.0
    assert motion[1, names.index("abs_direction_change_rad")] == 0.0
    assert result.audit["motion_rebased_windows"] == 1


def test_spatial_social_motion_is_rebased_inside_each_window() -> None:
    frames = pd.DataFrame(
        {
            "object_track_key": ["track-a"] * 3,
            "frame_index": [0, 1, 2],
            "timestamp_sec": [0.0, 1.0 / 30.0, 2.0 / 30.0],
            "cx_n": [0.0, 0.5, 0.6],
            "cy_n": [0.0, 0.0, 0.0],
            "bw_n": [0.2, 0.2, 0.2],
            "bh_n": [0.1, 0.1, 0.1],
            "nearest_pig_id": ["ID_2"] * 3,
            "roi_feeder_available": [True, True, True],
            "roi_drinker_available": [False, False, False],
            "roi_toy_available": [True, True, True],
            "nearest_dist_n": [0.5, 0.2, 0.1],
            "partner_distance_delta_n": [0.0, -0.3, -99.0],
            "approach_speed_n_per_second": [0.0, 9.0, 99.0],
            "retreat_speed_n_per_second": [0.0, 0.0, 99.0],
            "pair_contact_with_nearest": [True, True, True],
            "social_density_near_count": [0.0, 0.0, 0.0],
            "aggression_score_proxy_per_second": [0.0, 99.0, 99.0],
            "speed_n_per_second": [0.0, 15.0, 99.0],
            "bbox_valid": [True, True, True],
        }
    )
    windows = _windows(start=1, end=2)

    result = export_spatial_sequences(windows, frames)
    names = result.feature_names["social_relation"]
    social = result.arrays["social_relation"][0]
    assert social[0, names.index("partner_distance_delta_n")] == 0.0
    assert social[0, names.index("approach_speed_n_per_second")] == 0.0
    assert social[0, names.index("aggression_score_proxy_per_second")] == 0.0
    assert social[1, names.index("partner_distance_delta_n")] == pytest.approx(
        -0.1
    )
    assert social[1, names.index("approach_speed_n_per_second")] == pytest.approx(
        3.0
    )
    assert social[1, names.index("aggression_score_proxy_per_second")] == (
        pytest.approx(6.0)
    )
    assert result.arrays["roi_validity_mask"][0, 0].tolist() == [1.0, 0.0, 1.0]
    assert result.arrays["social_validity_mask"][0, 0] == 1.0
    selected = {
        feature
        for group_features in result.feature_names.values()
        for feature in group_features
    }
    assert "nearest_pig_id" not in selected
    assert result.audit["social_rebased_windows"] == 1


def test_sparse_s6_at16_uses_exact_selected_frames_and_sparse_pairs() -> None:
    indices = [0, 3, 6, 9, 12, 15]
    frames = pd.DataFrame(
        {
            "object_track_key": ["track-a"] * 16,
            "frame_index": list(range(16)),
            "timestamp_sec": [value / 30.0 for value in range(16)],
            "cx_n": [value / 30.0 for value in range(16)],
            "cy_n": [0.0] * 16,
            "bw_n": [0.2] * 16,
            "bh_n": [0.1] * 16,
            "area_n": [0.02] * 16,
            "aspect_ratio": [2.0] * 16,
            "speed_n_per_second": [999.0] * 16,
            "bbox_valid": [True] * 16,
        }
    )
    window_id = "track-a|view=S6@16|0-15"
    windows = pd.DataFrame(
        {
            "window_id": [window_id],
            "object_track_key": ["track-a"],
            "window_start_frame": [0],
            "window_end_frame": [15],
            "window_length_frames": [6],
            "feature_computation_grain": ["FINAL_VIEW_FEATURES"],
            "pair_scope_key": [window_id],
            "view_type": ["S6@16"],
            "sampling_pattern": [
                "uniform_sparse_offsets_0_3_6_9_12_15"
            ],
            "selected_frame_offsets": ["[0,3,6,9,12,15]"],
            "selected_frame_indices": ["[0,3,6,9,12,15]"],
            "selected_timestamps_seconds": [
                "[0.0,0.1,0.2,0.3,0.4,0.5]"
            ],
            "pair_delta_frames": ["[3,3,3,3,3]"],
            "pair_delta_seconds": ["[0.1,0.1,0.1,0.1,0.1]"],
            "pair_recomputed_for_view": [True],
            "aggregate_recomputed_for_view": [True],
        }
    )

    result = export_spatial_sequences(windows, frames)
    names = result.feature_names["motion_delta"]
    speed = result.arrays["motion_delta"][0, :, names.index(
        "speed_n_per_second"
    )]

    assert result.arrays["frame_index_sequence"][0].tolist() == indices
    assert result.arrays["adjacent_motion_pair_mask"][0].tolist() == [
        0.0,
    ] * 6
    assert result.arrays["sparse_velocity_pair_mask"][0].tolist() == [
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert speed.tolist() == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
