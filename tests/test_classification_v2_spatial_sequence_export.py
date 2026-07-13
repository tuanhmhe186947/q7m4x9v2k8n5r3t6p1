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
