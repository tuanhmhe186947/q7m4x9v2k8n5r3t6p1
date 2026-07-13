from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.image_context_index import (
    build_image_context_index,
)


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_uid": ["f0", "f1"],
            "source_type": ["legacy_recovered", "legacy_recovered"],
            "object_track_key": ["track-a", "track-a"],
            "frame_index": [0, 1],
            "crop_path": ["missing-0.jpg", "missing-1.jpg"],
        }
    )


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["track-a|win=2|0-1"],
            "source_type": ["legacy_recovered"],
            "object_track_key": ["track-a"],
            "window_start_frame": [0],
            "window_end_frame": [1],
            "window_length_frames": [2],
        }
    )


def _build(frames: pd.DataFrame, windows: pd.DataFrame, root: Path):
    return build_image_context_index(
        frames,
        windows,
        video_root=root / "videos",
        legacy_crop_root=root / "crops",
    )


def test_context_index_preserves_rows_even_when_media_is_missing(
    tmp_path: Path,
) -> None:
    result = _build(_frames(), _windows(), tmp_path)

    assert result.audit["input_frame_rows"] == 2
    assert result.audit["frame_rows"] == 2
    assert result.audit["frame_row_count_preserved"] is True
    assert result.audit["input_window_rows"] == 1
    assert result.audit["window_rows"] == 1
    assert result.audit["window_row_count_preserved"] is True
    assert result.audit["frame_unloadable_count"] == 2


def test_context_index_rejects_duplicate_track_frame_rows(tmp_path: Path) -> None:
    frames = _frames()
    frames.loc[1, "frame_index"] = 0

    with pytest.raises(ValueError, match="duplicate_frame_alignment_rows=2"):
        _build(frames, _windows(), tmp_path)


def test_context_index_rejects_null_track_key_instead_of_dropping_row(
    tmp_path: Path,
) -> None:
    frames = _frames()
    frames.loc[1, "object_track_key"] = pd.NA

    with pytest.raises(ValueError, match="Frame image-context contract failed"):
        _build(frames, _windows(), tmp_path)


def test_context_index_rejects_inconsistent_window_length(tmp_path: Path) -> None:
    windows = _windows()
    windows.loc[0, "window_length_frames"] = 3

    with pytest.raises(ValueError, match="Window image-context contract failed"):
        _build(_frames(), windows, tmp_path)
