from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.image_context_index import (
    MANDATORY_CVAT_MEDIA_BASENAME,
    audit_mandatory_cvat_video_case,
    build_image_context_index,
)


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_uid": ["f0", "f1"],
            "source_type": ["legacy_recovered", "legacy_recovered"],
            "dataset_id": ["fixture", "fixture"],
            "video_key": ["video-a", "video-a"],
            "object_track_key": ["track-a", "track-a"],
            "track_id": ["track-a", "track-a"],
            "pig_id": ["ID_1", "ID_1"],
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


def _mandatory_cvat_case(media_basename: str) -> pd.DataFrame:
    frame_indices = list(range(678, 684))
    return pd.DataFrame(
        {
            "video_key": ["Pigs291119_000231"] * len(frame_indices),
            "pig_id": ["ID_4"] * len(frame_indices),
            "frame_index": frame_indices,
            "resolved_media_path": [f"C:/videos/{media_basename}"]
            * len(frame_indices),
            "image_context_loadable": [True] * len(frame_indices),
        }
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
    assert result.frame_manifest["frame_uid"].is_unique
    assert "scene_frame_uid_sequence" in result.window_manifest


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


def test_mandatory_cvat_video_case_accepts_exact_resolved_interval() -> None:
    audit = audit_mandatory_cvat_video_case(
        _mandatory_cvat_case(MANDATORY_CVAT_MEDIA_BASENAME)
    )

    assert audit["ok"] is True
    assert audit["rows"] == 6
    assert audit["observed_frame_indices"] == list(range(678, 684))
    assert audit["resolved_media_basenames"] == [MANDATORY_CVAT_MEDIA_BASENAME]


def test_mandatory_cvat_video_case_rejects_loadable_wrong_basename() -> None:
    audit = audit_mandatory_cvat_video_case(
        _mandatory_cvat_case("Pigs291119_000231.mp4")
    )

    assert audit["ok"] is False
    assert audit["unloadable_rows"] == 0
    assert any("resolved_media_basename_mismatch" in error for error in audit["errors"])


def test_mandatory_cvat_video_case_rejects_incomplete_frame_set() -> None:
    frames = _mandatory_cvat_case(MANDATORY_CVAT_MEDIA_BASENAME).iloc[:-1]

    audit = audit_mandatory_cvat_video_case(frames)

    assert audit["ok"] is False
    assert any("row_count_mismatch" in error for error in audit["errors"])
    assert any("frame_set_mismatch" in error for error in audit["errors"])
