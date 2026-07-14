from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets import (
    image_sequence_dataset as image_sequence_dataset_module,
)
from pig_behavior.classification_v2.datasets.image_context_index import (
    MANDATORY_CVAT_MEDIA_BASENAME,
    audit_image_context_identifier_contract,
    audit_mandatory_cvat_video_case,
    build_image_context_index,
)
from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)
from pig_behavior.classification_v2.datasets.legacy_unreviewed_development import (
    LEGACY_DEVELOPMENT_SCOPE,
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
            "resolved_media_path": [f"C:/videos/{media_basename}"] * len(frame_indices),
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
    identifier_audit = audit_image_context_identifier_contract(
        result.frame_manifest,
        result.window_manifest,
    )
    assert identifier_audit["status"] == "v2"
    assert identifier_audit["valid"] is True


def test_context_index_uses_legacy_video_bbox_when_crop_is_missing(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "legacy.mp4"
    video_path.write_bytes(b"video-placeholder")
    frames = _frames()
    frames["source_video_path"] = str(video_path)
    frames[["x1", "y1", "x2", "y2"]] = [1.0, 2.0, 9.0, 12.0]

    result = _build(frames, _windows(), tmp_path)

    assert result.audit["frame_loadable_count"] == 2
    assert result.audit["image_context_source_counts"] == {
        "legacy_video_bbox": 2,
    }
    assert result.frame_manifest["resolved_media_path"].eq(str(video_path)).all()
    assert result.frame_manifest["image_context_loadable"].all()


def test_context_index_propagates_explicit_lineage_claim(tmp_path: Path) -> None:
    frames = _frames()
    windows = _windows()
    frames["lineage_scope"] = LEGACY_DEVELOPMENT_SCOPE
    frames["human_review_complete"] = False
    windows["lineage_scope"] = LEGACY_DEVELOPMENT_SCOPE
    windows["human_review_complete"] = False

    result = _build(frames, windows, tmp_path)

    assert set(result.frame_manifest["lineage_scope"]) == {
        LEGACY_DEVELOPMENT_SCOPE
    }
    assert not result.frame_manifest["human_review_complete"].any()
    assert set(result.window_manifest["lineage_scope"]) == {
        LEGACY_DEVELOPMENT_SCOPE
    }
    assert not result.window_manifest["human_review_complete"].any()
    assert result.audit["lineage_scope"] == LEGACY_DEVELOPMENT_SCOPE
    assert result.audit["human_review_complete"] is False


def test_image_dataset_dispatches_legacy_video_bbox_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = object.__new__(ClassificationV2ImageSequenceDataset)
    dataset.config = ImageSequenceDatasetConfig(image_size=8)
    expected = np.zeros((3, 8, 8), dtype=np.float32)
    monkeypatch.setattr(dataset, "_load_video_bbox_crop", lambda _frame: expected)

    observed = dataset._load_frame_image(
        {
            "source_type": "legacy_recovered",
            "image_context_source": "legacy_video_bbox",
            "resolved_media_path": "legacy.mp4",
        }
    )

    assert observed is expected


def test_image_dataset_bounds_and_releases_video_captures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            self.released = True

    created: dict[str, FakeCapture] = {}

    def open_capture(path: str) -> FakeCapture:
        capture = FakeCapture()
        created[path] = capture
        return capture

    monkeypatch.setattr(
        image_sequence_dataset_module,
        "cv2",
        SimpleNamespace(VideoCapture=open_capture),
    )
    dataset = object.__new__(ClassificationV2ImageSequenceDataset)
    dataset.config = ImageSequenceDatasetConfig(video_capture_cache_size=1)
    dataset._capture_cache = OrderedDict()
    dataset._capture_next_frame = {}
    dataset._decoded_video_frame = {}
    dataset.video_capture_open_count = 0
    dataset.video_capture_eviction_count = 0
    dataset.peak_open_video_captures = 0

    first = dataset._get_video_capture("first.mp4")
    dataset._capture_next_frame["first.mp4"] = 4
    dataset._decoded_video_frame["first.mp4"] = (
        3,
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    second = dataset._get_video_capture("second.mp4")

    assert first is created["first.mp4"]
    assert second is created["second.mp4"]
    assert created["first.mp4"].released is True
    assert list(dataset._capture_cache) == ["second.mp4"]
    assert "first.mp4" not in dataset._capture_next_frame
    assert "first.mp4" not in dataset._decoded_video_frame
    assert dataset.video_capture_audit() == {
        "video_capture_cache_size": 1,
        "active_video_captures": 1,
        "peak_open_video_captures": 1,
        "video_capture_open_count": 2,
        "video_capture_eviction_count": 1,
    }


def test_context_index_preserves_input_window_order(tmp_path: Path) -> None:
    windows = pd.DataFrame(
        {
            "window_id": ["track-a|win=1|1-1", "track-a|win=1|0-0"],
            "source_type": ["legacy_recovered", "legacy_recovered"],
            "object_track_key": ["track-a", "track-a"],
            "window_start_frame": [1, 0],
            "window_end_frame": [1, 0],
            "window_length_frames": [1, 1],
        }
    )

    result = _build(_frames(), windows, tmp_path)

    assert result.window_manifest["window_id"].tolist() == windows["window_id"].tolist()
    assert result.audit["window_order_preserved"] is True


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
    audit = audit_mandatory_cvat_video_case(_mandatory_cvat_case(MANDATORY_CVAT_MEDIA_BASENAME))

    assert audit["ok"] is True
    assert audit["rows"] == 6
    assert audit["observed_frame_indices"] == list(range(678, 684))
    assert audit["resolved_media_basenames"] == [MANDATORY_CVAT_MEDIA_BASENAME]


def test_mandatory_cvat_video_case_rejects_loadable_wrong_basename() -> None:
    audit = audit_mandatory_cvat_video_case(_mandatory_cvat_case("Pigs291119_000231.mp4"))

    assert audit["ok"] is False
    assert audit["unloadable_rows"] == 0
    assert any("resolved_media_basename_mismatch" in error for error in audit["errors"])


def test_mandatory_cvat_video_case_rejects_incomplete_frame_set() -> None:
    frames = _mandatory_cvat_case(MANDATORY_CVAT_MEDIA_BASENAME).iloc[:-1]

    audit = audit_mandatory_cvat_video_case(frames)

    assert audit["ok"] is False
    assert any("row_count_mismatch" in error for error in audit["errors"])
    assert any("frame_set_mismatch" in error for error in audit["errors"])


def test_image_context_identifier_audit_reads_explicit_legacy_manifest() -> None:
    frames = pd.DataFrame({"frame_uid": ["old-scene"]})
    windows = pd.DataFrame({"frame_uid_sequence": ["old-scene"]})

    audit = audit_image_context_identifier_contract(frames, windows)

    assert audit["status"] == "legacy_compatible"
    assert audit["valid"] is True


def test_image_context_identifier_audit_rejects_partial_v2_manifest() -> None:
    frames = pd.DataFrame(
        {
            "scene_frame_uid": ["scene-0"],
            "frame_uid": ["object-0"],
        }
    )
    windows = pd.DataFrame(
        {
            "scene_frame_uid_sequence": ["scene-0"],
            "frame_uid_sequence": ["object-0"],
        }
    )

    audit = audit_image_context_identifier_contract(frames, windows)

    assert audit["status"] == "invalid_v2"
    assert audit["valid"] is False
    assert audit["errors"] == ["partial_identifier_v2_without_version"]
