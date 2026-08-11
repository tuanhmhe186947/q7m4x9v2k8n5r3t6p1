from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
)
from pig_behavior.classification_v2.training.cvat_media_resolution import (
    CvatMediaResolutionError,
    attach_registered_cvat_media_paths,
    runtime_media_path,
)


def _cvat_frames(
    source_video_path: object = "data/videos/Pigs291119_000302_30fps.mp4",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml"] * 2,
            "source_video_key": ["Pigs291119_000302_30fps"] * 2,
            "source_video_path": [source_video_path] * 2,
            "frame_index": [4, 5],
            "x1": [1.0, 2.0],
            "x2": [4.0, 5.0],
        }
    )


def test_cvat_registration_preserves_scientific_and_frame_identity() -> None:
    frames = _cvat_frames()
    resolved = attach_registered_cvat_media_paths(frames)
    assert resolved["scientific_media_id"].tolist() == [
        "Pigs291119_000302_30fps",
        "Pigs291119_000302_30fps",
    ]
    assert resolved["registered_relative_media_path"].tolist() == [
        "data/videos/Pigs291119_000302_30fps.mp4",
        "data/videos/Pigs291119_000302_30fps.mp4",
    ]
    assert resolved["frame_index"].tolist() == frames["frame_index"].tolist()
    assert resolved[["x1", "x2"]].equals(frames[["x1", "x2"]])


def test_cvat_registration_rejects_blank_and_ambiguous_authority_mappings() -> None:
    with pytest.raises(CvatMediaResolutionError, match="blank"):
        attach_registered_cvat_media_paths(_cvat_frames(""))
    ambiguous = _cvat_frames()
    ambiguous.loc[1, "source_video_path"] = "data/videos/other.mp4"
    with pytest.raises(CvatMediaResolutionError, match="non-unique"):
        attach_registered_cvat_media_paths(ambiguous)


def test_runtime_root_changes_only_host_realization(tmp_path: Path) -> None:
    relative = "data/videos/Pigs291119_000302_30fps.mp4"
    first = runtime_media_path(input_root=tmp_path / "first", registered_relative_path=relative)
    second = runtime_media_path(input_root=tmp_path / "second", registered_relative_path=relative)
    assert first.as_posix().endswith(relative)
    assert second.as_posix().endswith(relative)
    assert first != second


def test_dataset_rejects_opaque_cvat_context_key_before_opening_media() -> None:
    dataset = object.__new__(ClassificationV2ImageSequenceDataset)
    dataset.config = ImageSequenceDatasetConfig(media_root=Path("input-root"))
    with pytest.raises(ValueError, match="opaque CVAT"):
        dataset._resolve_media_path(
            {
                "source_type": "cvat_tracking_xml",
                "resolved_media_path": "reviewed_rgb_v1/cvat_tracking_xml|opaque",
            }
        )


def test_legacy_media_resolution_is_unchanged(tmp_path: Path) -> None:
    dataset = object.__new__(ClassificationV2ImageSequenceDataset)
    dataset.config = ImageSequenceDatasetConfig(media_root=tmp_path)
    assert dataset._resolve_media_path(
        {"source_type": "legacy_recovered", "resolved_media_path": "crops/one.jpg"}
    ) == tmp_path / "crops" / "one.jpg"
