from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.training.legacy_development_l6_full_frame_cache import (
    DATASET_ID,
    IMAGE_SIZE,
    LINEAGE_SCOPE,
    RESIZE_POLICY,
    SOURCE_TYPE,
    _index_row,
    _validate_scene_metadata,
)


def _scene_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scene_frame_uid": ["scene-0", "scene-0", "scene-1"],
            "video_key": ["video-a", "video-a", "video-a"],
            "frame_index": [0, 0, 1],
            "resolved_media_path": ["video.mp4"] * 3,
            "image_width": [1920] * 3,
            "image_height": [1080] * 3,
            "source_type": [SOURCE_TYPE] * 3,
            "dataset_id": [DATASET_ID] * 3,
            "lineage_scope": [LINEAGE_SCOPE] * 3,
            "human_review_complete": [False] * 3,
            "full_frame_context_available": [True] * 3,
        }
    )


def test_full_frame_scene_metadata_accepts_actor_duplicates() -> None:
    _validate_scene_metadata(_scene_rows())


def test_full_frame_scene_metadata_rejects_conflicting_paths() -> None:
    rows = _scene_rows()
    rows.loc[1, "resolved_media_path"] = "other.mp4"

    with pytest.raises(ValueError, match="metadata conflicts"):
        _validate_scene_metadata(rows)


def test_full_frame_letterbox_index_is_exact_and_centered() -> None:
    item = SimpleNamespace(
        scene_frame_uid="scene-0",
        selection_order=7,
        video_key="video-a",
        frame_index=12,
    )

    row = _index_row(
        item,
        packed_row=3,
        width=1920,
        height=1080,
    )

    assert row["packed_row"] == 3
    assert row["selection_order"] == 7
    assert row["resized_width"] == IMAGE_SIZE
    assert row["resized_height"] == 126
    assert row["pad_left"] == row["pad_right"] == 0
    assert row["pad_top"] == row["pad_bottom"] == 49
    assert row["resize_policy"] == RESIZE_POLICY
    assert np.prod([row["source_width"], row["source_height"]]) > 0
