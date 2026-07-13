from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.interaction_context_index import (
    InteractionContextIndexConfig,
    build_interaction_context_index,
)


def _write_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write one internally aligned actor-window context fixture."""

    image_frames = pd.DataFrame(
        {
            "scene_frame_uid": ["scene-0", "scene-1"],
            "frame_uid": ["frame-0", "frame-1"],
            "image_context_id": ["context-0", "context-1"],
            "full_frame_context_available": [True, True],
            "partner_context_available": [True, True],
            "interaction_partner_count": [1, 1],
            "interaction_partner_ids": ["ID_2", "ID_2"],
        }
    )
    image_windows = pd.DataFrame(
        {
            "window_id": ["window-0"],
            "source_type": ["cvat_tracking_xml"],
            "dataset_id": ["dataset-0"],
            "video_key": ["video-0"],
            "object_track_key": ["track-0"],
            "pig_id": ["ID_1"],
            "track_id": [1],
            "window_start_frame": [0],
            "window_end_frame": [1],
            "scene_frame_uid_sequence": ["scene-0|scene-1"],
            "frame_uid_sequence": ["frame-0|frame-1"],
            "image_context_id_sequence": ["context-0;;context-1"],
        }
    )
    split = pd.DataFrame(
        {
            "window_id": ["window-0"],
            "behavior_window_label": ["fight"],
        }
    )
    image_frames.to_csv(root / "image_frame_context_manifest.csv", index=False)
    image_windows.to_csv(root / "image_window_context_manifest.csv", index=False)
    split.to_csv(root / "split_manifest.csv", index=False)
    return image_frames, image_windows, split


def _build(root: Path):
    return build_interaction_context_index(
        InteractionContextIndexConfig(root=root, output_dir=root / "output")
    )


def test_interaction_context_index_preserves_aligned_window(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    result = _build(tmp_path)

    assert len(result.manifest) == 1
    assert result.audit["duplicate_source_image_context_id_rows"] == 0
    assert result.audit["errors"] == []
    assert result.manifest.loc[0, "scene_partner_context_status"] == "ready"
    alignment = result.audit["window_alignment"]
    assert alignment["valid"] is True
    assert alignment["comparisons"]["interaction_context_windows"][
        "order_mismatch_rows"
    ] == 0


def test_interaction_context_index_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    _build(tmp_path)

    with pytest.raises(FileExistsError, match="--overwrite"):
        _build(tmp_path)


def test_interaction_context_index_rejects_duplicate_actor_frame_key(
    tmp_path: Path,
) -> None:
    image_frames, _, _ = _write_inputs(tmp_path)
    duplicate = pd.concat([image_frames, image_frames.iloc[[0]]], ignore_index=True)
    duplicate.to_csv(tmp_path / "image_frame_context_manifest.csv", index=False)

    with pytest.raises(ValueError, match="duplicate_image_context_id_rows"):
        _build(tmp_path)


def test_interaction_context_index_requires_exact_window_key_set(
    tmp_path: Path,
) -> None:
    _, _, split = _write_inputs(tmp_path)
    split.loc[len(split)] = ["window-missing", "stand"]
    split.to_csv(tmp_path / "split_manifest.csv", index=False)

    with pytest.raises(ValueError, match="split_windows_missing_image_context=1"):
        _build(tmp_path)


def test_interaction_context_index_rejects_window_order_mismatch(
    tmp_path: Path,
) -> None:
    _, image_windows, split = _write_inputs(tmp_path)
    second_window = image_windows.copy()
    second_window["window_id"] = "window-1"
    image_windows = pd.concat(
        [image_windows, second_window],
        ignore_index=True,
    )
    second_split = split.copy()
    second_split["window_id"] = "window-1"
    split = pd.concat([second_split, split], ignore_index=True)
    image_windows.to_csv(
        tmp_path / "image_window_context_manifest.csv",
        index=False,
    )
    split.to_csv(tmp_path / "split_manifest.csv", index=False)

    with pytest.raises(
        ValueError,
        match="image_split_window_order_mismatch_rows=2",
    ):
        _build(tmp_path)


def test_interaction_context_index_rejects_frame_context_mismatch(
    tmp_path: Path,
) -> None:
    _, image_windows, _ = _write_inputs(tmp_path)
    image_windows.loc[0, "frame_uid_sequence"] = "frame-1|frame-0"
    image_windows.to_csv(
        tmp_path / "image_window_context_manifest.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="frame_context_sequence_mismatches=2"):
        _build(tmp_path)


def test_interaction_context_index_rejects_scene_context_mismatch(
    tmp_path: Path,
) -> None:
    _, image_windows, _ = _write_inputs(tmp_path)
    image_windows.loc[0, "scene_frame_uid_sequence"] = "scene-1|scene-0"
    image_windows.to_csv(
        tmp_path / "image_window_context_manifest.csv",
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="scene_context_sequence_mismatches=2",
    ):
        _build(tmp_path)
