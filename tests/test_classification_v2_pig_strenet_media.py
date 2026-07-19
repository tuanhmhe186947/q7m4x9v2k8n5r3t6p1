from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.pig_strenet_media import (
    FrameMediaResolver,
)


def _load_builder_script():
    path = Path(__file__).parents[1] / (
        "scripts/classification_v2/03_image_cache_context/"
        "classification_v2_build_pig_strenet_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location("pig_strenet_builder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_video(path: Path, *, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        6.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = 20 + index * 20
        frame[:, :, 1] = 60 + index * 10
        frame[:, :, 2] = 120
        writer.write(frame)
    writer.release()


def _video_row(video_key: str = "sample") -> dict[str, object]:
    return {
        "video_key": video_key,
        "source_video_key": video_key,
        "source_video_path": "",
        "frame_index": 2,
        "image_width": 64,
        "image_height": 48,
        "x1": 8.0,
        "y1": 6.0,
        "x2": 32.0,
        "y2": 30.0,
        "crop_path": "",
        "scene_frame_uid": "scene-2",
    }


def test_video_frame_and_bbox_crop_are_resolved_with_cache(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video_path = video_root / "sample_30fps.avi"
    _write_video(video_path)

    with FrameMediaResolver(
        video_root=video_root,
        legacy_crop_root=tmp_path,
    ) as resolver:
        row = _video_row()
        scene = resolver.read_scene(row)
        assert scene.available
        assert scene.source_kind == "video_frame"
        assert scene.image_rgb is not None
        assert scene.image_rgb.shape == (48, 64, 3)

        cached = resolver.read_scene(row)
        assert cached.available
        actor = resolver.read_actor(row, image_size=16)
        assert actor.available
        assert actor.source_kind == "video_bbox_crop"
        assert actor.image_rgb is not None
        assert actor.image_rgb.shape == (16, 16, 3)
        manifest = resolver.manifest()

    assert manifest["valid"]
    assert manifest["source_file_count"] == 1
    assert manifest["runtime_counts"]["video_decode_count"] == 1
    assert manifest["runtime_counts"]["video_frame_cache_hits"] >= 1
    assert manifest["background_as_temporal_scene_used"] is False


def test_existing_actor_crop_has_priority_over_video(tmp_path: Path) -> None:
    crop_path = tmp_path / "actor.jpg"
    crop = np.zeros((12, 20, 3), dtype=np.uint8)
    crop[:, :, 1] = 255
    assert cv2.imwrite(str(crop_path), crop)

    row = _video_row()
    row["crop_path"] = str(crop_path)
    with FrameMediaResolver(
        video_root=tmp_path / "missing-videos",
        legacy_crop_root=tmp_path,
    ) as resolver:
        result = resolver.read_actor(row, image_size=10)
        manifest = resolver.manifest()

    assert result.available
    assert result.source_kind == "actor_crop_file"
    assert result.image_rgb is not None
    assert result.image_rgb.shape == (10, 10, 3)
    assert manifest["runtime_counts"].get("video_decode_count", 0) == 0


def test_static_background_scene_candidate_is_rejected(tmp_path: Path) -> None:
    background = tmp_path / "background.png"
    assert cv2.imwrite(str(background), np.zeros((48, 64, 3), dtype=np.uint8))
    row = _video_row()
    row["scene_image_path"] = str(background)
    row["video_key"] = "missing"
    row["source_video_key"] = "missing"
    result = FrameMediaResolver(
        video_root=tmp_path / "missing-videos",
        legacy_crop_root=tmp_path,
    )
    scene = result.read_scene(row)
    manifest = result.manifest()

    assert not scene.available
    assert scene.status == "forbidden_static_scene_candidate"
    assert manifest["background_as_temporal_scene_used"] is False
    assert manifest["rejected_static_scene_candidates"] == [str(background.resolve())]


def test_scene_resolution_accepts_explicit_frame_image(tmp_path: Path) -> None:
    image_path = tmp_path / "frame_000002.png"
    assert cv2.imwrite(str(image_path), np.zeros((48, 64, 3), dtype=np.uint8))
    row = _video_row()
    row["scene_image_path"] = str(image_path)
    row["video_key"] = "missing"
    row["source_video_key"] = "missing"

    resolver = FrameMediaResolver(
        video_root=tmp_path / "missing-videos",
        legacy_crop_root=tmp_path,
    )
    scene = resolver.read_scene(pd.Series(row))
    manifest = resolver.manifest()

    assert scene.available
    assert scene.source_kind == "scene_image_file"
    assert manifest["source_file_count"] == 1


def test_pixel_artifact_writers_use_video_for_difference_and_scene_roi(
    tmp_path: Path,
) -> None:
    builder = _load_builder_script()
    video_root = tmp_path / "videos"
    video_root.mkdir()
    _write_video(video_root / "sample_30fps.avi", frame_count=3)
    rows = []
    for frame_index in range(2):
        row = _video_row()
        row.update(
            {
                "frame_index": frame_index,
                "frame_uid": f"frame-{frame_index}",
                "scene_frame_uid": f"scene-{frame_index}",
            }
        )
        rows.append(row)
    frames = pd.DataFrame(rows)
    slots = pd.DataFrame(
        [
            {
                "pair_id": "pair-1",
                "global_slot_index": 0,
                "slot_role": "history",
                "frame_uid": "frame-0",
                "frame_available": True,
            },
            {
                "pair_id": "pair-1",
                "global_slot_index": 1,
                "slot_role": "target",
                "frame_uid": "frame-1",
                "frame_available": True,
            },
        ]
    )
    visual = pd.DataFrame(
        [
            {
                "visual_context_id": "ctx-1",
                "pair_id": "pair-1",
                "slot_index": 0,
                "roi_class": "feeder",
                "frame_uid": "frame-0",
                "actor_roi_visual_available": True,
                "union_x1": 4.0,
                "union_y1": 4.0,
                "union_x2": 35.0,
                "union_y2": 35.0,
            }
        ]
    )
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    with FrameMediaResolver(
        video_root=video_root,
        legacy_crop_root=tmp_path,
    ) as media:
        difference = builder._write_difference_artifacts(
            slots,
            output_dir,
            frames=frames,
            media=media,
            image_size=8,
        )
        roi = builder._write_roi_visual_pixel_artifacts(
            visual,
            frames,
            output_dir,
            media=media,
            image_size=8,
        )

    assert difference["status"] == "PASS"
    assert difference["maps_shape"] == [1, 1, 8, 8]
    assert roi["status"] == "PASS"
    assert roi["available_rows"] == 1
    assert (output_dir / "difference_pixel_index.csv").is_file()
    assert (output_dir / "roi_visual_union_patches_uint8.npy").is_file()


def test_bounded_target_selection_keeps_source_rows_available() -> None:
    builder = _load_builder_script()
    frames = pd.DataFrame(
        {
            "temporal_unit_key": ["unit-a", "unit-a", "unit-b"],
            "frame_index": [0, 1, 0],
        }
    )
    selected = builder._select_target_unit_keys(frames, 1)

    assert selected == {"unit-a"}
    assert len(frames) == 3
