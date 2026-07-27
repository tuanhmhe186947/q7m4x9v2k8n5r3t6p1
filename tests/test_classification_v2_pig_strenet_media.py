from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.pig_strenet_artifact_run import (
    audit_pig_strenet_artifact_run,
)
from pig_behavior.classification_v2.datasets.pig_strenet_media import (
    FrameMediaResolver,
)
from pig_behavior.classification_v2.datasets.pig_strenet_publication import (
    checkpointed_sha256,
    publish_media_manifest,
    recover_media_manifest,
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


def test_progress_reporter_is_atomic_and_marks_computed(tmp_path: Path) -> None:
    builder = _load_builder_script()
    progress_path = tmp_path / "pig_strenet_progress.json"
    reporter = builder._ProgressReporter(
        progress_path,
        input_csv=tmp_path / "input.csv",
        run_scope="full",
    )
    reporter("build_social_graph", 10, 20)

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["status"] == "RUNNING"
    assert payload["phase"] == "build_social_graph"
    assert payload["completed"] == 10
    assert payload["total"] == 20

    reporter.complete()
    completed = json.loads(progress_path.read_text(encoding="utf-8"))
    assert completed["status"] == "COMPUTED"
    assert completed["phase"] == "publication"
    assert not progress_path.with_suffix(".tmp").exists()


def test_media_hash_checkpoint_reuses_exact_file_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"first")
    checkpoint = tmp_path / "publication.sqlite3"
    first = checkpointed_sha256(media, checkpoint_path=checkpoint)

    def unexpected_hash(*_: object, **__: object) -> str:
        raise AssertionError("cached exact file identity must not be rehashed")

    module = importlib.import_module(
        "pig_behavior.classification_v2.datasets."
        "pig_strenet_publication"
    )
    monkeypatch.setattr(module, "_sha256_file_with_retry", unexpected_hash)
    second = checkpointed_sha256(media, checkpoint_path=checkpoint)

    assert second == first


def test_video_publication_binds_used_frames_without_hashing_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "large-source.mp4"
    video.write_bytes(b"container-identity")
    module = importlib.import_module(
        "pig_behavior.classification_v2.datasets."
        "pig_strenet_publication"
    )

    def unexpected_hash(*_: object, **__: object) -> str:
        raise AssertionError("video containers must not be fully rehashed")

    monkeypatch.setattr(module, "_sha256_file_with_retry", unexpected_hash)
    payload = publish_media_manifest(
        tmp_path / "media_manifest.json",
        video_root=tmp_path,
        legacy_crop_root=tmp_path,
        video_index_aliases=1,
        usage={
            str(video): {
                "source_kind_counts": {"video_frame": 2},
                "frame_indices": {3, 7},
            }
        },
        status_counts={"ok": 2},
        runtime_counts={"video_decode_count": 2},
        rejected_scene_candidates=[],
        checkpoint_path=tmp_path / "publication.sqlite3",
    )

    source = payload["sources"][0]
    assert payload["valid"]
    assert payload["derived_pixel_video_authority_count"] == 1
    assert payload["full_file_sha256_count"] == 0
    assert source["sha256"] is None
    assert source["authority_valid"]
    assert source["authority_mode"] == (
        "stat_and_used_frames_plus_derived_pixels"
    )
    assert source["derived_pixel_artifact_binding_required"] is True
    assert source["frame_index_count"] == 2


def test_recovered_media_manifest_is_complete_and_resumable(
    tmp_path: Path,
) -> None:
    media = tmp_path / "frame.jpg"
    media.write_bytes(b"deterministic-media")
    difference = tmp_path / "difference_pixel_index.csv"
    roi = tmp_path / "roi_visual_union_patch_index.csv"
    common = {
        "pixel_available": True,
        "pixel_source_kind": "actor_crop_file",
        "pixel_media_path": str(media),
        "pixel_frame_index": 3,
    }
    pd.DataFrame(
        [{**common, "frame_available": True}]
    ).to_csv(difference, index=False)
    pd.DataFrame(
        [{**common, "pixel_geometry_expected": True}]
    ).to_csv(roi, index=False)
    output = tmp_path / "media_manifest.json"
    checkpoint = tmp_path / "publication.sqlite3"

    payload = recover_media_manifest(
        output,
        video_root=tmp_path,
        legacy_crop_root=tmp_path,
        video_index_aliases=0,
        provenance_paths=[difference, roi],
        checkpoint_path=checkpoint,
    )
    repeated = recover_media_manifest(
        output,
        video_root=tmp_path,
        legacy_crop_root=tmp_path,
        video_index_aliases=0,
        provenance_paths=[difference, roi],
        checkpoint_path=checkpoint,
    )

    assert payload["valid"]
    assert payload["source_file_count"] == 1
    assert payload["sources"][0]["sha256"] == repeated["sources"][0]["sha256"]
    assert all(
        item["missing_required_rows"] == 0
        for item in payload["provenance_audit"]
    )


def test_production_builder_recovers_publication_without_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder_script()
    video_root = tmp_path / "videos"
    video_root.mkdir()
    _write_video(video_root / "sample_30fps.avi", frame_count=12)
    input_csv = tmp_path / "frames.csv"
    _publication_frames().to_csv(input_csv, index=False)
    output_dir = tmp_path / "artifacts"
    argv = [
        "pig-strenet-builder",
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--video-root",
        str(video_root),
        "--legacy-crop-root",
        str(tmp_path),
        "--run-scope",
        "full",
        "--social-checkpoint-pairs",
        "1",
    ]
    media_module = importlib.import_module(
        "pig_behavior.classification_v2.datasets.pig_strenet_media"
    )

    def fail_publication(*_: object, **__: object) -> dict[str, object]:
        raise OSError(22, "synthetic publication interruption")

    with monkeypatch.context() as context:
        context.setattr(
            media_module,
            "publish_media_manifest",
            fail_publication,
        )
        context.setattr(sys, "argv", argv)
        with pytest.raises(OSError, match="synthetic publication"):
            builder.main()

    pair_hash = _sha256(output_dir / "pair_manifest.csv")
    monkeypatch.setattr(
        sys,
        "argv",
        [*argv, "--recover-publication"],
    )
    builder.main()

    assert _sha256(output_dir / "pair_manifest.csv") == pair_hash
    assert (output_dir / "media_manifest.json").is_file()
    assert (output_dir / "artifact_manifest.json").is_file()
    assert (output_dir / "run_manifest.json").is_file()
    assert json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )["publication_recovery"]["computation_reexecuted"] is False
    gate = audit_pig_strenet_artifact_run(
        output_dir,
        input_csv=input_csv,
        expected_run_scope="full",
        require_roi_visual=False,
    )
    assert gate["status"] == "PASS"


def _publication_frames() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for frame_index in range(12):
        for actor_index in range(2):
            track = f"track-{actor_index}"
            row: dict[str, object] = {
                "object_track_key": track,
                "temporal_unit_key": f"cvat_tracking_xml|unit-{actor_index}",
                "source_type": "cvat_tracking_xml",
                "dataset_id": "synthetic",
                "video_key": "sample",
                "source_video_key": "sample",
                "source_video_path": "",
                "frame_index": frame_index,
                "frame_uid": f"{track}|frame-{frame_index}",
                "scene_frame_uid": f"scene-{frame_index}",
                "relative_frame_index": frame_index,
                "label_window_start": 6,
                "label_window_end": 11,
                "human_review_complete": False,
                "behavior_label": "fight",
                "lineage_scope": "synthetic",
                "crop_path": "",
                "image_width": 64,
                "image_height": 48,
                "x1": 4.0 + actor_index * 20.0,
                "y1": 5.0,
                "x2": 22.0 + actor_index * 20.0,
                "y2": 32.0,
                "cx_n": 0.25 + actor_index * 0.30,
                "cy_n": 0.4,
                "speed_n_per_frame": 0.01,
                "displacement_n": 0.01,
                "abs_accel_n_per_frame2": 0.001,
                "abs_direction_change_rad": 0.1,
                "nearest_dist_n": 0.3,
                "nearest_dist_delta": -0.001,
                "approach_speed_n_per_frame": 0.001,
                "separation_speed_n_per_frame": 0.0,
                "nearest_pair_iou": 0.0,
                "nearest_pair_overlap_ratio": 0.0,
                "pair_contact_with_nearest": False,
                "nearest_track_id": f"track-{1 - actor_index}",
                "timestamp_sec": frame_index / 6.0,
            }
            for roi_name in ("feeder", "drinker", "toy"):
                row.update(
                    {
                        f"roi_{roi_name}_available": False,
                        f"roi_{roi_name}_min_dist_n": 0.0,
                        f"roi_{roi_name}_max_overlap_ratio": 0.0,
                        f"roi_{roi_name}_max_iou": 0.0,
                        f"roi_{roi_name}_center_inside": False,
                        f"roi_{roi_name}_near": False,
                        f"roi_{roi_name}_contact": False,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
