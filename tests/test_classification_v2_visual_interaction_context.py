from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.visual_interaction_context import (
    CACHE_KEY_POLICY,
    RESIZE_POLICY,
    VisualInteractionCacheConfig,
    _cache_relative_path,
    _decode_union_crop,
    _resolve_context_geometry,
    _same_frame_actor_lookup,
    _select_target_frames,
    _validate_frames,
    _validate_partial_audit,
    _write_or_validate_cache_image,
)


def test_cache_relative_path_hashes_long_context_id() -> None:
    context_id = "legacy-context-" + ("x" * 300)

    path = _cache_relative_path(context_id)

    assert path == _cache_relative_path(context_id)
    assert path != _cache_relative_path(context_id + "y")
    assert path.parent.name == path.stem[:2]
    assert path.suffix == ".npy"
    assert len(path.name) == 68


def test_resume_reuses_only_identical_orphan_cache_image(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orphan.npy"
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    _write_or_validate_cache_image(path, image, resume=False)

    _write_or_validate_cache_image(path, image.copy(), resume=True)

    with pytest.raises(ValueError, match="resume cache image differs"):
        _write_or_validate_cache_image(
            path,
            np.zeros_like(image),
            resume=True,
        )


def test_decode_union_crop_evicts_lru_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCapture:
        def __init__(self, path: str) -> None:
            self.path = path
            self.released = False

        def isOpened(self) -> bool:
            return True

        def set(self, _property: int, _value: int) -> None:
            return None

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    created: dict[str, FakeCapture] = {}

    def capture_factory(path: str) -> FakeCapture:
        capture = FakeCapture(path)
        created[path] = capture
        return capture

    monkeypatch.setattr(
        "pig_behavior.classification_v2.datasets."
        "visual_interaction_context.cv2.VideoCapture",
        capture_factory,
    )
    captures: dict[str, FakeCapture] = {}
    decoded: dict[str, tuple[int, np.ndarray]] = {}
    next_frame: dict[str, int] = {}

    for path in ("video-a.mp4", "video-b.mp4"):
        image, *_ = _decode_union_crop(
            row={"resolved_media_path": path, "frame_index": 3},
            union_bbox=(0.0, 0.0, 8.0, 8.0),
            captures=captures,
            decoded=decoded,
            next_frame=next_frame,
            image_size=8,
            max_open_videos=1,
        )
        assert image is not None

    assert created["video-a.mp4"].released is True
    assert list(captures) == ["video-b.mp4"]
    assert list(decoded) == ["video-b.mp4"]
    assert list(next_frame) == ["video-b.mp4"]


def _row(
    *,
    media: Path,
    track_id: str,
    nearest_track_id: str,
    clip_id: str = "burst-a",
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> dict[str, object]:
    return {
        "image_context_id": f"context-{clip_id}-{track_id}",
        "source_type": "legacy_recovered",
        "dataset_id": "legacy_recovered_16f",
        "video_key": "video-a",
        "clip_id": clip_id,
        "resolved_media_path": str(media),
        "frame_index": 13,
        "track_id": track_id,
        "nearest_track_id": nearest_track_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def test_legacy_union_geometry_uses_same_clip_partner(tmp_path: Path) -> None:
    media = tmp_path / "color.mp4"
    media.touch()
    actor = _row(
        media=media,
        track_id="tracklet-1",
        nearest_track_id="tracklet-2",
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=40.0,
    )
    partner = _row(
        media=media,
        track_id="tracklet-2",
        nearest_track_id="tracklet-1",
        x1=40.0,
        y1=10.0,
        x2=60.0,
        y2=50.0,
    )
    lookup = _same_frame_actor_lookup(pd.DataFrame([actor, partner]))

    result = _resolve_context_geometry(actor, lookup, 0.1)

    assert result["status"] == "geometry_ready"
    assert result["partner_track_id"] == "tracklet-2"
    assert result["union_bbox"] == pytest.approx((5.0, 6.0, 65.0, 54.0))


def test_union_geometry_does_not_cross_legacy_bursts(tmp_path: Path) -> None:
    media = tmp_path / "color.mp4"
    media.touch()
    actor = _row(
        media=media,
        track_id="tracklet-1",
        nearest_track_id="tracklet-2",
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=40.0,
    )
    other_burst_partner = _row(
        media=media,
        track_id="tracklet-2",
        nearest_track_id="tracklet-1",
        clip_id="burst-b",
        x1=40.0,
        y1=10.0,
        x2=60.0,
        y2=50.0,
    )
    lookup = _same_frame_actor_lookup(pd.DataFrame([actor, other_burst_partner]))

    result = _resolve_context_geometry(actor, lookup, 0.1)

    assert result["status"] == "missing_nearest_partner_bbox"
    assert result["union_bbox"] is None


def test_visual_context_contract_requires_clip_identity(tmp_path: Path) -> None:
    media = tmp_path / "color.mp4"
    media.touch()
    row = _row(
        media=media,
        track_id="tracklet-1",
        nearest_track_id="tracklet-2",
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=40.0,
    )
    row.pop("clip_id")

    with pytest.raises(ValueError, match="missing columns"):
        _validate_frames(pd.DataFrame([row]))


def test_selection_renders_only_target_but_keeps_partner_lookup(
    tmp_path: Path,
) -> None:
    media = tmp_path / "color.mp4"
    media.touch()
    actor = _row(
        media=media,
        track_id="tracklet-1",
        nearest_track_id="tracklet-2",
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=40.0,
    )
    partner = _row(
        media=media,
        track_id="tracklet-2",
        nearest_track_id="tracklet-1",
        x1=40.0,
        y1=10.0,
        x2=60.0,
        y2=50.0,
    )
    frames = pd.DataFrame([actor, partner])
    selection_path = tmp_path / "selection.csv"
    pd.DataFrame(
        {"image_context_id": [actor["image_context_id"]]}
    ).to_csv(selection_path, index=False)

    lookup = _same_frame_actor_lookup(frames)
    selected = _select_target_frames(frames, selection_path)
    result = _resolve_context_geometry(
        selected.iloc[0].to_dict(),
        lookup,
        0.1,
    )

    assert selected["image_context_id"].tolist() == [actor["image_context_id"]]
    assert result["status"] == "geometry_ready"


def test_selection_rejects_missing_context_id(tmp_path: Path) -> None:
    media = tmp_path / "color.mp4"
    media.touch()
    actor = _row(
        media=media,
        track_id="tracklet-1",
        nearest_track_id="tracklet-2",
        x1=10.0,
        y1=20.0,
        x2=30.0,
        y2=40.0,
    )
    selection_path = tmp_path / "selection.csv"
    pd.DataFrame({"image_context_id": ["missing"]}).to_csv(
        selection_path,
        index=False,
    )

    with pytest.raises(ValueError, match="selection missing IDs"):
        _select_target_frames(pd.DataFrame([actor]), selection_path)


def test_selection_rejects_empty_manifest(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.csv"
    pd.DataFrame({"image_context_id": []}).to_csv(selection_path, index=False)

    with pytest.raises(ValueError, match="must not be empty"):
        _select_target_frames(pd.DataFrame(), selection_path)


def test_resume_rejects_changed_selection_hash(tmp_path: Path) -> None:
    frame_context = tmp_path / "frames.csv"
    frame_context.touch()
    selection_path = tmp_path / "selection.csv"
    selection_path.touch()
    config = VisualInteractionCacheConfig(
        frame_context_csv=frame_context,
        output_dir=tmp_path / "cache",
        selection_csv=selection_path,
    )
    audit_path = tmp_path / "visual_context_cache_audit.partial.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": ("classification_v2_visual_interaction_cache_partial_v1"),
                "frame_context_sha256": "frame-hash",
                "selection_sha256": "old-selection-hash",
                "selected_rows": 1,
                "image_size": 64,
                "padding_ratio": 0.15,
                "resize_policy": RESIZE_POLICY,
                "cache_key_policy": CACHE_KEY_POLICY,
                "max_open_videos": config.max_open_videos,
                "source_type_filter": None,
                "max_contexts": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selection_sha256"):
        _validate_partial_audit(
            audit_path,
            config,
            frame_context_sha256="frame-hash",
            selection_sha256="new-selection-hash",
            selected_rows=1,
        )
