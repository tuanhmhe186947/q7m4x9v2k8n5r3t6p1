from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.visual_interaction_context import (
    _resolve_context_geometry,
    _same_frame_actor_lookup,
    _validate_frames,
)


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
    lookup = _same_frame_actor_lookup(
        pd.DataFrame([actor, other_burst_partner])
    )

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
