from __future__ import annotations

import pandas as pd

from pig_behavior.classification_v2.datasets.temporal_smoke_scope import (
    TemporalSmokeScopeConfig,
    select_temporal_smoke_scope,
)


def _legacy_block(
    clip_id: str,
    actor_labels: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for actor_index, (pig_id, behavior) in enumerate(actor_labels.items()):
        for relative_frame in range(16):
            rows.append(
                {
                    "source_type": "legacy_recovered",
                    "dataset_id": "legacy",
                    "video_key": f"video-{clip_id}",
                    "clip_id": clip_id,
                    "frame_uid": f"{clip_id}-f{relative_frame}",
                    "frame_index": relative_frame,
                    "relative_frame_index": relative_frame,
                    "pig_id": pig_id,
                    "track_id": f"track-{actor_index}",
                    "behavior": behavior,
                }
            )
    return rows


def _cvat_block(
    anchor: int,
    actor_labels: dict[str, str],
    *,
    frames_per_actor: int = 6,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for actor_index, (pig_id, behavior) in enumerate(actor_labels.items()):
        for offset in range(frames_per_actor):
            frame_index = anchor + offset
            rows.append(
                {
                    "source_type": "cvat_tracking_xml",
                    "dataset_id": "cvat",
                    "video_key": "video-cvat",
                    "clip_id": "",
                    "frame_uid": f"cvat-f{frame_index}",
                    "frame_index": frame_index,
                    "relative_frame_index": frame_index,
                    "pig_id": pig_id,
                    "track_id": f"track-{actor_index}",
                    "behavior": behavior,
                }
            )
    return rows


def test_smoke_scope_keeps_complete_scene_blocks_for_both_sources() -> None:
    rows = [
        *_legacy_block("rich", {"ID_1": "eat", "ID_2": "fight"}),
        *_legacy_block("single", {"ID_3": "stand"}),
        *_cvat_block(0, {"ID_1": "move", "ID_2": "sitting"}),
        *_cvat_block(6, {"ID_1": "stand"}),
    ]

    selected, audit = select_temporal_smoke_scope(
        pd.DataFrame(rows),
        config=TemporalSmokeScopeConfig(blocks_per_source=1),
    )

    assert audit["errors"] == []
    assert audit["selected_block_counts"] == {
        "cvat_tracking_xml": 1,
        "legacy_recovered": 1,
    }
    assert audit["selected_multi_actor_block_counts"] == {
        "cvat_tracking_xml": 1,
        "legacy_recovered": 1,
    }
    assert len(selected.loc[selected["source_type"].eq("legacy_recovered")]) == 32
    assert len(selected.loc[selected["source_type"].eq("cvat_tracking_xml")]) == 12

    legacy = selected.loc[selected["source_type"].eq("legacy_recovered")]
    legacy_sizes = legacy.groupby(["clip_id", "pig_id"])[
        "relative_frame_index"
    ].nunique()
    assert legacy_sizes.eq(16).all()
    cvat = selected.loc[selected["source_type"].eq("cvat_tracking_xml")]
    cvat_sizes = cvat.groupby("pig_id")["frame_index"].nunique()
    assert cvat_sizes.eq(6).all()


def test_smoke_scope_audits_and_skips_incomplete_candidate_block() -> None:
    rows = [
        *_legacy_block("legacy-valid", {"ID_1": "lying"}),
        *_cvat_block(0, {"ID_1": "move"}),
        *_cvat_block(
            6,
            {"ID_2": "fight"},
            frames_per_actor=5,
        ),
    ]

    selected, audit = select_temporal_smoke_scope(
        pd.DataFrame(rows),
        config=TemporalSmokeScopeConfig(blocks_per_source=2),
    )

    assert audit["errors"] == []
    assert audit["invalid_block_counts"] == {"cvat_tracking_xml": 1}
    assert any(
        warning == "smoke_blocks_below_requested=cvat_tracking_xml:1/2"
        for warning in audit["warnings"]
    )
    cvat_frames = selected.loc[
        selected["source_type"].eq("cvat_tracking_xml"),
        "frame_index",
    ]
    assert sorted(cvat_frames.unique().tolist()) == list(range(6))


def test_smoke_scope_supports_explicit_legacy_only_contract() -> None:
    rows = [
        *_legacy_block("legacy-rich", {"ID_1": "eat", "ID_2": "fight"}),
        *_legacy_block("legacy-stand", {"ID_3": "stand"}),
    ]

    selected, audit = select_temporal_smoke_scope(
        pd.DataFrame(rows),
        config=TemporalSmokeScopeConfig(
            blocks_per_source=1,
            required_sources=("legacy_recovered",),
        ),
    )

    assert audit["errors"] == []
    assert audit["parameters"]["required_sources"] == ["legacy_recovered"]
    assert audit["selected_block_counts"] == {"legacy_recovered": 1}
    assert set(selected["source_type"]) == {"legacy_recovered"}
    assert len(selected) == 32
