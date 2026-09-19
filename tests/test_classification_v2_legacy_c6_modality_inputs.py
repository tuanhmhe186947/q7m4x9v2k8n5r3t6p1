from __future__ import annotations

from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.training.legacy_c6_modality_inputs import (
    C6_OFFSETS,
    LINEAGE_SCOPE,
    prepare_legacy_c6_modality_context,
)


def _tables(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    media_path = tmp_path / "color.mp4"
    media_path.write_bytes(b"synthetic-path-contract")
    units: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for unit_index, role in enumerate(("train", "validation")):
        target_key = f"unit-{unit_index}"
        units.append(
            {
                "temporal_unit_key": target_key,
                "position": unit_index,
                "window_id": f"window-{unit_index}",
                "l5_role": role,
                "lineage_scope": LINEAGE_SCOPE,
                "human_review_complete": False,
            }
        )
        for offset in range(16):
            frame_index = 100 * (unit_index + 1) + offset
            scene_uid = f"scene-{unit_index}-{offset}"
            for actor_index in range(2):
                track_id = f"track-{unit_index}-{actor_index}"
                partner_id = f"track-{unit_index}-{1 - actor_index}"
                frames.append(
                    {
                        "temporal_unit_key": (
                            target_key
                            if actor_index == 0
                            else f"partner-unit-{unit_index}"
                        ),
                        "scene_frame_uid": scene_uid,
                        "frame_uid": (
                            f"frame-{unit_index}-{actor_index}-{offset}"
                        ),
                        "source_type": "legacy_recovered",
                        "dataset_id": "legacy_recovered_16f",
                        "video_key": f"video-{unit_index}",
                        "clip_id": f"clip-{unit_index}",
                        "object_track_key": track_id,
                        "pig_id": f"ID_{actor_index + 1}",
                        "track_id": track_id,
                        "nearest_track_id": partner_id,
                        "frame_index": frame_index,
                        "relative_frame_index": offset,
                        "source_video_path": str(media_path),
                        "x1": 10.0 + actor_index * 20.0,
                        "y1": 10.0,
                        "x2": 30.0 + actor_index * 20.0,
                        "y2": 40.0,
                        "bbox_valid": True,
                        "lineage_scope": LINEAGE_SCOPE,
                        "human_review_complete": False,
                    }
                )
    return pd.DataFrame(frames), pd.DataFrame(units)


def test_prepares_only_selected_c6_slots_with_same_scene_partners(
    tmp_path: Path,
) -> None:
    frames, units = _tables(tmp_path)

    prepared = prepare_legacy_c6_modality_context(frames, units)

    assert prepared.audit["valid"] is True
    assert prepared.audit["selected_native_units"] == 2
    assert prepared.audit["c6_target_slots"] == 12
    assert len(prepared.window_context) == 2
    assert len(prepared.union_selection) == 12
    assert len(prepared.full_frame_selection) == 12
    assert len(prepared.frame_context) == 24


def test_selected_unit_order_wins_when_frames_carry_l5_columns(
    tmp_path: Path,
) -> None:
    frames, units = _tables(tmp_path)
    frame_order = frames["temporal_unit_key"].map(
        {"unit-0": 101, "unit-1": 100}
    )
    frames = frames.assign(
        position=frame_order,
        window_id="frame-window",
        l5_role="frame-role",
    )

    prepared = prepare_legacy_c6_modality_context(frames, units)

    assert prepared.audit["valid"] is True
    assert prepared.window_context["temporal_unit_key"].tolist() == [
        "unit-0",
        "unit-1",
    ]
    assert prepared.frame_context["resolved_media_exists"].all()
    assert prepared.frame_context["image_context_id"].is_unique
    expected_offsets = "|".join(
        str(100 + value) for value in C6_OFFSETS
    )
    assert (
        prepared.window_context.iloc[0]["expected_frame_indices"]
        == expected_offsets
    )
    assert prepared.audit["outer_holdout_media_reads"] == 0
