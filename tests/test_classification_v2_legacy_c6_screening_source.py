from __future__ import annotations

import pandas as pd

from pig_behavior.classification_v2.datasets.legacy_c6_screening_source import (
    LINEAGE_SCOPE,
    REVIEW_STATUS,
    select_legacy_c6_screening_source,
)


def _frames() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_index in range(3):
        for actor_index in range(2):
            for frame_index in range(16):
                rows.append(
                    {
                        "source_type": "legacy_recovered",
                        "dataset_id": "legacy_recovered_16f",
                        "video_key": f"day/video_{group_index}",
                        "clip_id": f"group_{group_index}",
                        "track_id": f"track_{group_index}_{actor_index}",
                        "pig_id": f"ID_{actor_index}",
                        "frame_uid": (
                            f"object::{group_index}::{actor_index}::{frame_index}"
                        ),
                        "scene_frame_uid": f"scene::{group_index}::{frame_index}",
                        "relative_frame_index": frame_index,
                        "behavior": "eat" if actor_index == 0 else "lying",
                        "bbox_valid": True,
                        "include_in_training": True,
                        "use_for_main_eval": True,
                        "crop_path": f"crop_{group_index}_{actor_index}_{frame_index}.jpg",
                    }
                )
    return pd.DataFrame(rows)


def test_selects_only_complete_groups_and_attaches_claims() -> None:
    result = select_legacy_c6_screening_source(_frames(), max_groups=2)

    assert result.audit["valid"] is True
    assert result.audit["groups"] == 2
    assert result.audit["actors"] == 4
    assert len(result.frames) == 64
    assert set(result.frames["lineage_scope"]) == {LINEAGE_SCOPE}
    assert not result.frames["human_review_complete"].any()
    assert set(result.frames["review_status"]) == {REVIEW_STATUS}
    counts = result.frames.groupby(["video_key", "clip_id", "pig_id"]).size()
    assert set(counts) == {16}


def test_filters_only_incomplete_actor_in_selected_full_source() -> None:
    frame = _frames().iloc[:-1].copy()
    result = select_legacy_c6_screening_source(frame)

    assert result.audit["excluded_actor_unit_count"] == 1
    assert result.audit["rows_dropped_inside_selected_groups"] == 15
    assert len(result.frames) == 80


def test_filters_actor_excluded_by_source_policy() -> None:
    frame = _frames()
    excluded = frame["track_id"].eq("track_0_0")
    frame.loc[excluded, "include_in_training"] = False
    frame.loc[excluded, "use_for_main_eval"] = False

    result = select_legacy_c6_screening_source(frame)

    assert result.audit["excluded_actor_unit_count"] == 1
    assert result.audit["rows_dropped_inside_selected_groups"] == 16
    assert "track_0_0" not in set(result.frames["track_id"])
    reasons = result.audit["excluded_actor_units"][0]["reasons"]
    assert reasons == [
        "source_excluded_from_training",
        "source_excluded_from_main_eval",
    ]


def test_cascade_filters_social_actor_missing_partner_after_source_filter() -> None:
    frame = _frames()
    group_zero = frame["clip_id"].eq("group_0")
    excluded = group_zero & frame["pig_id"].eq("ID_0")
    social = group_zero & frame["pig_id"].eq("ID_1")
    frame.loc[excluded, "include_in_training"] = False
    frame.loc[excluded, "use_for_main_eval"] = False
    frame.loc[social, "behavior"] = "social-nose"

    result = select_legacy_c6_screening_source(frame)

    assert result.audit["excluded_actor_unit_count"] == 2
    assert result.audit["rows_dropped_inside_selected_groups"] == 32
    assert set(result.frames["clip_id"]) == {"group_1", "group_2"}
    assert result.audit["excluded_actor_units"][1]["reasons"] == [
        "missing_interaction_partner_after_source_filter"
    ]
