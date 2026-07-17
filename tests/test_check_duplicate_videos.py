from __future__ import annotations

import pandas as pd
import pytest

from legacy_burst_recovery.check_duplicate_videos import (
    audit_duplicate_videos,
    choose_source_column,
    normalize_source_video_key,
)
from legacy_burst_recovery.make_nodup_legacy_csvs import build_nodup_tables


def test_normalize_source_video_key_supports_legacy_paths() -> None:
    assert normalize_source_video_key(
        "/content/pigs051219/PIGS051219/000328/color.mp4"
    ) == "pigs051219/000328"
    assert normalize_source_video_key(
        r"C:\data\Pigs281119_000085_30fps.mp4"
    ) == "pigs281119/000085"
    assert normalize_source_video_key("pigs281119/85") == "pigs281119/000085"
    assert normalize_source_video_key("not-a-video") == ""


def test_auto_source_column_prefers_direct_path_over_cached_key() -> None:
    frame = pd.DataFrame(
        {
            "source_video_key": ["pigs000000/000000"],
            "video_final": ["/pigs281119/PIGS281119/000085/color.mp4"],
        }
    )

    assert choose_source_column(frame, "auto") == "video_final"


def test_audit_returns_only_policy_hits_without_dropping_input_rows() -> None:
    legacy = pd.DataFrame(
        {
            "group_id": ["g1", "g2", "g3"],
            "pig_id": ["ID_1", "ID_2", "ID_3"],
            "video_final": [
                "/pigs281119/PIGS281119/000085/color.mp4",
                "/pigs291119/PIGS291119/000302/color.mp4",
                "/pigs301119/PIGS301119/000330/color.mp4",
            ],
        }
    )
    exclusions = pd.DataFrame(
        {"source_video_key": ["pigs281119/000085", "pigs291119/302"]}
    )

    preview, audit = audit_duplicate_videos(
        legacy,
        exclusions,
        source_column="video_final",
    )

    assert len(legacy) == 3
    assert len(preview) == 2
    assert preview["group_id"].tolist() == ["g1", "g2"]
    assert preview["source_video_key_audit"].tolist() == [
        "pigs281119/000085",
        "pigs291119/000302",
    ]
    assert audit["status"] == "DUPLICATES_FOUND"
    assert audit["counts"]["duplicate_rows"] == 2
    assert audit["counts"]["duplicate_group_pig"] == 2


def test_audit_fails_closed_on_unresolved_source() -> None:
    legacy = pd.DataFrame({"video_final": ["unknown/path.mp4"]})
    exclusions = pd.DataFrame({"source_video_key": ["pigs281119/000085"]})

    with pytest.raises(ValueError, match="unresolved source keys=1"):
        audit_duplicate_videos(
            legacy,
            exclusions,
            source_column="video_final",
        )


def test_audit_can_report_unresolved_without_treating_it_as_a_hit() -> None:
    legacy = pd.DataFrame(
        {
            "video_final": [
                "unknown/path.mp4",
                "/pigs281119/PIGS281119/000085/color.mp4",
            ]
        }
    )
    exclusions = pd.DataFrame({"source_video_key": ["pigs281119/000085"]})

    preview, audit = audit_duplicate_videos(
        legacy,
        exclusions,
        source_column="video_final",
        allow_unresolved=True,
    )

    assert len(preview) == 1
    assert audit["counts"]["unresolved_source_rows"] == 1
    assert audit["warnings"] == ["unresolved_source_rows=1"]


def test_nodup_filter_preserves_row_accounting_and_actor_pair_scope() -> None:
    center = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "group_id": ["g1", "g2"],
            "pig_id": ["ID_1", "ID_2"],
            "video_final": [
                "/pigs281119/PIGS281119/000085/color.mp4",
                "/pigs291119/PIGS291119/000302/color.mp4",
            ],
        }
    )
    bbox = pd.DataFrame(
        {
            "group_id": ["g1", "g1", "g2"],
            "pig_id": ["ID_1", "ID_1", "ID_2"],
            "video_final": [
                "/pigs281119/PIGS281119/000085/color.mp4",
                "/pigs999999/PIGS999999/000001/color.mp4",
                "/pigs291119/PIGS291119/000302/color.mp4",
            ],
        }
    )
    exclusions = pd.DataFrame({"source_video_key": ["pigs281119/000085"]})

    result = build_nodup_tables(center, bbox, exclusions)

    assert len(result["center_keep"]) + len(result["center_duplicate"]) == 2
    assert len(result["bbox_keep"]) + len(result["bbox_duplicate"]) == 3
    assert result["center_duplicate"]["group_id"].tolist() == ["g1"]
    assert result["bbox_duplicate"]["group_id"].tolist() == ["g1", "g1"]
    assert result["summary"]["duplicate_group_pig"] == 1
