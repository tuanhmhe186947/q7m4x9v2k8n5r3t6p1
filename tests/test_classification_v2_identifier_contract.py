from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.identifiers import (
    FRAME_OBJECT_IDENTIFIER_VERSION,
    audit_frame_object_identifiers,
    ensure_frame_object_identifiers,
    scene_frame_key,
)


def _legacy_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_type": ["cvat_tracking_xml", "cvat_tracking_xml"],
            "dataset_id": ["fixture", "fixture"],
            "video_key": ["video-a", "video-a"],
            "frame_index": [10, 10],
            "frame_uid": ["video-a::f000010", "video-a::f000010"],
            "image_key": ["video-a::f000010", "video-a::f000010"],
            "track_id": ["1", "2"],
            "pig_id": ["ID_1", "ID_2"],
        },
        index=[8, 3],
    )


def test_identifier_migration_preserves_rows_and_separates_scene_from_actor() -> None:
    rows = _legacy_rows()

    migrated = ensure_frame_object_identifiers(rows, source_name="fixture")
    audit = audit_frame_object_identifiers(migrated)

    assert migrated.index.tolist() == [8, 3]
    assert len(migrated) == len(rows)
    assert migrated["scene_frame_uid"].nunique() == 1
    assert migrated["frame_uid"].nunique() == 2
    assert migrated["identifier_schema_version"].eq(
        FRAME_OBJECT_IDENTIFIER_VERSION
    ).all()
    assert audit["valid"] is True
    assert audit["scene_frames"] == 1
    assert audit["frame_objects"] == 2


def test_identifier_migration_rejects_duplicate_actor_in_scene() -> None:
    rows = _legacy_rows()
    rows["track_id"] = "1"

    with pytest.raises(ValueError, match="duplicate_frame_uid=2"):
        ensure_frame_object_identifiers(rows, source_name="fixture")


def test_identifier_migration_rejects_missing_actor_key() -> None:
    rows = _legacy_rows().iloc[[0]].copy()
    rows["track_id"] = ""
    rows["pig_id"] = ""

    with pytest.raises(ValueError, match="missing_actor_key=1"):
        ensure_frame_object_identifiers(rows, source_name="fixture")


def test_current_identifier_contract_is_idempotent() -> None:
    migrated = ensure_frame_object_identifiers(
        _legacy_rows(),
        source_name="fixture",
    )

    repeated = ensure_frame_object_identifiers(migrated, source_name="fixture")

    pd.testing.assert_frame_equal(repeated, migrated)


def test_scene_frame_key_reads_old_and_new_artifacts() -> None:
    old = _legacy_rows()
    new = ensure_frame_object_identifiers(old, source_name="fixture")

    assert scene_frame_key(old).tolist() == old["frame_uid"].tolist()
    assert scene_frame_key(new).tolist() == new["scene_frame_uid"].tolist()


def test_source_namespace_prevents_cross_source_object_collision() -> None:
    first = _legacy_rows().iloc[[0]].copy()
    second = first.copy()
    second["source_type"] = "legacy_recovered"

    first_ids = ensure_frame_object_identifiers(first, source_name="first")
    second_ids = ensure_frame_object_identifiers(second, source_name="second")

    assert first_ids.iloc[0]["scene_frame_uid"] != second_ids.iloc[0][
        "scene_frame_uid"
    ]
    assert first_ids.iloc[0]["frame_uid"] != second_ids.iloc[0]["frame_uid"]
