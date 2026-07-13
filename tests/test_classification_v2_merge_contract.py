from __future__ import annotations

from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.identifiers import (
    FRAME_OBJECT_IDENTIFIER_VERSION,
)
from pig_behavior.classification_v2.merge_sources import (
    audit_merged_frame_objects,
    merge_frame_object_sources,
    read_canonical_frame_object_csv,
)
from pig_behavior.classification_v2.schema import CANONICAL_FRAME_OBJECT_COLUMNS


def _canonical_row(frame_index: int = 0) -> pd.DataFrame:
    row = {column: pd.NA for column in CANONICAL_FRAME_OBJECT_COLUMNS}
    row.update(
        {
            "source_type": "cvat_tracking_xml",
            "dataset_id": "fixture",
            "video_key": "video-a",
            "frame_uid": f"video-a::f{frame_index:06d}",
            "object_id_in_image": 1,
            "frame_index": frame_index,
            "track_id": "1",
            "pig_id": "ID_1",
            "behavior": "stand",
            "bbox_valid": True,
        }
    )
    return pd.DataFrame([row], columns=CANONICAL_FRAME_OBJECT_COLUMNS)


def test_merge_preserves_sum_of_source_rows() -> None:
    first = _canonical_row(0)
    second = _canonical_row(1)

    merged = merge_frame_object_sources(
        [first, second],
        source_names=["first", "second"],
    )

    assert len(merged) == len(first) + len(second)


def test_merge_migrates_scene_level_legacy_keys_to_object_keys() -> None:
    first = _canonical_row(0)
    second = _canonical_row(0)
    second["track_id"] = "2"
    second["pig_id"] = "ID_2"

    merged = merge_frame_object_sources(
        [first, second],
        source_names=["first", "second"],
    )

    assert merged["scene_frame_uid"].nunique() == 1
    assert merged["frame_uid"].nunique() == 2
    assert merged["identifier_schema_version"].eq(
        FRAME_OBJECT_IDENTIFIER_VERSION
    ).all()


def test_reader_migrates_old_canonical_csv_without_row_loss(tmp_path: Path) -> None:
    old = pd.concat([_canonical_row(0), _canonical_row(1)], ignore_index=True)
    old = old.drop(columns=["identifier_schema_version", "scene_frame_uid"])
    csv_path = tmp_path / "old_canonical.csv"
    old.to_csv(csv_path, index=False)

    migrated = read_canonical_frame_object_csv(csv_path)

    assert len(migrated) == len(old)
    assert migrated["scene_frame_uid"].nunique() == 2
    assert migrated["frame_uid"].nunique() == 2
    assert migrated["identifier_schema_version"].eq(
        FRAME_OBJECT_IDENTIFIER_VERSION
    ).all()


def test_merge_audit_rejects_duplicate_frame_object_rows() -> None:
    duplicate = pd.concat(
        [_canonical_row(0), _canonical_row(0)],
        ignore_index=True,
    )

    audit = audit_merged_frame_objects(duplicate)

    assert audit["duplicate_frame_object_rows"] == 2
    assert "duplicate_frame_object_rows=2" in audit["errors"]
