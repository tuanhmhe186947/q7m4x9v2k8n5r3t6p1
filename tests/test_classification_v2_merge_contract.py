from __future__ import annotations

import pandas as pd

from pig_behavior.classification_v2.merge_sources import (
    audit_merged_frame_objects,
    merge_frame_object_sources,
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


def test_merge_audit_rejects_duplicate_frame_object_rows() -> None:
    duplicate = pd.concat(
        [_canonical_row(0), _canonical_row(0)],
        ignore_index=True,
    )

    audit = audit_merged_frame_objects(duplicate)

    assert audit["duplicate_frame_object_rows"] == 2
    assert "duplicate_frame_object_rows=2" in audit["errors"]
