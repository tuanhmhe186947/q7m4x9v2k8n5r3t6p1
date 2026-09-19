from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.identifiers import (
    FRAME_OBJECT_IDENTIFIER_VERSION,
    audit_frame_object_identifiers,
    ensure_frame_object_identifiers,
    ensure_object_track_keys,
    scene_frame_key,
)
from pig_behavior.classification_v2.contracts.model_io import (
    validate_model_input_columns,
)
from pig_behavior.classification_v2.features.frame_local import (
    build_frame_local_primitives,
)
from pig_behavior.classification_v2.merge_sources import (
    merge_frame_object_sources,
)
from pig_behavior.classification_v2.sources.cvat_tracking_xml import (
    load_cvat_tracking_xml,
)
from pig_behavior.classification_v2.sources.legacy_recovered_csv import (
    load_legacy_frame_objects,
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


def test_scene_frame_key_rejects_partial_v2_scene_identity() -> None:
    rows = ensure_frame_object_identifiers(_legacy_rows(), source_name="fixture")
    rows.loc[rows.index[0], "scene_frame_uid"] = ""

    with pytest.raises(ValueError, match="missing_scene_frame_uid=1"):
        scene_frame_key(rows)


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


def test_legacy_source_emits_unique_object_ids_for_shared_scene(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame(
        {
            "image_key": ["legacy-scene-0", "legacy-scene-0"],
            "image_name": ["frame.jpg", "frame.jpg"],
            "source_video_key": ["legacy-video", "legacy-video"],
            "tracklet_id": ["track-1", "track-2"],
            "pig_id": ["ID_1", "ID_2"],
            "behavior": ["stand", "fight"],
            "frame_index": [0, 0],
            "x1": [10.0, 40.0],
            "y1": [10.0, 10.0],
            "x2": [30.0, 60.0],
            "y2": [30.0, 30.0],
        }
    )
    source_path = tmp_path / "legacy_frame_object_annotations.csv"
    source.to_csv(source_path, index=False)

    rows = load_legacy_frame_objects(source_path, dataset_id="legacy-fixture")

    assert len(rows) == 2
    assert rows["scene_frame_uid"].nunique() == 1
    assert rows["frame_uid"].nunique() == 2
    assert rows["global_context_pig_count"].eq(2).all()
    assert rows["identifier_schema_version"].eq(
        FRAME_OBJECT_IDENTIFIER_VERSION
    ).all()


def test_identifier_columns_are_forbidden_from_model_x() -> None:
    audit = validate_model_input_columns(
        [
            "speed_mean_window",
            "frame_uid",
            "scene_frame_uid",
            "identifier_schema_version",
        ]
    )

    assert audit["valid"] is False
    assert audit["forbidden_columns"] == [
        "frame_uid",
        "identifier_schema_version",
        "scene_frame_uid",
    ]


@pytest.mark.parametrize(
    "column",
    [
        "target_roi_contact",
        "target_roi_distance",
        "target_roi_contact_ratio_unit",
        "label_selected_target_roi_class",
        "roi_target_identity",
        "behavior_selected_roi_contact",
    ],
)
def test_target_selected_roi_is_forbidden_from_model_x(
    column: str,
) -> None:
    audit = validate_model_input_columns(["speed_mean_window", column])
    assert audit["valid"] is False
    assert audit["forbidden_columns"] == [column]


def test_label_independent_roi_is_allowed_in_model_x() -> None:
    audit = validate_model_input_columns(["roi_feeder_min_dist_n"])
    assert audit["valid"] is True


def test_source_merge_creates_deterministic_scoped_object_track_keys(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy_frame_object_annotations.csv"
    pd.DataFrame(
        {
            "image_key": ["legacy-scene", "legacy-scene"],
            "image_name": ["frame.jpg", "frame.jpg"],
            "source_video_key": ["video-a", "video-a"],
            "tracklet_id": ["track-1", "track-2"],
            "pig_id": ["ID_1", "ID_1"],
            "behavior": ["stand", "stand"],
            "frame_index": [0, 0],
            "x1": [10.0, 40.0],
            "y1": [10.0, 10.0],
            "x2": [30.0, 60.0],
            "y2": [30.0, 30.0],
        }
    ).to_csv(legacy_path, index=False)
    legacy = load_legacy_frame_objects(
        legacy_path,
        dataset_id="legacy-fixture",
    )
    legacy.loc[legacy.index[0], "pig_id"] = ""
    cvat_path = tmp_path / "cvat.xml"
    cvat_path.write_text(
        """
<annotations>
  <meta><task><id>1</id><name>video-b</name><size>1</size>
    <original_size><width>100</width><height>100</height></original_size>
  </task></meta>
  <track id="7" label="Pig_1" source="manual">
    <box frame="0" outside="0" xtl="10" ytl="10" xbr="30" ybr="30">
      <attribute name="Behavior">stand</attribute>
      <attribute name="Hidden">No</attribute>
    </box>
  </track>
</annotations>
""".strip(),
        encoding="utf-8",
    )
    cvat = load_cvat_tracking_xml(
        cvat_path,
        video_key="video-b",
        dataset_id="cvat-fixture",
    )
    merged = merge_frame_object_sources([legacy, cvat])
    assert merged["object_track_key"].str.strip().ne("").all()
    legacy_keys = merged.loc[
        merged["source_type"].eq("legacy_recovered"),
        "object_track_key",
    ]
    assert legacy_keys.nunique() == 2

    same_track_other_video = legacy.iloc[[0]].copy()
    same_track_other_video["video_key"] = "video-c"
    same_track_other_video["object_track_key"] = ""
    merged_videos = merge_frame_object_sources(
        [legacy.iloc[[0]], same_track_other_video]
    )
    assert merged_videos["object_track_key"].nunique() == 2

    shuffled = merge_frame_object_sources(
        [legacy.sample(frac=1.0, random_state=9), cvat]
    )
    expected = merged.set_index("frame_uid")["object_track_key"].sort_index()
    actual = shuffled.set_index("frame_uid")["object_track_key"].sort_index()
    pd.testing.assert_series_equal(actual, expected)


def test_source_merge_rejects_pig_id_only_identity() -> None:
    rows = ensure_frame_object_identifiers(
        _legacy_rows().iloc[[0]],
        source_name="pig-only-fixture",
    )
    rows["track_id"] = ""
    rows["object_id_in_image"] = ""
    rows["object_track_key"] = ""
    with pytest.raises(ValueError, match="missing_object_track_authority"):
        ensure_object_track_keys(
            rows,
            source_name="pig-only-fixture",
        )


def test_frame_local_preserves_source_merge_object_track_key(
    tmp_path: Path,
) -> None:
    source = _legacy_rows()
    source["object_id_in_image"] = source["track_id"]
    source["behavior"] = "stand"
    source["hidden"] = "No"
    source["bbox_valid"] = True
    source["x1"] = [10.0, 40.0]
    source["y1"] = 10.0
    source["x2"] = [30.0, 60.0]
    source["y2"] = 30.0
    source["image_width"] = 100
    source["image_height"] = 100
    merged = merge_frame_object_sources(
        [source],
        strict_schema=False,
    )
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "feeder"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 5, 5],
                        "segmentation": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    mask_path = tmp_path / "mask.png"
    pen_mask = np.full((100, 100), 255, dtype=np.uint8)
    pen_mask[:5, :] = 0
    cv2.imwrite(
        str(mask_path),
        pen_mask,
    )
    frame_local = build_frame_local_primitives(
        merged,
        roi_coco_path=roi_path,
        pen_mask_path=mask_path,
        expected_pen_mask_sha256=None,
    )
    assert frame_local["object_track_key"].tolist() == merged[
        "object_track_key"
    ].tolist()
