from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pig_behavior.classification_v2.review.identity_continuity_adjudication import (
    CORRECTED_BBOX_MODE,
)
from pig_behavior.classification_v2.review.identity_continuity_apply import (
    IdentitySourceApplyError,
    apply_identity_adjudication,
    sha256_file,
)
from pig_behavior.classification_v2.review.mini_cvat_adjudication import (
    MINI_CVAT_SCHEMA,
)


def _annotation(
    actor_id: str,
    track_id: str,
    frame_index: int,
    bbox: tuple[float, float, float, float],
) -> dict[str, object]:
    return {
        "actor_scope_id": actor_id,
        "frame_index": frame_index,
        "source_frame_index": frame_index,
        "original_object_track_key": f"object-{track_id}",
        "original_track_id": track_id,
        "original_pig_id": actor_id,
        "reviewed_pig_id": actor_id,
        "bbox_mode": CORRECTED_BBOX_MODE,
        "x1": bbox[0],
        "y1": bbox[1],
        "x2": bbox[2],
        "y2": bbox[3],
        "original_hidden": "No",
        "reviewed_hidden": "No",
    }


def _write_sidecar(path: Path) -> None:
    payload = {
        "schema": MINI_CVAT_SCHEMA,
        "reviewer": "reviewer-a",
        "source_type": "legacy_recovered",
        "dataset_id": "legacy_recovered_16f",
        "video_key": "scene/001",
        "editable_actor_ids": ["ID_4", "ID_5"],
        "frame_indices": [12, 15],
        "actor_attributes": [
            {
                "actor_scope_id": "ID_4",
                "original_pig_id": "ID_4",
                "reviewed_pig_id": "ID_4",
                "original_behavior": "fight",
                "reviewed_behavior": "move",
            },
            {
                "actor_scope_id": "ID_5",
                "original_pig_id": "ID_5",
                "reviewed_pig_id": "ID_5",
                "original_behavior": "move",
                "reviewed_behavior": "fight",
            },
        ],
        "frame_annotations": [
            _annotation("ID_4", "track-4", 12, (11.0, 12.0, 31.0, 32.0)),
            _annotation("ID_5", "track-5", 12, (41.0, 42.0, 61.0, 62.0)),
            _annotation("ID_4", "track-4", 15, (15.0, 16.0, 35.0, 36.0)),
            _annotation("ID_5", "track-5", 15, (45.0, 46.0, 65.0, 66.0)),
        ],
        "behavior_decision_ledger_touched": "NO",
        "source_annotations_changed": "NO",
        "model_x_forbidden": "YES",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_dense_csv(path: Path) -> None:
    fieldnames = [
        "group_id",
        "tracklet_id",
        "frame_index",
        "pig_id",
        "behavior",
        "behavior_coarse",
        "hidden",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_width",
        "image_height",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "cx",
        "cy",
        "bbox_source",
        "label_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in (12, 15):
            for actor_id, track_id, behavior, x1 in (
                ("ID_4", "track-4", "fight", 10.0),
                ("ID_5", "track-5", "move", 40.0),
            ):
                writer.writerow(
                    {
                        "group_id": "burst-a",
                        "tracklet_id": track_id,
                        "frame_index": frame_index,
                        "pig_id": actor_id,
                        "behavior": behavior,
                        "behavior_coarse": behavior,
                        "hidden": "No",
                        "x1": x1,
                        "y1": 10.0,
                        "x2": x1 + 10.0,
                        "y2": 20.0,
                        "image_width": 100,
                        "image_height": 80,
                        "bbox_w": 10.0,
                        "bbox_h": 10.0,
                        "bbox_area": 100.0,
                        "cx": x1 + 5.0,
                        "cy": 15.0,
                        "bbox_source": "detector",
                        "label_source": "legacy",
                    }
                )


def _write_xml(path: Path) -> None:
    root = ET.Element("annotations")
    for frame_index in (12, 15):
        image = ET.SubElement(
            root,
            "image",
            {
                "id": str(frame_index),
                "name": f"burst-a_f{frame_index}_k0.jpg",
                "width": "100",
                "height": "80",
            },
        )
        for actor_id, behavior, x1 in (
            ("ID_4", "fight", 10.0),
            ("ID_5", "move", 40.0),
        ):
            box = ET.SubElement(
                image,
                "box",
                {
                    "label": "Pig",
                    "source": "file",
                    "occluded": "0",
                    "xtl": str(x1),
                    "ytl": "10",
                    "xbr": str(x1 + 10.0),
                    "ybr": "20",
                    "z_order": "0",
                },
            )
            for name, value in (
                ("ID", actor_id),
                ("Behavior", behavior),
                ("Hidden", "No"),
            ):
                attribute = ET.SubElement(box, "attribute", {"name": name})
                attribute.text = value
    ET.ElementTree(root).write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


def _xml_value(box: ET.Element, name: str) -> str:
    for attribute in box.findall("attribute"):
        if attribute.attrib["name"] == name:
            return attribute.text or ""
    raise AssertionError(name)


def test_apply_updates_dense_csv_and_original_xml_with_backups(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "mini_cvat_adjudication.json"
    dense_csv = tmp_path / "legacy_dense_tracklet_map.csv"
    xml_path = tmp_path / "annotations.xml"
    audit_root = tmp_path / "audit"
    _write_sidecar(sidecar)
    _write_dense_csv(dense_csv)
    _write_xml(xml_path)
    dense_before = sha256_file(dense_csv)
    xml_before = sha256_file(xml_path)

    result = apply_identity_adjudication(
        sidecar_path=sidecar,
        csv_paths=(dense_csv,),
        xml_path=xml_path,
        audit_root=audit_root,
    )

    assert result.group_id == "burst-a"
    assert result.changed_target_count == 2
    assert result.manifest_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "APPLIED"
    assert manifest["behavior_decision_ledger_touched"] == "NO"
    assert manifest["source_annotations_changed"] == "YES"
    assert {row["before_sha256"] for row in manifest["targets"]} == {
        dense_before,
        xml_before,
    }
    assert all(Path(row["backup_path"]).is_file() for row in manifest["targets"])

    with dense_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    id4_frame12 = next(
        row
        for row in rows
        if row["tracklet_id"] == "track-4"
        and row["frame_index"] == "12"
    )
    assert id4_frame12["behavior"] == "move"
    assert id4_frame12["x1"] == "11"
    assert id4_frame12["bbox_w"] == "20"
    assert id4_frame12["bbox_area"] == "400"
    assert id4_frame12["bbox_source"] == "human_identity_adjudication"
    assert id4_frame12["identity_review_status"] == "APPLIED"

    tree = ET.parse(xml_path)
    frame12 = next(
        image
        for image in tree.getroot().findall("image")
        if image.attrib["name"] == "burst-a_f12_k0.jpg"
    )
    id4_box = next(
        box
        for box in frame12.findall("box")
        if _xml_value(box, "ID") == "ID_4"
    )
    assert _xml_value(id4_box, "Behavior") == "move"
    assert id4_box.attrib["xtl"] == "11"
    assert id4_box.attrib["source"] == "manual"


def test_derived_feature_csv_is_rejected_without_modification(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "mini_cvat_adjudication.json"
    derived_csv = tmp_path / "native_review_evidence.csv"
    xml_path = tmp_path / "annotations.xml"
    _write_sidecar(sidecar)
    _write_dense_csv(derived_csv)
    rows = list(csv.DictReader(derived_csv.open(encoding="utf-8", newline="")))
    fieldnames = list(rows[0]) + ["motion_schema_version"]
    with derived_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["motion_schema_version"] = "motion.v1"
            writer.writerow(row)
    _write_xml(xml_path)
    csv_before = sha256_file(derived_csv)
    xml_before = sha256_file(xml_path)

    with pytest.raises(
        IdentitySourceApplyError,
        match="derived_feature_csv_requires_feature_rebuild",
    ):
        apply_identity_adjudication(
            sidecar_path=sidecar,
            csv_paths=(derived_csv,),
            xml_path=xml_path,
            audit_root=tmp_path / "audit",
            group_id="burst-a",
        )

    assert sha256_file(derived_csv) == csv_before
    assert sha256_file(xml_path) == xml_before
