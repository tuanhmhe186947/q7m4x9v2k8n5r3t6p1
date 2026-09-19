from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

from pig_behavior.classification_v2.sources.cvat_annotation_quality import (
    audit_legacy_task_export,
    audit_tracking_xml,
    combine_annotation_audits,
)


def _add_image_box(
    image: ET.Element,
    *,
    pig_id: str,
    behavior: str,
    bbox: tuple[float, float, float, float],
) -> None:
    box = ET.SubElement(
        image,
        "box",
        {
            "label": "Pig",
            "source": "manual",
            "xtl": str(bbox[0]),
            "ytl": str(bbox[1]),
            "xbr": str(bbox[2]),
            "ybr": str(bbox[3]),
        },
    )
    for name, value in [
        ("ID", pig_id),
        ("Behavior", behavior),
        ("Hidden", "No"),
    ]:
        attribute = ET.SubElement(box, "attribute", {"name": name})
        attribute.text = value


def _write_legacy_task(root: Path) -> None:
    task_dir = root / "task_0"
    (task_dir / "data").mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"subset": "Train"}),
        encoding="utf-8",
    )

    manifest_rows = []
    xml_root = ET.Element("annotations")
    frame_id = 0
    for group_id in ["burst_color_duplicate_0", "burst_color_switch_0"]:
        for slot in range(6):
            image_name = f"{group_id}_f{slot * 3}_k{slot}.jpg"
            manifest_rows.append(
                {
                    "name": image_name.removesuffix(".jpg"),
                    "extension": ".jpg",
                    "width": 1280,
                    "height": 720,
                }
            )
            image = ET.SubElement(
                xml_root,
                "image",
                {
                    "id": str(frame_id),
                    "name": image_name,
                    "width": "1280",
                    "height": "720",
                },
            )
            if group_id == "burst_color_duplicate_0":
                _add_duplicate_group_boxes(image, slot=slot)
            else:
                _add_switch_group_box(image, slot=slot)
            frame_id += 1

    manifest_text = "\n".join(
        json.dumps(row) for row in manifest_rows
    )
    (task_dir / "data" / "manifest.jsonl").write_text(
        f"{manifest_text}\n",
        encoding="utf-8",
    )
    ET.ElementTree(xml_root).write(
        task_dir / "annotations.xml",
        encoding="utf-8",
        xml_declaration=True,
    )


def _add_duplicate_group_boxes(image: ET.Element, *, slot: int) -> None:
    _add_image_box(
        image,
        pig_id="ID_1",
        behavior="fight",
        bbox=(880 + slot, 200, 1000 + slot, 460),
    )
    second_id = "ID_1" if slot == 4 else "ID_8"
    _add_image_box(
        image,
        pig_id=second_id,
        behavior="eat",
        bbox=(430 + slot, 300, 640 + slot, 600),
    )


def _add_switch_group_box(image: ET.Element, *, slot: int) -> None:
    pig_id = "ID_5" if slot < 4 else "ID_4"
    _add_image_box(
        image,
        pig_id=pig_id,
        behavior="stand",
        bbox=(100 + slot, 100, 220 + slot, 300),
    )


def _add_track_box(
    track: ET.Element,
    *,
    frame_id: int,
    pig_id: str,
) -> None:
    box = ET.SubElement(
        track,
        "box",
        {
            "frame": str(frame_id),
            "outside": "0",
            "xtl": str(100 + frame_id),
            "ytl": "100",
            "xbr": str(220 + frame_id),
            "ybr": "300",
        },
    )
    for name, value in [
        ("ID", pig_id),
        ("Behavior", "stand"),
        ("Hidden", "No"),
    ]:
        attribute = ET.SubElement(box, "attribute", {"name": name})
        attribute.text = value


def _write_tracking_xml(path: Path, *, duplicate_last_frame: bool) -> None:
    root = ET.Element("annotations")
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    for name, value in [
        ("name", "Synthetic_tracking"),
        ("size", "2"),
        ("start_frame", "0"),
        ("stop_frame", "1"),
        ("source", "Synthetic_tracking.mp4"),
    ]:
        element = ET.SubElement(task, name)
        element.text = value
    original_size = ET.SubElement(task, "original_size")
    width = ET.SubElement(original_size, "width")
    width.text = "1280"
    height = ET.SubElement(original_size, "height")
    height.text = "720"

    track_1 = ET.SubElement(
        root,
        "track",
        {"id": "1", "label": "Pig_1"},
    )
    track_2 = ET.SubElement(
        root,
        "track",
        {"id": "2", "label": "Pig_2"},
    )
    for frame_id in range(2):
        _add_track_box(track_1, frame_id=frame_id, pig_id="ID_1")
        second_id = (
            "ID_1"
            if duplicate_last_frame and frame_id == 1
            else "ID_2"
        )
        _add_track_box(track_2, frame_id=frame_id, pig_id=second_id)

    ET.ElementTree(root).write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


def test_legacy_audit_reports_exact_frames_and_identity_candidates(
    tmp_path: Path,
) -> None:
    _write_legacy_task(tmp_path)

    report = audit_legacy_task_export(tmp_path)

    assert report["status"] == "FAIL"
    assert report["summary"]["duplicate_anchor_identity_rows"] == 2
    assert report["summary"]["incomplete_authority_actor_keys"] == 2
    assert report["summary"]["actors_absent_authority_frame"] == 1

    duplicate = next(
        issue
        for issue in report["issues"]
        if issue["code"] == "duplicate_anchor_identity"
    )
    assert duplicate["frame_id"] == 4
    assert duplicate["frame_position_1based"] == 5
    assert duplicate["total_frames"] == 12
    assert duplicate["image_name"].endswith("_f12_k4.jpg")

    duplicate_candidate = next(
        issue
        for issue in report["issues"]
        if issue["code"]
        == "probable_duplicate_identity_substitution"
    )
    mapping = duplicate_candidate["evidence"]["suggested_mapping"]
    eat_mapping = next(
        item for item in mapping if item["behavior"] == "eat"
    )
    assert eat_mapping["suggested_id"] == "ID_8"
    assert duplicate_candidate["evidence"]["auto_fix_safe"] is False

    sequence_candidate = next(
        issue
        for issue in report["issues"]
        if issue["code"] == "probable_sequence_identity_substitution"
    )
    assert sequence_candidate["pig_id"] == "ID_4"
    assert sequence_candidate["evidence"]["suggested_id"] == "ID_5"
    assert sequence_candidate["observed_slots"] == [4, 5]


def test_tracking_xml_audit_reports_duplicate_and_missing_identity(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "tracking.xml"
    _write_tracking_xml(xml_path, duplicate_last_frame=True)

    report = audit_tracking_xml(
        xml_path,
        expected_pig_ids=("ID_1", "ID_2"),
    )

    assert report["status"] == "FAIL"
    duplicate = next(
        issue
        for issue in report["issues"]
        if issue["code"] == "duplicate_pig_id_in_frame"
    )
    assert duplicate["frame_id"] == 1
    assert duplicate["frame_position_1based"] == 2
    assert duplicate["total_frames"] == 2
    assert duplicate["image_name"] == "Synthetic_tracking__f000001.jpg"

    missing = next(
        issue
        for issue in report["issues"]
        if issue["code"] == "missing_expected_pig_id_in_frame"
    )
    assert missing["pig_id"] == "ID_2"
    assert missing["frame_id"] == 1

    candidate = next(
        issue
        for issue in report["issues"]
        if issue["code"]
        == "probable_duplicate_identity_substitution"
    )
    suggested = {
        row["track_label"]: row["suggested_id"]
        for row in candidate["evidence"]["suggested_mapping"]
    }
    assert suggested == {"Pig_1": "ID_1", "Pig_2": "ID_2"}
    assert candidate["evidence"]["auto_fix_safe"] is False


def test_clean_tracking_xml_and_combined_report_pass(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "tracking.xml"
    _write_tracking_xml(xml_path, duplicate_last_frame=False)

    source = audit_tracking_xml(
        xml_path,
        expected_pig_ids=("ID_1", "ID_2"),
    )
    combined = combine_annotation_audits([source])

    assert source["status"] == "PASS"
    assert source["issues"] == []
    assert combined["status"] == "PASS"
    assert combined["summary"]["issue_count"] == 0


def test_minor_interpolation_overshoot_is_informational(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "tracking.xml"
    _write_tracking_xml(xml_path, duplicate_last_frame=False)
    tree = ET.parse(xml_path)
    first_box = tree.getroot().find("./track/box")
    assert first_box is not None
    first_box.set("ybr", "721.0")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    source = audit_tracking_xml(
        xml_path,
        expected_pig_ids=("ID_1", "ID_2"),
    )

    assert source["status"] == "PASS"
    issue = next(
        item
        for item in source["issues"]
        if item["code"] == "bbox_minor_boundary_overshoot"
    )
    assert issue["severity"] == "info"
    assert issue["evidence"]["max_overshoot_px"] == 1.0


def test_tracking_boundary_attention_does_not_claim_bbox_is_wrong(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "tracking.xml"
    _write_tracking_xml(xml_path, duplicate_last_frame=False)
    tree = ET.parse(xml_path)
    first_box = tree.getroot().find("./track/box")
    assert first_box is not None
    first_box.set("ybr", "750.0")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    source = audit_tracking_xml(
        xml_path,
        expected_pig_ids=("ID_1", "ID_2"),
    )

    assert source["status"] == "PASS"
    issue = next(
        item
        for item in source["issues"]
        if item["code"] == "bbox_boundary_attention"
    )
    assert issue["severity"] == "info"
    assert issue["evidence"]["requires_visual_attention"] is True
    assert issue["evidence"]["review_status_inferred"] is False
