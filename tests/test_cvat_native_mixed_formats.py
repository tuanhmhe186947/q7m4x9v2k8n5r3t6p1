from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from legacy_burst_recovery.cvat_behavior_overlay import load_cvat_legacy_rows
from pig_behavior.data.cvat_native import (
    load_all_cvat_tasks,
    load_cvat_task,
    parse_attrs,
)


def _write_task(
    root: Path,
    task_name: str,
    *,
    json_behavior: str,
    xml_behavior: str | None = None,
    xml_image_name: str | None = None,
) -> Path:
    task_dir = root / task_name
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True)
    image_name = f"burst_{task_name}_f0_k0.jpg"

    manifest = [
        {"version": "1.1"},
        {"type": "images"},
        {
            "name": Path(image_name).stem,
            "extension": ".jpg",
            "width": 1280,
            "height": 720,
        },
    ]
    (data_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item) for item in manifest) + "\n",
        encoding="utf-8",
    )
    (task_dir / "task.json").write_text(
        json.dumps({"name": task_name, "subset": "Train"}),
        encoding="utf-8",
    )

    shape = {
        "type": "rectangle",
        "label": "Pig",
        "frame": 0,
        "outside": False,
        "points": [10.0, 20.0, 30.0, 40.0],
        "attributes": [
            {"name": "ID", "value": "ID_1"},
            {"name": "Behavior", "value": json_behavior},
            {"name": "Hidden", "value": "No"},
        ],
    }
    (task_dir / "annotations.json").write_text(
        json.dumps([{"version": 0, "shapes": [shape]}]),
        encoding="utf-8",
    )

    if xml_behavior is not None:
        root_element = ET.Element("annotations")
        image = ET.SubElement(
            root_element,
            "image",
            {
                "id": "0",
                "name": xml_image_name or image_name,
                "width": "1280",
                "height": "720",
            },
        )
        box = ET.SubElement(
            image,
            "box",
            {
                "label": "Pig",
                "source": "manual",
                "xtl": "11.0",
                "ytl": "21.0",
                "xbr": "31.0",
                "ybr": "41.0",
            },
        )
        for name, value in [
            ("ID", "ID_1"),
            ("Behavior", xml_behavior),
            ("Hidden", "Yes"),
        ]:
            attribute = ET.SubElement(box, "attribute", {"name": name})
            attribute.text = value
        ET.ElementTree(root_element).write(
            task_dir / "annotations.xml",
            encoding="utf-8",
            xml_declaration=True,
        )
    return task_dir


def test_missing_hidden_attribute_is_not_silently_changed_to_no() -> None:
    parsed = parse_attrs(
        [
            {"name": "ID", "value": "ID_1"},
            {"name": "Behavior", "value": "stand"},
        ]
    )

    assert parsed["Hidden"] is None


def test_xml_is_authority_when_json_and_xml_both_exist(tmp_path: Path) -> None:
    task_dir = _write_task(
        tmp_path,
        "task_0",
        json_behavior="stand",
        xml_behavior="fight",
    )

    loaded = load_cvat_task(task_dir)

    assert len(loaded) == 1
    row = loaded.iloc[0]
    assert row["annotation_format"] == "xml"
    assert Path(row["annotation_path"]).name == "annotations.xml"
    assert row["behavior"] == "fight"
    assert row["hidden"] == "Yes"
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == (
        11.0,
        21.0,
        31.0,
        41.0,
    )


def test_mixed_xml_and_json_tasks_share_one_normalized_schema(
    tmp_path: Path,
) -> None:
    _write_task(
        tmp_path,
        "task_0",
        json_behavior="stand",
        xml_behavior="fight",
    )
    _write_task(
        tmp_path,
        "task_1",
        json_behavior="drink",
    )

    loaded = load_all_cvat_tasks(tmp_path).sort_values("task")

    assert len(loaded) == 2
    assert loaded["annotation_format"].tolist() == ["xml", "json"]
    assert loaded["behavior"].tolist() == ["fight", "drink"]
    assert loaded["pig_id"].tolist() == ["ID_1", "ID_1"]


def test_xml_manifest_name_mismatch_fails_closed(tmp_path: Path) -> None:
    task_dir = _write_task(
        tmp_path,
        "task_0",
        json_behavior="stand",
        xml_behavior="fight",
        xml_image_name="wrong_f0_k0.jpg",
    )

    with pytest.raises(ValueError, match="XML image/manifest mismatch"):
        load_cvat_task(task_dir)


def test_provenance_hashes_only_selected_annotation_authority(
    tmp_path: Path,
) -> None:
    task_0 = _write_task(
        tmp_path,
        "task_0",
        json_behavior="stand",
        xml_behavior="fight",
    )
    task_1 = _write_task(
        tmp_path,
        "task_1",
        json_behavior="drink",
    )

    loaded, source_files = load_cvat_legacy_rows(tmp_path)
    annotation_paths = {
        Path(item["path"])
        for item in source_files
        if Path(item["path"]).name.startswith("annotations.")
    }

    assert len(loaded) == 2
    assert task_0 / "annotations.xml" in annotation_paths
    assert task_0 / "annotations.json" not in annotation_paths
    assert task_1 / "annotations.json" in annotation_paths
