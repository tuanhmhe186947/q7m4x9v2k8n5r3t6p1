"""CVAT video XML export for fixed-ID pig tracks."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
from xml.dom import minidom
from xml.etree import ElementTree as ET

from pig_behavior.tracking.constants import ID_VALUES, PIG_LABEL_SCHEMA
from pig_behavior.tracking.refinement import _shape_attributes_dict


def _xml_child(parent: ET.Element, tag: str, text: Any = "") -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = str(text)
    return child


def _append_cvat_xml_label(parent: ET.Element, label: dict[str, Any]) -> None:
    label_el = ET.SubElement(parent, "label")
    _xml_child(label_el, "name", label["name"])
    _xml_child(label_el, "type", label.get("type", "any"))
    attrs_el = ET.SubElement(label_el, "attributes")
    for attribute in label.get("attributes", []):
        attr_el = ET.SubElement(attrs_el, "attribute")
        _xml_child(attr_el, "name", attribute["name"])
        _xml_child(attr_el, "mutable", str(bool(attribute["mutable"])))
        _xml_child(attr_el, "input_type", attribute["input_type"])
        _xml_child(attr_el, "default_value", attribute["default_value"])
        _xml_child(attr_el, "values", "\n".join(attribute.get("values", [])))


def write_cvat_video_xml(
    path: Path,
    shapes: list[dict[str, Any]],
    video_path: Path,
    frame_width: int,
    frame_height: int,
    frame_count: int,
) -> None:
    """Write native CVAT for video 1.1 XML with real track elements."""
    root = ET.Element("annotations")
    _xml_child(root, "version", "1.1")

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    _xml_child(task, "id", 0)
    _xml_child(task, "name", video_path.stem)
    _xml_child(task, "size", frame_count)
    _xml_child(task, "mode", "interpolation")
    _xml_child(task, "overlap", 0)
    _xml_child(task, "bugtracker", "")
    _xml_child(task, "flipped", "False")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    _xml_child(task, "created", now)
    _xml_child(task, "updated", now)

    labels_el = ET.SubElement(task, "labels")
    for label in PIG_LABEL_SCHEMA:
        _append_cvat_xml_label(labels_el, label)

    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    _xml_child(segment, "id", 0)
    _xml_child(segment, "start", 0)
    _xml_child(segment, "stop", max(0, frame_count - 1))
    _xml_child(segment, "url", "")

    owner = ET.SubElement(task, "owner")
    _xml_child(owner, "username", "auto")
    _xml_child(owner, "email", "")

    original_size = ET.SubElement(task, "original_size")
    _xml_child(original_size, "width", int(frame_width))
    _xml_child(original_size, "height", int(frame_height))
    _xml_child(meta, "dumped", now)

    shapes_by_track: dict[int, list[dict[str, Any]]] = {
        fixed_id: [] for fixed_id in range(1, len(ID_VALUES) + 1)
    }
    for shape in shapes:
        fixed_id = int(str(shape["label"]).removeprefix("Pig_"))
        shapes_by_track[fixed_id].append(shape)

    for fixed_id in range(1, len(ID_VALUES) + 1):
        track = ET.SubElement(
            root,
            "track",
            {
                "id": str(fixed_id),
                "label": f"Pig_{fixed_id}",
                "source": "auto",
            },
        )
        for shape in sorted(shapes_by_track[fixed_id], key=lambda item: item["frame"]):
            x1, y1, x2, y2 = [float(value) for value in shape["points"]]
            attributes = _shape_attributes_dict(shape)
            outside_val = "1" if shape.get("outside", False) else "0"
            occluded_val = "1" if shape.get("occluded", False) else "0"
            box = ET.SubElement(
                track,
                "box",
                {
                    "frame": str(int(shape["frame"])),
                    "xtl": f"{x1:.2f}",
                    "ytl": f"{y1:.2f}",
                    "xbr": f"{x2:.2f}",
                    "ybr": f"{y2:.2f}",
                    "outside": outside_val,
                    "occluded": occluded_val,
                    "keyframe": "1",
                },
            )
            for name in ("ID", "Behavior", "Hidden"):
                _xml_child(box, "attribute", attributes.get(name, "")).set(
                    "name",
                    name,
                )

    raw_xml = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(raw_xml).toprettyxml(
        indent="  ",
        encoding="utf-8",
    )
    path.write_bytes(pretty_xml)


__all__ = [
    "_append_cvat_xml_label",
    "_xml_child",
    "write_cvat_video_xml",
]
