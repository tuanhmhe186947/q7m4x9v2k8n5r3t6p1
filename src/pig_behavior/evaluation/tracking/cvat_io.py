"""Logic for CVAT XML parsing, task size, and task name."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from .frame_window import frame_is_in_bounds, validate_frame_bounds


@dataclass(slots=True)
class TrackingObject:
    """One box in a frame."""

    frame: int
    obj_id: str
    bbox: tuple[float, float, float, float]
    hidden: bool = False
    source_track_id: str = ""
    label: str = ""


def id_from_label(label: str, fallback: str) -> str:
    """Convert CVAT labels like Pig_3 to stable IDs like ID_3."""
    match = re.search(r"(?:pig|id)[_\-\s]*(\d+)", label, flags=re.IGNORECASE)
    if match:
        return f"ID_{int(match.group(1))}"
    return fallback


def box_hidden(box_el: ET.Element) -> bool:
    """Return whether a CVAT box has Hidden=Yes."""
    for attr in box_el.findall("attribute"):
        if attr.attrib.get("name") == "Hidden":
            return (attr.text or "").strip().lower() == "yes"
    return False


def box_id(box_el: ET.Element, track_label: str, track_id: str) -> str:
    """Read object identity from box attribute, track label, or track id."""
    for attr in box_el.findall("attribute"):
        if attr.attrib.get("name") == "ID" and attr.text:
            return attr.text.strip()
    return id_from_label(track_label, fallback=f"track_{track_id}")


def is_outside(box_el: ET.Element) -> bool:
    """Return whether a CVAT box is marked outside."""
    return str(box_el.attrib.get("outside", "0")).lower() in {"1", "true", "yes"}


def parse_cvat_video_xml(
    xml_path: Path,
    *,
    include_hidden: bool = False,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> dict[int, list[TrackingObject]]:
    """Parse CVAT boxes inside optional inclusive frame bounds."""
    validate_frame_bounds(start_frame, end_frame)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    by_frame: dict[int, list[TrackingObject]] = defaultdict(list)

    for track_el in root.findall("track"):
        track_id = str(track_el.attrib.get("id", ""))
        label = str(track_el.attrib.get("label", ""))
        for box_el in track_el.findall("box"):
            if is_outside(box_el):
                continue
            frame = int(box_el.attrib["frame"])
            if not frame_is_in_bounds(frame, start_frame, end_frame):
                continue
            hidden = box_hidden(box_el)
            if hidden and not include_hidden:
                continue
            bbox = (
                float(box_el.attrib["xtl"]),
                float(box_el.attrib["ytl"]),
                float(box_el.attrib["xbr"]),
                float(box_el.attrib["ybr"]),
            )
            obj = TrackingObject(
                frame=frame,
                obj_id=box_id(box_el, label, track_id),
                bbox=bbox,
                hidden=hidden,
                source_track_id=track_id,
                label=label,
            )
            by_frame[frame].append(obj)

    return dict(sorted(by_frame.items()))


def read_cvat_task_size(xml_path: Path) -> int | None:
    """Read task size from CVAT XML metadata."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    size_el = root.find("./meta/task/size")
    if size_el is None or size_el.text is None:
        return None
    try:
        return int(size_el.text)
    except ValueError:
        return None


def read_task_name(xml_path: Path) -> str:
    """Read CVAT task name from XML."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return ""
    name_el = root.find("./meta/task/name")
    return name_el.text or "" if name_el is not None else ""
