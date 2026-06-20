"""Temporary CVAT XML hotfix for a known ID swap.

This script swaps the boxes of two CVAT video tracks from a given frame onward.
It is intended for one-off manual repair when two identities were swapped in an
already exported CVAT 1.1 XML file.
"""

from __future__ import annotations

import argparse
import copy
import shutil
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = (
    PROJECT_ROOT
    / "outputs"
    / "id_tracking"
    / "Pigs291119_000302_30fps"
    / "Pigs291119_000302_30fps_annotations_cvat_video_1_1.xml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Swap two CVAT video tracks from a start time/frame onward.",
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--track-a", type=int, default=5)
    parser.add_argument("--track-b", type=int, default=7)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start-second", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def find_track(root: ET.Element, fixed_id: int) -> ET.Element:
    label = f"Pig_{fixed_id}"
    for track in root.findall("track"):
        if track.get("id") == str(fixed_id) or track.get("label") == label:
            return track
    raise ValueError(f"Could not find CVAT track for {label}.")


def boxes_by_frame(track: ET.Element) -> dict[int, ET.Element]:
    boxes: dict[int, ET.Element] = {}
    for box in track.findall("box"):
        frame = int(box.attrib["frame"])
        if frame in boxes:
            raise ValueError(
                f"Track {track.get('label')} has duplicate box at frame {frame}."
            )
        boxes[frame] = box
    return boxes


def attribute_children(box: ET.Element) -> dict[str, ET.Element]:
    return {
        child.attrib["name"]: child
        for child in box.findall("attribute")
        if "name" in child.attrib
    }


def ensure_attribute(box: ET.Element, name: str) -> ET.Element:
    attrs = attribute_children(box)
    if name in attrs:
        return attrs[name]
    child = ET.SubElement(box, "attribute", {"name": name})
    child.text = ""
    return child


def swap_box_payload(
    box_a: ET.Element,
    box_b: ET.Element,
    track_a: int,
    track_b: int,
) -> None:
    """Swap geometry/state and mutable attributes while keeping track IDs fixed."""
    frame_a = box_a.attrib["frame"]
    frame_b = box_b.attrib["frame"]

    attrs_a = {key: value for key, value in box_a.attrib.items() if key != "frame"}
    attrs_b = {key: value for key, value in box_b.attrib.items() if key != "frame"}
    for key in list(box_a.attrib):
        if key != "frame":
            del box_a.attrib[key]
    for key in list(box_b.attrib):
        if key != "frame":
            del box_b.attrib[key]
    box_a.attrib.update(attrs_b)
    box_b.attrib.update(attrs_a)
    box_a.attrib["frame"] = frame_a
    box_b.attrib["frame"] = frame_b

    child_values_a = {
        name: copy.deepcopy(child.text)
        for name, child in attribute_children(box_a).items()
        if name != "ID"
    }
    child_values_b = {
        name: copy.deepcopy(child.text)
        for name, child in attribute_children(box_b).items()
        if name != "ID"
    }
    for name in sorted(set(child_values_a) | set(child_values_b)):
        ensure_attribute(box_a, name).text = child_values_b.get(name, "")
        ensure_attribute(box_b, name).text = child_values_a.get(name, "")

    ensure_attribute(box_a, "ID").text = f"ID_{track_a}"
    ensure_attribute(box_b, "ID").text = f"ID_{track_b}"


def main() -> int:
    args = parse_args()
    xml_path = args.xml.resolve()
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    start_frame = (
        args.start_frame
        if args.start_frame is not None
        else int(round(args.start_second * args.fps))
    )
    tree = ET.parse(xml_path)
    root = tree.getroot()
    track_a = find_track(root, args.track_a)
    track_b = find_track(root, args.track_b)
    boxes_a = boxes_by_frame(track_a)
    boxes_b = boxes_by_frame(track_b)
    frames = sorted(set(boxes_a) & set(boxes_b))
    frames_to_swap = [frame for frame in frames if frame >= start_frame]
    if not frames_to_swap:
        raise ValueError(f"No common boxes found from frame {start_frame}.")

    missing_a = sorted(
        frame
        for frame in set(boxes_b)
        if frame >= start_frame and frame not in boxes_a
    )
    missing_b = sorted(
        frame
        for frame in set(boxes_a)
        if frame >= start_frame and frame not in boxes_b
    )
    if missing_a or missing_b:
        raise ValueError(
            "Tracks do not have matching frame coverage after start frame: "
            f"missing track {args.track_a} frames={missing_a[:5]}, "
            f"missing track {args.track_b} frames={missing_b[:5]}"
        )

    if args.dry_run:
        print(
            "[DRY-RUN] would swap "
            f"track {args.track_a}<->{args.track_b} from frame {start_frame}; "
            f"frames={len(frames_to_swap)}"
        )
        return 0

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = xml_path.with_name(
            f"{xml_path.stem}.bak_before_swap_{args.track_a}_{args.track_b}_"
            f"from_frame_{start_frame}_{stamp}{xml_path.suffix}"
        )
        shutil.copy2(xml_path, backup_path)
        print(f"[OK] backup: {backup_path}")

    for frame in frames_to_swap:
        swap_box_payload(boxes_a[frame], boxes_b[frame], args.track_a, args.track_b)

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(
        "[OK] fixed: "
        f"{xml_path} | swapped track {args.track_a}<->{args.track_b} "
        f"from frame {start_frame} ({args.start_second:.2f}s at {args.fps:.2f} fps), "
        f"frames={len(frames_to_swap)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
