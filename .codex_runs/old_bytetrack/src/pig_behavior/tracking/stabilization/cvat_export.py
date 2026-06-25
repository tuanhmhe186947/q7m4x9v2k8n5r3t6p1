"""CVAT video XML export with outside/gap tracking for stable annotations."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

import numpy as np

from pig_behavior.tracking.constants import PIG_LABEL_SCHEMA
from pig_behavior.tracking.exporters.cvat_xml import _append_cvat_xml_label, _xml_child


def write_stable_cvat_xml(
    path: Path | str,
    stable_tracks: dict[int, dict[int, tuple[np.ndarray, str, bool]]],
    video_path: Path | str,
    frame_width: int,
    frame_height: int,
    frame_count: int,
    expected_pigs: int = 8,
) -> None:
    """Writes stable CVAT 1.1 XML with explicit outside="1" gaps for missing frames.

    Args:
        path: Output path for the CVAT XML file.
        stable_tracks: Dict mapping track_id to a dict of {frame: (bbox, behavior_str, is_hidden)}.
        video_path: Path to the processed video.
        frame_width: Width of the video frames.
        frame_height: Height of the video frames.
        frame_count: Total frame count of the video.
        expected_pigs: Number of pigs/tracks to export (default 8).
    """
    path = Path(path)
    video_path = Path(video_path)

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

    # If any stable IDs are outside PIG_LABEL_SCHEMA (e.g. > 8), we can add custom labels
    # but the standard CVAT schema is 1-8. Let's stick to 1 to max(8, expected_pigs)
    max_track_id = max([expected_pigs] + list(stable_tracks.keys()))
    if max_track_id > 8:
        # Dynamically append labels if needed, or map extra tracks into a warning
        # Normally max_track_id is 8 or less because expected_pigs = 8.
        pass

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

    for track_id in range(1, max_track_id + 1):
        track_data = stable_tracks.get(track_id, {})
        track = ET.SubElement(
            root,
            "track",
            {
                "id": str(track_id),
                "label": f"Pig_{track_id}",
                "source": "auto",
            },
        )

        if not track_data:
            continue

        sorted_frames = sorted(list(track_data.keys()))
        first_frame = sorted_frames[0]
        last_frame = sorted_frames[-1]

        # Slide through all frames from first to last to write interpolation keyframes
        # and mark outside="1" / outside="0" transitions.
        is_outside = False

        for f in range(first_frame, frame_count):
            if f in track_data:
                bbox = track_data[f][0]
                behavior = track_data[f][2]
                is_hidden = track_data[f][3]
                x1, y1, x2, y2 = bbox

                # Transitioning from outside to inside
                if is_outside or f == first_frame:
                    is_outside = False
                    box = ET.SubElement(
                        track,
                        "box",
                        {
                            "frame": str(f),
                            "xtl": f"{x1:.2f}",
                            "ytl": f"{y1:.2f}",
                            "xbr": f"{x2:.2f}",
                            "ybr": f"{y2:.2f}",
                            "outside": "0",
                            "occluded": "1" if is_hidden else "0",
                            "keyframe": "1",
                        },
                    )
                    # Add attributes
                    for name, val in [
                        ("ID", f"ID_{track_id}"),
                        ("Behavior", behavior),
                        ("Hidden", "Yes" if is_hidden else "No"),
                    ]:
                        _xml_child(box, "attribute", val).set("name", name)
                else:
                    # Write regular keyframe (optional, but let's write it to lock the bbox)
                    # For CVAT XML, writing every frame is robust and simple.
                    box = ET.SubElement(
                        track,
                        "box",
                        {
                            "frame": str(f),
                            "xtl": f"{x1:.2f}",
                            "ytl": f"{y1:.2f}",
                            "xbr": f"{x2:.2f}",
                            "ybr": f"{y2:.2f}",
                            "outside": "0",
                            "occluded": "1" if is_hidden else "0",
                            "keyframe": "1",
                        },
                    )
                    for name, val in [
                        ("ID", f"ID_{track_id}"),
                        ("Behavior", behavior),
                        ("Hidden", "Yes" if is_hidden else "No"),
                    ]:
                        _xml_child(box, "attribute", val).set("name", name)

            else:
                # Missing from track data
                if not is_outside:
                    is_outside = True
                    # Write outside="1" transition at the first missing frame
                    # Re-use last known box or dummy
                    prev_f = max(first_frame, f - 1)
                    if prev_f in track_data:
                        bbox = track_data[prev_f][0]
                        behavior = track_data[prev_f][2]
                        is_hidden = track_data[prev_f][3]
                    else:
                        bbox = np.zeros(4, dtype=np.float32)
                        behavior = "lying"
                        is_hidden = False

                    x1, y1, x2, y2 = bbox
                    box = ET.SubElement(
                        track,
                        "box",
                        {
                            "frame": str(f),
                            "xtl": f"{x1:.2f}",
                            "ytl": f"{y1:.2f}",
                            "xbr": f"{x2:.2f}",
                            "ybr": f"{y2:.2f}",
                            "outside": "1",
                            "occluded": "0",
                            "keyframe": "1",
                        },
                    )
                    for name, val in [
                        ("ID", f"ID_{track_id}"),
                        ("Behavior", behavior),
                        ("Hidden", "Yes" if is_hidden else "No"),
                    ]:
                        _xml_child(box, "attribute", val).set("name", name)

                    # Past the last frame, the track stays outside so we can stop writing keyframes.
                    if f > last_frame:
                        break

    raw_xml = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(raw_xml).toprettyxml(
        indent="  ",
        encoding="utf-8",
    )
    path.write_bytes(pretty_xml)
