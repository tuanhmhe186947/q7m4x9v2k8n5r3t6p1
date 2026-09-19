"""Parse CVAT task exports from JSON or image-oriented XML annotations."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_BEHAVIORS = [
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
]

PIG_LABEL_PREFIX = "pig"
ANNOTATION_SOURCE_PRIORITY = ("xml", "json")


def load_behaviors_from_project(project_json: Path) -> list[str]:
    """Read allowed Behavior values from CVAT project.json when available."""
    if not project_json.exists():
        return DEFAULT_BEHAVIORS
    with project_json.open("r", encoding="utf-8") as f:
        project = json.load(f)
    for label in project.get("labels", []):
        for attr in label.get("attributes", []):
            if attr.get("name") == "Behavior" and attr.get("values"):
                return list(attr["values"])
    return DEFAULT_BEHAVIORS


def parse_attrs(attrs: Any) -> dict[str, str | None]:
    """Normalize CVAT attribute list/dict to ID, Behavior, Hidden fields."""
    parsed: dict[str, str | None] = {"ID": None, "Behavior": None, "Hidden": None}
    if isinstance(attrs, dict):
        for key in parsed:
            parsed[key] = attrs.get(key, parsed[key])
    elif isinstance(attrs, list):
        for attr in attrs:
            name = attr.get("name")
            if name in parsed:
                parsed[name] = attr.get("value")
    return parsed


def parse_burst_from_filename(img_name: str) -> tuple[str, int]:
    """Parse burst group and sequence order from frame file names."""
    stem = Path(img_name).stem
    parts = stem.split("_")
    order = 0
    if parts and parts[-1].startswith("k"):
        try:
            order = int(parts[-1][1:])
        except ValueError:
            order = 0
    if len(parts) >= 2 and parts[-2].startswith("f"):
        group_id = "_".join(parts[:-2])
    else:
        group_id = stem
    return group_id, order


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load CVAT imageset manifest entries that map frame index to image name."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    frames = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "name" in item and "extension" in item:
                frames.append(item)
    return frames


def frame_file_name(frame_info: dict[str, Any]) -> str:
    """Return image file name from a CVAT manifest row."""
    name = frame_info["name"]
    ext = frame_info.get("extension", "")
    return name if name.lower().endswith(ext.lower()) else f"{name}{ext}"


def select_cvat_annotation_source(task_dir: Path) -> tuple[str, Path]:
    """Select one annotation authority for a task without mixing formats."""
    candidates = {
        "xml": task_dir / "annotations.xml",
        "json": task_dir / "annotations.json",
    }
    for annotation_format in ANNOTATION_SOURCE_PRIORITY:
        path = candidates[annotation_format]
        if path.exists():
            return annotation_format, path
    raise FileNotFoundError(
        f"No annotations.xml or annotations.json found under {task_dir}"
    )


def _xml_attributes(box: ET.Element) -> list[dict[str, str | None]]:
    return [
        {
            "name": attribute.get("name"),
            "value": attribute.text,
        }
        for attribute in box.findall("attribute")
    ]


def _load_xml_shapes(
    annotations_path: Path,
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    root = ET.parse(annotations_path).getroot()
    images = root.findall("image")
    if not images:
        raise ValueError(
            f"CVAT XML must use image annotations, not tracks: {annotations_path}"
        )

    shapes: list[dict[str, Any]] = []
    seen_frame_ids: set[int] = set()
    for image in images:
        frame_idx = int(image.get("id", "-1"))
        if frame_idx in seen_frame_ids:
            raise ValueError(
                f"Duplicate XML image id {frame_idx} in {annotations_path}"
            )
        seen_frame_ids.add(frame_idx)
        if frame_idx < 0 or frame_idx >= len(manifest):
            raise ValueError(
                f"XML image id {frame_idx} is outside the task manifest"
            )

        manifest_name = frame_file_name(manifest[frame_idx])
        xml_name = str(image.get("name", ""))
        if manifest_name != xml_name:
            raise ValueError(
                "XML image/manifest mismatch at frame "
                f"{frame_idx}: {xml_name!r} != {manifest_name!r}"
            )

        for box in image.findall("box"):
            shapes.append(
                {
                    "type": "rectangle",
                    "label": box.get("label", ""),
                    "frame": frame_idx,
                    "outside": False,
                    "points": [
                        box.get("xtl"),
                        box.get("ytl"),
                        box.get("xbr"),
                        box.get("ybr"),
                    ],
                    "attributes": _xml_attributes(box),
                    "source": box.get("source"),
                }
            )
    return shapes


def _load_json_shapes(annotations_path: Path) -> list[dict[str, Any]]:
    with annotations_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)
    return [
        shape
        for annotation_obj in annotations
        for shape in annotation_obj.get("shapes", [])
    ]


def load_cvat_task(task_dir: Path) -> pd.DataFrame:
    """Load one CVAT native task folder into a flat dataframe."""
    task_json_path = task_dir / "task.json"
    manifest_path = task_dir / "data" / "manifest.jsonl"

    if not task_json_path.exists():
        print(f"[WARN] skip incomplete task: {task_dir}")
        return pd.DataFrame()

    try:
        annotation_format, annotations_path = select_cvat_annotation_source(
            task_dir
        )
    except FileNotFoundError:
        print(f"[WARN] skip incomplete task: {task_dir}")
        return pd.DataFrame()

    with task_json_path.open("r", encoding="utf-8") as f:
        task_json = json.load(f)

    manifest = load_manifest(manifest_path)
    burst_first_task_frame: dict[str, int] = {}
    for task_frame, frame_info in enumerate(manifest):
        manifest_image_name = frame_file_name(frame_info)
        manifest_group_id, _ = parse_burst_from_filename(
            manifest_image_name
        )
        burst_first_task_frame.setdefault(manifest_group_id, task_frame)
    if annotation_format == "xml":
        shapes = _load_xml_shapes(annotations_path, manifest)
    else:
        shapes = _load_json_shapes(annotations_path)
    subset = task_json.get("subset") or task_dir.name
    rows: list[dict[str, Any]] = []

    for shape in shapes:
        if shape.get("type") != "rectangle":
            continue
        if shape.get("outside") is True:
            continue
        label = str(shape.get("label", ""))
        if PIG_LABEL_PREFIX and not label.lower().startswith(PIG_LABEL_PREFIX):
            continue

        frame_idx = int(shape.get("frame", -1))
        if frame_idx < 0 or frame_idx >= len(manifest):
            print(f"[WARN] skip invalid frame {frame_idx} in {task_dir.name}")
            continue

        points = shape.get("points", [])
        if len(points) != 4:
            continue
        x1, y1, x2, y2 = map(float, points)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        frame_info = manifest[frame_idx]
        img_name = frame_file_name(frame_info)
        group_id, order = parse_burst_from_filename(img_name)
        first_task_frame = burst_first_task_frame[group_id]
        raw_attrs = shape.get("attributes", [])
        attrs = parse_attrs(raw_attrs)
        hidden_value = attrs["Hidden"]
        if hidden_value is None:
            hidden_value = "No"

        rows.append(
            {
                "task": task_dir.name,
                "subset": subset,
                "frame": frame_idx,
                "img_name": img_name,
                "image_path": str(task_dir / "data" / img_name),
                "width": frame_info.get("width"),
                "height": frame_info.get("height"),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "pig_id": attrs["ID"],
                "behavior": attrs["Behavior"],
                "hidden": hidden_value,
                "hidden_attribute_present": _has_attribute(
                    raw_attrs,
                    "Hidden",
                ),
                "group_id": group_id,
                "order": order,
                "burst_first_task_frame": first_task_frame,
                "is_burst_first_task_frame": frame_idx == first_task_frame,
                "category_name": label,
                "source": shape.get("source"),
                "annotation_format": annotation_format,
                "annotation_path": str(annotations_path),
            }
        )

    df = pd.DataFrame(rows)
    print(f"[LOAD] {task_dir.name}: {len(df)} boxes | subset={subset!r}")
    return df


def _has_attribute(attrs: Any, name: str) -> bool:
    if isinstance(attrs, dict):
        return name in attrs
    if isinstance(attrs, list):
        return any(
            isinstance(item, dict) and item.get("name") == name
            for item in attrs
        )
    return False


def load_all_cvat_tasks(export_root: Path) -> pd.DataFrame:
    """Load all task_* folders under the CVAT export root."""
    task_dirs = sorted(p for p in export_root.glob("task_*") if p.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No task_* folders found under {export_root}")

    frames = [load_cvat_task(task_dir) for task_dir in task_dirs]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        raise ValueError("No CVAT rectangle annotations were loaded.")
    return df
