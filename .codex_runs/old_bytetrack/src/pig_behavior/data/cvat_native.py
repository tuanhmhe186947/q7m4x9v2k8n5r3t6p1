"""Parse CVAT native export formats (task.json, annotations.json, manifest.jsonl)."""

from __future__ import annotations

import json
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
    parsed: dict[str, str | None] = {"ID": None, "Behavior": None, "Hidden": "No"}
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


def load_cvat_task(task_dir: Path) -> pd.DataFrame:
    """Load one CVAT native task folder into a flat dataframe."""
    task_json_path = task_dir / "task.json"
    annotations_path = task_dir / "annotations.json"
    manifest_path = task_dir / "data" / "manifest.jsonl"

    if not task_json_path.exists() or not annotations_path.exists():
        print(f"[WARN] skip incomplete task: {task_dir}")
        return pd.DataFrame()

    with task_json_path.open("r", encoding="utf-8") as f:
        task_json = json.load(f)
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    manifest = load_manifest(manifest_path)
    subset = task_json.get("subset") or task_dir.name
    rows: list[dict[str, Any]] = []

    for annotation_obj in annotations:
        for shape in annotation_obj.get("shapes", []):
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
            attrs = parse_attrs(shape.get("attributes", []))

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
                    "hidden": attrs["Hidden"] or "No",
                    "group_id": group_id,
                    "order": order,
                    "category_name": label,
                    "source": shape.get("source"),
                }
            )

    df = pd.DataFrame(rows)
    print(f"[LOAD] {task_dir.name}: {len(df)} boxes | subset={subset!r}")
    return df


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
