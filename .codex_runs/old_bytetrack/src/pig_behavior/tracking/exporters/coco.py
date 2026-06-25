"""COCO annotation export for pig tracking shapes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pig_behavior.tracking.constants import BEHAVIOR_VALUES, ID_VALUES
from pig_behavior.tracking.refinement import _shape_attributes_dict


def write_coco_annotation_json(
    path: Path,
    shapes: list[dict[str, Any]],
    video_path: Path,
    frame_width: int,
    frame_height: int,
    default_behavior: str,
    description: str = "Pig ID tracking annotations exported as COCO 1.0",
) -> None:
    """Write COCO 1.0 instances with Pig_N categories and CVAT attributes."""
    frames = sorted({int(shape["frame"]) for shape in shapes})
    image_id_by_frame = {frame: idx + 1 for idx, frame in enumerate(frames)}
    category_id_by_name = {f"Pig_{idx}": idx for idx in range(1, len(ID_VALUES) + 1)}

    images = [
        {
            "id": image_id_by_frame[frame],
            "file_name": f"frame_{frame:06d}.jpg",
            "width": int(frame_width),
            "height": int(frame_height),
            "frame": int(frame),
            "video": video_path.name,
        }
        for frame in frames
    ]

    annotations = []
    for annotation_id, shape in enumerate(shapes, start=1):
        x1, y1, x2, y2 = [float(value) for value in shape["points"]]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        label = str(shape["label"])
        attributes = _shape_attributes_dict(shape)
        fixed_id = int(label.removeprefix("Pig_"))
        attributes["track_id"] = fixed_id
        attributes["instance_id"] = fixed_id
        attributes.setdefault("Behavior", default_behavior)
        attributes.setdefault("Hidden", "No")
        attributes["TrackSource"] = str(shape.get("_track_source", "unknown"))
        attributes["NeedsReview"] = "Yes" if shape.get("_needs_review") else "No"
        attributes["Refined"] = "Yes" if shape.get("_refined") else "No"
        attributes["RefineReason"] = str(shape.get("_refine_reason", ""))
        attributes["MotionState"] = str(shape.get("_motion_state", "unknown"))
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id_by_frame[int(shape["frame"])],
                "category_id": category_id_by_name[label],
                "track_id": fixed_id,
                "instance_id": fixed_id,
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(width, 2),
                    round(height, 2),
                ],
                "area": round(width * height, 2),
                "segmentation": [],
                "iscrowd": 0,
                "score": float(shape.get("score", 1.0)),
                "attributes": attributes,
            }
        )

    categories = [
        {
            "id": idx,
            "name": f"Pig_{idx}",
            "supercategory": "Pig",
            "attributes": {
                "ID": [f"ID_{idx}"],
                "Behavior": BEHAVIOR_VALUES,
                "Hidden": ["No", "Yes"],
            },
        }
        for idx in range(1, len(ID_VALUES) + 1)
    ]

    payload = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


__all__ = ["write_coco_annotation_json"]
