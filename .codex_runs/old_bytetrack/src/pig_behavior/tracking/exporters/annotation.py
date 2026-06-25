"""CVAT shape JSON export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def strip_internal_shape_keys(shape: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in shape.items() if not key.startswith("_")}


def write_annotation_json(path: Path, shapes: list[dict[str, Any]]) -> None:
    payload = [
        {
            "version": 0,
            "tags": [],
            "shapes": [strip_internal_shape_keys(shape) for shape in shapes],
            "tracks": [],
        }
    ]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "strip_internal_shape_keys",
    "write_annotation_json",
]
