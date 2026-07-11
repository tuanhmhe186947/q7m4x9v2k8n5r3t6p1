"""Build reusable actor-partner visual context without behavior-label gating.

CVAT rows use the union of the actor and nearest-partner boxes from the same
video frame. Legacy crop-only rows remain represented with an unavailable mask;
the module never fabricates scene context and never drops unavailable rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.datasets.image_sequence_dataset import letterbox_rgb_uint8

RESIZE_POLICY = "actor_nearest_partner_union_letterbox_rgb_pad_black_v1"


@dataclass(frozen=True, slots=True)
class VisualInteractionCacheConfig:
    frame_context_csv: Path
    output_dir: Path
    image_size: int = 64
    padding_ratio: float = 0.15
    max_contexts: int | None = None
    source_type: str | None = None
    preview_limit: int = 100


def build_visual_interaction_cache(config: VisualInteractionCacheConfig) -> dict[str, Any]:
    """Materialize actor-partner union crops and write a complete audit manifest."""

    _validate_config(config)
    frames = pd.read_csv(config.frame_context_csv, low_memory=False)
    _validate_frames(frames)
    if config.source_type:
        frames = frames[frames["source_type"].astype(str).eq(config.source_type)].copy()
    frames = frames.sort_values(
        ["resolved_media_path", "frame_index", "object_track_key"], kind="mergesort"
    ).reset_index(drop=True)
    if config.max_contexts is not None:
        frames = frames.head(config.max_contexts).copy()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = config.output_dir / f"actor_partner_rgb_{config.image_size}_letterbox"
    preview_root = config.output_dir / "preview_actor_partner_jpg"
    cache_root.mkdir(parents=True, exist_ok=True)
    preview_root.mkdir(parents=True, exist_ok=True)

    lookup = _same_frame_actor_lookup(frames)
    captures: dict[str, cv2.VideoCapture] = {}
    decoded: dict[str, tuple[int, np.ndarray]] = {}
    next_frame: dict[str, int] = {}
    manifest_rows: list[dict[str, Any]] = []
    decode_count = seek_count = reuse_count = 0
    previews_written = 0
    try:
        for row in frames.to_dict("records"):
            result = _resolve_context_geometry(row, lookup, config.padding_ratio)
            image: np.ndarray | None = None
            if result["status"] == "geometry_ready":
                image, did_decode, did_seek, did_reuse = _decode_union_crop(
                    row=row,
                    union_bbox=result["union_bbox"],
                    captures=captures,
                    decoded=decoded,
                    next_frame=next_frame,
                    image_size=config.image_size,
                )
                decode_count += did_decode
                seek_count += did_seek
                reuse_count += did_reuse
                if image is None:
                    result["status"] = "video_decode_or_crop_failed"

            context_id = _visual_context_id(str(row["image_context_id"]))
            cache_rel = ""
            preview_rel = ""
            if image is not None:
                rel = _cache_relative_path(context_id)
                (cache_root / rel).parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_root / rel, image)
                cache_rel = str(Path(cache_root.name) / rel)
                if previews_written < config.preview_limit:
                    preview_name = f"{previews_written:04d}_{context_id[:12]}.jpg"
                    Image.fromarray(image).save(preview_root / preview_name, quality=92)
                    preview_rel = str(Path(preview_root.name) / preview_name)
                    previews_written += 1
                result["status"] = "ready"

            manifest_rows.append(
                {
                    "visual_context_id": context_id,
                    "image_context_id": str(row["image_context_id"]),
                    "source_type": str(row["source_type"]),
                    "video_key": str(row["video_key"]),
                    "frame_index": int(row["frame_index"]),
                    "actor_track_id": str(row["track_id"]),
                    "partner_track_id": result["partner_track_id"],
                    "context_kind": "actor_nearest_partner_union",
                    "visual_context_available": image is not None,
                    "visual_context_status": result["status"],
                    "cache_path": cache_rel,
                    "preview_path": preview_rel,
                    "image_size": config.image_size,
                    "resize_policy": RESIZE_POLICY,
                    **result["audit_geometry"],
                }
            )
    finally:
        for capture in captures.values():
            capture.release()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = config.output_dir / "visual_context_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    duplicate_ids = int(manifest["visual_context_id"].duplicated().sum())
    available = int(manifest["visual_context_available"].sum())
    status_counts = manifest["visual_context_status"].value_counts(dropna=False).to_dict()
    audit = {
        "schema_version": "classification_v2_visual_interaction_cache_audit_v1",
        "frame_context_csv": str(config.frame_context_csv),
        "manifest_csv": str(manifest_path),
        "selected_rows": int(len(manifest)),
        "available_rows": available,
        "unavailable_rows": int(len(manifest) - available),
        "duplicate_visual_context_id": duplicate_ids,
        "status_counts": status_counts,
        "image_size": config.image_size,
        "padding_ratio": config.padding_ratio,
        "resize_policy": RESIZE_POLICY,
        "video_decode_count": decode_count,
        "video_seek_count": seek_count,
        "video_frame_reuse_count": reuse_count,
        "previews_written": previews_written,
        "label_gated": False,
        "rows_dropped_for_missing_context": 0,
        "errors": [] if duplicate_ids == 0 else [f"duplicate_visual_context_id={duplicate_ids}"],
        "warnings": ["legacy crop-only rows are expected to have visual_context_available=false"],
    }
    audit["valid"] = not audit["errors"] and len(manifest) > 0
    (config.output_dir / "visual_context_cache_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def _validate_config(config: VisualInteractionCacheConfig) -> None:
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if not 0 <= config.padding_ratio <= 1:
        raise ValueError("padding_ratio must be in [0, 1]")
    if config.max_contexts is not None and config.max_contexts <= 0:
        raise ValueError("max_contexts must be positive")


def _validate_frames(frames: pd.DataFrame) -> None:
    required = {
        "image_context_id", "source_type", "video_key", "resolved_media_path",
        "frame_index", "track_id", "nearest_track_id", "x1", "y1", "x2", "y2",
    }
    missing = sorted(required.difference(frames.columns))
    if missing:
        raise ValueError(f"frame context manifest missing columns: {missing}")
    duplicates = int(frames["image_context_id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"duplicate image_context_id rows: {duplicates}")


def _same_frame_actor_lookup(frames: pd.DataFrame) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Index partner boxes by video/frame/track; track IDs are local to a video."""

    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in frames.to_dict("records"):
        frame_index = pd.to_numeric(row["frame_index"], errors="coerce")
        if pd.isna(frame_index):
            continue
        lookup[(str(row["video_key"]), int(frame_index), _canonical_id(row["track_id"]))] = row
    return lookup


def _resolve_context_geometry(
    row: dict[str, Any], lookup: dict[tuple[str, int, str], dict[str, Any]], padding_ratio: float
) -> dict[str, Any]:
    empty_geometry = {name: np.nan for name in ["union_x1", "union_y1", "union_x2", "union_y2"]}
    if str(row["source_type"]) != "cvat_tracking_xml":
        return {
            "status": "source_has_no_full_frame_media",
            "partner_track_id": "",
            "union_bbox": None,
            "audit_geometry": empty_geometry,
        }
    media = str(row.get("resolved_media_path", "")).strip()
    if not media or not Path(media).exists():
        return {"status": "missing_video", "partner_track_id": "", "union_bbox": None, "audit_geometry": empty_geometry}
    partner_id = _canonical_id(row.get("nearest_track_id"))
    frame_index = int(pd.to_numeric(row["frame_index"], errors="raise"))
    partner = lookup.get((str(row["video_key"]), frame_index, partner_id))
    if not partner_id or partner is None:
        return {
            "status": "missing_nearest_partner_bbox",
            "partner_track_id": partner_id,
            "union_bbox": None,
            "audit_geometry": empty_geometry,
        }
    actor_box = _valid_box(row)
    partner_box = _valid_box(partner)
    if actor_box is None or partner_box is None:
        return {
            "status": "invalid_actor_or_partner_bbox",
            "partner_track_id": partner_id,
            "union_bbox": None,
            "audit_geometry": empty_geometry,
        }
    x1 = min(actor_box[0], partner_box[0])
    y1 = min(actor_box[1], partner_box[1])
    x2 = max(actor_box[2], partner_box[2])
    y2 = max(actor_box[3], partner_box[3])
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    union = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
    audit = dict(zip(["union_x1", "union_y1", "union_x2", "union_y2"], union, strict=True))
    return {"status": "geometry_ready", "partner_track_id": partner_id, "union_bbox": union, "audit_geometry": audit}


def _decode_union_crop(
    *,
    row: dict[str, Any],
    union_bbox: tuple[float, float, float, float],
    captures: dict[str, cv2.VideoCapture],
    decoded: dict[str, tuple[int, np.ndarray]],
    next_frame: dict[str, int],
    image_size: int,
) -> tuple[np.ndarray | None, int, int, int]:
    path = str(row["resolved_media_path"])
    target = int(row["frame_index"])
    capture = captures.get(path)
    if capture is None:
        capture = cv2.VideoCapture(path)
        captures[path] = capture
    if not capture.isOpened():
        return None, 0, 0, 0
    did_decode = did_seek = did_reuse = 0
    cached = decoded.get(path)
    if cached is not None and cached[0] == target:
        frame = cached[1]
        did_reuse = 1
    else:
        if next_frame.get(path) != target:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            did_seek = 1
        ok, frame = capture.read()
        if not ok or frame is None:
            return None, 0, did_seek, 0
        decoded[path] = (target, frame)
        next_frame[path] = target + 1
        did_decode = 1
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = union_bbox
    ix1 = max(0, min(width, int(np.floor(x1))))
    iy1 = max(0, min(height, int(np.floor(y1))))
    ix2 = max(0, min(width, int(np.ceil(x2))))
    iy2 = max(0, min(height, int(np.ceil(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None, did_decode, did_seek, did_reuse
    rgb = cv2.cvtColor(frame[iy1:iy2, ix1:ix2], cv2.COLOR_BGR2RGB)
    return letterbox_rgb_uint8(rgb, image_size), did_decode, did_seek, did_reuse


def _valid_box(row: dict[str, Any]) -> tuple[float, float, float, float] | None:
    values = [pd.to_numeric(row.get(name), errors="coerce") for name in ["x1", "y1", "x2", "y2"]]
    if any(pd.isna(value) for value in values):
        return None
    x1, y1, x2, y2 = map(float, values)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _canonical_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _visual_context_id(image_context_id: str) -> str:
    return hashlib.sha1(f"visual_partner_v1|{image_context_id}".encode()).hexdigest()


def _cache_relative_path(context_id: str) -> Path:
    return Path(context_id[:2]) / f"{context_id}.npy"
