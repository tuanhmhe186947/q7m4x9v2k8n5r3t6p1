"""Build reusable actor-partner visual context without behavior-label gating.

Rows from any source use the union of actor and nearest-partner boxes only when
the same-frame video, clip, and partner geometry are available. The module
never fabricates scene context and never drops unavailable rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.datasets.image_sequence_dataset import letterbox_rgb_uint8

RESIZE_POLICY = "actor_nearest_partner_union_letterbox_rgb_pad_black_v1"
CACHE_KEY_POLICY = "sha256_image_context_id_v1"


@dataclass(frozen=True, slots=True)
class VisualInteractionCacheConfig:
    frame_context_csv: Path
    output_dir: Path
    selection_csv: Path | None = None
    image_size: int = 64
    padding_ratio: float = 0.15
    max_contexts: int | None = None
    source_type: str | None = None
    preview_limit: int = 100
    checkpoint_every: int = 1000
    max_open_videos: int = 4
    resume: bool = False


def build_visual_interaction_cache(config: VisualInteractionCacheConfig) -> dict[str, Any]:
    """Materialize actor-partner union crops and write a complete audit manifest."""

    _validate_config(config)
    frame_context_sha256 = _file_sha256(config.frame_context_csv)
    selection_sha256 = (
        _file_sha256(config.selection_csv)
        if config.selection_csv is not None
        else ""
    )
    frames = pd.read_csv(config.frame_context_csv, low_memory=False)
    _require_unchanged_file(
        config.frame_context_csv,
        expected_sha256=frame_context_sha256,
        description="frame context manifest",
    )
    _validate_frames(frames)
    if config.source_type:
        frames = frames[frames["source_type"].astype(str).eq(config.source_type)].copy()
    lookup = _same_frame_actor_lookup(frames)
    frames = _select_target_frames(frames, config.selection_csv)
    if config.selection_csv is not None:
        _require_unchanged_file(
            config.selection_csv,
            expected_sha256=selection_sha256,
            description="visual context selection",
        )
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

    captures: dict[str, cv2.VideoCapture] = {}
    decoded: dict[str, tuple[int, np.ndarray]] = {}
    next_frame: dict[str, int] = {}
    manifest_rows: list[dict[str, Any]] = []
    decode_count = seek_count = reuse_count = 0
    peak_open_video_count = 0
    previews_written = 0
    partial_manifest_path = config.output_dir / "visual_context_manifest.partial.csv"
    partial_audit_path = config.output_dir / "visual_context_cache_audit.partial.json"
    completed_context_ids: set[str] = set()
    resumed_rows = 0
    if config.resume and partial_manifest_path.exists():
        partial = pd.read_csv(partial_manifest_path, low_memory=False)
        _validate_partial_audit(
            partial_audit_path,
            config,
            frame_context_sha256=frame_context_sha256,
            selection_sha256=selection_sha256,
            selected_rows=len(frames),
        )
        _validate_partial_manifest(
            partial,
            config,
            selected_image_context_ids=set(frames["image_context_id"].astype(str)),
        )
        manifest_rows = partial.to_dict("records")
        completed_context_ids = set(partial["visual_context_id"].astype(str))
        resumed_rows = len(manifest_rows)
        previews_written = int(
            sum(
                bool(str(value).strip())
                for value in partial["preview_path"].fillna("")
            )
        )
    try:
        for row_number, row in enumerate(frames.to_dict("records"), start=1):
            context_id = _visual_context_id(str(row["image_context_id"]))
            if context_id in completed_context_ids:
                continue
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
                    max_open_videos=config.max_open_videos,
                )
                peak_open_video_count = max(
                    peak_open_video_count,
                    len(captures),
                )
                decode_count += did_decode
                seek_count += did_seek
                reuse_count += did_reuse
                if image is None:
                    result["status"] = "video_decode_or_crop_failed"

            cache_rel = ""
            preview_rel = ""
            if image is not None:
                rel = _cache_relative_path(context_id)
                cache_path = cache_root / rel
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                _write_or_validate_cache_image(
                    cache_path,
                    image,
                    resume=config.resume,
                )
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
                    "lineage_scope": str(row["lineage_scope"]),
                    "human_review_complete": _strict_bool_value(
                        row["human_review_complete"],
                        name="human_review_complete",
                    ),
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
            completed_context_ids.add(context_id)
            if config.checkpoint_every and row_number % config.checkpoint_every == 0:
                _write_partial_checkpoint(
                    manifest_rows,
                    partial_manifest_path,
                    partial_audit_path,
                    config,
                    decode_count=decode_count,
                    seek_count=seek_count,
                    reuse_count=reuse_count,
                    selected_rows=len(frames),
                    frame_context_sha256=frame_context_sha256,
                    selection_sha256=selection_sha256,
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
    lineage_scopes = sorted(manifest["lineage_scope"].astype(str).unique())
    reviewed_values = sorted(
        _strict_bool_series(
            manifest["human_review_complete"],
            name="human_review_complete",
        )
        .unique()
        .tolist()
    )
    claim_errors: list[str] = []
    if len(lineage_scopes) != 1 or not lineage_scopes[0]:
        claim_errors.append(f"lineage_scope_values={lineage_scopes}")
    if len(reviewed_values) != 1:
        claim_errors.append(f"human_review_complete_values={reviewed_values}")
    audit = {
        "schema_version": "classification_v2_visual_interaction_cache_audit_v1",
        "frame_context_csv": str(config.frame_context_csv),
        "frame_context_sha256": frame_context_sha256,
        "manifest_csv": str(manifest_path),
        "selected_rows": int(len(manifest)),
        "available_rows": available,
        "unavailable_rows": int(len(manifest) - available),
        "duplicate_visual_context_id": duplicate_ids,
        "status_counts": status_counts,
        "lineage_scope": lineage_scopes[0] if len(lineage_scopes) == 1 else "",
        "human_review_complete": (
            reviewed_values[0] if len(reviewed_values) == 1 else None
        ),
        "image_size": config.image_size,
        "padding_ratio": config.padding_ratio,
        "resize_policy": RESIZE_POLICY,
        "cache_key_policy": CACHE_KEY_POLICY,
        "max_open_videos": config.max_open_videos,
        "peak_open_videos": peak_open_video_count,
        "video_decode_count": decode_count,
        "video_seek_count": seek_count,
        "video_frame_reuse_count": reuse_count,
        "previews_written": previews_written,
        "resume_requested": bool(config.resume),
        "resumed_rows": int(resumed_rows),
        "label_gated": False,
        "rows_dropped_for_missing_context": 0,
        "selection_csv": (str(config.selection_csv) if config.selection_csv is not None else ""),
        "selection_sha256": selection_sha256,
        "errors": (
            claim_errors
            if duplicate_ids == 0
            else [*claim_errors, f"duplicate_visual_context_id={duplicate_ids}"]
        ),
        "warnings": [],
    }
    audit["valid"] = not audit["errors"] and len(manifest) > 0
    (config.output_dir / "visual_context_cache_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    _write_partial_checkpoint(
        manifest.to_dict("records"),
        partial_manifest_path,
        partial_audit_path,
        config,
        decode_count=decode_count,
        seek_count=seek_count,
        reuse_count=reuse_count,
        selected_rows=len(frames),
        frame_context_sha256=frame_context_sha256,
        selection_sha256=selection_sha256,
        complete=True,
    )
    return audit


def _validate_config(config: VisualInteractionCacheConfig) -> None:
    if not config.frame_context_csv.is_file():
        raise FileNotFoundError(
            "frame context manifest does not exist: "
            f"{config.frame_context_csv}"
        )
    if config.image_size <= 0:
        raise ValueError("image_size must be positive")
    if not 0 <= config.padding_ratio <= 1:
        raise ValueError("padding_ratio must be in [0, 1]")
    if config.max_contexts is not None and config.max_contexts <= 0:
        raise ValueError("max_contexts must be positive")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if config.max_open_videos <= 0:
        raise ValueError("max_open_videos must be positive")
    if config.selection_csv is not None and not config.selection_csv.is_file():
        raise FileNotFoundError(f"visual context selection does not exist: {config.selection_csv}")


def _validate_frames(frames: pd.DataFrame) -> None:
    required = {
        "image_context_id",
        "source_type",
        "dataset_id",
        "video_key",
        "clip_id",
        "resolved_media_path",
        "frame_index",
        "track_id",
        "nearest_track_id",
        "lineage_scope",
        "human_review_complete",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    missing = sorted(required.difference(frames.columns))
    if missing:
        raise ValueError(f"frame context manifest missing columns: {missing}")
    duplicates = int(frames["image_context_id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"duplicate image_context_id rows: {duplicates}")


def _same_frame_actor_lookup(
    frames: pd.DataFrame,
) -> dict[tuple[str, str, str, str, int, str], dict[str, Any]]:
    """Index partner boxes without crossing source, dataset, video, or clip."""

    lookup: dict[
        tuple[str, str, str, str, int, str],
        dict[str, Any],
    ] = {}
    for row in frames.to_dict("records"):
        frame_index = pd.to_numeric(row["frame_index"], errors="coerce")
        if pd.isna(frame_index):
            continue
        key = _context_lookup_key(
            row,
            frame_index=int(frame_index),
            track_id=_canonical_id(row["track_id"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate actor-partner lookup key: {key}")
        lookup[key] = row
    return lookup


def _select_target_frames(
    frames: pd.DataFrame,
    selection_csv: Path | None,
) -> pd.DataFrame:
    if selection_csv is None:
        return frames.copy()
    selection = pd.read_csv(selection_csv, low_memory=False)
    if set(selection.columns) != {"image_context_id"}:
        raise ValueError("visual context selection must contain only image_context_id")
    identifiers = selection["image_context_id"].fillna("").astype(str)
    if identifiers.empty:
        raise ValueError("visual context selection must not be empty")
    if identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError("visual context selection IDs must be unique and nonblank")
    frame_ids = frames["image_context_id"].fillna("").astype(str)
    positions = pd.Series(
        np.arange(len(selection), dtype=np.int64),
        index=identifiers,
    )
    selected = frames.loc[frame_ids.isin(positions.index)].copy()
    if len(selected) != len(selection):
        missing = sorted(set(identifiers) - set(selected["image_context_id"]))
        raise ValueError(f"visual context selection missing IDs: {missing[:5]}")
    selected["_selection_order"] = selected["image_context_id"].map(positions)
    selected = selected.sort_values("_selection_order", kind="mergesort")
    return selected.drop(columns="_selection_order").reset_index(drop=True)


def _resolve_context_geometry(
    row: dict[str, Any],
    lookup: dict[
        tuple[str, str, str, str, int, str],
        dict[str, Any],
    ],
    padding_ratio: float,
) -> dict[str, Any]:
    empty_geometry = {name: np.nan for name in ["union_x1", "union_y1", "union_x2", "union_y2"]}
    media = str(row.get("resolved_media_path", "")).strip()
    if not media or not Path(media).exists():
        return {
            "status": "missing_video",
            "partner_track_id": "",
            "union_bbox": None,
            "audit_geometry": empty_geometry,
        }
    partner_id = _canonical_id(row.get("nearest_track_id"))
    frame_index = int(pd.to_numeric(row["frame_index"], errors="raise"))
    partner_key = _context_lookup_key(
        row,
        frame_index=frame_index,
        track_id=partner_id,
    )
    partner = lookup.get(partner_key)
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
    return {
        "status": "geometry_ready",
        "partner_track_id": partner_id,
        "union_bbox": union,
        "audit_geometry": audit,
    }


def _context_lookup_key(
    row: dict[str, Any],
    *,
    frame_index: int,
    track_id: str,
) -> tuple[str, str, str, str, int, str]:
    return (
        str(row.get("source_type", "")),
        str(row.get("dataset_id", "")),
        str(row.get("video_key", "")),
        str(row.get("clip_id", "")),
        frame_index,
        track_id,
    )


def _decode_union_crop(
    *,
    row: dict[str, Any],
    union_bbox: tuple[float, float, float, float],
    captures: dict[str, cv2.VideoCapture],
    decoded: dict[str, tuple[int, np.ndarray]],
    next_frame: dict[str, int],
    image_size: int,
    max_open_videos: int,
) -> tuple[np.ndarray | None, int, int, int]:
    path = str(row["resolved_media_path"])
    target = int(row["frame_index"])
    capture = captures.pop(path, None)
    if capture is None:
        while len(captures) >= max_open_videos:
            stale_path = next(iter(captures))
            stale_capture = captures.pop(stale_path)
            stale_capture.release()
            decoded.pop(stale_path, None)
            next_frame.pop(stale_path, None)
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


def _strict_bool_value(value: Any, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{name} contains an invalid boolean value")


def _strict_bool_series(values: pd.Series, *, name: str) -> pd.Series:
    return values.map(lambda value: _strict_bool_value(value, name=name)).astype(bool)


def _single_manifest_value(frame: pd.DataFrame, column: str) -> str:
    values = sorted(frame[column].fillna("").astype(str).unique())
    if len(values) != 1 or not values[0]:
        raise ValueError(f"partial visual cache {column} values={values}")
    return values[0]


def _single_manifest_bool(frame: pd.DataFrame, column: str) -> bool:
    values = _strict_bool_series(frame[column], name=column).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"partial visual cache {column} values={values}")
    return bool(values[0])


def _visual_context_id(image_context_id: str) -> str:
    return hashlib.sha1(f"visual_partner_v1|{image_context_id}".encode()).hexdigest()


def _cache_relative_path(context_id: str) -> Path:
    digest = hashlib.sha256(context_id.encode("utf-8")).hexdigest()
    return Path(digest[:2]) / f"{digest}.npy"


def _validate_partial_audit(
    audit_path: Path,
    config: VisualInteractionCacheConfig,
    *,
    frame_context_sha256: str,
    selection_sha256: str,
    selected_rows: int,
) -> None:
    """Reject resume when the immutable cache-build contract changed."""

    if not audit_path.is_file():
        raise ValueError("partial visual cache audit is missing")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "classification_v2_visual_interaction_cache_partial_v1",
        "frame_context_sha256": frame_context_sha256,
        "selection_sha256": selection_sha256,
        "selected_rows": int(selected_rows),
        "image_size": int(config.image_size),
        "padding_ratio": float(config.padding_ratio),
        "resize_policy": RESIZE_POLICY,
        "cache_key_policy": CACHE_KEY_POLICY,
        "max_open_videos": int(config.max_open_videos),
        "source_type_filter": config.source_type,
        "max_contexts": config.max_contexts,
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatches:
        raise ValueError(f"partial visual cache audit contract mismatch: {sorted(mismatches)}")


def _validate_partial_manifest(
    partial: pd.DataFrame,
    config: VisualInteractionCacheConfig,
    *,
    selected_image_context_ids: set[str],
) -> None:
    """Reject resume state whose cache contract differs from the requested run."""

    required = {
        "visual_context_id",
        "image_context_id",
        "cache_path",
        "preview_path",
        "lineage_scope",
        "human_review_complete",
        "image_size",
        "resize_policy",
    }
    missing = sorted(required.difference(partial.columns))
    if missing:
        raise ValueError(f"partial visual cache manifest missing columns: {missing}")
    if partial["visual_context_id"].duplicated().any():
        raise ValueError("partial visual cache manifest has duplicate visual_context_id")
    _single_manifest_value(partial, "lineage_scope")
    _single_manifest_bool(partial, "human_review_complete")
    partial_ids = set(partial["image_context_id"].fillna("").astype(str))
    if not partial_ids.issubset(selected_image_context_ids):
        raise ValueError("partial visual cache escaped the selected context IDs")
    if pd.to_numeric(partial["image_size"], errors="coerce").ne(config.image_size).any():
        raise ValueError("partial visual cache image_size does not match requested image_size")
    if partial["resize_policy"].astype(str).ne(RESIZE_POLICY).any():
        raise ValueError("partial visual cache resize_policy mismatch")
    available = partial["visual_context_available"].astype(str).str.lower().isin({"true", "1"})
    missing_files = partial.loc[available, "cache_path"].astype(str).map(
        lambda value: not (config.output_dir / value).exists()
    )
    if missing_files.any():
        raise ValueError(
            "partial visual cache has missing tensor files: "
            f"{int(missing_files.sum())}"
        )


def _write_partial_checkpoint(
    manifest_rows: list[dict[str, Any]],
    manifest_path: Path,
    audit_path: Path,
    config: VisualInteractionCacheConfig,
    *,
    decode_count: int,
    seek_count: int,
    reuse_count: int,
    selected_rows: int,
    frame_context_sha256: str,
    selection_sha256: str,
    complete: bool = False,
) -> None:
    """Persist resumable state without mutating source data or dropping failures."""

    partial = pd.DataFrame(manifest_rows)
    partial.to_csv(manifest_path, index=False)
    payload = {
        "schema_version": "classification_v2_visual_interaction_cache_partial_v1",
        "selected_rows": int(selected_rows),
        "completed_rows": int(len(partial)),
        "complete": bool(complete),
        "frame_context_csv": str(config.frame_context_csv),
        "frame_context_sha256": frame_context_sha256,
        "image_size": int(config.image_size),
        "padding_ratio": float(config.padding_ratio),
        "resize_policy": RESIZE_POLICY,
        "cache_key_policy": CACHE_KEY_POLICY,
        "max_open_videos": int(config.max_open_videos),
        "source_type_filter": config.source_type,
        "lineage_scope": _single_manifest_value(
            partial,
            "lineage_scope",
        ),
        "human_review_complete": _single_manifest_bool(
            partial,
            "human_review_complete",
        ),
        "max_contexts": config.max_contexts,
        "selection_csv": (str(config.selection_csv) if config.selection_csv is not None else ""),
        "selection_sha256": selection_sha256,
        "video_decode_count_this_process": int(decode_count),
        "video_seek_count_this_process": int(seek_count),
        "video_frame_reuse_count_this_process": int(reuse_count),
    }
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_or_validate_cache_image(
    path: Path,
    image: np.ndarray,
    *,
    resume: bool,
) -> None:
    """Write one cache row or verify an orphan created after a checkpoint."""

    if path.exists():
        if not resume:
            raise FileExistsError(path)
        existing = np.load(path, allow_pickle=False)
        if (
            existing.shape != image.shape
            or existing.dtype != image.dtype
            or not np.array_equal(existing, image)
        ):
            raise ValueError(f"resume cache image differs: {path}")
        return
    with path.open("xb") as handle:
        np.save(handle, image, allow_pickle=False)


def _require_unchanged_file(
    path: Path,
    *,
    expected_sha256: str,
    description: str,
) -> None:
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{description} changed while building visual cache")
