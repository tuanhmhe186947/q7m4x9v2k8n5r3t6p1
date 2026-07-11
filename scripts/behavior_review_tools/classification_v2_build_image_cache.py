from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    chw_float_to_hwc_uint8,
    context_cache_relative_path,
)

RESIZE_POLICY = "letterbox_preserve_aspect_rgb_pad_black_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable classification_v2 actor crop image cache.")
    parser.add_argument(
        "--frame-context-csv",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/image_frame_context_manifest.csv"),
    )
    parser.add_argument(
        "--window-context-csv",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/image_window_context_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/image_cache_v2_letterbox"),
        help="Canonical reusable actor crop cache root; use one audited letterbox cache instead of smoke-specific roots.",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--source-type", default=None, help="Optional source_type filter for targeted smoke builds.")
    parser.add_argument("--preview-jpg", action="store_true", help="Write readable JPEG previews for audit.")
    parser.add_argument("--preview-limit", type=int, default=500, help="Maximum preview JPEGs to write.")
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Write partial manifest every N contexts.")
    parser.add_argument(
        "--resume-from-partial",
        action="store_true",
        help="Resume from manifest.partial.csv/cache_audit.partial.json instead of iterating completed rows again.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audit = build_image_cache(
        frame_context_csv=args.frame_context_csv,
        window_context_csv=args.window_context_csv,
        output_dir=args.output_dir,
        image_size=args.image_size,
        max_contexts=args.max_contexts,
        source_type=args.source_type,
        preview_jpg=args.preview_jpg,
        preview_limit=args.preview_limit,
        checkpoint_every=args.checkpoint_every,
        resume_from_partial=args.resume_from_partial,
        overwrite=args.overwrite,
    )
    print(json.dumps(audit, indent=2))


def build_image_cache(
    *,
    frame_context_csv: Path,
    window_context_csv: Path,
    output_dir: Path,
    image_size: int,
    max_contexts: int | None,
    source_type: str | None,
    preview_jpg: bool,
    preview_limit: int,
    checkpoint_every: int,
    resume_from_partial: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Materialize audited crops so training does not repeatedly seek videos."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if max_contexts is not None and max_contexts <= 0:
        raise ValueError("max_contexts must be positive when provided")
    if preview_limit < 0:
        raise ValueError("preview_limit must be non-negative")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / f"actor_rgb_{image_size}_letterbox"
    preview_root = output_dir / f"preview_jpg_{image_size}_letterbox"
    cache_root.mkdir(parents=True, exist_ok=True)
    if preview_jpg:
        preview_root.mkdir(parents=True, exist_ok=True)
    dataset = ClassificationV2ImageSequenceDataset(
        ImageSequenceDatasetConfig(
            frame_context_csv=frame_context_csv,
            window_context_csv=window_context_csv,
            image_size=image_size,
            require_complete=False,
            image_cache_size=0,
        )
    )
    frame = dataset.frames.copy()
    if source_type:
        frame = frame[frame["source_type"].astype(str).eq(source_type)].copy()
    if max_contexts is not None:
        frame = frame.head(int(max_contexts)).copy()
    selected_context_rows = len(frame)
    frame = _sort_for_media_locality(frame)

    manifest_rows: list[dict[str, Any]] = []
    loaded = 0
    failed = 0
    skipped_existing = 0
    previews_written = 0
    partial_manifest_path = output_dir / "manifest.partial.csv"
    partial_audit_path = output_dir / "cache_audit.partial.json"
    resumed_manifest_rows = 0
    resume_missing_cache_rows = 0
    completed_context_ids: set[str] = set()
    if resume_from_partial:
        resume_state = _load_partial_resume_state(
            manifest_path=partial_manifest_path,
            audit_path=partial_audit_path,
            image_size=image_size,
            source_type=source_type,
        )
        manifest_rows = resume_state["manifest_rows"]
        previews_written = resume_state["previews_written"]
        resumed_manifest_rows = len(manifest_rows)
        resume_missing_cache_rows = int(resume_state["missing_cache_rows"])
        completed_context_ids = {str(row["image_context_id"]) for row in manifest_rows}
        selected_context_ids = set(frame["image_context_id"].astype(str))
        unexpected_context_ids = completed_context_ids.difference(selected_context_ids)
        if unexpected_context_ids:
            raise ValueError(
                "partial cache contains image_context_id values outside the selected input set: "
                f"{len(unexpected_context_ids)}"
            )
    try:
        for row_index, row in enumerate(frame.itertuples(index=False), start=1):
            row_dict = row._asdict()
            context_id = str(row_dict["image_context_id"])
            if context_id in completed_context_ids:
                continue
            rel_path = context_cache_relative_path(context_id)
            cache_path = cache_root / rel_path
            image_uint8: np.ndarray | None = None
            if cache_path.exists() and not overwrite:
                skipped_existing += 1
            else:
                image_chw = dataset._load_frame_image(row_dict)
                if image_chw is None:
                    failed += 1
                    continue
                image_uint8 = chw_float_to_hwc_uint8(image_chw)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, image_uint8)
                loaded += 1
            letterbox_meta = _letterbox_metadata_from_bbox(row_dict, image_size)
            preview_rel_path = ""
            if preview_jpg and previews_written < preview_limit:
                preview_rel = _readable_preview_relative_path(row_dict, context_id)
                preview_path = preview_root / preview_rel
                if overwrite or not preview_path.exists():
                    if image_uint8 is None:
                        try:
                            image_uint8 = np.load(cache_path)
                        except Exception:
                            image_uint8 = None
                    if image_uint8 is not None:
                        preview_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(image_uint8).save(preview_path, quality=92)
                if preview_path.exists():
                    preview_rel_path = str(Path(f"preview_jpg_{image_size}_letterbox") / preview_rel)
                    previews_written += 1
            manifest_rows.append(
                {
                    "image_context_id": context_id,
                    "cache_path": str(Path(f"actor_rgb_{image_size}_letterbox") / rel_path),
                    "preview_path": preview_rel_path,
                    "image_size": int(image_size),
                    "cache_format": "npy_uint8_rgb_hwc",
                    "resize_policy": RESIZE_POLICY,
                    "source_type": str(row_dict.get("source_type", "")),
                    "dataset_id": str(row_dict.get("dataset_id", "")),
                    "video_key": str(row_dict.get("video_key", "")),
                    "pig_id": str(row_dict.get("pig_id", "")),
                    "track_id": str(row_dict.get("track_id", "")),
                    "frame_index": row_dict.get("frame_index", ""),
                    "x1": row_dict.get("x1", ""),
                    "y1": row_dict.get("y1", ""),
                    "x2": row_dict.get("x2", ""),
                    "y2": row_dict.get("y2", ""),
                    **letterbox_meta,
                }
            )
            completed_context_ids.add(context_id)
            if checkpoint_every and row_index % checkpoint_every == 0:
                _write_cache_checkpoint(
                    manifest_rows=manifest_rows,
                    manifest_path=partial_manifest_path,
                    audit_path=partial_audit_path,
                    output_dir=output_dir,
                    cache_root=cache_root,
                    preview_root=preview_root if preview_jpg else None,
                    image_size=image_size,
                    selected_context_rows=selected_context_rows,
                    loaded=loaded,
                    skipped_existing=skipped_existing,
                    failed=failed,
                    previews_written=previews_written,
                    source_type=source_type,
                    checkpoint_row_index=row_index,
                    video_decode_count=dataset.video_decode_count,
                    video_seek_count=dataset.video_seek_count,
                    video_frame_reuse_count=dataset.video_frame_reuse_count,
                )
    finally:
        dataset.close()

    manifest = pd.DataFrame(manifest_rows).sort_values("image_context_id", kind="mergesort")
    duplicate_context_rows = int(manifest["image_context_id"].duplicated().sum())
    missing_context_rows = int(selected_context_rows - len(manifest))
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    audit = {
        "schema_version": "classification_v2_image_cache_audit_v1",
        "frame_context_csv": str(frame_context_csv),
        "window_context_csv": str(window_context_csv),
        "output_dir": str(output_dir),
        "cache_root": str(cache_root),
        "preview_root": str(preview_root) if preview_jpg else None,
        "manifest_csv": str(manifest_path),
        "image_size": int(image_size),
        "selected_context_rows": int(selected_context_rows),
        "manifest_rows": int(len(manifest)),
        "missing_context_rows": int(missing_context_rows),
        "duplicate_context_rows": int(duplicate_context_rows),
        "resume_from_partial": bool(resume_from_partial),
        "resumed_manifest_rows": int(resumed_manifest_rows),
        "resume_missing_cache_rows": int(resume_missing_cache_rows),
        "loaded_rows": int(loaded),
        "skipped_existing_rows": int(skipped_existing),
        "failed_rows": int(failed),
        "preview_jpg_enabled": bool(preview_jpg),
        "preview_limit": int(preview_limit),
        "previews_written": int(previews_written),
        "source_type_filter": source_type,
        "cache_format": "npy_uint8_rgb_hwc",
        "resize_policy": RESIZE_POLICY,
        "processing_order": "source_media_frame_context_v1",
        "resume_key_policy": "image_context_id_v1",
        "video_decode_count": int(dataset.video_decode_count),
        "video_seek_count": int(dataset.video_seek_count),
        "video_frame_reuse_count": int(dataset.video_frame_reuse_count),
        "valid": bool(
            len(manifest) > 0
            and missing_context_rows == 0
            and duplicate_context_rows == 0
            and failed == 0
        ),
    }
    (output_dir / "cache_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _load_partial_resume_state(
    *,
    manifest_path: Path,
    audit_path: Path,
    image_size: int,
    source_type: str | None,
) -> dict[str, Any]:
    """Load an interrupted cache build only when its lineage matches this run."""

    if not manifest_path.exists() or not audit_path.exists():
        return {"manifest_rows": [], "previews_written": 0, "missing_cache_rows": 0}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if int(audit.get("image_size", -1)) != int(image_size):
        raise ValueError(f"partial cache image_size mismatch: {audit.get('image_size')} != {image_size}")
    if str(audit.get("resize_policy", "")) != RESIZE_POLICY:
        raise ValueError(f"partial cache resize_policy mismatch: {audit.get('resize_policy')} != {RESIZE_POLICY}")
    if audit.get("source_type_filter") != source_type:
        raise ValueError(
            f"partial cache source_type_filter mismatch: {audit.get('source_type_filter')} != {source_type}"
        )
    manifest = pd.read_csv(manifest_path, low_memory=False)
    required = {"image_context_id", "cache_path", "image_size", "cache_format", "resize_policy"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"partial cache manifest missing columns: {missing}")
    duplicate_context = int(manifest["image_context_id"].duplicated().sum())
    if duplicate_context:
        raise ValueError(f"partial cache manifest duplicate image_context_id rows: {duplicate_context}")
    size_mismatch = int(pd.to_numeric(manifest["image_size"], errors="coerce").ne(image_size).sum())
    if size_mismatch:
        raise ValueError(f"partial cache manifest image_size mismatch rows: {size_mismatch}")
    policy_mismatch = int(manifest["resize_policy"].astype(str).ne(RESIZE_POLICY).sum())
    if policy_mismatch:
        raise ValueError(f"partial cache manifest resize_policy mismatch rows: {policy_mismatch}")
    base = manifest_path.parent
    cache_exists = manifest["cache_path"].astype(str).map(
        lambda value: (Path(value) if Path(value).is_absolute() else base / value).exists()
    )
    missing_cache_rows = int((~cache_exists).sum())
    manifest = manifest[cache_exists].copy()
    return {
        "manifest_rows": manifest.to_dict("records"),
        "previews_written": int(audit.get("previews_written", 0)),
        "missing_cache_rows": missing_cache_rows,
    }


def _write_cache_checkpoint(
    *,
    manifest_rows: list[dict[str, Any]],
    manifest_path: Path,
    audit_path: Path,
    output_dir: Path,
    cache_root: Path,
    preview_root: Path | None,
    image_size: int,
    selected_context_rows: int,
    loaded: int,
    skipped_existing: int,
    failed: int,
    previews_written: int,
    source_type: str | None,
    checkpoint_row_index: int,
    video_decode_count: int,
    video_seek_count: int,
    video_frame_reuse_count: int,
) -> None:
    """Persist partial cache lineage so interrupted long builds are auditable."""

    pd.DataFrame(manifest_rows).sort_values("image_context_id", kind="mergesort").to_csv(manifest_path, index=False)
    audit = {
        "schema_version": "classification_v2_image_cache_partial_audit_v1",
        "output_dir": str(output_dir),
        "cache_root": str(cache_root),
        "preview_root": str(preview_root) if preview_root is not None else None,
        "manifest_csv": str(manifest_path),
        "image_size": int(image_size),
        "selected_context_rows": int(selected_context_rows),
        "checkpoint_row_index": int(checkpoint_row_index),
        "manifest_rows": int(len(manifest_rows)),
        "loaded_rows": int(loaded),
        "skipped_existing_rows": int(skipped_existing),
        "failed_rows": int(failed),
        "previews_written": int(previews_written),
        "source_type_filter": source_type,
        "cache_format": "npy_uint8_rgb_hwc",
        "resize_policy": RESIZE_POLICY,
        "processing_order": "source_media_frame_context_v1",
        "resume_key_policy": "image_context_id_v1",
        "video_decode_count": int(video_decode_count),
        "video_seek_count": int(video_seek_count),
        "video_frame_reuse_count": int(video_frame_reuse_count),
        "complete": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def _readable_preview_relative_path(row: dict[str, Any], context_id: str) -> Path:
    """Build a human-readable preview path while keeping a hash suffix unique."""

    source = _safe_segment(str(row.get("source_type", "unknown_source")))
    video = _safe_segment(str(row.get("video_key", "unknown_video")))
    pig = _safe_segment(str(row.get("pig_id", row.get("track_id", "unknown_pig"))))
    frame_value = row.get("frame_index", "")
    try:
        frame_text = f"frame_{int(float(frame_value)):06d}"
    except Exception:
        frame_text = "frame_unknown"
    short_id = context_cache_relative_path(context_id).stem[:10]
    return Path(source) / video / pig / f"{frame_text}_{short_id}.jpg"


def _sort_for_media_locality(frame: pd.DataFrame) -> pd.DataFrame:
    """Order requests for one-open-per-video sequential decoding.

    The deterministic context ID remains the cache key. Sorting changes only
    I/O order; it does not change labels, rows, bbox values, or output IDs.
    """

    ordered = frame.copy()
    ordered["_cache_frame_index"] = pd.to_numeric(ordered["frame_index"], errors="coerce")
    ordered["_cache_media_path"] = ordered["resolved_media_path"].fillna("").astype(str)
    ordered = ordered.sort_values(
        ["source_type", "_cache_media_path", "_cache_frame_index", "image_context_id"],
        kind="mergesort",
        na_position="last",
    )
    return ordered.drop(columns=["_cache_frame_index", "_cache_media_path"]).reset_index(drop=True)


def _letterbox_metadata_from_bbox(row: dict[str, Any], image_size: int) -> dict[str, Any]:
    """Record resize geometry so cache users can audit aspect preservation."""

    x1 = pd.to_numeric(row.get("x1"), errors="coerce")
    y1 = pd.to_numeric(row.get("y1"), errors="coerce")
    x2 = pd.to_numeric(row.get("x2"), errors="coerce")
    y2 = pd.to_numeric(row.get("y2"), errors="coerce")
    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
        return _empty_letterbox_metadata()
    crop_width = max(0.0, float(x2) - float(x1))
    crop_height = max(0.0, float(y2) - float(y1))
    if crop_width <= 0.0 or crop_height <= 0.0:
        return _empty_letterbox_metadata(crop_width=crop_width, crop_height=crop_height)
    scale = min(float(image_size) / crop_width, float(image_size) / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    pad_left = int((image_size - resized_width) // 2)
    pad_top = int((image_size - resized_height) // 2)
    return {
        "source_crop_width": float(crop_width),
        "source_crop_height": float(crop_height),
        "source_crop_aspect_ratio": float(crop_width / crop_height),
        "letterbox_scale": float(scale),
        "letterbox_resized_width": int(resized_width),
        "letterbox_resized_height": int(resized_height),
        "letterbox_pad_left": int(pad_left),
        "letterbox_pad_top": int(pad_top),
        "letterbox_pad_right": int(image_size - resized_width - pad_left),
        "letterbox_pad_bottom": int(image_size - resized_height - pad_top),
    }


def _empty_letterbox_metadata(crop_width: float | None = None, crop_height: float | None = None) -> dict[str, Any]:
    return {
        "source_crop_width": "" if crop_width is None else float(crop_width),
        "source_crop_height": "" if crop_height is None else float(crop_height),
        "source_crop_aspect_ratio": "",
        "letterbox_scale": "",
        "letterbox_resized_width": "",
        "letterbox_resized_height": "",
        "letterbox_pad_left": "",
        "letterbox_pad_top": "",
        "letterbox_pad_right": "",
        "letterbox_pad_bottom": "",
    }


def _safe_segment(value: str) -> str:
    """Keep preview paths readable and Windows-safe."""

    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned[:120] if cleaned else "unknown"


if __name__ == "__main__":
    main()
