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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_v2/image_cache_v2"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--source-type", default=None, help="Optional source_type filter for targeted smoke builds.")
    parser.add_argument("--preview-jpg", action="store_true", help="Write readable JPEG previews for audit.")
    parser.add_argument("--preview-limit", type=int, default=500, help="Maximum preview JPEGs to write.")
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
    overwrite: bool,
) -> dict[str, Any]:
    """Materialize audited crops so training does not repeatedly seek videos."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if max_contexts is not None and max_contexts <= 0:
        raise ValueError("max_contexts must be positive when provided")
    if preview_limit < 0:
        raise ValueError("preview_limit must be non-negative")
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

    manifest_rows: list[dict[str, Any]] = []
    loaded = 0
    failed = 0
    skipped_existing = 0
    previews_written = 0
    try:
        for row in frame.itertuples(index=False):
            row_dict = row._asdict()
            context_id = str(row_dict["image_context_id"])
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
                }
            )
    finally:
        dataset.close()

    manifest = pd.DataFrame(manifest_rows).sort_values("image_context_id", kind="mergesort")
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
        "selected_context_rows": int(len(frame)),
        "manifest_rows": int(len(manifest)),
        "loaded_rows": int(loaded),
        "skipped_existing_rows": int(skipped_existing),
        "failed_rows": int(failed),
        "preview_jpg_enabled": bool(preview_jpg),
        "preview_limit": int(preview_limit),
        "previews_written": int(previews_written),
        "source_type_filter": source_type,
        "cache_format": "npy_uint8_rgb_hwc",
        "resize_policy": RESIZE_POLICY,
        "valid": bool(len(manifest) > 0 and failed == 0),
    }
    (output_dir / "cache_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


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


def _safe_segment(value: str) -> str:
    """Keep preview paths readable and Windows-safe."""

    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned[:120] if cleaned else "unknown"


if __name__ == "__main__":
    main()
