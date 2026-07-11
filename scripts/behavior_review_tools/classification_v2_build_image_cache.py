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

    manifest_rows: list[dict[str, Any]] = []
    loaded = 0
    failed = 0
    skipped_existing = 0
    previews_written = 0
    partial_manifest_path = output_dir / "manifest.partial.csv"
    partial_audit_path = output_dir / "cache_audit.partial.json"
    start_row_index = 1
    resumed_manifest_rows = 0
    if resume_from_partial:
        resume_state = _load_partial_resume_state(
            manifest_path=partial_manifest_path,
            audit_path=partial_audit_path,
            image_size=image_size,
            source_type=source_type,
        )
        manifest_rows = resume_state["manifest_rows"]
        start_row_index = resume_state["start_row_index"]
        previews_written = resume_state["previews_written"]
        resumed_manifest_rows = len(manifest_rows)
    try:
        for row_index, row in enumerate(frame.itertuples(index=False), start=1):
            if row_index < start_row_index:
                continue
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
            if checkpoint_every and row_index % checkpoint_every == 0:
                _write_cache_checkpoint(
                    manifest_rows=manifest_rows,
                    manifest_path=partial_manifest_path,
                    audit_path=partial_audit_path,
                    output_dir=output_dir,
                    cache_root=cache_root,
                    preview_root=preview_root if preview_jpg else None,
                    image_size=image_size,
                    selected_context_rows=len(frame),
                    loaded=loaded,
                    skipped_existing=skipped_existing,
                    failed=failed,
                    previews_written=previews_written,
                    source_type=source_type,
                    checkpoint_row_index=row_index,
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
        "resume_from_partial": bool(resume_from_partial),
        "start_row_index": int(start_row_index),
        "resumed_manifest_rows": int(resumed_manifest_rows),
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


def _load_partial_resume_state(
    *,
    manifest_path: Path,
    audit_path: Path,
    image_size: int,
    source_type: str | None,
) -> dict[str, Any]:
    """Load an interrupted cache build only when its lineage matches this run."""

    if not manifest_path.exists() or not audit_path.exists():
        return {"manifest_rows": [], "start_row_index": 1, "previews_written": 0}
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
    checkpoint_row_index = int(audit.get("checkpoint_row_index", len(manifest)))
    if checkpoint_row_index < len(manifest):
        raise ValueError(
            f"partial audit checkpoint_row_index {checkpoint_row_index} is behind manifest rows {len(manifest)}"
        )
    return {
        "manifest_rows": manifest.to_dict("records"),
        "start_row_index": checkpoint_row_index + 1,
        "previews_written": int(audit.get("previews_written", 0)),
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


def _safe_segment(value: str) -> str:
    """Keep preview paths readable and Windows-safe."""

    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned[:120] if cleaned else "unknown"


if __name__ == "__main__":
    main()
