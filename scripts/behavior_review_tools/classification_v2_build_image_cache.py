from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.datasets.image_sequence_dataset import (
    ClassificationV2ImageSequenceDataset,
    ImageSequenceDatasetConfig,
    chw_float_to_hwc_uint8,
    context_cache_relative_path,
)


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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audit = build_image_cache(
        frame_context_csv=args.frame_context_csv,
        window_context_csv=args.window_context_csv,
        output_dir=args.output_dir,
        image_size=args.image_size,
        max_contexts=args.max_contexts,
        source_type=args.source_type,
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
    overwrite: bool,
) -> dict[str, Any]:
    """Materialize audited crops so training does not repeatedly seek videos."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if max_contexts is not None and max_contexts <= 0:
        raise ValueError("max_contexts must be positive when provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / f"actor_rgb_{image_size}"
    cache_root.mkdir(parents=True, exist_ok=True)
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
    try:
        for row in frame.itertuples(index=False):
            row_dict = row._asdict()
            context_id = str(row_dict["image_context_id"])
            rel_path = context_cache_relative_path(context_id)
            cache_path = cache_root / rel_path
            if cache_path.exists() and not overwrite:
                skipped_existing += 1
            else:
                image_chw = dataset._load_frame_image(row_dict)
                if image_chw is None:
                    failed += 1
                    continue
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, chw_float_to_hwc_uint8(image_chw))
                loaded += 1
            manifest_rows.append(
                {
                    "image_context_id": context_id,
                    "cache_path": str(Path(f"actor_rgb_{image_size}") / rel_path),
                    "image_size": int(image_size),
                    "cache_format": "npy_uint8_rgb_hwc",
                    "source_type": str(row_dict.get("source_type", "")),
                    "video_key": str(row_dict.get("video_key", "")),
                    "frame_index": row_dict.get("frame_index", ""),
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
        "manifest_csv": str(manifest_path),
        "image_size": int(image_size),
        "selected_context_rows": int(len(frame)),
        "manifest_rows": int(len(manifest)),
        "loaded_rows": int(loaded),
        "skipped_existing_rows": int(skipped_existing),
        "failed_rows": int(failed),
        "source_type_filter": source_type,
        "cache_format": "npy_uint8_rgb_hwc",
        "valid": bool(len(manifest) > 0 and failed == 0),
    }
    (output_dir / "cache_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


if __name__ == "__main__":
    main()
