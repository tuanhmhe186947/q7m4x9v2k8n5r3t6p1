from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.datasets.image_context_index import (
    FRAME_CONTEXT_COLUMNS,
    WINDOW_CONTEXT_COLUMNS,
    build_image_context_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build classification_v2 image-context index manifests."
    )
    parser.add_argument(
        "--frame-features-csv",
        type=Path,
        default=Path("outputs/classification_v2/review_policy/reviewed_frame_features.csv"),
    )
    parser.add_argument(
        "--window-manifest-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/classification_v2/train_ready_windows")
    )
    parser.add_argument("--video-root", type=Path, default=Path("data/videos"))
    parser.add_argument(
        "--legacy-crop-root",
        type=Path,
        default=Path("data/raw/legacy_full_multigt_masked_nodup_16f/crops"),
    )
    parser.add_argument("--max-frame-rows", type=int, default=None)
    parser.add_argument(
        "--lineage-scope",
        default=None,
        help="Optional explicit profile claim propagated to both manifests.",
    )
    parser.add_argument(
        "--human-review-complete",
        choices=("true", "false"),
        default=None,
        help="Required with --lineage-scope; use false for unreviewed lanes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived image-context artifacts explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.lineage_scope is None) != (args.human_review_complete is None):
        raise ValueError(
            "--lineage-scope and --human-review-complete must be provided together"
        )
    frame_path = args.output_dir / "image_frame_context_manifest.csv"
    window_path = args.output_dir / "image_window_context_manifest.csv"
    audit_path = args.output_dir / "image_context_index_audit.json"
    require_output_paths_available(
        [frame_path, window_path, audit_path],
        overwrite=args.overwrite,
    )
    frame_header = pd.read_csv(args.frame_features_csv, nrows=0).columns.tolist()
    window_header = pd.read_csv(args.window_manifest_csv, nrows=0).columns.tolist()
    frame_usecols = [c for c in FRAME_CONTEXT_COLUMNS if c in frame_header]
    window_usecols = [c for c in WINDOW_CONTEXT_COLUMNS if c in window_header]

    frames = pd.read_csv(args.frame_features_csv, usecols=frame_usecols, low_memory=False)
    if args.max_frame_rows is not None:
        if args.max_frame_rows <= 0:
            raise ValueError("--max-frame-rows must be > 0")
        frames = frames.head(args.max_frame_rows).copy()
    windows = pd.read_csv(args.window_manifest_csv, usecols=window_usecols, low_memory=False)
    if args.lineage_scope is not None:
        human_review_complete = args.human_review_complete == "true"
        frames["lineage_scope"] = args.lineage_scope
        frames["human_review_complete"] = human_review_complete
        windows["lineage_scope"] = args.lineage_scope
        windows["human_review_complete"] = human_review_complete

    index = build_image_context_index(
        frames,
        windows,
        video_root=args.video_root,
        legacy_crop_root=args.legacy_crop_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index.frame_manifest.to_csv(frame_path, index=False)
    index.window_manifest.to_csv(window_path, index=False)
    audit = {
        "frame_features_csv": str(args.frame_features_csv),
        "window_manifest_csv": str(args.window_manifest_csv),
        "video_root": str(args.video_root),
        "legacy_crop_root": str(args.legacy_crop_root),
        "image_frame_context_manifest": str(frame_path),
        "image_window_context_manifest": str(window_path),
        **index.audit,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
