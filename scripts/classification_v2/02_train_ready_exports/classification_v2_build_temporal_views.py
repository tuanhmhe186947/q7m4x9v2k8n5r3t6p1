"""Build audited fixed-six and native temporal-view manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.features.temporal_views import (
    build_temporal_views,
    write_temporal_view_outputs,
)


def parse_args() -> argparse.Namespace:
    """Parse explicit lineage paths; no canonical fallback is allowed."""

    parser = argparse.ArgumentParser(
        description="Build classification_v2 temporal views after harmonization."
    )
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--harmonized-frame-csv", type=Path, required=True)
    parser.add_argument("--temporal-interval-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build in memory first, then persist only a valid complete packet."""

    args = parse_args()
    windows = pd.read_csv(args.window_manifest, low_memory=False)
    frames = pd.read_csv(args.harmonized_frame_csv, low_memory=False)
    intervals = pd.read_csv(args.temporal_interval_csv, low_memory=False)
    result = build_temporal_views(windows, frames, intervals)
    summary = {
        **result.audit,
        "dry_run": bool(args.dry_run),
        "output_dir": str(args.output_dir),
    }
    if not args.dry_run:
        summary["output_paths"] = write_temporal_view_outputs(
            result,
            args.output_dir,
            overwrite=args.overwrite,
            input_artifacts={
                "window_manifest": args.window_manifest,
                "harmonized_frames": args.harmonized_frame_csv,
                "temporal_intervals": args.temporal_interval_csv,
            },
        )
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
