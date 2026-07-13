from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.datasets.native_temporal_units import (
    build_native_temporal_units,
    json_default,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one-row-per-temporal-unit classification_v2 dataset."
    )
    parser.add_argument(
        "--intervals-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/sequence_features/"
            "temporal_label_intervals.csv"
        ),
    )
    parser.add_argument(
        "--reviewed-frame-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/review_policy/reviewed_frame_features.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/native_temporal_units"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.intervals_csv.exists():
        raise FileNotFoundError(args.intervals_csv)
    if not args.reviewed_frame_csv.exists():
        raise FileNotFoundError(args.reviewed_frame_csv)

    intervals = pd.read_csv(args.intervals_csv, low_memory=False)
    reviewed_frames = pd.read_csv(args.reviewed_frame_csv, low_memory=False)
    tables = build_native_temporal_units(intervals, reviewed_frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "native_temporal_unit_manifest.csv"
    audit_path = args.output_dir / "native_temporal_unit_audit.json"
    audit = {
        **tables.audit,
        "intervals_csv": str(args.intervals_csv),
        "reviewed_frame_csv": str(args.reviewed_frame_csv),
        "native_temporal_unit_manifest_csv": str(manifest_path),
        "native_temporal_unit_audit_json": str(audit_path),
    }
    audit["native_temporal_unit_manifest_written"] = not bool(audit["errors"])
    if not audit["errors"]:
        tables.manifest.to_csv(manifest_path, index=False)
    audit_path.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        ),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=json_default))
    if audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
