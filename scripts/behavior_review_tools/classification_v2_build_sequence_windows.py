"""CLI wrapper for temporal harmonization and sequence-window features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.features.sequence_windows import (
    audit_sequence_windows,
    build_sequence_windows,
)
from pig_behavior.classification_v2.features.temporal_harmonization import audit_temporal_harmonization


def _parse_window_lengths(value: str) -> list[int]:
    parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("window length list must not be empty")
    out = []
    for p in parts:
        try:
            n = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid window length: {p}") from exc
        if n <= 0:
            raise argparse.ArgumentTypeError("window lengths must be > 0")
        out.append(n)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build temporal label intervals and long-format 6/8/12/16 sequence-window "
            "features from enhanced frame features."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--harmonized-frame-csv", type=Path, default=None)
    parser.add_argument("--temporal-intervals-csv", type=Path, default=None)
    parser.add_argument("--sequence-window-manifest-csv", type=Path, default=None)
    parser.add_argument("--sequence-window-features-csv", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--window-lengths", type=_parse_window_lengths, default=[6, 8, 12, 16])
    parser.add_argument("--legacy-window-stride", type=int, default=3)
    parser.add_argument("--cvat-window-stride-intervals", type=int, default=1)
    parser.add_argument("--cvat-label-stride", type=int, default=6)
    parser.add_argument("--legacy-expected-sequence-length", type=int, default=16)
    parser.add_argument("--default-fps", type=float, default=None)
    parser.add_argument("--min-bbox-valid-ratio", type=float, default=1.0)
    parser.add_argument("--max-hidden-ratio-main", type=float, default=0.5)
    parser.add_argument("--min-spatiotemporal-valid-ratio", type=float, default=1.0)
    parser.add_argument("--exclude-mixed-windows", action="store_true")
    parser.add_argument("--max-windows-per-track", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be > 0")

    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    harmonized, intervals, windows = build_sequence_windows(
        df,
        window_lengths=args.window_lengths,
        legacy_window_stride=args.legacy_window_stride,
        cvat_window_stride_intervals=args.cvat_window_stride_intervals,
        cvat_label_stride=args.cvat_label_stride,
        legacy_expected_sequence_length=args.legacy_expected_sequence_length,
        default_fps=args.default_fps,
        min_bbox_valid_ratio=args.min_bbox_valid_ratio,
        max_hidden_ratio_main=args.max_hidden_ratio_main,
        min_spatiotemporal_valid_ratio=args.min_spatiotemporal_valid_ratio,
        include_mixed_windows=not args.exclude_mixed_windows,
        max_windows_per_track=args.max_windows_per_track,
    )

    output_dir = args.output_dir
    harmonized_csv = args.harmonized_frame_csv or output_dir / "training_ready_frame_features_harmonized_preview.csv"
    intervals_csv = args.temporal_intervals_csv or output_dir / "temporal_label_intervals.csv"
    manifest_csv = args.sequence_window_manifest_csv or output_dir / "sequence_window_manifest.csv"
    features_csv = args.sequence_window_features_csv or output_dir / "sequence_window_features.csv"
    audit_json = args.audit_json or output_dir / "sequence_window_audit.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    harmonized_csv.parent.mkdir(parents=True, exist_ok=True)
    intervals_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_json.parent.mkdir(parents=True, exist_ok=True)

    harmonized.to_csv(harmonized_csv, index=False)
    intervals.to_csv(intervals_csv, index=False)
    windows.to_csv(manifest_csv, index=False)
    # At this stage manifest and feature table intentionally share rows. Keeping
    # a separate file path preserves the future contract if visual-path columns
    # or train-only columns are split later.
    windows.to_csv(features_csv, index=False)

    temporal_audit = audit_temporal_harmonization(harmonized, intervals)
    window_audit = audit_sequence_windows(windows, intervals)
    audit = {
        "input_csv": str(args.input_csv),
        "harmonized_frame_csv": str(harmonized_csv),
        "temporal_intervals_csv": str(intervals_csv),
        "sequence_window_manifest_csv": str(manifest_csv),
        "sequence_window_features_csv": str(features_csv),
        "parameters": {
            "window_lengths": args.window_lengths,
            "legacy_window_stride": args.legacy_window_stride,
            "cvat_window_stride_intervals": args.cvat_window_stride_intervals,
            "cvat_label_stride": args.cvat_label_stride,
            "legacy_expected_sequence_length": args.legacy_expected_sequence_length,
            "default_fps": args.default_fps,
            "min_bbox_valid_ratio": args.min_bbox_valid_ratio,
            "max_hidden_ratio_main": args.max_hidden_ratio_main,
            "min_spatiotemporal_valid_ratio": args.min_spatiotemporal_valid_ratio,
            "include_mixed_windows": not args.exclude_mixed_windows,
            "max_windows_per_track": args.max_windows_per_track,
            "max_rows": args.max_rows,
        },
        "temporal_harmonization": temporal_audit,
        "sequence_windows": window_audit,
        "errors": temporal_audit.get("errors", []) + window_audit.get("errors", []),
        "warnings": temporal_audit.get("warnings", []) + window_audit.get("warnings", []),
    }
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {harmonized_csv} rows={len(harmonized)} cols={len(harmonized.columns)}")
    print(f"[OK] wrote {intervals_csv} rows={len(intervals)}")
    print(f"[OK] wrote {manifest_csv} rows={len(windows)}")
    print(f"[OK] wrote {features_csv} rows={len(windows)}")
    print(f"[OK] wrote {audit_json}")
    if audit["errors"]:
        print(f"[ERRORS] {audit['errors']}")
    if audit["warnings"]:
        print(f"[WARNINGS] {audit['warnings']}")


if __name__ == "__main__":
    main()
