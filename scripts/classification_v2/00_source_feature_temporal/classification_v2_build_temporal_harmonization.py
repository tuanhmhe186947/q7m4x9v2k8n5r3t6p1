"""CLI wrapper for temporal label harmonization in classification_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.features.temporal_harmonization import (
    audit_temporal_harmonization,
    build_temporal_label_intervals,
    harmonize_temporal_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harmonize source-specific temporal labels before sequence-window "
            "generation. Legacy is treated as 16f constant; CVAT XML as anchor "
            "labels covering 6-frame intervals by default."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--intervals-csv", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--cvat-label-stride", type=int, default=6)
    parser.add_argument("--legacy-expected-sequence-length", type=int, default=16)
    parser.add_argument("--legacy-min-complete-ratio", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived output and audit files explicitly.",
    )
    return parser.parse_args()


def _write_audit(path: Path, audit: dict[str, object]) -> None:
    """Persist audit evidence before success or fail-closed exit."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fail_if_audit_has_errors(
    audit: dict[str, object],
    audit_path: Path,
) -> None:
    """Write the failed audit and prevent invalid temporal CSVs."""

    errors = audit.get("errors") or []
    if not errors:
        return
    _write_audit(audit_path, audit)
    print(f"[ERRORS] {errors}")
    raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    require_output_paths_available(
        [args.output_csv, args.intervals_csv, args.audit_json],
        overwrite=args.overwrite,
    )
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be > 0")

    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    out = harmonize_temporal_labels(
        df,
        cvat_label_stride=args.cvat_label_stride,
        legacy_expected_sequence_length=args.legacy_expected_sequence_length,
        legacy_min_complete_ratio=args.legacy_min_complete_ratio,
    )
    intervals = build_temporal_label_intervals(
        out,
        cvat_label_stride=args.cvat_label_stride,
        legacy_expected_sequence_length=args.legacy_expected_sequence_length,
    )
    audit = audit_temporal_harmonization(out, intervals)
    audit["input_csv"] = str(args.input_csv)
    audit["output_csv"] = str(args.output_csv)
    audit["intervals_csv"] = str(args.intervals_csv)
    audit["parameters"] = {
        "cvat_label_stride": args.cvat_label_stride,
        "legacy_expected_sequence_length": args.legacy_expected_sequence_length,
        "legacy_min_complete_ratio": args.legacy_min_complete_ratio,
        "max_rows": args.max_rows,
        "overwrite": args.overwrite,
    }

    _fail_if_audit_has_errors(audit, args.audit_json)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.intervals_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    intervals.to_csv(args.intervals_csv, index=False)
    _write_audit(args.audit_json, audit)

    print(f"[OK] wrote {args.output_csv} rows={len(out)} cols={len(out.columns)}")
    print(f"[OK] wrote {args.intervals_csv} rows={len(intervals)}")
    print(f"[OK] wrote {args.audit_json}")
    if audit.get("warnings"):
        print(f"[WARNINGS] {audit['warnings']}")


if __name__ == "__main__":
    main()
