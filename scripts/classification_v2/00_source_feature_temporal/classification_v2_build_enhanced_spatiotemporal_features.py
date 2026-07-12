"""CLI wrapper for enhanced spatio-temporal features in classification_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.features.spatiotemporal import (
    audit_enhanced_spatiotemporal_features,
    build_enhanced_spatiotemporal_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build enhanced spatio-temporal/social/shape features after geometry+ROI "
            "and before review template generation."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--cvat-label-stride", type=int, default=6)
    parser.add_argument("--legacy-expected-sequence-length", type=int, default=16)
    parser.add_argument("--social-near-distance-n", type=float, default=0.08)
    parser.add_argument("--social-contact-iou-threshold", type=float, default=0.01)
    parser.add_argument("--social-contact-overlap-threshold", type=float, default=0.05)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)

    df = pd.read_csv(args.input_csv, low_memory=False)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be > 0")
        df = df.head(args.max_rows).copy()

    out = build_enhanced_spatiotemporal_features(
        df,
        cvat_label_stride=args.cvat_label_stride,
        legacy_expected_sequence_length=args.legacy_expected_sequence_length,
        social_near_distance_n=args.social_near_distance_n,
        social_contact_iou_threshold=args.social_contact_iou_threshold,
        social_contact_overlap_threshold=args.social_contact_overlap_threshold,
    )
    audit = audit_enhanced_spatiotemporal_features(out)
    audit["input_csv"] = str(args.input_csv)
    audit["output_csv"] = str(args.output_csv)
    audit["parameters"] = {
        "cvat_label_stride": args.cvat_label_stride,
        "legacy_expected_sequence_length": args.legacy_expected_sequence_length,
        "social_near_distance_n": args.social_near_distance_n,
        "social_contact_iou_threshold": args.social_contact_iou_threshold,
        "social_contact_overlap_threshold": args.social_contact_overlap_threshold,
        "max_rows": args.max_rows,
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {args.output_csv} rows={len(out)} cols={len(out.columns)}")
    print(f"[OK] wrote {args.audit_json}")
    if audit.get("errors"):
        print(f"[ERRORS] {audit['errors']}")
    if audit.get("warnings"):
        print(f"[WARNINGS] {audit['warnings']}")


if __name__ == "__main__":
    main()
