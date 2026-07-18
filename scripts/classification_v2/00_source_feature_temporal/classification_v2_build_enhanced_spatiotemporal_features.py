"""CLI wrapper for enhanced spatio-temporal features in classification_v2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.features.pen_context import (
    DEFAULT_PEN_MASK_SHA256,
    audit_pen_context_features,
    build_pen_context_features,
)
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
    parser.add_argument("--stationary-speed-threshold", type=float, default=0.002)
    parser.add_argument("--active-speed-threshold", type=float, default=0.006)
    parser.add_argument("--turning-angle-threshold-deg", type=float, default=30.0)
    parser.add_argument(
        "--pen-mask",
        type=Path,
        default=Path("data/annotations/scene/mask.png"),
        help="Fixed-camera pen calibration mask used for pen-context features.",
    )
    parser.add_argument("--pen-mask-threshold", type=int, default=127)
    parser.add_argument(
        "--expected-pen-mask-sha256",
        type=str,
        default=DEFAULT_PEN_MASK_SHA256,
        help="Required fail-closed SHA-256 for fixed-camera mask calibration.",
    )
    parser.add_argument(
        "--pen-near-boundary-clearance-ratio",
        type=float,
        default=1.0,
        help="Near-wall threshold measured in actor half-diagonal units.",
    )
    parser.add_argument(
        "--pen-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append hash-bound pen-boundary features. Use --no-pen-context "
            "only for an explicit feature ablation."
        ),
    )
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
    """Write the failed audit and prevent a poisoned CSV from being emitted."""

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
        [args.output_csv, args.audit_json],
        overwrite=args.overwrite,
    )

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
        stationary_speed_threshold=args.stationary_speed_threshold,
        active_speed_threshold=args.active_speed_threshold,
        turning_angle_threshold_rad=math.radians(args.turning_angle_threshold_deg),
    )
    pen_audit: dict[str, object] | None = None
    if args.pen_context:
        if not str(args.expected_pen_mask_sha256 or "").strip():
            raise ValueError(
                "Pen context requires --expected-pen-mask-sha256"
            )
        out = build_pen_context_features(
            out,
            mask_path=args.pen_mask,
            mask_threshold=args.pen_mask_threshold,
            near_boundary_clearance_ratio=(
                args.pen_near_boundary_clearance_ratio
            ),
            expected_mask_sha256=args.expected_pen_mask_sha256,
        )
        pen_audit = audit_pen_context_features(
            out,
            mask_path=args.pen_mask,
            mask_threshold=args.pen_mask_threshold,
            near_boundary_clearance_ratio=(
                args.pen_near_boundary_clearance_ratio
            ),
            input_rows=len(df),
            expected_mask_sha256=args.expected_pen_mask_sha256,
        )
    audit = audit_enhanced_spatiotemporal_features(out)
    if pen_audit is not None:
        audit["pen_context"] = pen_audit
        audit["errors"] = [
            *audit.get("errors", []),
            *[f"pen_context={error}" for error in pen_audit["errors"]],
        ]
    audit["input_csv"] = str(args.input_csv)
    audit["output_csv"] = str(args.output_csv)
    audit["parameters"] = {
        "cvat_label_stride": args.cvat_label_stride,
        "legacy_expected_sequence_length": args.legacy_expected_sequence_length,
        "social_near_distance_n": args.social_near_distance_n,
        "social_contact_iou_threshold": args.social_contact_iou_threshold,
        "social_contact_overlap_threshold": args.social_contact_overlap_threshold,
        "stationary_speed_threshold": args.stationary_speed_threshold,
        "active_speed_threshold": args.active_speed_threshold,
        "turning_angle_threshold_deg": args.turning_angle_threshold_deg,
        "pen_context": args.pen_context,
        "pen_mask": str(args.pen_mask) if args.pen_context else None,
        "pen_mask_threshold": args.pen_mask_threshold,
        "expected_pen_mask_sha256": args.expected_pen_mask_sha256,
        "pen_near_boundary_clearance_ratio": (
            args.pen_near_boundary_clearance_ratio
        ),
        "max_rows": args.max_rows,
        "overwrite": args.overwrite,
    }

    _fail_if_audit_has_errors(audit, args.audit_json)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    _write_audit(args.audit_json, audit)

    print(f"[OK] wrote {args.output_csv} rows={len(out)} cols={len(out.columns)}")
    print(f"[OK] wrote {args.audit_json}")
    if audit.get("warnings"):
        print(f"[WARNINGS] {audit['warnings']}")


if __name__ == "__main__":
    main()
