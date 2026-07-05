"""Dataset build helpers for classification_v2.

Current stable entry point:
- start from frame_features_roi.csv / spatiotemporal_frame_features_roi.csv
- add/apply ROI behavior review policy
- write training_ready_frame_features.csv

Earlier steps are currently run by scripts:
- classification_v2_merge_sources.py
- classification_v2_apply_context_policy.py
- classification_v2_build_geometry_features.py
- classification_v2_build_roi_features.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.features.review_policy import (
    add_roi_label_review_attributes,
    apply_behavior_review_decisions,
    build_behavior_review_template,
)
from pig_behavior.classification_v2.validation import validate_reviewed_frame_features


def build_reviewed_frame_features(
    *,
    roi_features_csv: Path,
    output_csv: Path,
    audit_json: Path | None = None,
    review_csv: Path | None = None,
    review_template_csv: Path | None = None,
    conflicts_csv: Path | None = None,
    pending_csv: Path | None = None,
    pending_policy: str = "auto",
    include_weak_in_training: bool = False,
) -> pd.DataFrame:
    """Build reviewed frame-level features from ROI feature CSV."""
    if pending_policy not in {"auto", "exclude"}:
        raise ValueError("pending_policy must be 'auto' or 'exclude'.")

    if not roi_features_csv.exists():
        raise FileNotFoundError(roi_features_csv)

    print(f"reading ROI feature CSV: {roi_features_csv}")
    df = pd.read_csv(roi_features_csv, low_memory=False)

    print("adding automatic ROI label-review attributes...")
    reviewed = add_roi_label_review_attributes(df)

    if review_template_csv is not None:
        template = build_behavior_review_template(
            reviewed,
            only_review_required=True,
        )
        review_template_csv.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(review_template_csv, index=False)
        print(f"saved review template: {review_template_csv} rows={len(template)}")

    review_decisions = None
    if review_csv is not None:
        if not review_csv.exists():
            raise FileNotFoundError(review_csv)
        print(f"reading manual review CSV: {review_csv}")
        review_decisions = pd.read_csv(review_csv, low_memory=False)

    print("applying review decisions...")
    reviewed = apply_behavior_review_decisions(
        reviewed,
        review_decisions,
        pending_policy=pending_policy,
        include_weak_in_training=include_weak_in_training,
    )

    print("validating reviewed frame features...")
    audit = validate_reviewed_frame_features(reviewed)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if audit["errors"]:
        raise ValueError(f"Review policy audit errors: {audit['errors']}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(output_csv, index=False)
    print(f"saved training-ready frame features: {output_csv}")

    if audit_json is not None:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved audit: {audit_json}")

    if conflicts_csv is not None:
        conflicts = reviewed[
            reviewed["roi_consistency_status"].isin(
                [
                    "target_roi_near_no_contact",
                    "target_roi_far",
                    "target_roi_unavailable",
                ]
            )
        ].copy()
        conflicts_csv.parent.mkdir(parents=True, exist_ok=True)
        conflicts.to_csv(conflicts_csv, index=False)
        print(f"saved ROI conflict rows: {conflicts_csv} rows={len(conflicts)}")

    if pending_csv is not None:
        pending = reviewed[reviewed["review_decision"].eq("pending")].copy()
        pending_csv.parent.mkdir(parents=True, exist_ok=True)
        pending.to_csv(pending_csv, index=False)
        print(f"saved pending rows: {pending_csv} rows={len(pending)}")

    return reviewed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build reviewed training-ready frame features from "
            "classification_v2 ROI feature CSV."
        )
    )
    parser.add_argument("--roi-features-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument("--review-template-csv", type=Path, default=None)
    parser.add_argument("--conflicts-csv", type=Path, default=None)
    parser.add_argument("--pending-csv", type=Path, default=None)
    parser.add_argument(
        "--pending-policy",
        choices=["auto", "exclude"],
        default="auto",
    )
    parser.add_argument(
        "--include-weak-in-training",
        action="store_true",
        help="Allow weak samples for robust training flags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_reviewed_frame_features(
        roi_features_csv=args.roi_features_csv,
        output_csv=args.output_csv,
        audit_json=args.audit_json,
        review_csv=args.review_csv,
        review_template_csv=args.review_template_csv,
        conflicts_csv=args.conflicts_csv,
        pending_csv=args.pending_csv,
        pending_policy=args.pending_policy,
        include_weak_in_training=args.include_weak_in_training,
    )


if __name__ == "__main__":
    main()