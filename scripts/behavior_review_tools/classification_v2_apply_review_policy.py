"""CLI wrapper for applying classification_v2 ROI/behavior review policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from pig_behavior.classification_v2.build_dataset import build_reviewed_frame_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply review policy after classification_v2 ROI features and "
            "write training_ready_frame_features.csv."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
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
        help=(
            "auto keeps rows using automatic policy when manual review is missing; "
            "exclude removes still-pending rows from final training flags."
        ),
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
        roi_features_csv=args.input_csv,
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