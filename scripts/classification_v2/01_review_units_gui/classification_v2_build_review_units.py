from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.review.review_unit_builder import (
    ReviewUnitConfig,
    build_review_units,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical review units for classification_v2."
    )
    parser.add_argument(
        "--intervals-csv",
        type=Path,
        default=Path(r"outputs/classification_v2/sequence_features/temporal_label_intervals.csv"),
    )
    parser.add_argument(
        "--sequence-window-manifest-csv",
        type=Path,
        default=Path(r"outputs/classification_v2/sequence_features/sequence_window_manifest.csv"),
    )
    parser.add_argument(
        "--window-review-manifest-csv",
        type=Path,
        default=Path(r"outputs/classification_v2/review_templates/full_review_manifest.csv"),
    )
    parser.add_argument(
        "--disable-window-review-overlay",
        action="store_true",
        help=(
            "Do not read a canonical window-review manifest. Use this for a "
            "versioned rebuild that has no same-lineage window review artifact."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"outputs/classification_v2/review_units"),
    )
    parser.add_argument("--max-units-per-template", type=int, default=5000)
    args = parser.parse_args()

    window_review = args.window_review_manifest_csv
    if args.disable_window_review_overlay or not window_review.exists():
        window_review = None

    audit = build_review_units(
        ReviewUnitConfig(
            intervals_csv=args.intervals_csv,
            sequence_window_manifest_csv=args.sequence_window_manifest_csv,
            window_review_manifest_csv=window_review,
            output_dir=args.output_dir,
            max_units_per_template=args.max_units_per_template,
        )
    )

    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
