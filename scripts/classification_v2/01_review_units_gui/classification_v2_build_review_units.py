from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.review.behavior_review_selection import (
    BehaviorReviewSelectionConfig,
)
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
        "--native-evidence-csv",
        type=Path,
        required=True,
        help=(
            "Validated frame-level native evidence containing deterministic "
            "temporal-unit aggregate evidence."
        ),
    )
    parser.add_argument(
        "--sequence-window-manifest-csv",
        type=Path,
        default=None,
        help=(
            "Optional window manifest used only for affected-window coverage. "
            "Omit it for native-only review-unit construction."
        ),
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Build review units directly from temporal intervals without windows.",
    )
    parser.add_argument(
        "--window-review-manifest-csv",
        type=Path,
        default=None,
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
    parser.add_argument(
        "--max-units-per-template",
        type=int,
        default=0,
        help=(
            "Fail if a canonical template exceeds this size. Zero means no cap. "
            "Use pilot builders for sampled review instead of truncating canonical files."
        ),
    )
    parser.add_argument(
        "--include-all-retained-legacy-units",
        action="store_true",
        help=(
            "Deprecated diagnostic flag. Production selective review rejects "
            "global legacy selection."
        ),
    )
    parser.add_argument(
        "--full-native-unit-behavior-review",
        action="store_true",
        help=(
            "Deprecated diagnostic flag. Production selective review rejects "
            "global native-unit selection."
        ),
    )
    parser.add_argument(
        "--pig-strenet-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Optional exact-lineage Pig-STRENet artifact directory. When set, "
            "every temporal unit must have one review-evidence row."
        ),
    )
    parser.add_argument("--behavior-selection-seed", type=int, default=20260720)
    parser.add_argument("--behavior-random-per-stratum", type=int, default=5)
    parser.add_argument("--behavior-clean-control-per-stratum", type=int, default=0)
    parser.add_argument("--behavior-high-risk-fraction", type=float, default=0.0)
    parser.add_argument("--behavior-high-risk-cap", type=int, default=0)
    parser.add_argument("--behavior-high-risk-min-pool", type=int, default=20)
    parser.add_argument("--behavior-rare-census-max", type=int, default=0)
    args = parser.parse_args()

    if args.native_only and args.sequence_window_manifest_csv is not None:
        raise SystemExit("--native-only cannot be combined with --sequence-window-manifest-csv")
    window_manifest = None if args.native_only else args.sequence_window_manifest_csv
    window_review = args.window_review_manifest_csv
    if (
        args.disable_window_review_overlay
        or window_review is None
        or not window_review.exists()
    ):
        window_review = None
    if window_review is not None and window_manifest is None:
        raise SystemExit(
            "window-review overlay requires --sequence-window-manifest-csv; "
            "omit the overlay in --native-only mode"
        )

    audit = build_review_units(
        ReviewUnitConfig(
            intervals_csv=args.intervals_csv,
            output_dir=args.output_dir,
            native_evidence_csv=args.native_evidence_csv,
            sequence_window_manifest_csv=window_manifest,
            window_review_manifest_csv=window_review,
            max_units_per_template=args.max_units_per_template,
            include_all_retained_legacy_units=(
                args.include_all_retained_legacy_units
            ),
            include_all_retained_native_units=(
                args.full_native_unit_behavior_review
            ),
            pig_strenet_artifact_dir=args.pig_strenet_artifact_dir,
            behavior_selection=BehaviorReviewSelectionConfig(
                random_seed=args.behavior_selection_seed,
                random_per_stratum=args.behavior_random_per_stratum,
                clean_control_per_stratum=(
                    args.behavior_clean_control_per_stratum
                ),
                calibrated_high_risk_fraction=(
                    args.behavior_high_risk_fraction
                ),
                calibrated_high_risk_max_per_stratum=(
                    args.behavior_high_risk_cap
                ),
                calibrated_high_risk_min_pool=(
                    args.behavior_high_risk_min_pool
                ),
                rare_census_max_per_source_behavior=(
                    args.behavior_rare_census_max
                ),
            ),
        )
    )

    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
