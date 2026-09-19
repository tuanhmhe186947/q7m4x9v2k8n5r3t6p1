"""Build a resume-safe non-interaction view with corrected ROI direction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.safe_non_interaction_view import (
    build_roi_direction_corrected_noninteraction_view,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-view-csv", type=Path, required=True)
    parser.add_argument("--output-view-csv", type=Path, required=True)
    parser.add_argument("--output-audit-json", type=Path, required=True)
    parser.add_argument("--existing-decisions-csv", type=Path)
    return parser.parse_args()


def _existing_review_keys(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = set(pd.read_csv(path, nrows=0).columns)
    if "review_unit_id" not in columns:
        raise ValueError("existing decisions missing review_unit_id")
    ids = (
        pd.read_csv(path, usecols=["review_unit_id"])["review_unit_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("existing decisions require unique nonblank review keys")
    return set(ids)


def main() -> int:
    args = parse_args()
    if not args.safe_view_csv.is_file():
        raise FileNotFoundError(args.safe_view_csv)
    if args.output_view_csv.exists() or args.output_audit_json.exists():
        raise FileExistsError("refusing to overwrite corrected view artifacts")

    safe_view = pd.read_csv(args.safe_view_csv, low_memory=False)
    preserved = _existing_review_keys(args.existing_decisions_csv)
    result = build_roi_direction_corrected_noninteraction_view(
        safe_view,
        preserve_review_keys=preserved,
    )

    args.output_view_csv.parent.mkdir(parents=True, exist_ok=True)
    result.view.to_csv(args.output_view_csv, index=False, lineterminator="\n")
    audit = {
        **result.audit,
        "input_safe_view_path": str(args.safe_view_csv.resolve()),
        "input_safe_view_sha256": sha256_file(args.safe_view_csv),
        "existing_review_keys_requested": len(preserved),
        "output_view_path": str(args.output_view_csv.resolve()),
        "output_view_sha256": sha256_file(args.output_view_csv),
    }
    write_json(args.output_audit_json, audit)
    print(
        "ROI_DIRECTION_CORRECTED_NONINTERACTION_VIEW "
        f"rows={len(result.view)} "
        f"suppressed={audit['suppressed_roi_only_explore_count']} "
        f"preserved={audit['preserved_existing_review_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
