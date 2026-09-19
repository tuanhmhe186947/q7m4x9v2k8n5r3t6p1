"""Read-only post-apply check for hidden-reviewed frame features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROVENANCE_COLUMNS = {
    "hidden_before_review",
    "hidden_after_review",
    "hidden_review_status",
    "hidden_review_confidence",
    "hidden_review_reason",
    "hidden_source",
    "hidden_trust_status",
    "hidden_is_trusted",
    "hidden_review_item_id",
    "hidden_reviewer",
    "hidden_reviewed_at",
    "hidden_effective_for_policy",
    "hidden_review_available_mask",
    "visibility_quality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    return parser.parse_args()


def _series_values_equal(left: pd.Series, right: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        equal = left.eq(right) | (left.isna() & right.isna())
        return bool(equal.all())
    normalized_left = left.fillna("<NA>").astype(str)
    normalized_right = right.fillna("<NA>").astype(str)
    return normalized_left.equals(normalized_right)


def main() -> None:
    args = parse_args()
    before = pd.read_csv(
        args.input_csv,
        low_memory=False,
        float_precision="round_trip",
    )
    after = pd.read_csv(
        args.output_csv,
        low_memory=False,
        float_precision="round_trip",
    )
    errors: list[str] = []
    if len(before) != len(after):
        errors.append(f"row_count_changed={len(before)}->{len(after)}")
    missing = sorted(PROVENANCE_COLUMNS.difference(after.columns))
    if missing:
        errors.append(f"missing_provenance_columns={missing}")
    if "hidden_review_item_id" in after.columns:
        duplicate = int(after["hidden_review_item_id"].duplicated().sum())
        if duplicate:
            errors.append(f"duplicate_hidden_review_item_id={duplicate}")

    protected = sorted(
        set(before.columns).intersection(after.columns).difference(PROVENANCE_COLUMNS | {"hidden"})
    )
    changed_protected: list[str] = []
    if len(before) == len(after):
        for column in protected:
            if not _series_values_equal(before[column], after[column]):
                changed_protected.append(column)
    if changed_protected:
        errors.append(f"unexpected_changed_columns={changed_protected}")

    audit = {
        "input_rows": int(len(before)),
        "output_rows": int(len(after)),
        "protected_columns_checked": len(protected),
        "unexpected_changed_columns": changed_protected,
        "errors": errors,
        "warnings": [],
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"FAIL: {errors}")
    print("PASS: Hidden apply preserved rows and non-Hidden source columns.")


if __name__ == "__main__":
    main()
