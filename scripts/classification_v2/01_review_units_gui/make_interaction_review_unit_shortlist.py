from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(r"outputs/classification_v2/review_units/interaction_review_unit_template.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(r"outputs/classification_v2/review_units/interaction_review_unit_shortlist.csv"),
    )
    parser.add_argument("--target-partner-far", type=int, default=80)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    if df.empty:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print("empty input")
        return

    reason = df["review_reason"].fillna("").astype(str)
    must_mask = (
        reason.str.contains("interaction_contact_weak", regex=False)
        | reason.str.contains("interaction_label_temporal_not_stable", regex=False)
        | reason.str.contains("interaction_partner_missing", regex=False)
        | reason.str.contains("temporal_unit_not_stable", regex=False)
        | reason.str.contains("interaction_review", regex=False)
    )
    must = df[must_mask].copy()
    far = df[~must_mask & reason.str.contains("interaction_partner_far", regex=False)].copy()
    rest = df[~must_mask & ~df.index.isin(far.index)].copy()

    far_parts = []
    if not far.empty:
        total = len(far)
        for _, g in far.groupby(["source_type", "review_unit_type"], dropna=False):
            quota = max(3, math.ceil(args.target_partner_far * len(g) / total))
            far_parts.append(g.sort_values("review_priority", ascending=False).head(quota))
    far_sample = pd.concat(far_parts, ignore_index=True) if far_parts else pd.DataFrame(columns=df.columns)
    if len(far_sample) > args.target_partner_far:
        far_sample = far_sample.sort_values("review_priority", ascending=False).head(args.target_partner_far)

    # Keep a small high-priority sample of general candidates, especially fight.
    rest_sample = rest.sort_values("review_priority", ascending=False).head(60)
    short = pd.concat([must, far_sample, rest_sample], ignore_index=True)
    short = short.drop_duplicates("review_unit_id")
    short = short.sort_values("review_priority", ascending=False).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    short.to_csv(args.output_csv, index=False)
    audit = {
        "input_rows": int(len(df)),
        "shortlist_rows": int(len(short)),
        "reason_counts": short["review_reason"].value_counts(dropna=False).to_dict(),
        "behavior_counts": short["behavior_label"].value_counts(dropna=False).to_dict(),
        "source_counts": short["source_type"].value_counts(dropna=False).to_dict(),
        "unit_type_counts": short["review_unit_type"].value_counts(dropna=False).to_dict(),
        "output_csv": str(args.output_csv),
    }
    audit_path = args.output_csv.with_name(args.output_csv.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
