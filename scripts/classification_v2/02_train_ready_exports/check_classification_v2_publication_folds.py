from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check classification_v2 publication split leakage.")
    parser.add_argument(
        "--split-manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/publication_splits/publication_split_manifest.csv"),
    )
    parser.add_argument(
        "--recording-group-manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/publication_splits/recording_group_manifest.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/publication_splits/check_publication_folds_audit.json"),
    )
    parser.add_argument("--id-col", default="window_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.split_manifest_csv.exists():
        raise FileNotFoundError(args.split_manifest_csv)
    if not args.recording_group_manifest_csv.exists():
        raise FileNotFoundError(args.recording_group_manifest_csv)

    splits = pd.read_csv(args.split_manifest_csv, low_memory=False)
    groups = pd.read_csv(args.recording_group_manifest_csv, low_memory=False)
    required_split = {args.id_col, "recording_group_id", "split"}
    required_group = {"recording_group_id", "canonical_recording_date", "biological_subject_scope_known"}
    missing_split = sorted(required_split.difference(splits.columns))
    missing_group = sorted(required_group.difference(groups.columns))

    leakage = []
    if not missing_split:
        leakage = (
            splits.groupby("recording_group_id")["split"].nunique().loc[lambda s: s > 1].index.astype(str).tolist()
        )
    date_leakage = []
    if not missing_split and "canonical_recording_date" in groups.columns:
        merged = splits.merge(
            groups[["recording_group_id", "canonical_recording_date"]].drop_duplicates(),
            on="recording_group_id",
            how="left",
            validate="many_to_one",
        )
        date_leakage = (
            merged.groupby("canonical_recording_date")["split"]
            .nunique()
            .loc[lambda s: s > 1]
            .index.astype(str)
            .tolist()
        )

    errors = []
    if missing_split:
        errors.append(f"missing_split_columns={missing_split}")
    if missing_group:
        errors.append(f"missing_group_columns={missing_group}")
    if leakage:
        errors.append(f"recording_group_leakage={len(leakage)}")
    if date_leakage:
        errors.append(f"recording_date_leakage={len(date_leakage)}")

    audit = {
        "split_manifest_csv": str(args.split_manifest_csv),
        "recording_group_manifest_csv": str(args.recording_group_manifest_csv),
        "id_col": args.id_col,
        "rows": int(len(splits)),
        "unique_ids": int(splits[args.id_col].nunique(dropna=False)) if args.id_col in splits else 0,
        "recording_group_count": int(splits["recording_group_id"].nunique()) if "recording_group_id" in splits else 0,
        "split_rows": splits["split"].value_counts(dropna=False).to_dict() if "split" in splits else {},
        "recording_group_leakage_count": int(len(leakage)),
        "recording_group_leakage_sample": leakage[:50],
        "recording_date_leakage_count": int(len(date_leakage)),
        "recording_date_leakage_sample": date_leakage[:50],
        "biological_subject_scope_known": bool(
            groups["biological_subject_scope_known"].fillna(False).astype(bool).any()
        )
        if "biological_subject_scope_known" in groups
        else False,
        "warnings": ["pig_id_not_validated_as_cross_video_biological_identity"],
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
