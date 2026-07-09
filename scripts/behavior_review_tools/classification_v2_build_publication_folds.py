from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.metadata.recording_groups import (
    assign_publication_splits,
    build_recording_group_manifest,
    json_default,
    parse_ratios,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication-safe grouped train/val/test splits.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv"),
    )
    parser.add_argument(
        "--recording-group-manifest-csv",
        type=Path,
        default=None,
        help="Optional prebuilt recording_group_manifest.csv. If absent it is built from --manifest-csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/publication_splits"),
    )
    parser.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test ratios")
    parser.add_argument(
        "--group-level",
        choices=["recording_date", "recording_session", "video"],
        default="recording_date",
    )
    parser.add_argument("--label-col", default="behavior_window_label")
    parser.add_argument("--valid-col", default="window_valid_for_main_train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest_csv.exists():
        raise FileNotFoundError(args.manifest_csv)
    rows = pd.read_csv(args.manifest_csv, low_memory=False)

    if args.recording_group_manifest_csv is not None:
        if not args.recording_group_manifest_csv.exists():
            raise FileNotFoundError(args.recording_group_manifest_csv)
        group_manifest = pd.read_csv(args.recording_group_manifest_csv, low_memory=False)
        recording_group_audit = {"recording_group_manifest_csv": str(args.recording_group_manifest_csv)}
    else:
        group_tables = build_recording_group_manifest(rows, group_level=args.group_level)
        group_manifest = group_tables.manifest
        recording_group_audit = group_tables.audit

    ratios = parse_ratios(args.ratios)
    split_tables = assign_publication_splits(
        rows,
        group_manifest,
        ratios=ratios,
        label_col=args.label_col,
        valid_col=args.valid_col,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_path = args.output_dir / "recording_group_manifest.csv"
    split_path = args.output_dir / "publication_split_manifest.csv"
    audit_path = args.output_dir / "publication_split_audit.json"
    group_manifest.to_csv(group_path, index=False)
    split_tables.split_manifest.to_csv(split_path, index=False)
    audit = {
        "manifest_csv": str(args.manifest_csv),
        "recording_group_manifest_csv": str(group_path),
        "publication_split_manifest_csv": str(split_path),
        "publication_split_audit_json": str(audit_path),
        "group_level": args.group_level,
        "recording_group_audit": recording_group_audit,
        "publication_split_audit": split_tables.audit,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=json_default))
    if split_tables.audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
