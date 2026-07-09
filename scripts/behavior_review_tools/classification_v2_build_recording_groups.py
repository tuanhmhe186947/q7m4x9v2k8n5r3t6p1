from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.metadata.recording_groups import (
    build_recording_group_manifest,
    json_default,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build classification_v2 recording-group metadata manifest.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/classification_v2/sequence_features_reviewed/sequence_window_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/publication_splits"),
    )
    parser.add_argument("--manual-metadata-csv", type=Path, default=None)
    parser.add_argument(
        "--group-level",
        choices=["recording_date", "recording_session", "video"],
        default="recording_date",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    manual = None
    if args.manual_metadata_csv is not None:
        if not args.manual_metadata_csv.exists():
            raise FileNotFoundError(args.manual_metadata_csv)
        manual = pd.read_csv(args.manual_metadata_csv, low_memory=False)

    rows = pd.read_csv(args.input_csv, low_memory=False)
    tables = build_recording_group_manifest(rows, manual_metadata=manual, group_level=args.group_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "recording_group_manifest.csv"
    audit_path = args.output_dir / "recording_group_audit.json"
    tables.manifest.to_csv(manifest_path, index=False)
    audit = {
        **tables.audit,
        "input_csv": str(args.input_csv),
        "manual_metadata_csv": str(args.manual_metadata_csv) if args.manual_metadata_csv else None,
        "recording_group_manifest_csv": str(manifest_path),
        "recording_group_audit_json": str(audit_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=json_default))
    if audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
