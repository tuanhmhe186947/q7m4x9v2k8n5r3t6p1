"""Audit pre-review structural availability of final temporal views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.final_view_contract_audit import (
    audit_pre_review_structural_view_availability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.frame_csv.is_file():
        raise FileNotFoundError(args.frame_csv)
    frames = pd.read_csv(
        args.frame_csv,
        usecols=[
            "source_type",
            "object_track_key",
            "temporal_unit_key",
            "frame_index",
        ],
        low_memory=False,
    )
    audit = audit_pre_review_structural_view_availability(
        frames,
        source_fps=args.source_fps,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    args.output_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["errors"]:
        raise ValueError(f"view availability audit failed: {audit['errors']}")


if __name__ == "__main__":
    main()
