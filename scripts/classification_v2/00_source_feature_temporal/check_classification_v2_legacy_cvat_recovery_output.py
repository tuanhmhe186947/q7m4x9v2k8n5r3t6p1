"""Check a recovered legacy dense map against CVAT-derived recovery inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from legacy_burst_recovery.cvat_recovery_validation import (
    validate_cvat_recovered_dense,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-csv", type=Path, required=True)
    parser.add_argument("--anchor-csv", type=Path, required=True)
    parser.add_argument("--dense-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--filter-group-id")
    parser.add_argument("--bbox-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    center = pd.read_csv(args.center_csv, low_memory=False)
    anchors = pd.read_csv(args.anchor_csv, low_memory=False)
    dense = pd.read_csv(args.dense_csv, low_memory=False)
    if args.filter_group_id:
        center = center.loc[center["group_id"].astype(str).eq(args.filter_group_id)]
        anchors = anchors.loc[
            anchors["group_id"].astype(str).eq(args.filter_group_id)
        ]
        dense = dense.loc[dense["group_id"].astype(str).eq(args.filter_group_id)]
    audit = validate_cvat_recovered_dense(
        center,
        anchors,
        dense,
        bbox_tolerance=args.bbox_tolerance,
    )
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
