"""Build a frozen-input burst posture authority without active-ledger access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.behavior_posture import (
    SAFE_DERIVATION_BEHAVIOR_AUTHORITIES,
    build_burst_posture_authority,
)
from pig_behavior.classification_v2.review.post_review_learning import (
    assert_not_active_behavior_ledger_path,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build burst-level posture authority from explicit frozen or "
            "synthetic behavior inputs."
        )
    )
    parser.add_argument("--burst-behavior-csv", type=Path, required=True)
    parser.add_argument("--posture-overrides-csv", type=Path, default=None)
    parser.add_argument(
        "--behavior-label-authority",
        required=True,
        choices=sorted(SAFE_DERIVATION_BEHAVIOR_AUTHORITIES),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    paths = [args.burst_behavior_csv, args.output_csv, args.audit_json]
    if args.posture_overrides_csv is not None:
        paths.append(args.posture_overrides_csv)
    for path in paths:
        assert_not_active_behavior_ledger_path(path)

    bursts = pd.read_csv(args.burst_behavior_csv, low_memory=False)
    overrides = (
        pd.read_csv(args.posture_overrides_csv, low_memory=False)
        if args.posture_overrides_csv is not None
        else None
    )
    authority, audit = build_burst_posture_authority(
        bursts,
        overrides,
        behavior_label_authority=args.behavior_label_authority,
    )
    audit["inputs"] = {
        "burst_behavior_csv": {
            "path": str(args.burst_behavior_csv.resolve()),
            "sha256": sha256_file(args.burst_behavior_csv),
        },
        "posture_overrides_csv": (
            {
                "path": str(args.posture_overrides_csv.resolve()),
                "sha256": sha256_file(args.posture_overrides_csv),
            }
            if args.posture_overrides_csv is not None
            else None
        ),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    authority.to_csv(args.output_csv, index=False)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_csv": str(args.output_csv),
                "audit_json": str(args.audit_json),
                "rows": audit["rows"],
                "valid_rows": audit["valid_rows"],
                "unresolved_rows": audit["unresolved_rows"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
