"""Read-only integrity check for two-sided Hidden review templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_builder import (
    HiddenReviewConfig,
    audit_hidden_review_manifest,
    balanced_hidden_smoke_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--trusted-yes-per-stratum", type=int, default=1)
    parser.add_argument("--random-no-per-stratum", type=int, default=10)
    parser.add_argument("--clean-control-per-stratum", type=int, default=1)
    parser.add_argument("--max-high-risk-per-stratum", type=int, default=16)
    parser.add_argument("--high-risk-threshold", type=float, default=0.35)
    parser.add_argument("--clean-control-max-risk", type=float, default=0.10)
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="Use only to reproduce the builder smoke input scope.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = pd.read_csv(args.input_csv, low_memory=False)
    if args.max_rows_per_source is not None:
        if args.max_rows_per_source <= 0:
            raise ValueError("--max-rows-per-source must be > 0")
        frames = balanced_hidden_smoke_scope(
            frames,
            args.max_rows_per_source,
        )
    manifest = pd.read_csv(args.manifest_csv, low_memory=False)
    config = HiddenReviewConfig(
        random_seed=args.random_seed,
        trusted_yes_per_stratum=args.trusted_yes_per_stratum,
        random_no_per_stratum=args.random_no_per_stratum,
        clean_control_per_stratum=args.clean_control_per_stratum,
        max_high_risk_per_stratum=args.max_high_risk_per_stratum,
        high_risk_threshold=args.high_risk_threshold,
        clean_control_max_risk=args.clean_control_max_risk,
    )
    audit = audit_hidden_review_manifest(
        frames,
        manifest,
        config,
    )
    payload = json.dumps(audit, ensure_ascii=False, indent=2)
    print(payload)
    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(payload, encoding="utf-8")
    if audit["errors"]:
        raise SystemExit(f"FAIL: {audit['errors']}")
    print("PASS: Hidden review template covers both positive and negative audit.")


if __name__ == "__main__":
    main()
