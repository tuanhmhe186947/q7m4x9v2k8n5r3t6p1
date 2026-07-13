"""Fail-closed coverage check for Hidden review decision CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_builder import (
    audit_hidden_decision_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Smoke/debug only. Full data lineage must not use this flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv, low_memory=False)
    decisions = pd.read_csv(args.decisions_csv, low_memory=False)
    audit = audit_hidden_decision_coverage(
        manifest,
        decisions,
        require_resolved=not args.allow_unresolved,
    )
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["errors"]:
        raise SystemExit(f"FAIL: {audit['errors']}")
    print("PASS: every selected Hidden review item has one resolved decision.")


if __name__ == "__main__":
    main()
