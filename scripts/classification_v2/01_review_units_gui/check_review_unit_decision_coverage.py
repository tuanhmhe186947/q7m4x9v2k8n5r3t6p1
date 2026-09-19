"""Fail-closed coverage audit for classification_v2 human review decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.behavior_review_contract import (
    REQUIRED_DECISION_COLUMNS,
)
from pig_behavior.classification_v2.review.behavior_review_contract import (
    audit_decision_coverage as audit_decision_coverage_contract,
)

REQUIRED_COLUMNS = list(REQUIRED_DECISION_COLUMNS)


def audit_decision_coverage(
    review_manifest: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Delegate to the canonical behavior-review coverage contract."""

    return audit_decision_coverage_contract(
        review_manifest,
        decisions,
        require_complete=require_complete,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    review_manifest = pd.read_csv(args.review_manifest_csv, low_memory=False)
    parts = []
    missing_files = []
    for path in args.decisions_csv:
        if not path.exists():
            missing_files.append(str(path))
            continue
        parts.append(pd.read_csv(path, low_memory=False))
    decisions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    audit = audit_decision_coverage(
        review_manifest,
        decisions,
        require_complete=args.require_complete,
    )
    if missing_files:
        audit.setdefault("errors", []).append(f"missing_decision_files={missing_files}")

    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
