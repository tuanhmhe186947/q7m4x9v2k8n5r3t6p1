"""Carry audited Hidden decisions into a redesigned review workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_migration import (
    carry_forward_hidden_review_decisions,
)


def parse_args() -> argparse.Namespace:
    """Parse explicit source, destination, and overwrite contracts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-manifest-csv", type=Path, required=True)
    parser.add_argument("--current-manifest-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Carry decisions only after all identity and payload audits pass."""

    args = parse_args()
    inputs = [
        args.previous_manifest_csv,
        args.current_manifest_csv,
        args.decisions_csv,
    ]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    outputs = [args.output_decisions_csv, args.audit_json]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs exist; use --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )

    previous = pd.read_csv(args.previous_manifest_csv, low_memory=False)
    current = pd.read_csv(args.current_manifest_csv, low_memory=False)
    decisions = pd.read_csv(args.decisions_csv, low_memory=False)
    carried, audit = carry_forward_hidden_review_decisions(
        previous,
        current,
        decisions,
    )
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if audit["errors"]:
        raise SystemExit(f"FAIL: {audit['errors']}")
    args.output_decisions_csv.parent.mkdir(parents=True, exist_ok=True)
    carried.to_csv(args.output_decisions_csv, index=False)
    print(
        "[PASS] Hidden decisions carried without payload loss: "
        f"rows={len(carried)} audit={args.audit_json}"
    )


if __name__ == "__main__":
    main()
