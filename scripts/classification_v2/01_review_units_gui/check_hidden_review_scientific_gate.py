"""Fail-closed scientific gate for completed two-sided Hidden review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_science import (
    evaluate_hidden_scientific_gate,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--design-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Emit an explicitly non-authorizing partial report. The default "
            "requires a final PASS."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.manifest_csv, args.decisions_csv, args.design_json):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = pd.read_csv(args.manifest_csv, low_memory=False)
    decisions = pd.read_csv(args.decisions_csv, low_memory=False)
    design = json.loads(args.design_json.read_text(encoding="utf-8"))
    audit = evaluate_hidden_scientific_gate(
        manifest,
        decisions,
        design,
        manifest_sha256=sha256_file(args.manifest_csv),
        design_sha256=sha256_file(args.design_json),
    )
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "PASS" and not args.report_only:
        raise SystemExit(f"FAIL: Hidden scientific gate status={audit['status']}")
    if args.report_only:
        print("REPORT ONLY: this output cannot authorize a training snapshot.")
    else:
        print("PASS: Hidden uncertainty and predeclared quality thresholds pass.")


if __name__ == "__main__":
    main()
