"""Fail-closed coverage and decision-ledger checker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pig_behavior.tracking.gt_audit_review import atomic_write_json, coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--input-authority", required=True)
    parser.add_argument("--expected-gui-code-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = coverage(
        args.manifest,
        args.decisions,
        args.events,
        source_audit_path=args.source_audit,
        expected_gui_code_sha=args.expected_gui_code_sha,
        input_authority_path=args.input_authority,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["coverage_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
