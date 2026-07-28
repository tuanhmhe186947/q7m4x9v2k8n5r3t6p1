"""Independently validate published Behavior threshold candidate comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.behavior_threshold_audit import (
    independent_threshold_candidate_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--auto-carry-csv", type=Path, required=True)
    parser.add_argument("--threshold-registry-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universe = pd.read_csv(args.universe_csv, low_memory=False)
    candidates = pd.read_csv(args.candidate_csv, low_memory=False)
    auto_carry = pd.read_csv(args.auto_carry_csv, low_memory=False)
    registry = json.loads(
        args.threshold_registry_json.read_text(encoding="utf-8")
    )
    audit = independent_threshold_candidate_audit(
        universe,
        candidates,
        auto_carry,
        registry,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
