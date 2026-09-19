"""Freeze completed primary and control review copies as one authority."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.post_review_learning import (
    assert_not_active_behavior_ledger_path,
    bindings_from_paths,
    build_review_close_authority,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-scope-csv", type=Path, required=True)
    parser.add_argument("--primary-decisions-csv", type=Path, required=True)
    parser.add_argument("--primary-quality-csv", type=Path, required=True)
    parser.add_argument("--control-scope-csv", type=Path, required=True)
    parser.add_argument("--control-decisions-csv", type=Path, required=True)
    parser.add_argument("--control-quality-csv", type=Path, required=True)
    parser.add_argument("--expected-primary-count", type=int, default=2729)
    parser.add_argument("--minimum-control-count", type=int, default=120)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "primary_scope": args.primary_scope_csv,
        "primary_decisions": args.primary_decisions_csv,
        "primary_quality": args.primary_quality_csv,
        "control_scope": args.control_scope_csv,
        "control_decisions": args.control_decisions_csv,
        "control_quality": args.control_quality_csv,
    }
    assert_not_active_behavior_ledger_path(args.output_json)
    bindings = bindings_from_paths(paths)
    frames = {
        name: pd.read_csv(path, low_memory=False)
        for name, path in paths.items()
    }
    authority = build_review_close_authority(
        primary_scope=frames["primary_scope"],
        primary_decisions=frames["primary_decisions"],
        primary_quality=frames["primary_quality"],
        control_scope=frames["control_scope"],
        control_decisions=frames["control_decisions"],
        control_quality=frames["control_quality"],
        artifact_bindings=bindings,
        expected_primary_count=args.expected_primary_count,
        minimum_control_count=args.minimum_control_count,
    )
    write_json(args.output_json, authority)
    print("PASS: frozen post-review authority written")
    print(args.output_json)


if __name__ == "__main__":
    main()
