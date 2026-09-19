"""Predeclare the residual control review without reading review decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.post_review_learning import (
    ControlSelectionConfig,
    assert_not_active_behavior_ledger_path,
    build_post_review_control_scope,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-csv", type=Path, required=True)
    parser.add_argument("--primary-scope-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--stratum-column",
        action="append",
        dest="stratum_columns",
        help=(
            "Repeat for each declared stratum column. Defaults to behavior, "
            "source, and review-unit type. Recording date is optional."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.population_csv, args.primary_scope_csv, args.output_dir):
        assert_not_active_behavior_ledger_path(path)
    for path in (args.population_csv, args.primary_scope_csv):
        if not path.is_file():
            raise FileNotFoundError(path)

    columns = tuple(args.stratum_columns or ControlSelectionConfig().stratum_columns)
    config = ControlSelectionConfig(
        target_count=args.target_count,
        seed=args.seed,
        stratum_columns=columns,
    )
    population = pd.read_csv(args.population_csv, low_memory=False)
    primary = pd.read_csv(args.primary_scope_csv, low_memory=False)
    selected, audit = build_post_review_control_scope(
        population,
        primary,
        config=config,
    )
    audit["inputs"] = {
        "population": {
            "path": str(args.population_csv.resolve()),
            "sha256": sha256_file(args.population_csv),
        },
        "primary_scope": {
            "path": str(args.primary_scope_csv.resolve()),
            "sha256": sha256_file(args.primary_scope_csv),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scope_path = args.output_dir / "post_review_control_scope.csv"
    audit_path = args.output_dir / "post_review_control_scope_audit.json"
    selected.to_csv(scope_path, index=False)
    audit["output"] = {
        "path": str(scope_path.resolve()),
        "sha256": sha256_file(scope_path),
    }
    write_json(audit_path, audit)
    print(f"PASS: selected {len(selected)} residual controls")
    print(scope_path)
    print(audit_path)


if __name__ == "__main__":
    main()
