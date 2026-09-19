"""Build the primary five-fold Q2 grouped evaluation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.grouped_folds import build_grouped_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe Q2 outer and inner folds.")
    parser.add_argument(
        "--native-unit-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/native_temporal_units_publication_splits/publication_split_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/q2_grouped_folds"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    result = build_grouped_folds(
        pd.read_csv(args.native_unit_csv, low_memory=False),
        requested_folds=args.folds,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.assignments.to_csv(args.output_dir / "q2_outer_fold_assignments.csv", index=False)
    result.roles.to_csv(args.output_dir / "q2_outer_inner_roles.csv", index=False)
    audit = {
        "input_csv": str(args.native_unit_csv),
        "assignment_csv": str(args.output_dir / "q2_outer_fold_assignments.csv"),
        "roles_csv": str(args.output_dir / "q2_outer_inner_roles.csv"),
        **result.audit,
    }
    (args.output_dir / "q2_grouped_fold_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
