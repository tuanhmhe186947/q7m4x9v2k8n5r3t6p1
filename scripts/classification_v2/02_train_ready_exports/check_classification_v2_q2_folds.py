"""Re-audit persisted Q2 grouped folds independently of the builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.grouped_folds import audit_grouped_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Q2 grouped fold artifacts.")
    parser.add_argument("--fold-dir", type=Path, default=Path("outputs/classification_v2/q2_grouped_folds"))
    args = parser.parse_args()
    assignments = pd.read_csv(args.fold_dir / "q2_outer_fold_assignments.csv", low_memory=False)
    roles = pd.read_csv(args.fold_dir / "q2_outer_inner_roles.csv", low_memory=False)
    audit = audit_grouped_folds(assignments, roles, requested_folds=5, seed=20260710)
    errors = list(audit["errors"])
    expected_role_rows = len(assignments) * int(audit["selected_fold_count"])
    if len(roles) != expected_role_rows:
        errors.append(f"role_row_count={len(roles)}, expected={expected_role_rows}")
    if set(roles["role"].astype(str)) != {"train", "validation", "test"}:
        errors.append(f"role_values={sorted(roles['role'].astype(str).unique())}")
    result = {
        **audit,
        "role_rows": int(len(roles)),
        "expected_role_rows": expected_role_rows,
        "errors": errors,
        "valid": not errors,
    }
    (args.fold_dir / "check_q2_grouped_fold_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
