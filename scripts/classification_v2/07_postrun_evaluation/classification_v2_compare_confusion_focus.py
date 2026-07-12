from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.confusion_comparison import compare_confusion_focus


def main() -> None:
    """Compare aligned native-unit predictions with fold-cluster uncertainty."""

    parser = argparse.ArgumentParser(description="Compare classification_v2 confusion-focus outcomes.")
    parser.add_argument("--proposed-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--proposed-run-audit", type=Path, default=None)
    parser.add_argument("--baseline-run-audit", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proposed-pred-col", default="behavior_pred_calibrated")
    parser.add_argument("--baseline-pred-col", default="native_predicted_behavior")
    parser.add_argument("--proposed-confidence-col", default="calibrated_confidence")
    parser.add_argument("--expected-fold-count", type=int, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--sesoi-macro-f1", type=float, default=0.02)
    args = parser.parse_args()

    proposed = pd.read_csv(args.proposed_csv, low_memory=False)
    baseline = pd.read_csv(args.baseline_csv, low_memory=False)
    paper_inputs_verified = _paper_inputs_verified(args.proposed_run_audit, args.baseline_run_audit)
    report, hard_errors = compare_confusion_focus(
        proposed,
        baseline,
        proposed_pred_col=args.proposed_pred_col,
        baseline_pred_col=args.baseline_pred_col,
        proposed_confidence_col=args.proposed_confidence_col,
        expected_fold_count=args.expected_fold_count,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        high_confidence_threshold=args.high_confidence_threshold,
        sesoi_macro_f1=args.sesoi_macro_f1,
        paper_facing_inputs_verified=paper_inputs_verified,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "confusion_focus_comparison.json"
    errors_path = args.output_dir / "high_confidence_hard_errors.csv"
    hard_errors.to_csv(errors_path, index=False)
    report["proposed_csv"] = str(args.proposed_csv)
    report["baseline_csv"] = str(args.baseline_csv)
    report["proposed_run_audit"] = str(args.proposed_run_audit) if args.proposed_run_audit else None
    report["baseline_run_audit"] = str(args.baseline_run_audit) if args.baseline_run_audit else None
    report["high_confidence_hard_errors_csv"] = str(errors_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def _paper_inputs_verified(proposed_path: Path | None, baseline_path: Path | None) -> bool:
    """Require both parent model runs to have passed their explicit full-paper gates."""

    if proposed_path is None or baseline_path is None:
        return False
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in (proposed_path, baseline_path)]
    return all(
        audit.get("valid") is True and audit.get("run_mode") == "full" and audit.get("paper_facing_result") is True
        for audit in audits
    )


if __name__ == "__main__":
    main()
