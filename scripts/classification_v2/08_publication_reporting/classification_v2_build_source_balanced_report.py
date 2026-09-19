from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.source_balanced_reporting import (
    build_source_balanced_native_report,
)


def main() -> None:
    """Build source-balanced native-unit metrics and a row-preserving selection manifest."""

    parser = argparse.ArgumentParser(description="Build classification_v2 source-balanced report.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--window-metadata-csv", type=Path, required=True)
    parser.add_argument("--run-audit-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-fold-count", type=int, default=None)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_csv, low_memory=False)
    metadata = pd.read_csv(args.window_metadata_csv, usecols=["window_id", "source_type"], low_memory=False)
    run_verified = _paper_run_verified(args.run_audit_json)
    native_units, selection, report = build_source_balanced_native_report(
        predictions,
        metadata,
        expected_fold_count=args.expected_fold_count,
        paper_facing_run_verified=run_verified,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    units_path = args.output_dir / "native_unit_predictions_with_source.csv"
    selection_path = args.output_dir / "source_balanced_native_selection.csv"
    report_path = args.output_dir / "source_balanced_native_report.json"
    native_units.to_csv(units_path, index=False)
    selection.to_csv(selection_path, index=False)
    report.update(
        {
            "predictions_csv": str(args.predictions_csv),
            "window_metadata_csv": str(args.window_metadata_csv),
            "run_audit_json": str(args.run_audit_json) if args.run_audit_json else None,
            "native_unit_predictions_csv": str(units_path),
            "selection_manifest_csv": str(selection_path),
        }
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def _paper_run_verified(path: Path | None) -> bool:
    """Only a completed full-run audit can unlock paper-facing source reporting."""

    if path is None or not path.exists():
        return False
    audit = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        audit.get("valid") is True and audit.get("run_mode") == "full" and audit.get("paper_facing_result") is True
    )


if __name__ == "__main__":
    main()
