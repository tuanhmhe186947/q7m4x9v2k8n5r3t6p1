"""Create non-mutating decision summaries after coverage validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pig_behavior.tracking.gt_audit_review import atomic_write_json, load_rows

TOTAL_NAMES = {
    "PREDICTION_ERROR_CONFIRMED": "PREDICTION_ERRORS_CONFIRMED",
    "GT_IDENTITY_QUESTION": "GT_IDENTITY_QUESTIONS",
    "GT_BBOX_QUESTION": "GT_BBOX_QUESTIONS",
    "HIDDEN_LABEL_QUESTION": "HIDDEN_LABEL_QUESTIONS",
    "EVALUATOR_MATCHING_QUESTION": "EVALUATOR_MATCHING_QUESTIONS",
    "FRAGMENTATION_ONLY": "FRAGMENTATION_ONLY",
    "NO_MATERIAL_ISSUE_CONFIRMED": "NO_MATERIAL_ISSUE_CONFIRMED",
    "AMBIGUOUS_UNRESOLVED": "AMBIGUOUS_UNRESOLVED",
    "OTHER_REVIEW_QUESTION": "OTHER_REVIEW_QUESTIONS",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    coverage_result = json.loads(Path(args.coverage).read_text(encoding="utf-8"))
    if coverage_result.get("coverage_status") != "PASS":
        raise SystemExit("COVERAGE_NOT_PASS")
    manifest = {row["review_unit_id"]: row for row in load_rows(args.manifest)}
    decisions = load_rows(args.decisions)
    dimensions = [
        "primary_method_id",
        "selection_reasons",
        "error_category",
        "video_id",
        "Hidden_status",
        "episode_id",
    ]
    summary_rows = []
    for dimension in dimensions:
        counts = Counter(
            (manifest[row["review_unit_id"]][dimension], row["decision"], row["confidence"])
            for row in decisions
        )
        summary_rows.extend(
            {
                "dimension": dimension,
                "value": value,
                "decision": decision,
                "confidence": confidence,
                "count": count,
            }
            for (value, decision, confidence), count in sorted(counts.items())
        )
    totals = Counter(row["decision"] for row in decisions)
    result = {
        "review_units": len(manifest),
        "reviewed": len(decisions),
        "required_totals": {
            output_name: totals.get(decision, 0) for decision, output_name in TOTAL_NAMES.items()
        },
        "source_gt_and_predictions_unchanged": True,
        "scientific_reconciliation_required": True,
    }
    atomic_write_json(args.output_json, result)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
