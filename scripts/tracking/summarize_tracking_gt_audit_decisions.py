"""Create non-mutating decision summaries after coverage validation."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pig_behavior.tracking.gt_audit_review import load_rows, atomic_write_json

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--decisions", required=True); ap.add_argument("--coverage", required=True); ap.add_argument("--output-json", required=True); ap.add_argument("--output-csv", required=True)
    a = ap.parse_args(); cov = json.loads(Path(a.coverage).read_text(encoding="utf-8"))
    if cov.get("coverage_status") != "PASS": raise SystemExit("COVERAGE_NOT_PASS")
    manifest = {r["review_unit_id"]: r for r in load_rows(a.manifest)}; rows = load_rows(a.decisions)
    totals = Counter(r["decision"] for r in rows); by_method = Counter((manifest[r["review_unit_id"]]["primary_method_id"], r["decision"]) for r in rows)
    result = {"review_units": len(manifest), "reviewed": len(rows), "decision_counts": dict(totals), "by_method_decision": {f"{m}|{d}": n for (m,d),n in by_method.items()}, "source_gt_and_predictions_unchanged": True}
    atomic_write_json(a.output_json, result)
    with open(a.output_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method_id", "decision", "count"]); w.writerows((m,d,n) for (m,d),n in sorted(by_method.items()))
    return 0
if __name__ == "__main__": raise SystemExit(main())
