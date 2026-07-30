"""Fail-closed coverage and decision-ledger checker."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pig_behavior.tracking.gt_audit_review import coverage, atomic_write_json

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--decisions", required=True); ap.add_argument("--events", required=True); ap.add_argument("--output", required=True)
    a = ap.parse_args(); result = coverage(a.manifest, a.decisions, a.events); atomic_write_json(a.output, result); print(json.dumps(result, indent=2)); return 0 if result["coverage_status"] == "PASS" else 2
if __name__ == "__main__": raise SystemExit(main())
