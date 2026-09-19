from __future__ import annotations

import argparse
import json
from pathlib import Path

AUDIT_JSON = Path("outputs/classification_v2/train_ready_windows/train_ready_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a classification_v2 train-ready export audit."
    )
    parser.add_argument("--audit-json", type=Path, default=AUDIT_JSON)
    args = parser.parse_args()
    if not args.audit_json.exists():
        raise FileNotFoundError(args.audit_json)
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    feature_audit = audit.get("feature_selection", {})
    errors = list(feature_audit.get("errors", []))
    forbidden = list(feature_audit.get("forbidden_selected", []))
    rows = audit.get("rows", {})
    feature_count = int(feature_audit.get("feature_count", 0))

    print("audit_json =", args.audit_json)
    print("rows =", rows)
    print("feature_count =", feature_count)
    print("forbidden_selected =", forbidden)
    print("errors =", errors)

    if feature_count <= 0:
        errors.append("feature_count<=0")
    if forbidden:
        errors.append(f"forbidden_selected={forbidden}")
    if rows.get("input") != rows.get("X") or rows.get("input") != rows.get("y"):
        errors.append("row_count_mismatch")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
