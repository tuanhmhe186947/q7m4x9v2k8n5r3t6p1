from __future__ import annotations

import json
from pathlib import Path

AUDIT_JSON = Path("outputs/classification_v2/train_ready_windows/train_ready_audit.json")


def main() -> None:
    if not AUDIT_JSON.exists():
        raise FileNotFoundError(AUDIT_JSON)
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    feature_audit = audit.get("feature_selection", {})
    errors = list(feature_audit.get("errors", []))
    forbidden = list(feature_audit.get("forbidden_selected", []))
    rows = audit.get("rows", {})
    feature_count = int(feature_audit.get("feature_count", 0))

    print("audit_json =", AUDIT_JSON)
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
