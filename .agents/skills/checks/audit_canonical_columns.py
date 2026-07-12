"""Detect removed or reordered canonical columns between CSV schemas."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import finish, load_json, read_csv


def audit(before: Path, after: Path, contract: Path) -> dict[str, object]:
    """Compare headers while allowing explicitly noncanonical additions."""
    before_header, _ = read_csv(before)
    after_header, _ = read_csv(after)
    payload = load_json(contract)
    canonical = [str(value) for value in payload.get("columns", [])]
    missing_before = [name for name in canonical if name not in before_header]
    missing_after = [name for name in canonical if name not in after_header]
    before_order = [name for name in before_header if name in canonical]
    after_order = [name for name in after_header if name in canonical]
    order_changed = before_order != after_order
    errors: list[str] = []
    if not canonical:
        errors.append("empty_canonical_column_contract")
    if missing_before:
        errors.append(f"canonical_missing_before={missing_before}")
    if missing_after:
        errors.append(f"canonical_missing_after={missing_after}")
    if order_changed:
        errors.append("canonical_column_order_changed")
    return {
        "check": "canonical_columns",
        "missing_before": missing_before,
        "missing_after": missing_after,
        "order_changed": order_changed,
        "added_noncanonical": sorted(set(after_header) - set(before_header)),
        "removed_noncanonical": sorted(set(before_header) - set(after_header)),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-csv", type=Path, required=True)
    parser.add_argument("--after-csv", type=Path, required=True)
    parser.add_argument("--canonical-json", type=Path, required=True)
    args = parser.parse_args()
    return finish(audit(args.before_csv, args.after_csv, args.canonical_json))


if __name__ == "__main__":
    raise SystemExit(main())
