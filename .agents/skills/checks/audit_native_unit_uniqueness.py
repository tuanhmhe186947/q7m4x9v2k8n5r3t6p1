"""Audit native temporal-unit uniqueness and source-specific lengths."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _common import finish, read_csv

EXPECTED_LENGTHS = {"legacy_recovered": 16, "cvat_tracking_xml": 6}


def audit(path: Path, key: str) -> dict[str, object]:
    """Require one row per native key and the correct 16/6-frame contract."""
    header, rows = read_csv(path)
    required = {key, "source_type", "label_frame_count"}
    errors = [f"missing_column={name}" for name in sorted(required - set(header))]
    keys = [row.get(key, "").strip() for row in rows]
    duplicates = sorted(value for value, count in Counter(keys).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate_native_unit_count={len(duplicates)}")
    bad_lengths: list[str] = []
    for row in rows:
        source = row.get("source_type", "")
        expected = EXPECTED_LENGTHS.get(source)
        try:
            observed = int(row.get("label_frame_count", ""))
        except ValueError:
            observed = -1
        if expected is None or observed != expected:
            bad_lengths.append(row.get(key, ""))
    if bad_lengths:
        errors.append(f"bad_native_length_count={len(bad_lengths)}")
    return {
        "check": "native_unit_uniqueness",
        "rows": len(rows),
        "duplicate_keys": duplicates,
        "bad_length_keys": bad_lengths,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-csv", type=Path, required=True)
    parser.add_argument("--key-column", default="temporal_unit_key")
    args = parser.parse_args()
    return finish(audit(args.native_csv, args.key_column))


if __name__ == "__main__":
    raise SystemExit(main())
