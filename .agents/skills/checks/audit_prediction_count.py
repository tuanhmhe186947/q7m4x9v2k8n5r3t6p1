"""Compare prediction keys with the authoritative evaluation manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _common import finish, read_csv


def _keys(path: Path, key: str) -> tuple[list[str], list[str]]:
    header, rows = read_csv(path)
    if key not in header:
        return [], [f"{path}:missing_column={key}"]
    values = [row[key].strip() for row in rows]
    errors = [f"{path}:blank_key"] if any(not value for value in values) else []
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        errors.append(f"{path}:duplicate_key_count={len(duplicates)}")
    return values, errors


def audit(manifest: Path, predictions: Path, key: str) -> dict[str, object]:
    """Require exactly one prediction for every expected manifest key."""
    expected, errors = _keys(manifest, key)
    observed, prediction_errors = _keys(predictions, key)
    errors.extend(prediction_errors)
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing:
        errors.append(f"missing_prediction_count={len(missing)}")
    if unexpected:
        errors.append(f"unexpected_prediction_count={len(unexpected)}")
    return {
        "check": "prediction_count",
        "key_column": key,
        "manifest_rows": len(expected),
        "prediction_rows": len(observed),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--key-column", default="temporal_unit_key")
    args = parser.parse_args()
    return finish(audit(args.manifest_csv, args.predictions_csv, args.key_column))


if __name__ == "__main__":
    raise SystemExit(main())
