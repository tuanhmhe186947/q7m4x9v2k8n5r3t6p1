"""Check that optional modality references and availability masks agree."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import as_bool, finish, read_csv


def audit(path: Path, pairs: list[str]) -> dict[str, object]:
    """Validate value-column to mask-column pairs in a model manifest."""
    header, rows = read_csv(path)
    parsed: list[tuple[str, str]] = []
    errors: list[str] = []
    for pair in pairs:
        if ":" not in pair:
            errors.append(f"invalid_pair={pair}")
            continue
        value_col, mask_col = pair.split(":", 1)
        parsed.append((value_col, mask_col))
        for column in (value_col, mask_col):
            if column not in header:
                errors.append(f"missing_column={column}")
    inconsistencies: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=2):
        for value_col, mask_col in parsed:
            available = as_bool(row.get(mask_col, ""))
            has_value = bool(row.get(value_col, "").strip())
            if available is None or available != has_value:
                inconsistencies.append(
                    {
                        "csv_line": index,
                        "value_column": value_col,
                        "mask_column": mask_col,
                        "has_value": has_value,
                        "mask": row.get(mask_col, ""),
                    }
                )
    if inconsistencies:
        errors.append(f"modality_mask_mismatch_count={len(inconsistencies)}")
    return {
        "check": "missing_modality_masks",
        "rows": len(rows),
        "pairs": [f"{value}:{mask}" for value, mask in parsed],
        "inconsistencies": inconsistencies,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True)
    args = parser.parse_args()
    return finish(audit(args.manifest_csv, args.pair))


if __name__ == "__main__":
    raise SystemExit(main())
