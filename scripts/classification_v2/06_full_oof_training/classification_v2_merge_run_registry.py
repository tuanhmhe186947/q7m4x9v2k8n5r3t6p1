"""Merge isolated fold registry entries into one append-only CSV ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.run_lineage import (
    merge_registry_entries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge validated independent fold registry entries."
    )
    parser.add_argument(
        "--entry-json",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [str(path) for path in args.entry_json if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing registry entries={missing}")
    result = merge_registry_entries(
        args.entry_json,
        args.registry_csv,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
