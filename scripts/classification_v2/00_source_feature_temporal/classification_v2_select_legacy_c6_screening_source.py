"""Select complete legacy groups for C6 development screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.datasets.legacy_c6_screening_source import (
    select_legacy_c6_screening_source,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument(
        "--selection-salt",
        default="legacy_c6_complete_group_gate_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.output_csv, args.audit_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"legacy C6 source output exists: {path}")

    source = pd.read_csv(args.input_csv, low_memory=False)
    selected = select_legacy_c6_screening_source(
        source,
        max_groups=args.max_groups,
        selection_salt=args.selection_salt,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected.frames.to_csv(args.output_csv, index=False)
    audit = {
        **selected.audit,
        "input_path": str(args.input_csv),
        "input_sha256": file_sha256(args.input_csv),
        "output_path": str(args.output_csv),
        "output_sha256": file_sha256(args.output_csv),
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
