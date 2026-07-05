"""Apply final context policy to merged classification_v2 frame objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.features.context_policy import (
    apply_context_policy,
    audit_context_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize include/training/eval flags for classification_v2."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--expected-pig-count", type=int, default=8)
    parser.add_argument(
        "--require-full-8-for-eval",
        action="store_true",
        help="Only use full 8-pig context rows for main eval.",
    )
    parser.add_argument(
        "--no-recompute-context",
        action="store_true",
        help="Use existing context columns instead of recomputing from frame_uid/pig_id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"reading: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)

    print("applying context policy...")
    out = apply_context_policy(
        df,
        expected_pig_count=args.expected_pig_count,
        recompute_context=not args.no_recompute_context,
        require_full_8_for_eval=args.require_full_8_for_eval,
    )

    audit = audit_context_policy(out)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if audit["errors"]:
        raise ValueError(f"Context policy audit errors: {audit['errors']}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"saved csv: {args.output_csv}")

    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved audit: {args.audit_json}")


if __name__ == "__main__":
    main()