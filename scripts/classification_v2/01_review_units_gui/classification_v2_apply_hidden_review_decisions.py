"""Apply Hidden review decisions to a new derived frame-feature artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_builder import (
    apply_hidden_review_decisions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--confusion-audit-json", type=Path, required=True)
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Smoke/debug only. Full reviewed data must be fail-closed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = [args.output_csv, args.audit_json, args.confusion_audit_json]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; use --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )

    frames = pd.read_csv(args.input_csv, low_memory=False)
    manifest = pd.read_csv(args.manifest_csv, low_memory=False)
    decisions = pd.read_csv(args.decisions_csv, low_memory=False)
    reviewed, audit, confusion = apply_hidden_review_decisions(
        frames,
        manifest,
        decisions,
        require_resolved=not args.allow_unresolved,
    )

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(args.output_csv, index=False)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.confusion_audit_json.write_text(
        json.dumps(confusion, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[PASS] Hidden decisions applied without row loss: "
        f"rows={len(reviewed)} corrected={audit['corrected_hidden_rows']}"
    )


if __name__ == "__main__":
    main()
