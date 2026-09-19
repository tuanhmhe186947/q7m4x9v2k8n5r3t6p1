"""Independently check evidence semantics against both CSV schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.evidence_semantics import (
    build_evidence_semantics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-local-csv", required=True, type=Path)
    parser.add_argument("--native-evidence-csv", required=True, type=Path)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument("--semantics-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    expected = build_evidence_semantics(
        pd.read_csv(args.frame_local_csv, low_memory=False),
        pd.read_csv(args.native_evidence_csv, low_memory=False),
        lineage_id=args.lineage_id,
        code_authority_sha=args.code_authority_sha,
    )
    observed = json.loads(args.semantics_json.read_text(encoding="utf-8"))
    errors = list(expected["errors"])
    if observed != expected:
        errors.append("evidence_semantics_content_drift")
    audit = {
        "lineage_id": args.lineage_id,
        "code_authority_sha": args.code_authority_sha.lower(),
        "valid": not errors,
        "errors": errors,
        "semantics_match_independent_rebuild": observed == expected,
        "declared_field_count": len(observed.get("fields", {})),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
