"""Fail-closed checker for review_neighborhood_evidence.v1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.neighborhood_evidence import (
    require_review_neighborhood_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = pd.read_csv(args.evidence_csv, low_memory=False)
    metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    audit = require_review_neighborhood_evidence(evidence, metadata)
    rendered = json.dumps(audit, indent=2, ensure_ascii=False)
    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
