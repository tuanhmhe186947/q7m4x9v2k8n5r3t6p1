"""Independently check Classification V2 native review evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.features.native_evidence_contract import (
    check_native_review_evidence,
)
from pig_behavior.classification_v2.training.lineage_hashing import (
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--native-evidence-csv", required=True, type=Path)
    parser.add_argument("--producer-audit-json", required=True, type=Path)
    parser.add_argument("--contract-manifest", required=True, type=Path)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.input_csv,
        args.native_evidence_csv,
        args.producer_audit_json,
        args.contract_manifest,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    require_output_paths_available(
        [args.output_json],
        overwrite=args.overwrite,
    )
    source = pd.read_csv(args.input_csv, low_memory=False)
    output = pd.read_csv(args.native_evidence_csv, low_memory=False)
    producer_audit = _read_json(args.producer_audit_json)
    audit = check_native_review_evidence(
        source,
        output,
        producer_audit=producer_audit,
        code_sha=args.code_sha,
        input_sha256=file_sha256(args.input_csv),
        contract_manifest_sha256=file_sha256(args.contract_manifest),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["errors"]:
        raise SystemExit(2)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    main()
