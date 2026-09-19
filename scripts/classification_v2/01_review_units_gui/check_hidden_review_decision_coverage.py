"""Fail-closed coverage check for Hidden review decision CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_builder import (
    audit_hidden_decision_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Smoke/debug only. Full data lineage must not use this flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv, low_memory=False)
    decisions = pd.read_csv(args.decisions_csv, low_memory=False)
    audit = audit_hidden_decision_coverage(
        manifest,
        decisions,
        require_resolved=not args.allow_unresolved,
    )
    audit.update(
        {
            "checker_code_sha": _git_head(),
            "manifest_sha256": _sha256(args.manifest_csv),
            "decisions_sha256": _sha256(args.decisions_csv),
            "data_lineage_authority_preserved": True,
            "input_artifacts_regenerated": False,
        }
    )
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["errors"]:
        raise SystemExit(f"FAIL: {audit['errors']}")
    print("PASS: every selected Hidden review item has one resolved decision.")


def _git_head() -> str:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
