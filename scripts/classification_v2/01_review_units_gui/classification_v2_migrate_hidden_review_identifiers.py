"""Migrate Hidden review workload and decisions onto identifier v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_migration import (
    migrate_hidden_review_decisions,
    upgrade_hidden_review_manifest_identifiers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an audited, row-preserving identifier-v2 copy of a Hidden "
            "review workload and its existing human decisions."
        )
    )
    parser.add_argument("--legacy-manifest-csv", type=Path, required=True)
    parser.add_argument("--legacy-decisions-csv", type=Path, required=True)
    parser.add_argument("--output-manifest-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [args.legacy_manifest_csv, args.legacy_decisions_csv]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing migration inputs: {missing}")
    outputs = [
        args.output_manifest_csv,
        args.output_decisions_csv,
        args.mapping_csv,
        args.audit_json,
    ]
    _guard_outputs(outputs, overwrite=args.overwrite)

    legacy_manifest = pd.read_csv(args.legacy_manifest_csv, low_memory=False)
    decisions = pd.read_csv(args.legacy_decisions_csv, low_memory=False)
    upgraded = upgrade_hidden_review_manifest_identifiers(legacy_manifest)
    mapping, migrated, audit = migrate_hidden_review_decisions(
        legacy_manifest,
        upgraded,
        decisions,
    )
    audit.update(
        {
            "legacy_manifest_csv": str(args.legacy_manifest_csv),
            "legacy_manifest_sha256": _sha256(args.legacy_manifest_csv),
            "legacy_decisions_csv": str(args.legacy_decisions_csv),
            "legacy_decisions_sha256": _sha256(args.legacy_decisions_csv),
            "output_manifest_csv": str(args.output_manifest_csv),
            "output_decisions_csv": str(args.output_decisions_csv),
            "mapping_csv": str(args.mapping_csv),
            "output_manifest_sha256": None,
            "output_decisions_sha256": None,
            "mapping_sha256": None,
            "outputs_written": False,
        }
    )
    if not audit["valid"]:
        _write_json_atomic(args.audit_json, audit)
        raise ValueError(f"Hidden identifier migration failed: {audit['errors']}")

    _write_csv_atomic(args.output_manifest_csv, upgraded)
    _write_csv_atomic(args.output_decisions_csv, migrated)
    _write_csv_atomic(args.mapping_csv, mapping)
    audit["output_manifest_sha256"] = _sha256(args.output_manifest_csv)
    audit["output_decisions_sha256"] = _sha256(args.output_decisions_csv)
    audit["mapping_sha256"] = _sha256(args.mapping_csv)
    audit["outputs_written"] = True
    _write_json_atomic(args.audit_json, audit)
    print(
        "[PASS] Hidden identifier migration: "
        f"manifest={len(upgraded)} decisions={len(migrated)} "
        f"mapped={audit['mapped_decision_rows']}"
    )
    print(f"[PASS] audit={args.audit_json}")


def _guard_outputs(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Migration outputs already exist; use --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
