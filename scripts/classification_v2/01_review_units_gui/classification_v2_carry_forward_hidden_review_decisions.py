"""Carry audited Hidden decisions into a redesigned review workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.hidden_review_migration import (
    carry_forward_hidden_review_decisions,
)


def parse_args() -> argparse.Namespace:
    """Parse explicit source, destination, and overwrite contracts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-manifest-csv", type=Path, required=True)
    parser.add_argument("--current-manifest-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--output-decisions-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate identity/hash/decision coverage, write only the audit "
            "JSON, and never write the decision CSV."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Explicitly write carried decisions after all audits pass.",
    )
    return parser.parse_args()


def main() -> None:
    """Carry decisions only after all identity and payload audits pass."""

    args = parse_args()
    if args.dry_run and args.apply:
        raise SystemExit("choose exactly one of --dry-run or --apply")
    if not args.dry_run and not args.apply:
        raise SystemExit("explicit --dry-run or --apply is required")
    inputs = [
        args.previous_manifest_csv,
        args.current_manifest_csv,
        args.decisions_csv,
    ]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    outputs = [args.output_decisions_csv, args.audit_json]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs exist; use --overwrite explicitly: "
            + ", ".join(str(path) for path in existing)
        )

    previous = pd.read_csv(args.previous_manifest_csv, low_memory=False)
    current = pd.read_csv(args.current_manifest_csv, low_memory=False)
    decisions = pd.read_csv(args.decisions_csv, low_memory=False)
    carried, audit = carry_forward_hidden_review_decisions(
        previous,
        current,
        decisions,
    )
    audit.update(
        {
            "previous_manifest_csv": str(args.previous_manifest_csv),
            "previous_manifest_sha256": _sha256(
                args.previous_manifest_csv
            ),
            "current_manifest_csv": str(args.current_manifest_csv),
            "current_manifest_sha256": _sha256(args.current_manifest_csv),
            "source_decisions_csv": str(args.decisions_csv),
            "source_decisions_sha256": _sha256(args.decisions_csv),
            "output_decisions_csv": str(args.output_decisions_csv),
            "output_decisions_sha256": None,
            "output_written": False,
        }
    )
    if audit["errors"]:
        _write_json_atomic(args.audit_json, audit)
        raise SystemExit(f"FAIL: {audit['errors']}")
    if args.dry_run:
        audit["dry_run"] = True
        audit["output_written"] = False
        _write_json_atomic(args.audit_json, audit)
        print(
            "[PASS] Hidden carry-forward dry-run; no decision CSV written: "
            f"rows={len(carried)}"
        )
        return
    audit["apply_confirmed"] = True
    csv_payload = carried.to_csv(index=False).encode("utf-8")
    audit["output_decisions_sha256"] = hashlib.sha256(
        csv_payload
    ).hexdigest()
    audit["output_written"] = True
    _write_apply_transaction(
        args.output_decisions_csv,
        args.audit_json,
        csv_payload,
        audit,
    )
    print(
        "[PASS] Hidden decisions carried without payload loss: "
        f"rows={len(carried)} audit={args.audit_json}"
    )


def _write_apply_transaction(
    output_path: Path,
    audit_path: Path,
    csv_payload: bytes,
    audit: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    csv_temporary = output_path.with_name(
        f".{output_path.name}.{token}.tmp"
    )
    audit_temporary = audit_path.with_name(
        f".{audit_path.name}.{token}.tmp"
    )
    try:
        csv_temporary.write_bytes(csv_payload)
        audit_temporary.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        audit_temporary.replace(audit_path)
        try:
            csv_temporary.replace(output_path)
        except Exception as error:
            failed = dict(audit)
            failed["output_written"] = False
            failed["output_decisions_sha256"] = None
            failed["errors"] = [
                *list(audit.get("errors", [])),
                f"decision_csv_promotion_failed={type(error).__name__}",
            ]
            _write_json_atomic(audit_path, failed)
            raise
    finally:
        csv_temporary.unlink(missing_ok=True)
        audit_temporary.unlink(missing_ok=True)


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
