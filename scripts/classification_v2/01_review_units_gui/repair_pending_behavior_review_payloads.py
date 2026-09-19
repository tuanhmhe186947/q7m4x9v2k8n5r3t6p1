"""Repair contradictory payloads on pending behavior-review decisions.

A pending row is not a completed human decision. Correction, strength, action,
and weight values on that row must therefore not be interpreted downstream.
This operator clears only those fields, preserves notes, writes a byte-for-byte
backup, and records the previous payload in an audit JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.review.behavior_review_contract import (
    REQUIRED_DECISION_COLUMNS,
    canonicalize_decisions,
    normalize_text,
    validate_decision_semantics,
)

PAYLOAD_COLUMNS = (
    "manual_corrected_behavior",
    "manual_label_strength",
    "manual_training_action",
    "manual_sample_weight",
)


def repair_pending_payloads(
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clear unconfirmed payload from pending rows without changing row identity."""
    missing = sorted(set(REQUIRED_DECISION_COLUMNS) - set(decisions.columns))
    if missing:
        raise ValueError(f"decision CSV missing required columns: {missing}")

    before = decisions.copy(deep=True)
    out = decisions.copy(deep=True)
    decision = out["manual_review_decision"].map(normalize_text)
    pending = decision.eq("pending") | decision.eq("")
    payload_present = pd.Series(False, index=out.index)
    for column in PAYLOAD_COLUMNS:
        payload_present |= out[column].map(normalize_text).ne("")
    repair_mask = pending & payload_present

    repaired_rows = []
    for row_index in out.index[repair_mask]:
        row = out.loc[row_index]
        repaired_rows.append(
            {
                "row_index": int(row_index),
                "review_unit_id": normalize_text(row["review_unit_id"]),
                "behavior_label": normalize_text(row["behavior_label"]),
                "payload_before": {
                    column: normalize_text(row[column]) for column in PAYLOAD_COLUMNS
                },
            }
        )
        for column in PAYLOAD_COLUMNS:
            value: Any = pd.NA if column == "manual_sample_weight" else ""
            out.at[row_index, column] = value
        out.at[row_index, "manual_review_decision"] = "pending"

    if len(out) != len(before):
        raise AssertionError("pending payload repair changed row count")
    before_ids = before["review_unit_id"].map(normalize_text).tolist()
    after_ids = out["review_unit_id"].map(normalize_text).tolist()
    if before_ids != after_ids:
        raise AssertionError("pending payload repair changed review_unit_id order")

    changed_columns = []
    for column in out.columns:
        before_values = before[column].map(normalize_text)
        after_values = out[column].map(normalize_text)
        if not before_values.equals(after_values):
            changed_columns.append(column)
    allowed_changes = {"manual_review_decision", *PAYLOAD_COLUMNS}
    unexpected_changes = sorted(set(changed_columns) - allowed_changes)
    if unexpected_changes:
        raise AssertionError(f"unexpected changed columns: {unexpected_changes}")

    normalized, normalization_warnings = canonicalize_decisions(out)
    semantic_errors, semantic_warnings = validate_decision_semantics(
        normalized,
        require_complete=False,
    )
    audit = {
        "schema_version": "pending_behavior_payload_repair_v1",
        "rows_before": int(len(before)),
        "rows_after": int(len(out)),
        "pending_payload_rows_before": int(repair_mask.sum()),
        "pending_payload_rows_after": int(
            _pending_payload_mask(out).sum()
        ),
        "changed_columns": sorted(changed_columns),
        "repaired_rows": repaired_rows,
        "semantic_errors_after": semantic_errors,
        "warnings": normalization_warnings + semantic_warnings,
    }
    return out, audit


def _pending_payload_mask(decisions: pd.DataFrame) -> pd.Series:
    decision = decisions["manual_review_decision"].map(normalize_text)
    pending = decision.eq("pending") | decision.eq("")
    payload = pd.Series(False, index=decisions.index)
    for column in PAYLOAD_COLUMNS:
        payload |= decisions[column].map(normalize_text).ne("")
    return pending & payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8-sig",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            frame.to_csv(handle, index=False)
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--backup-csv", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the repair. Without this flag the command is read-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.decisions_csv
    if not path.exists():
        raise FileNotFoundError(path)
    audit_path = args.audit_json or path.with_name(
        f"{path.stem}.pending_payload_repair_audit.json"
    )
    backup_path = args.backup_csv or path.with_name(
        f"{path.stem}.before_pending_payload_repair.csv"
    )

    before_sha256 = _sha256(path)
    decisions = pd.read_csv(path, low_memory=False)
    repaired, audit = repair_pending_payloads(decisions)
    audit.update(
        {
            "mode": "apply" if args.apply else "dry_run",
            "decisions_csv": str(path),
            "backup_csv": str(backup_path),
            "sha256_before": before_sha256,
        }
    )

    if args.apply:
        if backup_path.exists():
            raise FileExistsError(
                f"backup already exists; refusing to overwrite: {backup_path}"
            )
        backup_path.write_bytes(path.read_bytes())
        _write_csv_atomic(repaired, path)
        audit["sha256_after"] = _sha256(path)
        audit["backup_sha256"] = _sha256(backup_path)
        audit["backup_matches_original"] = audit["backup_sha256"] == before_sha256
    else:
        audit["sha256_after"] = None
        audit["backup_sha256"] = None
        audit["backup_matches_original"] = None

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    if audit["semantic_errors_after"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
