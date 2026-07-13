"""Fail-closed coverage audit for classification_v2 human review decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

VALID_BEHAVIORS = {
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
}
VALID_DECISIONS = {"pending", "accept", "corrected", "exclude"}
VALID_ACTIONS = {
    "",
    "main_train",
    "keep",
    "correct_and_keep",
    "downweight",
    "low_weight_train",
    "exclude",
    "review_later",
}
REQUIRED_COLUMNS = [
    "review_item_id",
    "review_unit_id",
    "review_unit_type",
    "temporal_unit_key",
    "source_type",
    "dataset_id",
    "video_key",
    "pig_id",
    "track_id",
    "object_track_key",
    "unit_start_frame",
    "unit_end_frame",
    "display_frame_indices",
    "review_template",
    "behavior_label",
    "original_behavior",
    "review_reason",
    "apply_scope",
    "manual_review_decision",
    "manual_corrected_behavior",
    "manual_label_strength",
    "manual_training_action",
    "manual_sample_weight",
    "manual_note",
]


def _text(series: pd.Series) -> pd.Series:
    """Normalize nullable decision text without changing source files."""
    return series.fillna("").astype(str).str.strip()


def audit_decision_coverage(
    review_manifest: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    """Validate schema, uniqueness, coverage, and corrected-label semantics."""
    errors: list[str] = []
    warnings: list[str] = []
    if "review_unit_id" not in review_manifest.columns:
        return {"errors": ["review_manifest_missing_review_unit_id"], "warnings": []}

    expected_ids = _text(review_manifest["review_unit_id"])
    duplicate_manifest = int(expected_ids.duplicated(keep=False).sum())
    if duplicate_manifest:
        errors.append(f"duplicate_review_manifest_rows={duplicate_manifest}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in decisions.columns]
    if missing_columns:
        errors.append(f"missing_decision_columns={missing_columns}")
        return {
            "review_unit_rows": int(len(review_manifest)),
            "decision_rows": int(len(decisions)),
            "errors": errors,
            "warnings": warnings,
        }

    if "window_uid" in decisions.columns:
        errors.append("forbidden_window_uid_column")

    decision_ids = _text(decisions["review_unit_id"])
    duplicate_decisions = int(decision_ids.duplicated(keep=False).sum())
    if duplicate_decisions:
        errors.append(f"duplicate_decision_rows={duplicate_decisions}")

    expected_set = set(expected_ids)
    decision_set = set(decision_ids)
    missing_ids = sorted(expected_set - decision_set)
    unexpected_ids = sorted(decision_set - expected_set)
    decision_values = _text(decisions["manual_review_decision"])
    invalid_decisions = sorted(set(decision_values) - VALID_DECISIONS)
    pending_count = int(decision_values.eq("pending").sum())
    action_values = _text(decisions["manual_training_action"])
    invalid_actions = sorted(set(action_values) - VALID_ACTIONS)
    review_later_count = int(action_values.eq("review_later").sum())

    if unexpected_ids:
        errors.append(f"unexpected_review_unit_count={len(unexpected_ids)}")
    if invalid_decisions:
        errors.append(f"invalid_decisions={invalid_decisions}")
    if invalid_actions:
        errors.append(f"invalid_training_actions={invalid_actions}")
    if require_complete and missing_ids:
        errors.append(f"missing_review_unit_count={len(missing_ids)}")
    elif missing_ids:
        warnings.append(f"missing_review_unit_count={len(missing_ids)}")
    if require_complete and pending_count:
        errors.append(f"pending_review_unit_count={pending_count}")
    elif pending_count:
        warnings.append(f"pending_review_unit_count={pending_count}")
    if require_complete and review_later_count:
        errors.append(f"review_later_unit_count={review_later_count}")
    elif review_later_count:
        warnings.append(f"review_later_unit_count={review_later_count}")

    corrected_values = _text(decisions["manual_corrected_behavior"])
    corrected = decision_values.eq("corrected")
    invalid_corrected = corrected & ~corrected_values.isin(VALID_BEHAVIORS)
    pending_with_correction = decision_values.eq("pending") & corrected_values.ne("")
    if invalid_corrected.any():
        errors.append(f"invalid_corrected_behavior_rows={int(invalid_corrected.sum())}")
    if pending_with_correction.any():
        errors.append(f"pending_with_corrected_behavior={int(pending_with_correction.sum())}")

    return {
        "review_unit_rows": int(len(review_manifest)),
        "decision_rows": int(len(decisions)),
        "covered_review_units": int(len(expected_set & decision_set)),
        "missing_review_unit_count": int(len(missing_ids)),
        "unexpected_review_unit_count": int(len(unexpected_ids)),
        "duplicate_review_manifest_rows": duplicate_manifest,
        "duplicate_decision_rows": duplicate_decisions,
        "pending_review_unit_count": pending_count,
        "review_later_unit_count": review_later_count,
        "decision_counts": decision_values.value_counts(dropna=False).to_dict(),
        "training_action_counts": action_values.value_counts(dropna=False).to_dict(),
        "require_complete": require_complete,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest-csv", type=Path, required=True)
    parser.add_argument("--decisions-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    review_manifest = pd.read_csv(args.review_manifest_csv, low_memory=False)
    parts = []
    missing_files = []
    for path in args.decisions_csv:
        if not path.exists():
            missing_files.append(str(path))
            continue
        parts.append(pd.read_csv(path, low_memory=False))
    decisions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    audit = audit_decision_coverage(
        review_manifest,
        decisions,
        require_complete=args.require_complete,
    )
    if missing_files:
        audit.setdefault("errors", []).append(f"missing_decision_files={missing_files}")

    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
