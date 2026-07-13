from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.behavior_review_contract import (
    CANONICAL_BEHAVIORS,
    audit_decision_coverage,
    audit_review_unit_contract,
    canonicalize_decisions,
    validate_decision_semantics,
)

REVIEWED_PATH = Path(r"outputs\classification_v2\review_policy\reviewed_frame_features.csv")
AUDIT_PATH = Path(r"outputs\classification_v2\review_policy\apply_review_unit_decisions_audit.json")
COMBINED_PATH = Path(r"outputs\classification_v2\review_policy\review_unit_decisions_combined.csv")
REVIEW_MANIFEST_PATH = Path(
    r"outputs\classification_v2\review_units\full_review_unit_manifest.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit applied review decision outputs.")
    parser.add_argument("--reviewed-csv", type=Path, default=REVIEWED_PATH)
    parser.add_argument("--audit-json", type=Path, default=AUDIT_PATH)
    parser.add_argument("--combined-csv", type=Path, default=COMBINED_PATH)
    parser.add_argument("--source-frame-features-csv", type=Path)
    parser.add_argument(
        "--review-unit-manifest-csv",
        type=Path,
        default=REVIEW_MANIFEST_PATH,
    )
    args = parser.parse_args()
    reviewed_path = args.reviewed_csv
    audit_path = args.audit_json
    combined_path = args.combined_csv
    review_manifest_path = args.review_unit_manifest_csv
    errors: list[str] = []
    review_manifest: pd.DataFrame | None = None

    if not review_manifest_path.exists():
        errors.append(f"missing_review_unit_manifest={review_manifest_path}")
    else:
        review_manifest = pd.read_csv(review_manifest_path, low_memory=False)
        contract = audit_review_unit_contract(review_manifest)
        errors.extend(contract["errors"])

    print("=== FILES ===")
    for path in [reviewed_path, audit_path, combined_path]:
        print(path, "exists=", path.exists())
        if not path.exists():
            errors.append(f"missing_output={path}")

    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        print("\n=== AUDIT ===")
        print("errors=", audit.get("errors"))
        print("warnings=", audit.get("warnings"))
        print("rows=", audit.get("rows"))
        print(
            "apply_audit=",
            json.dumps(audit.get("apply_audit"), indent=2, ensure_ascii=False, default=str),
        )
        if audit.get("errors"):
            errors.append(f"apply_audit_errors={audit.get('errors')}")
        rows = audit.get("rows", {})
        if rows.get("frame_features") != rows.get("reviewed_frame_features"):
            errors.append("apply_audit_row_count_mismatch")
        apply_audit = audit.get("apply_audit", {})
        if apply_audit.get("missing_review_unit_count"):
            errors.append("apply_audit_has_unmatched_review_units")
        if apply_audit.get("duplicate_active_decision_rows"):
            errors.append("apply_audit_has_duplicate_active_decisions")

    if combined_path.exists():
        decisions = pd.read_csv(combined_path, low_memory=False)
        decisions, _ = canonicalize_decisions(decisions)
        if review_manifest is not None:
            coverage_audit = audit_decision_coverage(
                review_manifest,
                decisions,
                require_complete=True,
            )
            errors.extend(coverage_audit["errors"])
        else:
            decision_errors, _ = validate_decision_semantics(
                decisions,
                require_complete=True,
            )
            errors.extend(decision_errors)
        print("\n=== COMBINED DECISIONS ===")
        print("rows=", len(decisions))
        if "manual_review_decision" in decisions.columns:
            print(decisions["manual_review_decision"].value_counts(dropna=False).to_string())
        if "review_unit_id" in decisions.columns:
            duplicate_count = int(decisions["review_unit_id"].duplicated().sum())
            print("duplicate review_unit_id=", duplicate_count)
            if duplicate_count:
                errors.append(f"duplicate_review_unit_id={duplicate_count}")
        has_window_uid = "window_uid" in decisions.columns
        print("has window_uid=", has_window_uid)
        if has_window_uid:
            errors.append("forbidden_window_uid_column")

    if reviewed_path.exists():
        reviewed = pd.read_csv(reviewed_path, low_memory=False)
        print("\n=== REVIEWED FRAME FEATURES ===")
        print("rows=", len(reviewed), "cols=", len(reviewed.columns))
        for column in [
            "review_decision_applied",
            "review_manual_decision",
            "review_training_action",
            "review_include_in_training",
            "behavior_before_review",
            "behavior_after_review",
        ]:
            if column in reviewed.columns:
                print(f"\n{column}:")
                print(reviewed[column].value_counts(dropna=False).head(20).to_string())

        required = {"behavior_before_review", "behavior_after_review"}
        missing_required = sorted(required - set(reviewed.columns))
        if missing_required:
            errors.append(f"missing_reviewed_columns={missing_required}")
        if required.issubset(reviewed.columns):
            changed = reviewed[
                reviewed["behavior_before_review"]
                .astype(str)
                .ne(reviewed["behavior_after_review"].astype(str))
            ]
            cols = [
                "review_unit_id",
                "source_type",
                "video_key",
                "frame_index",
                "pig_id",
                "behavior_before_review",
                "behavior_after_review",
                "review_manual_decision",
                "review_corrected_behavior",
            ]
            cols = [col for col in cols if col in changed.columns]
            print("\n=== CHANGED LABEL SAMPLE ===")
            print(changed[cols].head(20).to_string(index=False))
            invalid_change = changed["review_manual_decision"].astype(str).ne("corrected")
            if invalid_change.any():
                errors.append(
                    f"label_changed_without_corrected_decision={int(invalid_change.sum())}"
                )

        invalid_behavior = sorted(
            set(reviewed["behavior"].dropna().astype(str)) - CANONICAL_BEHAVIORS
        )
        if invalid_behavior:
            errors.append(f"invalid_reviewed_behavior={invalid_behavior}")

        if review_manifest is not None:
            errors.extend(_applied_scope_errors(reviewed, review_manifest))

        if args.source_frame_features_csv is not None:
            source = pd.read_csv(args.source_frame_features_csv, low_memory=False)
            if len(source) != len(reviewed):
                errors.append(f"source_reviewed_row_mismatch={len(source)}:{len(reviewed)}")
            if "frame_uid" not in source.columns or "frame_uid" not in reviewed.columns:
                errors.append("frame_uid_missing_for_row_identity_audit")
            else:
                source_ids = source["frame_uid"].fillna("").astype(str)
                reviewed_ids = reviewed["frame_uid"].fillna("").astype(str)
                if source_ids.duplicated().any() or reviewed_ids.duplicated().any():
                    errors.append("duplicate_frame_uid_in_source_or_reviewed")
                if set(source_ids) != set(reviewed_ids):
                    errors.append("source_reviewed_frame_uid_set_mismatch")

    if errors:
        print("\n[FAIL] apply-output audit errors:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)
    print("\n[PASS] applied review outputs are valid")


def _applied_scope_errors(
    reviewed: pd.DataFrame,
    review_manifest: pd.DataFrame,
) -> list[str]:
    """Verify every applied decision touches exactly its canonical frame scope."""
    errors: list[str] = []
    required = {
        "review_decision_applied",
        "review_unit_id_applied",
        "source_type",
        "frame_index",
    }
    if not required.issubset(reviewed.columns):
        return [f"missing_applied_scope_columns={sorted(required - set(reviewed.columns))}"]
    applied_mask = reviewed["review_decision_applied"].astype(str).str.lower().isin(
        {"true", "1", "yes", "y"}
    )
    applied = reviewed[applied_mask].copy()
    manifest = review_manifest.set_index("review_unit_id", drop=False)
    for unit_id, rows in applied.groupby("review_unit_id_applied", sort=False):
        if unit_id not in manifest.index:
            errors.append(f"applied_unknown_review_unit={unit_id}")
            continue
        unit = manifest.loc[unit_id]
        start = int(unit["unit_start_frame"])
        end = int(unit["unit_end_frame"])
        observed = sorted(
            pd.to_numeric(rows["frame_index"], errors="coerce").dropna().astype(int)
        )
        if observed != list(range(start, end + 1)):
            errors.append(f"applied_frame_scope_mismatch={unit_id}")
        if not rows["source_type"].astype(str).eq(str(unit["source_type"])).all():
            errors.append(f"applied_source_scope_mismatch={unit_id}")
    return errors


if __name__ == "__main__":
    main()
