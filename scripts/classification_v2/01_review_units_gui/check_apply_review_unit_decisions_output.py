from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REVIEWED_PATH = Path(r"outputs\classification_v2\review_policy\reviewed_frame_features.csv")
AUDIT_PATH = Path(r"outputs\classification_v2\review_policy\apply_review_unit_decisions_audit.json")
COMBINED_PATH = Path(r"outputs\classification_v2\review_policy\review_unit_decisions_combined.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit applied review decision outputs.")
    parser.add_argument("--reviewed-csv", type=Path, default=REVIEWED_PATH)
    parser.add_argument("--audit-json", type=Path, default=AUDIT_PATH)
    parser.add_argument("--combined-csv", type=Path, default=COMBINED_PATH)
    parser.add_argument("--source-frame-features-csv", type=Path)
    args = parser.parse_args()
    reviewed_path = args.reviewed_csv
    audit_path = args.audit_json
    combined_path = args.combined_csv
    errors: list[str] = []

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

    if combined_path.exists():
        decisions = pd.read_csv(combined_path, low_memory=False)
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

        if args.source_frame_features_csv is not None:
            source_rows = len(pd.read_csv(args.source_frame_features_csv, usecols=["frame_uid"]))
            if source_rows != len(reviewed):
                errors.append(
                    f"source_reviewed_row_mismatch={source_rows}:{len(reviewed)}"
                )

    if errors:
        print("\n[FAIL] apply-output audit errors:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)
    print("\n[PASS] applied review outputs are valid")


if __name__ == "__main__":
    main()
