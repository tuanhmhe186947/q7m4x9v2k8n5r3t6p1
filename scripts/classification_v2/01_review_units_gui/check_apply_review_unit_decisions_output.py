from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REVIEWED_PATH = Path(r"outputs\classification_v2\review_policy\reviewed_frame_features.csv")
AUDIT_PATH = Path(r"outputs\classification_v2\review_policy\apply_review_unit_decisions_audit.json")
COMBINED_PATH = Path(r"outputs\classification_v2\review_policy\review_unit_decisions_combined.csv")


def main() -> None:
    print("=== FILES ===")
    for path in [REVIEWED_PATH, AUDIT_PATH, COMBINED_PATH]:
        print(path, "exists=", path.exists())

    if AUDIT_PATH.exists():
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        print("\n=== AUDIT ===")
        print("errors=", audit.get("errors"))
        print("warnings=", audit.get("warnings"))
        print("rows=", audit.get("rows"))
        print(
            "apply_audit=",
            json.dumps(audit.get("apply_audit"), indent=2, ensure_ascii=False, default=str),
        )

    if COMBINED_PATH.exists():
        decisions = pd.read_csv(COMBINED_PATH, low_memory=False)
        print("\n=== COMBINED DECISIONS ===")
        print("rows=", len(decisions))
        if "manual_review_decision" in decisions.columns:
            print(decisions["manual_review_decision"].value_counts(dropna=False).to_string())
        if "review_unit_id" in decisions.columns:
            print("duplicate review_unit_id=", decisions["review_unit_id"].duplicated().sum())
        print("has window_uid=", "window_uid" in decisions.columns)

    if REVIEWED_PATH.exists():
        reviewed = pd.read_csv(REVIEWED_PATH, low_memory=False)
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
        if required.issubset(reviewed.columns):
            changed = reviewed[
                reviewed["behavior_before_review"].astype(str).ne(reviewed["behavior_after_review"].astype(str))
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


if __name__ == "__main__":
    main()
