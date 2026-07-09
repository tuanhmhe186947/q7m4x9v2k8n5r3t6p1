from pathlib import Path

import pandas as pd

roots = {
    "roi": Path(r"outputs\classification_v2\review_policy\roi_review_unit_gui_pilot"),
    "motion": Path(r"outputs\classification_v2\review_policy\motion_review_unit_gui_pilot"),
    "posture": Path(r"outputs\classification_v2\review_policy\posture_review_unit_gui_pilot"),
    "interaction": Path(r"outputs\classification_v2\review_policy\interaction_review_unit_gui_pilot"),
}

required_cols = [
    "review_unit_id",
    "review_unit_type",
    "review_template",
    "behavior_label",
    "manual_review_decision",
    "manual_corrected_behavior",
    "manual_label_strength",
    "manual_training_action",
    "manual_sample_weight",
    "manual_note",
]

for name, root in roots.items():
    p = root / "behavior_unit_review_decisions.csv"
    print("\n==============================")
    print("GROUP:", name)
    print("FILE:", p)
    print("exists =", p.exists())

    if not p.exists():
        continue

    df = pd.read_csv(p, low_memory=False)

    print("rows =", len(df))
    print("columns =", df.columns.tolist())

    missing = [c for c in required_cols if c not in df.columns]
    print("missing required cols =", missing)

    print("has window_uid =", "window_uid" in df.columns)

    if "review_unit_id" in df.columns:
        print("duplicate review_unit_id =", df["review_unit_id"].duplicated().sum())

    if "behavior_label" in df.columns:
        print("\nbehavior counts:")
        print(df["behavior_label"].fillna("").value_counts(dropna=False).to_string())

    if "manual_review_decision" in df.columns:
        print("\ndecision counts:")
        print(df["manual_review_decision"].fillna("").value_counts(dropna=False).to_string())

    show_cols = [c for c in required_cols if c in df.columns]
    if show_cols:
        print("\nsample:")
        print(df[show_cols].head(10).to_string(index=False))
