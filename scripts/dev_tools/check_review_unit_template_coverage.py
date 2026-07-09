from pathlib import Path

import pandas as pd

root = Path(r"outputs\classification_v2\review_units")

files = {
    "all_units": root / "review_unit_manifest.csv",
    "full_review": root / "full_review_unit_manifest.csv",
    "interaction": root / "interaction_review_unit_template.csv",
    "motion": root / "motion_review_unit_template.csv",
    "posture": root / "posture_review_unit_template.csv",
    "roi": root / "roi_review_unit_template.csv",
    "temporal": root / "temporal_consistency_review_unit_template.csv",
}

dfs = {}
for name, path in files.items():
    if not path.exists():
        print("[MISSING]", name, path)
        continue
    dfs[name] = pd.read_csv(path, low_memory=False)
    print("\n==", name, "==")
    print("path =", path)
    print("rows =", len(dfs[name]))
    if "behavior_label" in dfs[name].columns:
        print(dfs[name]["behavior_label"].fillna("").value_counts().to_string())
    if "review_unit_id" in dfs[name].columns:
        print("duplicate review_unit_id =", dfs[name]["review_unit_id"].duplicated().sum())
    print("has window_uid =", "window_uid" in dfs[name].columns)

required_groups = {
    "interaction": {"fight", "social-nose"},
    "roi": {"eat", "drink", "playwithtoy"},
    "motion": {"move", "explore", "stand"},
    "posture": {"lying", "sitting", "stand"},
}

print("\n\n=== REQUIRED GROUP COVERAGE ===")
for group, labels in required_groups.items():
    df = dfs.get(group)
    if df is None:
        print(group, "MISSING FILE")
        continue

    present = set(df["behavior_label"].dropna().astype(str)) if "behavior_label" in df.columns else set()
    missing = sorted(labels - present)
    extra = sorted(present - labels)

    print("\n", group)
    print("required =", sorted(labels))
    print("present  =", sorted(present))
    print("missing  =", missing)
    print("extra    =", extra)

print("\n\n=== PLAYWITHTOY CHECK ===")
all_units = dfs.get("all_units")
full_review = dfs.get("full_review")
roi = dfs.get("roi")

if all_units is not None:
    p = all_units[all_units["behavior_label"].astype(str).eq("playwithtoy")]
    print("playwithtoy in all_units =", len(p))
    cols = [
        "review_unit_id",
        "source_type",
        "review_unit_type",
        "behavior_label",
        "review_reason",
        "review_priority",
        "roi_context_quality",
        "roi_target_class",
        "roi_target_near",
        "roi_target_contact",
    ]
    cols = [c for c in cols if c in p.columns]
    print(p[cols].head(20).to_string(index=False))

if full_review is not None:
    print(
        "playwithtoy in full_review =",
        int(full_review["behavior_label"].astype(str).eq("playwithtoy").sum()),
    )

if roi is not None:
    print(
        "playwithtoy in roi =",
        int(roi["behavior_label"].astype(str).eq("playwithtoy").sum()),
    )

print("\n\n=== FULL REVIEW UNION CHECK ===")
if full_review is not None:
    union_ids = set()
    for name in ["interaction", "motion", "posture", "roi", "temporal"]:
        df = dfs.get(name)
        if df is not None and "review_unit_id" in df.columns:
            union_ids.update(df["review_unit_id"].dropna().astype(str))

    full_ids = set(full_review["review_unit_id"].dropna().astype(str))
    print("union component ids =", len(union_ids))
    print("full ids =", len(full_ids))
    print("component not in full =", len(union_ids - full_ids))
    print("full not in components =", len(full_ids - union_ids))
