from pathlib import Path

import pandas as pd

SRC_DIR = Path(r"outputs\classification_v2\review_units")
OUT_DIR = Path(r"outputs\classification_v2\review_units\balanced_gui_pilots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECS = {
    "roi": {
        "input": SRC_DIR / "roi_review_unit_template.csv",
        "output": OUT_DIR / "roi_balanced_gui_pilot.csv",
        "labels": ["drink", "eat", "playwithtoy"],
    },
    "motion": {
        "input": SRC_DIR / "motion_review_unit_template.csv",
        "output": OUT_DIR / "motion_balanced_gui_pilot.csv",
        "labels": ["explore", "move", "stand"],
    },
    "posture": {
        "input": SRC_DIR / "posture_review_unit_template.csv",
        "output": OUT_DIR / "posture_balanced_gui_pilot.csv",
        "labels": ["lying", "sitting"],
    },
    "interaction": {
        "input": SRC_DIR / "interaction_review_unit_template.csv",
        "output": OUT_DIR / "interaction_balanced_gui_pilot.csv",
        "labels": ["fight", "social-nose"],
    },
}

PER_LABEL_MAX = 4
SOURCE_ORDER = ["cvat_tracking_xml", "legacy_recovered"]


def pick_balanced(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    parts = []

    for label in labels:
        d = df[df["behavior_label"].astype(str).eq(label)].copy()

        print(f"\nlabel={label} rows={len(d)}")
        if d.empty:
            print("  [MISSING]")
            continue

        if "source_type" in d.columns:
            print(d["source_type"].value_counts(dropna=False).to_string())

        chosen_indices = []

        # Prefer source diversity: one CVAT and one legacy if available.
        if "source_type" in d.columns:
            for src in SOURCE_ORDER:
                q = d[d["source_type"].astype(str).eq(src)].copy()
                if q.empty:
                    continue
                if "review_priority" in q.columns:
                    q = q.sort_values("review_priority", ascending=False)
                chosen_indices.append(q.index[0])

        # Fill remaining slots by priority/order.
        if "review_priority" in d.columns:
            d_sorted = d.sort_values("review_priority", ascending=False)
        else:
            d_sorted = d

        for idx in d_sorted.index:
            if len(chosen_indices) >= PER_LABEL_MAX:
                break
            if idx not in chosen_indices:
                chosen_indices.append(idx)

        parts.append(df.loc[chosen_indices].copy())

    if not parts:
        return df.head(0).copy()

    out = pd.concat(parts, ignore_index=True)

    # Stable useful order for GUI.
    label_order = {label: i for i, label in enumerate(labels)}
    out["_label_order"] = out["behavior_label"].astype(str).map(label_order).fillna(999)
    if "source_type" in out.columns:
        source_order = {src: i for i, src in enumerate(SOURCE_ORDER)}
        out["_source_order"] = out["source_type"].astype(str).map(source_order).fillna(999)
    else:
        out["_source_order"] = 999

    sort_cols = ["_label_order", "_source_order"]
    if "review_priority" in out.columns:
        out["_review_priority_sort"] = pd.to_numeric(out["review_priority"], errors="coerce").fillna(0)
        out = out.sort_values(sort_cols + ["_review_priority_sort"], ascending=[True, True, False])
        out = out.drop(columns=["_review_priority_sort"])
    else:
        out = out.sort_values(sort_cols)

    out = out.drop(columns=["_label_order", "_source_order"])

    if "review_unit_id" in out.columns:
        out = out.drop_duplicates("review_unit_id", keep="first")

    return out.reset_index(drop=True)


for name, spec in SPECS.items():
    print("\n\n==============================")
    print("GROUP:", name)
    print("INPUT:", spec["input"])

    if not spec["input"].exists():
        print("[MISSING FILE]", spec["input"])
        continue

    df = pd.read_csv(spec["input"], low_memory=False)

    print("rows =", len(df))
    print("behavior counts:")
    print(df["behavior_label"].fillna("").value_counts().to_string())

    out = pick_balanced(df, spec["labels"])

    out.to_csv(spec["output"], index=False, encoding="utf-8-sig")

    print("\nWROTE:", spec["output"])
    print("pilot rows =", len(out))
    if len(out):
        print("pilot behavior counts:")
        print(out["behavior_label"].fillna("").value_counts().to_string())
        if "source_type" in out.columns:
            print("pilot source counts:")
            print(out["source_type"].fillna("").value_counts().to_string())
        print(
            "duplicate review_unit_id =",
            out["review_unit_id"].duplicated().sum() if "review_unit_id" in out.columns else "NO COL",
        )
        print("has window_uid =", "window_uid" in out.columns)
