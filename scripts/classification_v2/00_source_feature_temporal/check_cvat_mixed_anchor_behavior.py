from pathlib import Path

import pandas as pd

VALID = {
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

enh_path = Path(r"outputs\classification_v2\frame_features\spatiotemporal_frame_features_enhanced.csv")
itv_path = Path(r"outputs\classification_v2\sequence_features\temporal_label_intervals.csv")

enh = pd.read_csv(enh_path, low_memory=False)
itv = pd.read_csv(itv_path, low_memory=False)

m = itv[
    itv["source_type"].astype(str).eq("cvat_tracking_xml") & itv["temporal_consistency_status"].astype(str).eq("mixed")
].copy()

print("mixed intervals =", len(m))

if m.empty:
    print("No mixed CVAT intervals.")
    raise SystemExit(0)

# Lấy anchor row trong enhanced:
# anchor row = frame_index == label_window_start trong cùng temporal_unit_key
cvat = enh[enh["source_type"].astype(str).eq("cvat_tracking_xml")].copy()

cvat["frame_index_num"] = pd.to_numeric(cvat["frame_index"], errors="coerce")
cvat["label_window_start_num"] = pd.to_numeric(cvat["label_window_start"], errors="coerce")

anchors = cvat[cvat["frame_index_num"].eq(cvat["label_window_start_num"])].copy()

anchor_cols = [
    "temporal_unit_key",
    "frame_index",
    "behavior",
    "label_anchor_frame_index",
    "label_window_start",
    "label_window_end",
]
anchor_cols = [c for c in anchor_cols if c in anchors.columns]

anchors = anchors[anchor_cols].rename(
    columns={
        "behavior": "anchor_behavior",
        "frame_index": "anchor_frame_index",
    }
)

# Nếu có duplicate anchor row cùng temporal_unit_key thì giữ dòng đầu,
# nhưng vẫn báo để biết.
dup_anchor_count = int(anchors["temporal_unit_key"].duplicated().sum())
anchors = anchors.drop_duplicates("temporal_unit_key", keep="first")

out = m.merge(
    anchors,
    on="temporal_unit_key",
    how="left",
)

out["anchor_behavior"] = out["anchor_behavior"].fillna("")
out["anchor_behavior_valid"] = out["anchor_behavior"].astype(str).isin(VALID)

print("\nanchor duplicate rows before drop =", dup_anchor_count)

print("\nanchor behavior counts in mixed intervals:")
print(out["anchor_behavior"].value_counts(dropna=False).to_string())

print("\nmixed intervals with valid anchor behavior =", int(out["anchor_behavior_valid"].sum()))
print("mixed intervals without valid anchor behavior =", int((~out["anchor_behavior_valid"]).sum()))

print(f"\nvalid anchor ratio = {out['anchor_behavior_valid'].mean():.4f}")

sample_cols = [
    "temporal_unit_key",
    "video_key",
    "pig_id",
    "track_id",
    "label_window_start",
    "label_window_end",
    "behavior_temporal_final",
    "temporal_consistency_status",
    "anchor_frame_index",
    "anchor_behavior",
    "anchor_behavior_valid",
]
sample_cols = [c for c in sample_cols if c in out.columns]

print("\nSAMPLE MIXED INTERVALS WITH ANCHOR:")
print(out[sample_cols].head(20).to_string(index=False))

out_dir = Path(r"outputs\classification_v2\audits")
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / "cvat_mixed_anchor_behavior_check.csv"
out.to_csv(out_csv, index=False, encoding="utf-8-sig")

print("\nWrote:", out_csv)
