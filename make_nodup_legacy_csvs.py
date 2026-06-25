import re
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\ironh\Downloads\PIG_Behavior_Project")

center_csv = ROOT / "old_burst_center_keyframes_combined.csv"
all_bbox_csv = ROOT / "old_burst_all_keyframe_bboxes_combined.csv"
exclude_csv = ROOT / "exclude_source_videos.csv"

out_center_keep = ROOT / "old_burst_center_keyframes_nodup_videos.csv"
out_bbox_keep = ROOT / "old_burst_all_keyframe_bboxes_nodup_videos.csv"

out_center_dup = ROOT / "duplicate_video_quarantine_center.csv"
out_bbox_dup = ROOT / "duplicate_video_quarantine_all_bboxes.csv"
out_audit = ROOT / "duplicate_video_filter_audit.csv"

ex = set(pd.read_csv(exclude_csv)["source_video_key"].astype(str).str.lower())

pat = re.compile(r"(pigs\d{6})/pigs\d{6}/(\d+)/color\.mp4", re.I)

def source_key(p):
    p = str(p).replace("\\", "/").lower()
    m = pat.search(p)
    return m.group(1) + "/" + m.group(2) if m else None

center = pd.read_csv(center_csv, low_memory=False)
bbox = pd.read_csv(all_bbox_csv, low_memory=False)

center["source_video_key"] = center["video_final"].map(source_key)
bbox["source_video_key"] = bbox["video_final"].map(source_key)

center["duplicate_video"] = center["source_video_key"].isin(ex)
bbox["duplicate_video"] = bbox["source_video_key"].isin(ex)

dup_pairs = set(
    map(
        tuple,
        center.loc[center["duplicate_video"], ["group_id", "pig_id"]].astype(str).values,
    )
)

def pair_is_dup(df):
    return [
        (str(g), str(p)) in dup_pairs
        for g, p in zip(df["group_id"], df["pig_id"])
    ]

bbox["duplicate_video_by_pair"] = pair_is_dup(bbox)
bbox["duplicate_video"] = bbox["duplicate_video"] | bbox["duplicate_video_by_pair"]

center_keep = center[~center["duplicate_video"]].copy()
center_dup = center[center["duplicate_video"]].copy()

bbox_keep = bbox[~bbox["duplicate_video"]].copy()
bbox_dup = bbox[bbox["duplicate_video"]].copy()

center_keep.to_csv(out_center_keep, index=False)
center_dup.to_csv(out_center_dup, index=False)

bbox_keep.to_csv(out_bbox_keep, index=False)
bbox_dup.to_csv(out_bbox_dup, index=False)

audit = (
    center_dup.groupby("source_video_key")
    .agg(
        duplicate_center_rows=("sample_id", "count"),
        duplicate_group_pig=("pig_id", "count"),
    )
    .reset_index()
    .sort_values("source_video_key")
)

audit.to_csv(out_audit, index=False)

print("CENTER")
print(" original rows:", len(center))
print(" keep rows    :", len(center_keep))
print(" dup rows     :", len(center_dup))
print(" keep group+pig:", center_keep.groupby(["group_id", "pig_id"]).ngroups)
print(" dup group+pig :", center_dup.groupby(["group_id", "pig_id"]).ngroups)

print("\nALL BBOX")
print(" original rows:", len(bbox))
print(" keep rows    :", len(bbox_keep))
print(" dup rows     :", len(bbox_dup))
print(" keep group+pig:", bbox_keep.groupby(["group_id", "pig_id"]).ngroups)
print(" dup group+pig :", bbox_dup.groupby(["group_id", "pig_id"]).ngroups)

print("\nDUPLICATE SOURCE VIDEOS")
print(center_dup["source_video_key"].value_counts().to_string())

print("\nSaved:")
print(out_center_keep)
print(out_bbox_keep)
print(out_center_dup)
print(out_bbox_dup)
print(out_audit)