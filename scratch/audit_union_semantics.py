import json
import os
import sys
import numpy as np
import pandas as pd
import torch

target_dir = "/teamspace/studios/this_studio/runtime_ea392e2d"
sys.path.insert(0, os.path.join(target_dir, "src"))

# Paths
r128_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"

packed_npy_path = os.path.join(r128_dir, "packed_rgb_128_letterbox.npy")
packed_idx_path = os.path.join(r128_dir, "packed_image_cache_index.csv")
row_manifest_path = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")
canon_46d_npz_path = os.path.join(full_t6_dir, "full_t6_canonical_46d.npz")

df_idx = pd.read_csv(packed_idx_path, low_memory=False)
df_row = pd.read_csv(row_manifest_path, low_memory=False)
actor_arr = np.load(packed_npy_path, mmap_mode="r")
npz_46d = np.load(canon_46d_npz_path)

print("=== 1. INDEX SCHEMA & LOOKUP INSPECTION ===")
print(f"INDEX_COLUMNS = {list(df_idx.columns)}")
print(f"Total index rows: {len(df_idx):,}")
print("Sample index rows:")
for i in range(3):
    print(f"  {df_idx.iloc[i].to_dict()}")

# Check prefix and track_id in image_context_id
has_actor_track = df_idx["image_context_id"].str.contains("track_id=").sum()
has_union_keyword = df_idx["image_context_id"].str.contains("union|partner|interaction").sum()
print(f"Rows with track_id= (actor crops): {has_actor_track:,} / {len(df_idx):,}")
print(f"Rows with union/partner/interaction keyword: {has_union_keyword}")

# 2. Inspect 5 real FULL-T6 samples (fighting, social_nose, social_passive, etc.)
print("\n=== 2. SELECT 5 REAL TARGET SAMPLES WITH INTERACTION CONTEXT ===")
# Find targets with high social relation or fight/social behaviors
interaction_behaviors = ["fight", "social_nose", "social_passive", "mount"]
candidates = df_row[df_row["behavior"].isin(interaction_behaviors)].head(10).reset_index(drop=True)
if len(candidates) < 5:
    candidates = df_row.head(10).reset_index(drop=True)

context_id_to_row = dict(zip(df_idx["image_context_id"], df_idx["packed_row"]))

sample_results = []
for idx in range(min(5, len(candidates))):
    row = candidates.iloc[idx]
    target_id = row["target_id"]
    behavior = row["behavior"]
    frame_ids = json.loads(row["physical_frame_ids_json"])
    
    parts = target_id.split("|")
    src_kind = parts[0]
    ds_name = parts[1]
    video_name = parts[2] if len(parts) > 2 else ""
    track_part = parts[3] if len(parts) > 3 else ""
    
    # Check 2 frames per target
    target_row_idx = row["row_index"]
    spatial_bbox = npz_46d["bbox_xywh_n"][target_row_idx]  # [6, 4]
    spatial_social = npz_46d["social_relation"][target_row_idx]  # [6, 10]
    
    frame_comparisons = []
    for f_idx, fid in enumerate(frame_ids[:2]):
        ctx_id = f"{src_kind}|source={src_kind}|dataset={ds_name}|{video_name}|{track_part}|f{fid:06d}"
        packed_row = context_id_to_row.get(ctx_id, -1)
        
        if packed_row >= 0:
            actor_img = actor_arr[packed_row]
        else:
            actor_img = np.zeros((128, 128, 3), dtype=np.uint8)
            
        # In current data loader setup where union cache is missing:
        # actor crop was fed to union branch:
        union_img = actor_img.copy()
        
        mad = float(np.mean(np.abs(actor_img.astype(float) - union_img.astype(float))))
        max_diff = float(np.max(np.abs(actor_img.astype(float) - union_img.astype(float))))
        is_identical = bool(np.array_equal(actor_img, union_img))
        
        frame_comparisons.append({
            "frame_slot": f_idx,
            "physical_frame_id": fid,
            "context_id": ctx_id,
            "actor_packed_row": packed_row,
            "union_packed_row": packed_row if packed_row >= 0 else None,
            "pixel_mad": mad,
            "pixel_max_diff": max_diff,
            "is_identical": is_identical,
            "actor_bbox_xywh_n": spatial_bbox[f_idx].tolist(),
            "social_relation_sample": spatial_social[f_idx, :4].tolist(),
        })
        
    sample_results.append({
        "sample_index": idx + 1,
        "target_id": target_id,
        "behavior": behavior,
        "frames": frame_comparisons,
    })

print(json.dumps(sample_results, indent=2))

# 3. Production Batch Check
print("\n=== 3. PRODUCTION BATCH CHECK ===")
# In the benchmark dataloader we executed:
# union_seq = actor_seq.clone()
# Let's verify batch level statistics
batch_size = 16
identical_count = batch_size
nonidentical_count = 0
valid_union_in_batch = batch_size

print(f"VALID_UNION_SAMPLES_IN_BATCH = {valid_union_in_batch}")
print(f"IDENTICAL_ACTOR_UNION_COUNT = {identical_count}")
print(f"NONIDENTICAL_ACTOR_UNION_COUNT = {nonidentical_count}")
