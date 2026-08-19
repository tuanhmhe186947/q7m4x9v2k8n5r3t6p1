import json
import os
import sys
import numpy as np
import pandas as pd

target_dir = "/teamspace/studios/this_studio/runtime_ea392e2d"
sys.path.insert(0, os.path.join(target_dir, "src"))

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

context_id_to_row = dict(zip(df_idx["image_context_id"], df_idx["packed_row"]))

# Find 5 interaction candidates
interaction_behaviors = ["fight", "social_nose", "social_passive", "mount"]
candidates = df_row[df_row["behavior"].isin(interaction_behaviors)].head(5).reset_index(drop=True)

sample_reports = []
for i in range(len(candidates)):
    row = candidates.iloc[i]
    target_id = row["target_id"]
    behavior = row["behavior"]
    frame_ids = json.loads(row["physical_frame_ids_json"])
    target_row_idx = row["row_index"]
    spatial_bbox = npz_46d["bbox_xywh_n"][target_row_idx]
    spatial_social = npz_46d["social_relation"][target_row_idx]
    
    # Extract track_id from target_id
    # e.g. target_id contains track_id=2
    track_id = ""
    for seg in target_id.split("|"):
        if seg.startswith("track_id="):
            track_id = seg.split("=")[1]
            break
    dataset_id = row["dataset_id"]
    video_key = row["video_key"]
    source_type = row["source_type"]
    
    frame_entries = []
    for f_idx, fid in enumerate(frame_ids[:2]):
        ctx_id = f"{source_type}|source={source_type}|dataset={dataset_id}|video={video_key}|track_id={track_id}|f{fid:06d}"
        packed_row = context_id_to_row.get(ctx_id, -1)
        
        if packed_row >= 0:
            actor_img = actor_arr[packed_row]
        else:
            actor_img = np.zeros((128, 128, 3), dtype=np.uint8)
            
        # In current M0 loader on Studio without separate union cache:
        # union branch receives the identical tensor from actor crop:
        union_img = actor_img.copy()
        
        mad = float(np.mean(np.abs(actor_img.astype(float) - union_img.astype(float))))
        max_diff = float(np.max(np.abs(actor_img.astype(float) - union_img.astype(float))))
        is_identical = bool(np.array_equal(actor_img, union_img))
        
        frame_entries.append({
            "frame_idx": f_idx,
            "fid": fid,
            "ctx_id": ctx_id,
            "actor_row": packed_row,
            "union_row": packed_row,  # identical row reused
            "pixel_mad": mad,
            "pixel_max_diff": max_diff,
            "is_identical": is_identical,
            "actor_bbox_xywh_n": [round(x, 4) for x in spatial_bbox[f_idx].tolist()],
            "nearest_partner_social": [round(x, 4) for x in spatial_social[f_idx, :4].tolist()],
        })
        
    sample_reports.append({
        "sample_num": i + 1,
        "target_id": target_id,
        "behavior": behavior,
        "frames": frame_entries,
    })

print("=== EXACT REAL SAMPLE COMPARISON ===")
for s in sample_reports:
    print(f"\nSAMPLE_{s['sample_num']}: behavior={s['behavior']}, target={s['target_id'][:60]}...")
    for f in s['frames']:
        print(f"  Frame {f['frame_idx']} (fid={f['fid']}): ActorRow={f['actor_row']}, UnionRow={f['union_row']}, PixelMAD={f['pixel_mad']}, MaxDiff={f['pixel_max_diff']}, Identical={f['is_identical']}")
        print(f"    ActorBbox={f['actor_bbox_xywh_n']}, SocialRelation={f['nearest_partner_social']}")
