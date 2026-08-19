import sys
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
import json
import numpy as np
import pandas as pd

r128_dir = "/teamspace/uploads/classification_v2/cloud_r128_recovery_20260817_gcp/r128_cache"
full_t6_dir = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817"

print("=== 1. Inspecting r128_cache/cache_manifest.json ===")
manifest_p = os.path.join(r128_dir, "cache_manifest.json")
if os.path.exists(manifest_p):
    with open(manifest_p, "r") as f:
        print(f.read()[:2000])

print("\\n=== 2. Inspecting packed_rgb_128_letterbox.npy ===")
npy_p = os.path.join(r128_dir, "packed_rgb_128_letterbox.npy")
if os.path.exists(npy_p):
    arr = np.load(npy_p, mmap_mode="r")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}, size: {os.path.getsize(npy_p):,} bytes")

print("\\n=== 3. Inspecting packed_image_cache_index.csv ===")
idx_p = os.path.join(r128_dir, "packed_image_cache_index.csv")
if os.path.exists(idx_p):
    df_idx = pd.read_csv(idx_p, nrows=20)
    print("Columns:", list(df_idx.columns))
    print(df_idx.head(5))
    df_idx_full = pd.read_csv(idx_p, low_memory=False)
    print(f"Total index rows: {len(df_idx_full):,}")
    if "crop_type" in df_idx_full.columns:
        print("crop_type counts:", df_idx_full["crop_type"].value_counts())
    if "context_type" in df_idx_full.columns:
        print("context_type counts:", df_idx_full["context_type"].value_counts())
    if "source" in df_idx_full.columns:
        print("source counts:", df_idx_full["source"].value_counts())

print("\\n=== 4. Inspecting T6_target_index.csv ===")
t6_idx_p = os.path.join(r128_dir, "T6_target_index.csv")
if os.path.exists(t6_idx_p):
    df_t6 = pd.read_csv(t6_idx_p, nrows=10)
    print("Columns:", list(df_t6.columns))
    print(df_t6.head(5))
    df_t6_full = pd.read_csv(t6_idx_p, low_memory=False)
    print(f"Total T6 rows: {len(df_t6_full):,}")

print("\\n=== 5. Inspecting full_t6_canonical_46d.npz ===")
npz_46d_p = os.path.join(full_t6_dir, "full_t6_canonical_46d.npz")
if os.path.exists(npz_46d_p):
    with np.load(npz_46d_p) as npz:
        print("NPZ keys:", list(npz.keys()))
        for k in ["bbox_xywh_n", "bbox_shape_n", "motion_delta", "roi_class_relation", "social_relation", "length_mask", "observed_mask"]:
            if k in npz:
                print(f"  {k}: shape={npz[k].shape}, dtype={npz[k].dtype}")

print("\\n=== 6. Searching for any visual interaction / union caches across /teamspace ===")
for root, dirs, files in os.walk("/teamspace"):
    for f in files:
        if "union" in f.lower() or "visual_interaction" in f.lower() or "visual_context" in f.lower():
            p = os.path.join(root, f)
            print(f"Found visual context file: {p} ({os.path.getsize(p):,} bytes)")
"""

print("Executing index inspection on remote Studio...")
res = studio.run(f"python3 -c '{remote_script}'")
print("=== REMOTE OUTPUT ===")
print(res)
