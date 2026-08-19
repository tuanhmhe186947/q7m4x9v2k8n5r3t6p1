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
        print(f.read())

print("=== 2. Inspecting packed_rgb_128_letterbox.npy ===")
npy_p = os.path.join(r128_dir, "packed_rgb_128_letterbox.npy")
if os.path.exists(npy_p):
    arr = np.load(npy_p, mmap_mode="r")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}, size: {os.path.getsize(npy_p):,} bytes")

print("=== 3. Inspecting packed_image_cache_index.csv ===")
idx_p = os.path.join(r128_dir, "packed_image_cache_index.csv")
if os.path.exists(idx_p):
    df_idx = pd.read_csv(idx_p, low_memory=False)
    print("Columns:", list(df_idx.columns))
    print("Shape:", df_idx.shape)
    print(df_idx.head(3))
    if "context_type" in df_idx.columns:
        print("context_type counts:\\n", df_idx["context_type"].value_counts())
    if "crop_type" in df_idx.columns:
        print("crop_type counts:\\n", df_idx["crop_type"].value_counts())

print("=== 4. Inspecting T6_target_index.csv ===")
t6_idx_p = os.path.join(r128_dir, "T6_target_index.csv")
if os.path.exists(t6_idx_p):
    df_t6 = pd.read_csv(t6_idx_p, low_memory=False)
    print("Columns:", list(df_t6.columns))
    print("Shape:", df_t6.shape)
    print(df_t6.head(3))

print("=== 5. Inspecting full_t6_canonical_46d.npz ===")
npz_46d_p = os.path.join(full_t6_dir, "full_t6_canonical_46d.npz")
if os.path.exists(npz_46d_p):
    with np.load(npz_46d_p) as npz:
        print("NPZ keys:", list(npz.keys()))
        for k in sorted(npz.keys()):
            print(f"  {k}: shape={npz[k].shape}, dtype={npz[k].dtype}")

print("=== 6. Inspecting full_t6_row_manifest.csv ===")
row_man_p = os.path.join(full_t6_dir, "full_t6_row_manifest.csv")
if os.path.exists(row_man_p):
    df_row = pd.read_csv(row_man_p, low_memory=False)
    print("Row manifest shape:", df_row.shape)
    print("Split counts:\\n", df_row["split"].value_counts())
    print("Fold counts:\\n", df_row["outer_fold_id"].value_counts())

print("=== 7. Inspecting full_t6_training_input_map_20260817.json ===")
input_map_p = os.path.join(full_t6_dir, "full_t6_training_input_map_20260817.json")
if os.path.exists(input_map_p):
    with open(input_map_p, "r") as f:
        print(f.read())
"""

print("Executing targeted remote inspection...")
res = studio.run(f"python3 -c '{remote_script}'")
print("=== REMOTE OUTPUT ===")
print(res)
