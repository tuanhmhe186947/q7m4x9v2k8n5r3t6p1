import json
from pathlib import Path
import numpy as np
import pandas as pd

# Load Union Cache & Index
union_dir = Path("outputs/classification_v2/full_t6_union_r128_20260818")
union_npy_p = union_dir / "packed_rgb_128_letterbox.npy"
union_idx_p = union_dir / "packed_image_cache_index.csv"
union_manifest_p = union_dir / "visual_context_manifest.csv"

union_arr = np.load(union_npy_p, mmap_mode="r")
df_union_idx = pd.read_csv(union_idx_p, low_memory=False)
df_union_man = pd.read_csv(union_manifest_p, low_memory=False)

# Actor Cache Path
actor_npy_p = Path("outputs/classification_v2/model_readiness_audit/pre_gpu_autoresearch_q2_6c2f204_20260804_084638/reviewed_rgb_v1/actor_rgb_128_full/packed_rgb_128_letterbox.npy")
print(f"Actor Cache Path: {actor_npy_p} ({actor_npy_p.stat().st_size:,} B)")
actor_arr = np.load(actor_npy_p, mmap_mode="r")

# Actor index
actor_idx_p = actor_npy_p.parent / "packed_image_cache_index.csv"
df_actor_idx = pd.read_csv(actor_idx_p, low_memory=False)
actor_map = dict(zip(df_actor_idx["image_context_id"], df_actor_idx["packed_row"]))
union_map = dict(zip(df_union_idx["image_context_id"], df_union_idx["packed_row"]))

# Filter to available union rows with valid geometry
avail_man = df_union_man[df_union_man["visual_context_available"] == True].copy()
print(f"Available union rows in manifest: {len(avail_man):,}")

# Sample 20 contexts across different datasets/videos
sampled_contexts = avail_man.sample(n=20, random_state=20260818).reset_index(drop=True)

comparison_results = []
identical_count = 0
nonidentical_count = 0

for i, row in sampled_contexts.iterrows():
    cid = row["image_context_id"]
    actor_row = actor_map.get(cid)
    union_row = union_map.get(cid)
    
    assert actor_row is not None, f"Actor row missing for {cid}"
    assert union_row is not None, f"Union row missing for {cid}"
    
    actor_img = np.asarray(actor_arr[actor_row])
    union_img = np.asarray(union_arr[union_row])
    
    mad = float(np.mean(np.abs(actor_img.astype(float) - union_img.astype(float))))
    max_diff = float(np.max(np.abs(actor_img.astype(float) - union_img.astype(float))))
    is_identical = bool(np.array_equal(actor_img, union_img))
    
    if is_identical:
        identical_count += 1
    else:
        nonidentical_count += 1
        
    res = {
        "sample": i + 1,
        "image_context_id": cid,
        "actor_row": int(actor_row),
        "union_row": int(union_row),
        "pixel_mad": round(mad, 4),
        "pixel_max_diff": round(max_diff, 4),
        "is_identical": is_identical,
        "actor_track_id": str(row["actor_track_id"]),
        "partner_track_id": str(row["partner_track_id"]),
        "union_bbox": [round(float(row[c]), 1) for c in ["union_x1", "union_y1", "union_x2", "union_y2"]],
    }
    comparison_results.append(res)

print("\n=== 20 SAMPLED CONTEXT COMPARISONS (ACTOR vs UNION) ===")
for r in comparison_results:
    print(f"Sample {r['sample']:02d}: ActorRow={r['actor_row']}, UnionRow={r['union_row']} | MAD={r['pixel_mad']:.2f}, MaxDiff={r['pixel_max_diff']:.0f} | Identical={r['is_identical']} | ActorID={r['actor_track_id']}, PartnerID={r['partner_track_id']}")
    print(f"   UnionBbox={r['union_bbox']}")

print(f"\nSummary: {nonidentical_count}/20 NON-IDENTICAL ({identical_count}/20 identical)")
