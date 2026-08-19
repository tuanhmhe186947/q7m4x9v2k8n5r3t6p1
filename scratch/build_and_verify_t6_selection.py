import ast
import hashlib
import json
from pathlib import Path
import pandas as pd

# 1. Load FULL-T6 manifest
manifest_p = Path("outputs/classification_v2/temporal_v2_canonical_authority_v1/full_temporal_window_manifest_release.csv")
df = pd.read_csv(manifest_p, low_memory=False)
df_t6 = df[df["view_id"] == "T6"].reset_index(drop=True)
print(f"Total T6 targets: {len(df_t6):,}")

# 2. Extract image_context_ids
all_context_ids = []
for _, r in df_t6.iterrows():
    st = r["source_type"]
    otk = r["object_track_key"]
    raw_fids = r["selected_frame_indices"]
    if isinstance(raw_fids, str):
        try:
            fids = json.loads(raw_fids)
        except Exception:
            fids = ast.literal_eval(raw_fids)
    else:
        fids = raw_fids
        
    for fid in fids:
        if fid >= 0:
            cid = f"{st}|{otk}|f{int(fid):06d}"
            all_context_ids.append(cid)

unique_cids = sorted(set(all_context_ids))
print(f"Total context slots across 33,287 T6 windows: {len(all_context_ids):,}")
print(f"Unique image_context_ids: {len(unique_cids):,}")

# 3. Verify against image_frame_context_manifest.csv
frame_manifest_p = Path("outputs/classification_v2/image_context_v2/image_frame_context_manifest.csv")
df_frames = pd.read_csv(frame_manifest_p, usecols=["image_context_id"])
frame_cid_set = set(df_frames["image_context_id"])
print(f"Total frames in frame manifest: {len(frame_cid_set):,}")

missing = [cid for cid in unique_cids if cid not in frame_cid_set]
print(f"Missing from frame manifest: {len(missing)}")
assert len(missing) == 0, f"Missing {len(missing)} context IDs!"

# 4. Save Selection CSV (ONLY one column: image_context_id)
out_dir = Path("outputs/classification_v2/full_t6_union_r128_20260818")
out_dir.mkdir(parents=True, exist_ok=True)
sel_p = out_dir / "full_t6_union_selection.csv"

df_sel = pd.DataFrame({"image_context_id": unique_cids})
df_sel.to_csv(sel_p, index=False)

sel_bytes = sel_p.read_bytes()
sel_sha = hashlib.sha256(sel_bytes).hexdigest()
print(f"\nSaved selection CSV: {sel_p}")
print(f"Selection SHA256: {sel_sha}")
print(f"Selection Rows: {len(df_sel):,}")
print(f"Columns: {list(df_sel.columns)}")
