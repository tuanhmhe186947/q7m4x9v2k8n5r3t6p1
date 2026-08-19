import hashlib
import json
from pathlib import Path
import pandas as pd

# Paths
win_ctx_p = Path("outputs/classification_v2/image_context_v2/image_window_context_manifest.csv")
full_t6_manifest_p = Path("outputs/classification_v2/full_t6_canonical_46d_20260816")

# Let's find full_temporal_window_manifest_release.csv or target_id list
# Or download from Drive / use local copy
print("Loading window context manifest...")
df_win = pd.read_csv(win_ctx_p, low_memory=False)
print(f"Total window context manifest rows: {len(df_win):,}")

# Filter to T6 windows
# Let's check window_id structure
print("Sample window_ids:")
print(df_win["window_id"].head())

# T6 windows have view=T6 or view_id=T6 or are referenced in full_t6
t6_windows = df_win[df_win["window_id"].str.contains(r"view=T6|view_id=T6|\bT6\b", regex=True)].copy()
print(f"T6 matching windows in window manifest: {len(t6_windows):,}")

# If we have 33,287 T6 windows
assert len(t6_windows) == 33287, f"Expected 33,287 windows, got {len(t6_windows)}"

# Extract all image_context_id in order
all_context_ids = []
for seq in t6_windows["image_context_id_sequence"]:
    for cid in str(seq).split(";;"):
        cid = cid.strip()
        if cid:
            all_context_ids.append(cid)

unique_cids = sorted(set(all_context_ids))
print(f"Total context references in T6 windows: {len(all_context_ids):,}")
print(f"Unique image_context_ids in T6 selection: {len(unique_cids):,}")

# Save selection CSV (contains ONLY image_context_id)
out_dir = Path("outputs/classification_v2/full_t6_union_r128_20260818")
out_dir.mkdir(parents=True, exist_ok=True)
sel_p = out_dir / "full_t6_union_selection.csv"

df_sel = pd.DataFrame({"image_context_id": unique_cids})
df_sel.to_csv(sel_p, index=False)

sel_bytes = sel_p.read_bytes()
sel_sha = hashlib.sha256(sel_bytes).hexdigest()
print(f"Selection saved to: {sel_p}")
print(f"Selection SHA256: {sel_sha}")
print(f"Selection rows: {len(df_sel):,}")
print(f"Columns: {list(df_sel.columns)}")
