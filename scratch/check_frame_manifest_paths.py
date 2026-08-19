import pandas as pd
from pathlib import Path

df_frames = pd.read_csv("outputs/classification_v2/image_context_v2/image_frame_context_manifest.csv", nrows=10)
print("Columns:", df_frames.columns.tolist())
for i, r in df_frames.iterrows():
    p = Path(str(r["resolved_media_path"]))
    print(f"Row {i}: video_key={r['video_key']}, media_path={p}, exists={p.exists()}")

# Find all unique video keys and check local path resolution
df_all = pd.read_csv("outputs/classification_v2/image_context_v2/image_frame_context_manifest.csv", usecols=["video_key", "resolved_media_path"])
unique_videos = df_all.drop_duplicates(subset=["video_key"])
print(f"\nTotal unique videos in frame context manifest: {len(unique_videos)}")

local_videos = list(Path("data").rglob("*.mp4"))
local_stems = {v.stem: v for v in local_videos}

for _, r in unique_videos.iterrows():
    vk = r["video_key"]
    orig_p = Path(str(r["resolved_media_path"]))
    stem = orig_p.stem
    resolved = local_stems.get(stem) or local_stems.get(str(vk))
    print(f"  video_key: {vk} | orig: {orig_p} (exists={orig_p.exists()}) | resolved_local: {resolved}")
