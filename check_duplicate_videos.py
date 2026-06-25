import re
import pandas as pd

legacy = r"C:\Users\ironh\Downloads\PIG_Behavior_Project\old_burst_center_keyframes_combined.csv"
excl = r"C:\Users\ironh\Downloads\PIG_Behavior_Project\exclude_source_videos.csv"

df = pd.read_csv(legacy, low_memory=False)
ex = set(pd.read_csv(excl)["source_video_key"].astype(str).str.lower())

pat = re.compile(r"(pigs\d{6})/pigs\d{6}/(\d+)/color\.mp4", re.I)

def key(p):
    p = str(p).replace("\\", "/").lower()
    m = pat.search(p)
    return m.group(1) + "/" + m.group(2) if m else None

df["source_video_key"] = df["video_final"].map(key)
hit = df[df["source_video_key"].isin(ex)]

print("legacy rows=", len(df))
print("duplicate rows=", len(hit))
print("duplicate group+pig=", hit.groupby(["group_id", "pig_id"]).ngroups if len(hit) else 0)

if len(hit):
    print("\nDuplicate source videos:")
    print(hit["source_video_key"].value_counts().to_string())

    out = r"C:\Users\ironh\Downloads\PIG_Behavior_Project\duplicate_video_preview.csv"
    hit.to_csv(out, index=False)
    print("\nSaved:", out)
else:
    print("NO DUPLICATE VIDEO FOUND")