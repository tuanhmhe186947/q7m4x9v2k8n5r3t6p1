from pathlib import Path

import pandas as pd

df = pd.read_csv(
    r"outputs\classification_v2\frame_features\spatiotemporal_frame_features_enhanced.csv",
    low_memory=False,
)

legacy_root = Path(r"data\raw\legacy_full_multigt_masked_nodup_16f\crops")
video_root = Path(r"data\videos")

print("rows =", len(df))
print("\nsource counts:")
print(df["source_type"].value_counts(dropna=False).to_string())

print("\ncolumns:")
for c in ["crop_path", "video_key", "frame_index", "x1", "y1", "x2", "y2"]:
    print(c, c in df.columns)


def rel_after_crops(x):
    s = str(x).replace("/", "\\")
    if "\\crops\\" in s:
        return s.split("\\crops\\", 1)[1]
    return Path(s).name


print("\n=== LEGACY crop-file check ===")
legacy = df[df["source_type"].astype(str).eq("legacy_recovered")].copy()
print("legacy rows =", len(legacy))

if "crop_path" in legacy.columns:
    sample = legacy.head(200)
    direct = sample["crop_path"].map(lambda x: Path(str(x)).exists() if pd.notna(x) else False)
    resolved = sample["crop_path"].map(lambda x: (legacy_root / rel_after_crops(x)).exists() if pd.notna(x) else False)

    print("direct crop_path exists on first 200:")
    print(direct.value_counts(dropna=False).to_string())

    print("resolved via legacy_root exists on first 200:")
    print(resolved.value_counts(dropna=False).to_string())

    print("legacy_root =", legacy_root, "exists =", legacy_root.exists())

print("\n=== CVAT video-frame-bbox check ===")
cvat = df[df["source_type"].astype(str).eq("cvat_tracking_xml")].copy()
print("cvat rows =", len(cvat))
print("video_root =", video_root, "exists =", video_root.exists())

print("\nCVAT video_key sample:")
print(cvat["video_key"].dropna().astype(str).drop_duplicates().head(20).to_string(index=False))

print("\nCVAT bbox validity sample:")
bbox_cols = ["x1", "y1", "x2", "y2", "frame_index"]
missing = [c for c in bbox_cols if c not in cvat.columns]
if missing:
    print("missing bbox/frame cols:", missing)
else:
    sample = cvat.head(200).copy()
    for c in bbox_cols:
        sample[c] = pd.to_numeric(sample[c], errors="coerce")
    bbox_ok = (
        sample["frame_index"].notna()
        & sample["x1"].notna()
        & sample["y1"].notna()
        & sample["x2"].notna()
        & sample["y2"].notna()
        & (sample["x2"] > sample["x1"])
        & (sample["y2"] > sample["y1"])
    )
    print(bbox_ok.value_counts(dropna=False).to_string())

print("\nConclusion:")
print("- legacy_recovered should use crop_file mode")
print("- cvat_tracking_xml should use video_frame_bbox mode unless we materialize crop cache later")
