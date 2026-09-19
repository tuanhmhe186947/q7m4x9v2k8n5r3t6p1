import os
from pathlib import Path

import pandas as pd

enh_path = Path(
    "outputs/classification_v2/frame_features/"
    "spatiotemporal_frame_features_enhanced.csv"
)

candidate_roots = [
    Path(
        os.environ.get(
            "CLASSIFICATION_V2_LEGACY_CROP_ROOT",
            "outputs/legacy_16f_rebuild/"
            "legacy_16f_rebuild_20260718_v2/06_full_recovery/crops",
        )
    ).parent,
    Path(
        os.environ.get(
            "CLASSIFICATION_V2_LEGACY_CROP_ROOT",
            "outputs/legacy_16f_rebuild/"
            "legacy_16f_rebuild_20260718_v2/06_full_recovery/crops",
        )
    ),
]

df = pd.read_csv(enh_path, low_memory=False)
legacy = df[df["source_type"].astype(str).eq("legacy_recovered")].copy()

print("legacy rows =", len(legacy))
print("\nCandidate roots:")
for root in candidate_roots:
    print(root, "exists =", root.exists())

print("\nSample original crop_path:")
print(legacy["crop_path"].dropna().head(5).to_string(index=False))


def suffix_after_known_root(path_str: str) -> str:
    s = str(path_str).replace("/", "\\")
    markers = [
        "\\outputs\\legacy_full_multigt_masked_nodup_16f\\crops\\",
        "\\data\\raw\\legacy_full_multigt_masked_nodup_16f\\crops\\",
        "\\legacy_full_multigt_masked_nodup_16f\\crops\\",
        "\\legacy_full_multigt_masked_nodup_16f\\",
    ]
    for marker in markers:
        if marker in s:
            return s.split(marker, 1)[1]
    return Path(s).name


sample = legacy["crop_path"].dropna().head(200).astype(str).tolist()

print("\nResolve counts on first 200 legacy crop_path:")
for root in candidate_roots:
    ok = 0
    examples = []
    for p in sample:
        rel = suffix_after_known_root(p)
        cand = root / rel
        if cand.exists():
            ok += 1
            if len(examples) < 3:
                examples.append(cand)
    print("\nROOT =", root)
    print("ok =", ok, "/", len(sample))
    print("examples:")
    for e in examples:
        print(" ", e)
