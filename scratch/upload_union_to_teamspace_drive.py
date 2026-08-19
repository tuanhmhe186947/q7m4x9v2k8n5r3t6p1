import time
from pathlib import Path
import lightning_sdk

ts = lightning_sdk.Teamspace(name="pig-project", user="ironheart211224")

local_dir = Path("outputs/classification_v2/full_t6_union_r128_20260818")
files_to_upload = [
    "full_t6_union_selection.csv",
    "visual_context_manifest.csv",
    "visual_context_cache_audit.json",
    "packed_image_cache_index.csv",
    "packed_image_cache_audit.json",
    "packed_rgb_128_letterbox.npy",
]

remote_prefix = "classification_v2/full_t6_union_r128_20260818"

print(f"Uploading 6 union artifacts to Teamspace Drive under {remote_prefix}...")
for fname in files_to_upload:
    local_p = local_dir / fname
    assert local_p.exists(), f"Missing file: {local_p}"
    remote_p = f"{remote_prefix}/{fname}"
    sz = local_p.stat().st_size
    print(f"\nUploading {fname} ({sz:,} bytes) -> {remote_p}...")
    t0 = time.perf_counter()
    ts.upload_file(str(local_p), remote_p, progress_bar=True)
    elapsed = time.perf_counter() - t0
    print(f"  Finished in {elapsed:.1f}s ({sz / (1024*1024) / max(elapsed, 0.001):.1f} MB/s)")

print("\nALL 6 UNION ARTIFACTS UPLOADED TO TEAMSPACE DRIVE!")
