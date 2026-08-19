import os
import time
from pathlib import Path
import lightning_sdk
from lightning_sdk import Machine

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

if studio.status != "Running":
    print("Starting Studio on CPU...")
    studio.start(machine=Machine.CPU)

# Create destination folder on studio
import base64
mkdir_code = 'import os; os.makedirs("/teamspace/studios/this_studio/full_t6_union_r128_20260818", exist_ok=True)'
b64 = base64.b64encode(mkdir_code.encode("utf-8")).decode("ascii")
studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64}\').decode(\'utf-8\'))"')

local_dir = Path("outputs/classification_v2/full_t6_union_r128_20260818")
files = [
    "full_t6_union_selection.csv",
    "visual_context_manifest.csv",
    "visual_context_cache_audit.json",
    "packed_image_cache_index.csv",
    "packed_image_cache_audit.json",
    "packed_rgb_128_letterbox.npy",
]

for f in files:
    local_p = local_dir / f
    sz = local_p.stat().st_size
    remote_p = f"/teamspace/studios/this_studio/full_t6_union_r128_20260818/{f}"
    print(f"\nUploading {f} ({sz:,} bytes) -> {remote_p}...")
    t0 = time.perf_counter()
    studio.upload_file(str(local_p), remote_p, progress_bar=True)
    el = time.perf_counter() - t0
    print(f"Uploaded {f} in {el:.1f}s ({sz / (1024*1024) / max(el, 0.001):.1f} MB/s)")

# Fix backslashes in filename if any on remote
fix_code = """
import os, shutil
dest_dir = "/teamspace/studios/this_studio/full_t6_union_r128_20260818"
for f in os.listdir("/teamspace/studios/this_studio"):
    for target in ["full_t6_union_selection.csv", "visual_context_manifest.csv", "visual_context_cache_audit.json", "packed_image_cache_index.csv", "packed_image_cache_audit.json", "packed_rgb_128_letterbox.npy"]:
        if target in f and f != "full_t6_union_r128_20260818":
            src = os.path.join("/teamspace/studios/this_studio", f)
            dest = os.path.join(dest_dir, target)
            if os.path.exists(src) and src != dest:
                shutil.move(src, dest)
                print(f"Moved {src} -> {dest}")

print("=== FINAL FILES IN full_t6_union_r128_20260818 ===")
for f in sorted(os.listdir(dest_dir)):
    p = os.path.join(dest_dir, f)
    print(f"  {f}: {os.path.getsize(p):,} bytes")
"""
b64_fix = base64.b64encode(fix_code.encode("utf-8")).decode("ascii")
res = studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64_fix}\').decode(\'utf-8\'))"')
print(res)
