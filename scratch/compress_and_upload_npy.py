import base64
import gzip
import shutil
import time
from pathlib import Path
import lightning_sdk

src_npy = Path("outputs/classification_v2/full_t6_union_r128_20260818/packed_rgb_128_letterbox.npy")
gz_npy = Path("outputs/classification_v2/full_t6_union_r128_20260818/packed_rgb_128_letterbox.npy.gz")

if not gz_npy.exists():
    print(f"Compressing {src_npy} ({src_npy.stat().st_size / (1024**3):.2f} GB) to gzip level 1...")
    t0 = time.perf_counter()
    with open(src_npy, "rb") as f_in:
        with gzip.open(gz_npy, "wb", compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out, length=64 * 1024 * 1024)
    el = time.perf_counter() - t0
    print(f"Compressed in {el:.1f}s -> {gz_npy.stat().st_size / (1024**3):.2f} GB ({gz_npy.stat().st_size:,} bytes)!")
else:
    print(f"Compressed file already exists: {gz_npy} ({gz_npy.stat().st_size / (1024**3):.2f} GB)")

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_dest = "/teamspace/studios/this_studio/full_t6_union_r128_20260818/packed_rgb_128_letterbox.npy.gz"
print(f"\nUploading {gz_npy.name} ({gz_npy.stat().st_size:,} bytes) -> {remote_dest}...")
t0 = time.perf_counter()
studio.upload_file(str(gz_npy), remote_dest, progress_bar=True)
el = time.perf_counter() - t0
print(f"Uploaded in {el:.1f}s ({gz_npy.stat().st_size / (1024*1024) / max(el, 0.001):.1f} MB/s)!")

# Extract on remote Studio
extract_script = """
import os, subprocess, time

studio_root = "/teamspace/studios/this_studio"
union_dir = "/teamspace/studios/this_studio/full_t6_union_r128_20260818"
os.makedirs(union_dir, exist_ok=True)

# Find where the gz file landed
gz_path = None
for root, dirs, files in os.walk(studio_root):
    for f in files:
        if "packed_rgb_128_letterbox.npy.gz" in f:
            gz_path = os.path.join(root, f)
            break

print(f"Found uploaded gz file: {gz_path}")
if gz_path:
    target_npy = os.path.join(union_dir, "packed_rgb_128_letterbox.npy")
    print(f"Extracting {gz_path} -> {target_npy}...")
    t0 = time.perf_counter()
    subprocess.run(f"gzip -dc '{gz_path}' > '{target_npy}'", shell=True, check=True)
    el = time.perf_counter() - t0
    print(f"Decompressed in {el:.1f}s!")
    print(f"Extracted tensor size: {os.path.getsize(target_npy):,} bytes")
    if os.path.exists(gz_path):
        os.remove(gz_path)
        print("Removed gz archive.")

# Move any scattered files into full_t6_union_r128_20260818
for f in os.listdir(studio_root):
    for target in ["full_t6_union_selection.csv", "visual_context_manifest.csv", "visual_context_cache_audit.json", "packed_image_cache_index.csv", "packed_image_cache_audit.json"]:
        if target in f and f != "full_t6_union_r128_20260818":
            src = os.path.join(studio_root, f)
            dst = os.path.join(union_dir, target)
            if os.path.exists(src) and src != dst:
                os.rename(src, dst)
                print(f"Moved {src} -> {dst}")

print("=== FINAL FILES IN full_t6_union_r128_20260818 ===")
for f in sorted(os.listdir(union_dir)):
    p = os.path.join(union_dir, f)
    print(f"  {f}: {os.path.getsize(p):,} bytes")
"""
b64_ext = base64.b64encode(extract_script.encode("utf-8")).decode("ascii")
res = studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64_ext}\').decode(\'utf-8\'))"')
print(res)
