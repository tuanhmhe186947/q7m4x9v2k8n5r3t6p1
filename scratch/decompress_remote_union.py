import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

script = """
import os, subprocess, time

target_dir = "/teamspace/studios/this_studio/full_t6_union_r128_20260818"
os.makedirs(target_dir, exist_ok=True)

# Find the gz file
gz_file = None
for f in os.listdir("/teamspace/studios/this_studio"):
    if "packed_rgb_128_letterbox.npy.gz" in f:
        gz_file = os.path.join("/teamspace/studios/this_studio", f)
        print(f"Found gz file: {gz_file} ({os.path.getsize(gz_file):,} bytes)")
        break

if gz_file:
    out_npy = os.path.join(target_dir, "packed_rgb_128_letterbox.npy")
    print(f"Decompressing {gz_file} -> {out_npy}...")
    t0 = time.perf_counter()
    subprocess.run(f"gzip -dc '{gz_file}' > '{out_npy}'", shell=True, check=True)
    el = time.perf_counter() - t0
    print(f"Decompressed in {el:.1f}s!")
    print(f"Final npy size: {os.path.getsize(out_npy):,} bytes")
    os.remove(gz_file)
    print("Cleaned up gz archive.")

print("=== ALL ARTIFACTS IN full_t6_union_r128_20260818 ===")
for f in sorted(os.listdir(target_dir)):
    p = os.path.join(target_dir, f)
    print(f"  {f}: {os.path.getsize(p):,} bytes")
"""

b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
res = studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64}\').decode(\'utf-8\'))"')
print(res)
