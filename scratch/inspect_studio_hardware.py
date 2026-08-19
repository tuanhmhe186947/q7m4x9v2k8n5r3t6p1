import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
import psutil
import multiprocessing

print("=== Studio CPU & Memory Hardware ===")
print("CPU count (logical):", os.cpu_count())
print("CPU count (physical):", psutil.cpu_count(logical=False))
mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / (1024**3):.2f} GB, Available RAM: {mem.available / (1024**3):.2f} GB")

print("\\n=== Storage / Disk Free ===")
for path in ["/teamspace/studios/this_studio", "/teamspace/uploads"]:
    if os.path.exists(path):
        usage = psutil.disk_usage(path)
        print(f"{path}: Total {usage.total / (1024**3):.2f} GB, Used {usage.used / (1024**3):.2f} GB, Free {usage.free / (1024**3):.2f} GB")
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res)
