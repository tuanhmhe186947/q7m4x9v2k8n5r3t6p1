import lightning_sdk
from pathlib import Path

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

local_script = Path("scratch/preflight_benchmark.py")
print(f"Uploading {local_script} to Studio...")
studio.upload_file(str(local_script), "/teamspace/studios/this_studio/preflight_benchmark.py")
print("Upload complete! Now running preflight and benchmark on Studio...")

# Fix path on remote if uploaded with backslashes
move_cmd = """
import os, shutil
for f in os.listdir("/teamspace/studios/this_studio"):
    if "preflight_benchmark.py" in f and f != "preflight_benchmark.py":
        src = os.path.join("/teamspace/studios/this_studio", f)
        dest = "/teamspace/studios/this_studio/preflight_benchmark.py"
        shutil.move(src, dest)
        print(f"Moved {src} -> {dest}")
"""
import base64
b64_move = base64.b64encode(move_cmd.encode("utf-8")).decode("ascii")
studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64_move}\').decode(\'utf-8\'))"')

res = studio.run("python3 /teamspace/studios/this_studio/preflight_benchmark.py")
print("=== REMOTE OUTPUT ===")
print(res.encode("ascii", errors="replace").decode("ascii"))
