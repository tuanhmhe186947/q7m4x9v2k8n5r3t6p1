import lightning_sdk
from pathlib import Path

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

local_script = Path("scratch/audit_samples_exact.py")
print(f"Uploading {local_script} to Studio...")
studio.upload_file(str(local_script), "/teamspace/studios/this_studio/audit_samples_exact.py")

# Fix remote filename if backslash
import base64
move_cmd = """
import os, shutil
for f in os.listdir("/teamspace/studios/this_studio"):
    if "audit_samples_exact.py" in f and f != "audit_samples_exact.py":
        src = os.path.join("/teamspace/studios/this_studio", f)
        dest = "/teamspace/studios/this_studio/audit_samples_exact.py"
        shutil.move(src, dest)
"""
b64_move = base64.b64encode(move_cmd.encode("utf-8")).decode("ascii")
studio.run(f'python3 -c "import base64; exec(base64.b64decode(\'{b64_move}\').decode(\'utf-8\'))"')

res = studio.run("python3 /teamspace/studios/this_studio/audit_samples_exact.py")
print("=== REMOTE OUTPUT ===")
print(res.encode("ascii", errors="replace").decode("ascii"))
