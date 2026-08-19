import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
import subprocess

print("=== Python and Environment ===")
print("Python path:", subprocess.check_output(["which", "python3"]).decode().strip())
print("Torch version:")
try:
    import torch
    print(" ", torch.__version__, "CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("  error:", e)

print("\\n=== Directory Structure ===")
print("In this_studio:")
for f in os.listdir("/teamspace/studios/this_studio"):
    p = os.path.join("/teamspace/studios/this_studio", f)
    if os.path.isdir(p) and not f.startswith("."):
        print(f"  [DIR] {f}")
    elif not f.startswith("."):
        print(f"  [FILE] {f} ({os.path.getsize(p):,} B)")
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

print("Executing environment check on Studio...")
res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res)
