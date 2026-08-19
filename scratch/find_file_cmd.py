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
print("Finding file with find command...")
try:
    out = subprocess.check_output(["find", "/teamspace", "-name", "*ea392e2d*", "-o", "-name", "*runtime_m0*"], stderr=subprocess.DEVNULL).decode()
    print("Find /teamspace:", out)
except Exception as e:
    print("Error:", e)
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

print(studio.run(cmd))
