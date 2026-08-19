import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os

print("=== Checking existing run scripts in this_studio ===")
for root, dirs, files in os.walk("/teamspace/studios/this_studio"):
    for f in files:
        if f.endswith(".py") and ("r128" in f or "preflight" in f or "proof" in f or "t6" in f or "loader" in f):
            p = os.path.join(root, f)
            print(f"{p} ({os.path.getsize(p):,} B)")
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res)
