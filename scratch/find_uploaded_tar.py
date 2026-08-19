import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
print("Searching for runtime_m0_ea392e2d.tar.gz...")
for root, dirs, files in os.walk("/teamspace/studios/this_studio"):
    for f in files:
        if "ea392e2d" in f:
            print("Found:", os.path.join(root, f))
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

print(studio.run(cmd))
