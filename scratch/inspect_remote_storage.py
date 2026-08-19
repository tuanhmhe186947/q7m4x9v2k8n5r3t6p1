import sys
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import os
import sys

print("=== 1. Listing /teamspace/uploads/classification_v2 ===")
base = "/teamspace/uploads/classification_v2"
if os.path.exists(base):
    for root, dirs, files in os.walk(base):
        for f in files:
            p = os.path.join(root, f)
            try:
                size = os.path.getsize(p)
                print(f"{p} ({size:,} bytes)")
            except Exception as e:
                print(f"{p} (error: {e})")
else:
    print(f"{base} does not exist!")

print("\\n=== 2. Listing /teamspace/studios/this_studio ===")
print("pwd:", os.getcwd())
for item in os.listdir("/teamspace/studios/this_studio"):
    print(" ", item)
"""

print("Executing script on remote Studio...")
res = studio.run(f"python3 -c '{remote_script}'")
print("=== REMOTE OUTPUT ===")
print(res)
