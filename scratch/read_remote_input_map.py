import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import json

path = "/teamspace/uploads/classification_v2/full_t6_training_authority_20260817/full_t6_training_input_map_20260817.json"
with open(path) as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res)
