import base64
import lightning_sdk

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

remote_script = """
import subprocess
import os

target_dir = "/teamspace/studios/this_studio/runtime_ea392e2d"
os.chdir(target_dir)

cmd = ["python3", "scripts/classification_v2/04_baselines_smokes/verify_m0_contract.py"]
env = os.environ.copy()
env["PYTHONPATH"] = os.path.join(target_dir, "src")

res = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("RETURNCODE:", res.returncode)
print("--- STDOUT ---")
print(res.stdout)
print("--- STDERR ---")
print(res.stderr)
assert res.returncode == 0, "verify_m0_contract failed!"
"""

b64_code = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{b64_code}\').decode(\'utf-8\'))"'

print("Executing M0 contract verification on Studio...")
res = studio.run(cmd)
print("=== REMOTE OUTPUT ===")
print(res.encode("ascii", errors="replace").decode("ascii"))
