import lightning_sdk
from lightning_sdk import Machine

studio = lightning_sdk.Studio(
    name="training-pig-project-L4",
    teamspace="pig-project",
    user="ironheart211224",
)

print(f"Current studio status: {studio.status}")
print("Starting studio on Machine.CPU...")
studio.start(machine=Machine.CPU)
print(f"Studio started. Status: {studio.status}, Machine: {studio.machine}")

out = studio.run("""
ls -la /teamspace/studios/this_studio/full_t6_union_r128_20260818/
""")
print("=== UNION CACHE CHECK ===")
print(out)
