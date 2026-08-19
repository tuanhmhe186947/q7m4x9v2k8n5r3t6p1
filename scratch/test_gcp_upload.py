import time
from pathlib import Path
import lightning_sdk

ts = lightning_sdk.Teamspace(name="pig-project", user="ironheart211224")

local_dir = Path("outputs/classification_v2/full_t6_union_r128_20260818")
test_file = local_dir / "packed_image_cache_audit.json"
remote_path = "classification_v2/full_t6_union_r128_20260818/packed_image_cache_audit.json"

print(f"Testing upload of {test_file.name} to {remote_path} with cloud_account='gcp-lightning-public-prod'...")
ts.upload_file(str(test_file), remote_path, cloud_account="gcp-lightning-public-prod", progress_bar=True)
print("Test upload succeeded!")
