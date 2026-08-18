import json
import shutil
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path("src").resolve()))

from pig_behavior.classification_v2.datasets.visual_interaction_context import (
    VisualInteractionCacheConfig,
    build_visual_interaction_cache,
)

frame_ctx_p = Path("outputs/classification_v2/image_context_v2/image_frame_context_manifest.csv")
sel_p = Path("outputs/classification_v2/full_t6_union_r128_20260818/full_t6_union_selection.csv")
out_dir = Path("outputs/classification_v2/full_t6_union_r128_20260818")

assert frame_ctx_p.exists(), f"Frame context manifest not found: {frame_ctx_p}"
assert sel_p.exists(), f"Selection CSV not found: {sel_p}"

config = VisualInteractionCacheConfig(
    frame_context_csv=frame_ctx_p,
    output_dir=out_dir,
    selection_csv=sel_p,
    image_size=128,
    padding_ratio=0.15,
    max_contexts=None,
    checkpoint_every=5000,
    resume=True,
)

print(f"Starting FULL-T6 R128 Union Cache Build...")
print(f"  Frame Context Manifest: {frame_ctx_p}")
print(f"  Selection CSV: {sel_p}")
print(f"  Output Dir: {out_dir}")
print(f"  Image Size: 128x128")
print(f"  Padding Ratio: 0.15")

t0 = time.perf_counter()
audit = build_visual_interaction_cache(config)
elapsed = time.perf_counter() - t0

print(f"\nFULL-T6 Union Cache Build Completed in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")
print("Audit result summary:")
print(json.dumps(audit, indent=2))
