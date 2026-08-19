import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path("src").resolve()))

from pig_behavior.classification_v2.datasets.visual_interaction_context import (
    VisualInteractionCacheConfig,
    build_visual_interaction_cache,
)

frame_ctx_p = Path("outputs/classification_v2/image_context_v2/image_frame_context_manifest.csv")
sel_p = Path("outputs/classification_v2/full_t6_union_r128_20260818/full_t6_union_selection.csv")
test_out_dir = Path("outputs/classification_v2/test_union_10")

config = VisualInteractionCacheConfig(
    frame_context_csv=frame_ctx_p,
    output_dir=test_out_dir,
    selection_csv=sel_p,
    image_size=128,
    padding_ratio=0.15,
    max_contexts=10,
    resume=False,
)

print("Running test build_visual_interaction_cache on 10 contexts...")
audit = build_visual_interaction_cache(config)
print("Audit result:")
print(audit)
