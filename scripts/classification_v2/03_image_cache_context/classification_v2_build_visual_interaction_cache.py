from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.datasets.visual_interaction_context import (
    VisualInteractionCacheConfig,
    build_visual_interaction_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable actor-partner visual context cache.")
    parser.add_argument(
        "--frame-context-csv",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/image_frame_context_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_v2/visual_interaction_cache"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--padding-ratio", type=float, default=0.15)
    parser.add_argument("--max-contexts", type=int, default=None)
    parser.add_argument("--source-type", default=None)
    parser.add_argument("--preview-limit", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    audit = build_visual_interaction_cache(VisualInteractionCacheConfig(**vars(args)))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
