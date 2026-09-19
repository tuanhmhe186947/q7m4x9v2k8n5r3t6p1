from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.datasets.visual_interaction_selection import (
    build_visual_interaction_short_selection,
    load_visual_interaction_selection_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the exact legacy_16f short union-context targets."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_visual_interaction_selection_config(args.config)
    audit = build_visual_interaction_short_selection(config)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
