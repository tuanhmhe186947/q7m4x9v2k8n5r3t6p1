"""Build a reusable seek index over the frozen Temporal-v2 frame authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.temporal_v2_consumer import (
    build_target_frame_offset_index,
    verify_registered_canonical_authority,
)


def parse_args() -> argparse.Namespace:
    """Require the I-2 hard-link mapping and a new task-owned index path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--canonical-mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Index byte offsets only; no frame features or media are regenerated."""

    args = parse_args()
    registered = verify_registered_canonical_authority(
        args.authority_root,
        mapping_path=args.canonical_mapping,
    )
    result = build_target_frame_offset_index(
        args.authority_root,
        output_path=args.output,
    )
    print(json.dumps({"registered_authority": registered, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
