"""Freeze the promotion decision for the rebuild-bound C6 modality matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_c6_modality_promotion import (
    load_promotion_config,
    write_c6_modality_promotion_freeze,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_promotion_config(args.config)
    payload = write_c6_modality_promotion_freeze(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
