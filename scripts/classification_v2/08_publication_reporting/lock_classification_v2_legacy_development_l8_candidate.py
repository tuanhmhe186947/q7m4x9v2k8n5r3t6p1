"""Audit and lock the bounded legacy_16f L8 development candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_development_l8_candidate_lock import (
    lock_legacy_l8_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run every audit without creating the immutable output directory.",
    )
    args = parser.parse_args()
    result = lock_legacy_l8_candidate(
        args.config,
        write_outputs=not args.check_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
