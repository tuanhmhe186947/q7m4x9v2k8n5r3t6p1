"""Write the bounded legacy_16f L0-L8 goal completion handback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_development_goal_completion import (
    write_legacy_goal_completion_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run every L0-L8 audit without writing the handback artifact.",
    )
    args = parser.parse_args()
    result = write_legacy_goal_completion_audit(
        args.config,
        write_output=not args.check_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
