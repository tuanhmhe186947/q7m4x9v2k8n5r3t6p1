"""Write the hash-bound full-development ROI confirmation decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation import (
    legacy_development_l6_roi_relation_full_decision as decision,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    output, payload = decision.write_roi_relation_full_decision(
        args.config,
        project_root=args.project_root.resolve(),
    )
    print(json.dumps({**payload, "output_path": str(output)}, indent=2))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
