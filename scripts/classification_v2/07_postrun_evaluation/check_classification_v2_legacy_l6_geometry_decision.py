"""Write the paired native/video-cluster legacy L6 geometry decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_development_l6_geometry_decision import (
    write_geometry_decision,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    output, payload = write_geometry_decision(
        args.config,
        project_root=args.project_root,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": payload["status"],
                "decision": payload["decision"],
                "errors": payload["errors"],
                "valid": payload["valid"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
