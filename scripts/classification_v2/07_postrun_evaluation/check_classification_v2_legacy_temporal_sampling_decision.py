"""Evaluate the hash-bound legacy C6/C8/S6 temporal sampling matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation import (
    legacy_development_temporal_sampling_decision as temporal_decision,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = _parser().parse_args()
    output, result = temporal_decision.write_temporal_sampling_decision(
        args.config,
        project_root=args.project_root,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "ranking": result["ranking"],
                "decision": result["decision"],
                "errors": result["errors"],
                "valid": result["valid"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
