"""Write the hash-bound legacy L6 pen-context short decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_development_l6_pen_context_decision import (
    write_pen_context_short_decision,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = _parser().parse_args()
    output, payload = write_pen_context_short_decision(
        args.config,
        project_root=args.project_root,
    )
    summary = {
        "output": str(output),
        "status": payload["status"],
        "decision": payload["decision"]["decision"],
        "full_pen_context_expansion_authorized": payload["decision"][
            "full_pen_context_expansion_authorized"
        ],
        "errors": payload["errors"],
        "valid": payload["valid"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
