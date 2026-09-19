"""Write the paired legacy L6 full-frame-context short decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation import (
    legacy_development_l6_full_frame_context_decision as decision,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


def main() -> int:
    args = _parser().parse_args()
    output, payload = decision.write_full_frame_context_short_decision(
        args.config,
        project_root=args.project_root,
    )
    print(
        json.dumps(
            {
                "output_path": str(output),
                "status": payload["status"],
                "decision": payload["decision"]["decision"],
                "full_frame_context_expansion_authorized": payload[
                    "decision"
                ]["full_frame_context_expansion_authorized"],
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
