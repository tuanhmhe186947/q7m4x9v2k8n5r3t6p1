"""Build the hash-bound Stage A temporal-base transfer decision packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation import (
    legacy_development_temporal_base_selection_decision as base_decision,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = _parser().parse_args()
    output, result = base_decision.write_temporal_base_decision(
        args.config,
        project_root=args.project_root,
    )
    actions = {
        pair_id: comparison["transfer_decision"]["screening_action"]
        for pair_id, comparison in result["paired_comparisons"].items()
    }
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "transfer_actions": actions,
                "candidate_packet": result["full_data_candidate_packet"],
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
