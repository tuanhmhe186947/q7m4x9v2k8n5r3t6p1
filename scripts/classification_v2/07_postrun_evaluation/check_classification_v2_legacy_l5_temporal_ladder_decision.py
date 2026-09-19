"""Create the immutable legacy L5 temporal-ladder decision artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation import (
    legacy_development_l5_temporal_ladder_decision as ladder_decision,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and compare eight immutable legacy L5 temporal controls."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    result = ladder_decision.evaluate_temporal_ladder_matrix_decision(
        args.config,
        project_root=root,
    )
    output_path = ladder_decision.configured_output_path(args.config, root)
    if not args.print_only:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
