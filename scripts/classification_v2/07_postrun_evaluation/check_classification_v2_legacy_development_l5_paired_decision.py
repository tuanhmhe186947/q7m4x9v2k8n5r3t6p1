"""Create the immutable legacy L5 T1-versus-V1 paired decision artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_development_l5_paired_decision import (
    configured_output_path,
    evaluate_legacy_l5_paired_decision,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and compare immutable legacy L5 temporal controls."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/classification_v2/"
            "legacy_development_l5_t1_v1_paired_decision_v1.json"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    result = evaluate_legacy_l5_paired_decision(
        args.config,
        project_root=root,
    )
    output_path = configured_output_path(args.config, root)
    if not args.print_only:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"comparison artifact already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
