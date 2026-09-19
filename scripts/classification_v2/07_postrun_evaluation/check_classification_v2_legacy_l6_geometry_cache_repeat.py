"""Emit the immutable legacy L6 geometry-cache repeat gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_development_l6_geometry_cache_repeat import (
    configured_output_path,
    evaluate_geometry_cache_repeat,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two exact legacy_16f T6 geometry caches."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    result = evaluate_geometry_cache_repeat(args.config, project_root=root)
    output = configured_output_path(args.config, root)
    if not args.print_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
