from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l6_social_relation_cache import (
    configured_repeat_output_path,
    evaluate_social_relation_cache_repeat,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = evaluate_social_relation_cache_repeat(
        args.config,
        project_root=root,
    )
    output = configured_repeat_output_path(args.config, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({**result, "output_path": str(output)}, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
