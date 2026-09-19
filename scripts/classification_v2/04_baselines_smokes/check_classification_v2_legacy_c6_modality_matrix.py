"""Static, synthetic, and future authorized C6 modality matrix commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_c6_modality_matrix import (
    build_c6_modality_cache,
    evaluate_c6_short_matrix,
    load_c6_matrix_config,
    run_c6_repeat,
    static_c6_matrix_preflight,
    synthetic_c6_functional_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit or execute the frozen legacy C6 modality matrix.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--action",
        required=True,
        choices=(
            "static-preflight",
            "synthetic-preflight",
            "build-cache",
            "run-repeat",
            "evaluate",
        ),
    )
    parser.add_argument("--repeat-id", default="")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.action == "synthetic-preflight":
        if args.config is None:
            result = synthetic_c6_functional_preflight()
        else:
            config = load_c6_matrix_config(args.config)
            result = synthetic_c6_functional_preflight(
                experiment_family=str(
                    config.payload["experiment_contract"][
                        "changed_scientific_family"
                    ]
                ),
                modalities=tuple(config.payload["matrix"]["modalities"]),
            )
    else:
        if args.config is None:
            parser.error("--config is required for this action")
        config = load_c6_matrix_config(args.config)
        if args.action == "static-preflight":
            result = static_c6_matrix_preflight(config)
        elif args.action == "build-cache":
            result = build_c6_modality_cache(config)
        elif args.action == "run-repeat":
            if not args.repeat_id:
                parser.error("--repeat-id is required for run-repeat")
            result = run_c6_repeat(config, args.repeat_id)
        else:
            result = evaluate_c6_short_matrix(config)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if result.get("valid", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
