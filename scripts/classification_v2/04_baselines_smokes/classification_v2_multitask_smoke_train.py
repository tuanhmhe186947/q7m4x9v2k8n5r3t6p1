from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.multitask_smoke import (
    MultitaskSmokeConfig,
    run_multitask_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded classification_v2 multitask trainability smoke.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/classification_v2/model_smoke/multitask_visual_v3")
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    audit = run_multitask_smoke(
        MultitaskSmokeConfig(
            output_dir=args.output_dir,
            steps=args.steps,
            per_class=args.per_class,
            device=args.device,
        )
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
