from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.experiments.tabular_linear_baseline import (
    TabularLinearBaselineConfig,
    run_tabular_linear_baseline,
)


def main() -> None:
    """CLI entrypoint for B1: run a leakage-safe linear tabular ablation."""

    parser = argparse.ArgumentParser(description="Run classification_v2 B1 linear tabular whitelist baseline.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/classification_v2/model_smoke/tabular_linear_baseline")
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--bootstrap-iterations", type=int, default=100)
    args = parser.parse_args()
    result = run_tabular_linear_baseline(
        TabularLinearBaselineConfig(
            output_dir=args.output_dir,
            max_iter=args.max_iter,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    )
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
