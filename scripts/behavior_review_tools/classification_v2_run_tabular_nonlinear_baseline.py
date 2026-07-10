from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.experiments.tabular_nonlinear_baseline import (
    TabularNonlinearBaselineConfig,
    run_tabular_nonlinear_baseline,
)


def main() -> None:
    """CLI entrypoint for B2: run the nonlinear tabular whitelist ablation."""

    parser = argparse.ArgumentParser(description="Run classification_v2 B2 nonlinear tabular whitelist baseline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/tabular_nonlinear_baseline"),
    )
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--bootstrap-iterations", type=int, default=100)
    args = parser.parse_args()
    result = run_tabular_nonlinear_baseline(
        TabularNonlinearBaselineConfig(
            output_dir=args.output_dir,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    )
    print(json.dumps(result["audit"], indent=2))


if __name__ == "__main__":
    main()
