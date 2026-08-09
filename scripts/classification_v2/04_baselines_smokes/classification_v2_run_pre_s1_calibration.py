"""Launch only the authority-bound PRE-S1 calibration route.

This is deliberately not a generic trainer.  It has no role, model, temporal,
optimizer, batch-size, scheduler, metric, or output-root override flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.pre_s1_calibration import (
    create_calibration_plan,
    load_canonical_population,
    preflight_calibration,
    run_pre_s1_calibration,
)


def parse_args() -> argparse.Namespace:
    """Parse only non-scientific launch identity and resolved data locations."""

    parser = argparse.ArgumentParser(description="Run the PRE-S1 calibration only.")
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--data-bindings", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--execution-authorization", required=True)
    return parser.parse_args()


def main() -> None:
    """Resolve and run the immutable calibration contract."""

    args = parse_args()
    if args.execution_authorization != "PRE_S1_CALIBRATION_AUTHORIZED":
        raise SystemExit("PRE-S1 calibration requires its exact execution authorization token")
    plan = create_calibration_plan(
        args.authority,
        outputs_root=args.outputs_root.resolve(),
        run_id=args.run_id,
        device_name=args.device,
        data_bindings_path=args.data_bindings,
    )
    hashes = preflight_calibration(plan)
    population = load_canonical_population(plan, hashes)
    report = run_pre_s1_calibration(
        plan,
        population,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
