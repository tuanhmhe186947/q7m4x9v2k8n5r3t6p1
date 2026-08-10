"""Run exactly one separately authorized NVIDIA-L4 Stage-1 temporal arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training import stage1_temporal_screening as stage1


def parse_args() -> argparse.Namespace:
    """Parse a single immutable Stage-1 arm without scientific overrides."""

    parser = argparse.ArgumentParser(
        description="Run one authority-bound Stage-1 temporal-screening arm."
    )
    parser.add_argument("--execution-permit", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--binding-bundle", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--view", choices=tuple(stage1.VIEW_SPECS), required=True)
    parser.add_argument("--data-bindings", type=Path, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    """Reject unapproved hardware/authority before opening any RGB payload."""

    args = parse_args()
    plan = stage1.create_stage1_plan(
        args.authority,
        view=args.view,
        repository_root=args.repository_root,
        outputs_root=args.outputs_root,
        trial_id=args.trial_id,
        device_name="cuda",
        data_bindings_path=args.data_bindings,
        execution_permit_path=args.execution_permit,
        binding_bundle_path=args.binding_bundle,
        allow_consumed_execution_permit=args.resume_checkpoint is not None,
        engineering_smoke=False,
        allow_existing_output=args.resume_checkpoint is not None,
    )
    hashes = stage1.preflight_stage1(plan)
    population = stage1.load_stage1_population(plan, hashes)
    result = stage1.run_stage1_temporal_screening(
        plan,
        population,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
