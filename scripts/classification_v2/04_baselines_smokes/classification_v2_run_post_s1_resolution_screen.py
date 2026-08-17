"""Run one authority-bound post-S1 T6 pure-resolution arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.post_s1_resolution_screening import (
    MAX_STEPS,
    create_resolution_plan,
    load_resolution_population,
    run_resolution_arm,
)


def parse_args() -> argparse.Namespace:
    """Parse one registered R64, R128, or R160 arm without overrides."""

    parser = argparse.ArgumentParser(description="Run one post-S1 T6 pure-spatial-resolution arm.")
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--base-stage1-authority", required=True, type=Path)
    parser.add_argument("--host-binding-path", required=True, type=Path)
    parser.add_argument("--canonical-code-sha", required=True)
    parser.add_argument("--rgb-source-root", required=True, type=Path)
    parser.add_argument("--runtime-input-authority", required=True, type=Path)
    parser.add_argument("--runtime-input-binding", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--outputs-root", required=True, type=Path)
    parser.add_argument("--stage1-data-bindings", required=True, type=Path)
    parser.add_argument("--stage1-binding-bundle", required=True, type=Path)
    parser.add_argument("--execution-permit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--input-resolution", required=True, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260804,
        choices=(20260804, 20260805, 20260806),
        help="Registered matched seed for resolution screening.",
    )
    parser.add_argument(
        "--confirmation-authority",
        type=Path,
        default=None,
        help="Optional confirmation retention authority for confirmation seeds.",
    )
    return parser.parse_args()


def main() -> None:
    """Fail before media access when an arm or endpoint diverges from authority."""

    args = parse_args()
    plan = create_resolution_plan(
        args.authority,
        repository_root=args.repository_root,
        outputs_root=args.outputs_root,
        stage1_data_bindings_path=args.stage1_data_bindings,
        stage1_binding_bundle_path=args.stage1_binding_bundle,
        execution_permit_path=args.execution_permit,
        base_stage1_authority_path=args.base_stage1_authority,
        host_binding_path=args.host_binding_path,
        canonical_code_sha=args.canonical_code_sha,
        rgb_source_root=args.rgb_source_root,
        runtime_input_authority_path=args.runtime_input_authority,
        runtime_input_binding_path=args.runtime_input_binding,
        media_root=args.media_root,
        output_dir=args.output_dir,
        trial_id=args.trial_id,
        input_resolution=args.input_resolution,
        device_name=args.device,
        seed=args.seed,
        confirmation_authority_path=args.confirmation_authority,
    )
    population = load_resolution_population(plan)
    result = run_resolution_arm(plan, population, steps=args.steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
