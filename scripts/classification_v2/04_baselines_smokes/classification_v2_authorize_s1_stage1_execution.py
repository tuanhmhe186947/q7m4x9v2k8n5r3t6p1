"""Issue only registered Stage-1 single-use permits after authority checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.lightning_resource_identity import (
    LightningResourceIdentityError,
    validate_passing_lightning_resource_preflight,
)
from pig_behavior.classification_v2.training.stage1_execution_authorization import (
    ROTATION_MODE_CODE_ONLY,
    ROTATION_MODES,
    Stage1ExecutionAuthorizationError,
    create_stage1_execution_permits,
    rotate_stage1_execution_permits,
)


def parse_args() -> argparse.Namespace:
    """Parse only execution-path inputs; scientific controls are immutable."""

    parser = argparse.ArgumentParser(
        description="Issue per-arm Stage-1 L4 execution permits."
    )
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--binding-bundle", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--ttl-hours", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--confirmation-authority", type=Path)
    parser.add_argument("--lightning-resource-contract", type=Path, required=True)
    parser.add_argument("--lightning-resource-preflight", type=Path, required=True)
    parser.add_argument(
        "--view",
        action="append",
        choices=("T6", "T8", "T12", "T16"),
        help="Required only to spell the exact retained confirmation subset.",
    )
    parser.add_argument(
        "--rotate-view",
        action="append",
        choices=("T6", "T8", "T12", "T16"),
        help="Replace only a code-stale permit; repeat for each selected view.",
    )
    parser.add_argument(
        "--supersession-reason",
        help="Required with --rotate-view; recorded in immutable permit lineage.",
    )
    parser.add_argument(
        "--rotation-mode",
        choices=ROTATION_MODES,
        default=ROTATION_MODE_CODE_ONLY,
        help="Exact fail-closed permit rotation mode; default remains CODE_ONLY.",
    )
    parser.add_argument(
        "--predecessor-binding-bundle",
        type=Path,
        help=(
            "Required only for CONFIRMATION_PROVENANCE_REPAIR_20260810; "
            "must hash-match the active predecessor permit."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Create exact per-view permits, or report a fail-closed reason."""

    args = parse_args()
    try:
        validate_passing_lightning_resource_preflight(
            contract_path=args.lightning_resource_contract,
            preflight_path=args.lightning_resource_preflight,
        )
        if args.rotate_view:
            if args.view:
                raise Stage1ExecutionAuthorizationError(
                    "--view cannot be combined with --rotate-view"
                )
            if args.supersession_reason is None:
                raise Stage1ExecutionAuthorizationError(
                    "--supersession-reason is required with --rotate-view"
                )
            rotations = rotate_stage1_execution_permits(
                repository_root=args.repository_root,
                outputs_root=args.outputs_root,
                authority_path=args.authority,
                binding_bundle_path=args.binding_bundle,
                views=args.rotate_view,
                reason=args.supersession_reason,
                seed=args.seed,
                confirmation_authority_path=args.confirmation_authority,
                rotation_mode=args.rotation_mode,
                predecessor_binding_bundle_path=(
                    args.predecessor_binding_bundle
                ),
                ttl_hours=args.ttl_hours,
            )
            print(
                json.dumps(
                    {
                        "status": "ROTATED",
                        "single_use": True,
                        "rotation_mode": args.rotation_mode,
                        "rotations": {
                            view: {
                                "path": str(rotation.replacement.path),
                                "sha256": rotation.replacement.sha256,
                                "trial_id": rotation.replacement.payload["trial_id"],
                                "superseded_path": str(rotation.superseded_path),
                                "supersession_record": str(rotation.record_path),
                            }
                            for view, rotation in rotations.items()
                        },
                    },
                    indent=2,
                )
            )
            return 0
        if args.supersession_reason is not None:
            raise Stage1ExecutionAuthorizationError(
                "--supersession-reason requires --rotate-view"
            )
        if args.rotation_mode != ROTATION_MODE_CODE_ONLY:
            raise Stage1ExecutionAuthorizationError(
                "--rotation-mode requires --rotate-view"
            )
        if args.predecessor_binding_bundle is not None:
            raise Stage1ExecutionAuthorizationError(
                "--predecessor-binding-bundle requires --rotate-view"
            )
        permits = create_stage1_execution_permits(
            repository_root=args.repository_root,
            outputs_root=args.outputs_root,
            authority_path=args.authority,
            binding_bundle_path=args.binding_bundle,
            seed=args.seed,
            confirmation_authority_path=args.confirmation_authority,
            views=args.view,
            ttl_hours=args.ttl_hours,
        )
    except (LightningResourceIdentityError, Stage1ExecutionAuthorizationError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": "AUTHORIZED",
                "single_use": True,
                "permits": {
                    view: {
                        "path": str(permit.path),
                        "sha256": permit.sha256,
                        "trial_id": permit.payload["trial_id"],
                    }
                    for view, permit in permits.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
