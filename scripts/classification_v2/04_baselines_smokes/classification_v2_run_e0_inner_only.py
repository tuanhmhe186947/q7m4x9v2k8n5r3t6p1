"""Resolve or execute the frozen inner-only B3/T6 E0 contract.

The ``train`` mode is intentionally locked behind a separate external
authorization string. Phase 2B uses only ``inspect``, ``assert-outer-blocked``,
and the bounded CPU ``smoke`` mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.e0_inner_only import (
    E0ContractError,
    assert_e0_role_permitted,
    inspect_e0_execution_authority,
    load_e0_execution_authority,
    resolve_e0_data_paths,
    run_e0_inner_only_training,
    run_e0_local_smoke,
    verify_e0_execution_sources,
)

DEFAULT_AUTHORITY = Path(
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2/e0_execution_authority.json"
)
EXECUTION_AUTHORIZATION = "REQUIRED"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed E0 B3 T6 FOLD_3 inner-only launcher."
    )
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--data-bindings", type=Path)
    parser.add_argument("--use-authority-local-paths", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--mode",
        choices=("inspect", "assert-outer-blocked", "smoke", "train"),
        default="inspect",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=1)
    parser.add_argument("--execution-authorization")
    args = parser.parse_args()

    authority = load_e0_execution_authority(args.authority)
    verify_e0_execution_sources(
        authority,
        repository_root=Path(__file__).resolve().parents[3],
    )
    if args.mode == "inspect":
        _write_report(args.report, _inspection_report(args.authority))
        return
    if args.mode == "assert-outer-blocked":
        _assert_outer_test_blocked(authority)
        _write_report(args.report, {"outer_test_access": "BLOCKED", "valid": True})
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for smoke or train mode")
    paths = resolve_e0_data_paths(
        authority,
        bindings_path=args.data_bindings,
        use_authority_local_paths=args.use_authority_local_paths,
    )
    if args.mode == "smoke":
        if args.device != "cpu":
            parser.error("Phase 2B smoke mode is CPU-only")
        report = run_e0_local_smoke(
            args.authority,
            paths=paths,
            output_dir=args.output_dir,
            device_name=args.device,
            train_steps=args.smoke_steps,
        )
        _write_report(args.report, report)
        return
    if args.execution_authorization != EXECUTION_AUTHORIZATION:
        raise E0ContractError(
            "train mode requires a separate Phase-3 --execution-authorization REQUIRED"
        )
    report = run_e0_inner_only_training(
        args.authority,
        paths=paths,
        output_dir=args.output_dir,
        device_name=args.device,
        resume_checkpoint=args.resume_checkpoint,
    )
    _write_report(args.report, report)


def _inspection_report(authority_path: Path) -> dict[str, Any]:
    report = inspect_e0_execution_authority(authority_path)
    report.update(
        {
            "schema_version": "classification_v2.e0_execution_resolution.v1",
            "wrapper": "classification_v2_run_e0_inner_only.py",
            "e0_launch_uses_variant_full": False,
            "outer_test_access": "BLOCKED",
            "valid": True,
        }
    )
    return report


def _assert_outer_test_blocked(authority: dict[str, Any]) -> None:
    forbidden = authority["outer_test_prohibition"]
    if any(forbidden.values()):
        raise E0ContractError("E0 authority permits an outer-test operation")
    try:
        assert_e0_role_permitted("test")
    except E0ContractError:
        return
    raise E0ContractError("E0 failed to block its held-out test role")


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
