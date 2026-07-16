"""Preflight, run, and close the legacy L7 imbalance-policy matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.imbalance_losses import LOSS_POLICIES
from pig_behavior.classification_v2.training.legacy_development_l7_imbalance_config import (
    load_l7_imbalance_config,
    preflight_l7_imbalance_policy,
)
from pig_behavior.classification_v2.training.legacy_development_l7_imbalance_runtime import (
    audit_l7_imbalance_run,
    run_l7_imbalance_policy,
    write_l7_imbalance_repeat_gate,
    write_l7_imbalance_short_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--policy", choices=LOSS_POLICIES, required=True)
        if name == "run":
            child.add_argument("--run-id", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--result", type=Path, required=True)
    repeat = subparsers.add_parser("repeat")
    repeat.add_argument("--config", type=Path, required=True)
    repeat.add_argument("--policy", choices=LOSS_POLICIES, required=True)
    repeat.add_argument("--primary", type=Path, required=True)
    repeat.add_argument("--repeat", type=Path, required=True)
    repeat.add_argument("--output", type=Path, required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--config", type=Path, required=True)
    matrix.add_argument(
        "--gate",
        action="append",
        required=True,
        help="policy=repeat_gate_path; repeat three times",
    )
    matrix.add_argument("--output", type=Path, required=True)
    return parser


def _gate_mapping(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        policy, separator, path = value.partition("=")
        if not separator or policy not in LOSS_POLICIES or not path:
            raise ValueError(f"invalid --gate value={value!r}")
        if policy in mapping:
            raise ValueError(f"duplicate --gate policy={policy}")
        mapping[policy] = Path(path)
    return mapping


def main() -> int:
    args = _parser().parse_args()
    config = load_l7_imbalance_config(args.config)
    if args.command == "preflight":
        payload = preflight_l7_imbalance_policy(config, args.policy)
    elif args.command == "run":
        payload = run_l7_imbalance_policy(
            config,
            policy=args.policy,
            run_id=args.run_id,
        )
    elif args.command == "audit":
        payload = audit_l7_imbalance_run(config, result_path=args.result)
    elif args.command == "repeat":
        payload = write_l7_imbalance_repeat_gate(
            config,
            policy=args.policy,
            primary_result_path=args.primary,
            repeat_result_path=args.repeat,
            output_path=args.output,
        )
    else:
        gates = _gate_mapping(args.gate)
        if set(gates) != set(LOSS_POLICIES):
            raise ValueError("matrix requires exactly one gate per policy")
        payload = write_l7_imbalance_short_matrix(
            config,
            repeat_gate_paths=gates,
            output_path=args.output,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
