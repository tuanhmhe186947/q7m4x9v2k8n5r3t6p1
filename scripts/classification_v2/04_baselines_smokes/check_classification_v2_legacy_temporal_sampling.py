"""Run the controlled legacy-16f contiguous versus uniform sampling matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_temporal_sampling import (
    FULL_SCOPE,
    VIEW_SPECS,
    TemporalSamplingConfig,
    audit_temporal_sampling_short_matrix,
    execute_temporal_sampling_run,
    load_temporal_sampling_config,
    preflight_temporal_sampling,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run-view")
    actions.add_argument("--audit-short-matrix", action="store_true")
    parser.add_argument("--repeat-id")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_temporal_sampling_config(args.config)
    full_launch_gate = _validate_full_launch_gate(config)
    if args.preflight:
        payload = preflight_temporal_sampling(config)
        if full_launch_gate is not None:
            payload["full_launch_gate"] = full_launch_gate
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["valid"] else 1
    if args.audit_short_matrix:
        output, payload = audit_temporal_sampling_short_matrix(config)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": payload["status"],
                    "full_confirmation_authorized": payload[
                        "full_confirmation_authorized"
                    ],
                    "errors": payload["errors"],
                    "valid": payload["valid"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if payload["valid"] else 1
    if not args.repeat_id:
        raise ValueError("--repeat-id is required with --run-view")
    output, payload = execute_temporal_sampling_run(
        config,
        str(args.run_view),
        str(args.repeat_id),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": payload["status"],
                "view_id": payload["view_id"],
                "repeat_id": payload["repeat_id"],
                "metrics": payload["metrics"],
                "errors": payload["errors"],
                "valid": payload["valid"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["valid"] else 1


def _validate_full_launch_gate(
    config: TemporalSamplingConfig,
) -> dict[str, object] | None:
    if config.training_scope != FULL_SCOPE:
        return None
    contract = config.payload["experiment_contract"]
    required = (
        "short_gate_path",
        "short_gate_sha256",
        "short_gate_status",
        "short_config_sha256",
        "launch_script_path",
        "launch_script_sha256",
    )
    missing = [field for field in required if field not in contract]
    if missing:
        raise ValueError(f"full temporal sampling gate fields missing={missing}")
    path = config.repo_root / str(contract["short_gate_path"])
    if not path.is_file():
        raise FileNotFoundError(f"short temporal sampling gate missing: {path}")
    observed_hash = file_sha256(path)
    if observed_hash != contract["short_gate_sha256"]:
        raise ValueError("short temporal sampling gate hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_status = str(contract["short_gate_status"])
    if payload.get("status") != expected_status:
        raise ValueError("short temporal sampling gate status drift")
    if payload.get("config_sha256") != contract["short_config_sha256"]:
        raise ValueError("short temporal sampling config hash drift")
    if payload.get("full_confirmation_authorized") is not True:
        raise ValueError("short gate does not authorize full confirmation")
    if payload.get("valid") is not True or payload.get("errors") != []:
        raise ValueError("short temporal sampling gate is invalid")
    if set(payload.get("views", {})) != set(VIEW_SPECS):
        raise ValueError("short temporal sampling view set drift")
    launch_path = config.repo_root / str(contract["launch_script_path"])
    if not launch_path.is_file():
        raise FileNotFoundError(f"temporal sampling launcher missing: {launch_path}")
    launch_hash = file_sha256(launch_path)
    if launch_hash != contract["launch_script_sha256"]:
        raise ValueError("temporal sampling launcher hash drift")
    return {
        "path": str(path.resolve()),
        "sha256": observed_hash,
        "status": expected_status,
        "short_config_sha256": contract["short_config_sha256"],
        "launch_script_sha256": launch_hash,
        "full_confirmation_authorized": True,
        "valid": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
