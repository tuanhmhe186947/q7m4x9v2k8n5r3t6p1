"""Build isolated L5 frame-feature caches or audit their short repeat gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.legacy_development_l5 import (
    LegacyL5Config,
    load_legacy_l5_config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_feature_cache import (
    DEFAULT_CHECKPOINT_EVERY_ROWS,
    FEATURE_CONTROL_IDS,
    audit_legacy_l5_feature_preflight,
    audit_legacy_l5_feature_short_gate,
    build_legacy_l5_feature_cache,
    load_legacy_l5_feature_parents,
    write_legacy_l5_feature_short_gate,
)


def main() -> None:
    parser = _argument_parser()
    args = parser.parse_args()
    config = load_legacy_l5_config(args.config_json)
    if args.mode == "preflight":
        result = _build_preflight(args, config=config)
    elif args.mode == "build_cache":
        result = _build_cache(args, config=config)
    else:
        result = _build_short_gate(args, config=config)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(2)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run crash-bounded legacy-only L5 feature-cache gates."
    )
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("preflight", "build_cache", "short_gate"),
        required=True,
    )
    parser.add_argument("--control-id", choices=FEATURE_CONTROL_IDS)
    parser.add_argument("--scope", choices=("short", "full"))
    parser.add_argument("--run-id")
    parser.add_argument("--readiness-audit-json", type=Path)
    parser.add_argument("--short-cache-audit-json", type=Path)
    parser.add_argument("--full-cache-audit-json", type=Path)
    parser.add_argument("--weights-audit-json", type=Path)
    parser.add_argument("--vram-probe-audit-json", type=Path)
    parser.add_argument("--short-gate-audit-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint-every-rows",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY_ROWS,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--primary-result", action="append", default=[])
    parser.add_argument("--repeat-result", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    return parser


def _build_preflight(
    args: argparse.Namespace,
    *,
    config: LegacyL5Config,
) -> dict[str, Any]:
    required = (
        "readiness_audit_json",
        "short_cache_audit_json",
        "full_cache_audit_json",
        "weights_audit_json",
        "vram_probe_audit_json",
    )
    _require_arguments(args, required, mode="preflight")
    forbidden = (
        "control_id",
        "scope",
        "run_id",
        "short_gate_audit_json",
        "output_json",
    )
    if any(getattr(args, name) is not None for name in forbidden) or args.resume:
        raise ValueError("preflight accepts only immutable parent audits")
    parents = load_legacy_l5_feature_parents(
        config,
        readiness_audit_path=args.readiness_audit_json,
        short_cache_audit_path=args.short_cache_audit_json,
        full_cache_audit_path=args.full_cache_audit_json,
        weights_audit_path=args.weights_audit_json,
        vram_probe_audit_path=args.vram_probe_audit_json,
    )
    return audit_legacy_l5_feature_preflight(config, parents=parents)


def _build_cache(
    args: argparse.Namespace,
    *,
    config: LegacyL5Config,
) -> dict[str, Any]:
    required = (
        "control_id",
        "scope",
        "run_id",
        "readiness_audit_json",
        "short_cache_audit_json",
        "full_cache_audit_json",
        "weights_audit_json",
        "vram_probe_audit_json",
    )
    _require_arguments(args, required, mode="build_cache")
    if args.primary_result or args.repeat_result or args.output_json is not None:
        raise ValueError("build_cache does not accept short-gate result arguments")
    parents = load_legacy_l5_feature_parents(
        config,
        readiness_audit_path=args.readiness_audit_json,
        short_cache_audit_path=args.short_cache_audit_json,
        full_cache_audit_path=args.full_cache_audit_json,
        weights_audit_path=args.weights_audit_json,
        vram_probe_audit_path=args.vram_probe_audit_json,
    )
    return build_legacy_l5_feature_cache(
        config,
        control_id=args.control_id,
        scope=args.scope,
        run_id=args.run_id,
        output_dir=config.l5_output_root / args.run_id,
        parents=parents,
        device_name=args.device,
        checkpoint_every_rows=args.checkpoint_every_rows,
        resume=args.resume,
        short_gate_audit_path=args.short_gate_audit_json,
    )


def _build_short_gate(
    args: argparse.Namespace,
    *,
    config: LegacyL5Config,
) -> dict[str, Any]:
    _require_arguments(args, ("output_json",), mode="short_gate")
    forbidden = (
        "control_id",
        "scope",
        "run_id",
        "readiness_audit_json",
        "short_cache_audit_json",
        "full_cache_audit_json",
        "weights_audit_json",
        "vram_probe_audit_json",
        "short_gate_audit_json",
    )
    if any(getattr(args, name) is not None for name in forbidden) or args.resume:
        raise ValueError("short_gate accepts only paired results and output JSON")
    primary = _parse_control_paths(args.primary_result, name="primary")
    repeat = _parse_control_paths(args.repeat_result, name="repeat")
    result = audit_legacy_l5_feature_short_gate(
        config,
        primary_result_paths=primary,
        repeat_result_paths=repeat,
    )
    write_legacy_l5_feature_short_gate(
        config,
        output_path=args.output_json,
        payload=result,
    )
    return result


def _parse_control_paths(values: list[str], *, name: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        control_id, separator, raw_path = value.partition("=")
        if not separator or control_id not in FEATURE_CONTROL_IDS or not raw_path:
            raise ValueError(f"invalid {name} result binding: {value}")
        if control_id in parsed:
            raise ValueError(f"duplicate {name} result control: {control_id}")
        parsed[control_id] = Path(raw_path)
    missing = [value for value in FEATURE_CONTROL_IDS if value not in parsed]
    if missing:
        raise ValueError(f"missing {name} result controls: {missing}")
    return {control_id: parsed[control_id] for control_id in FEATURE_CONTROL_IDS}


def _require_arguments(
    args: argparse.Namespace,
    names: tuple[str, ...],
    *,
    mode: str,
) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise ValueError(f"{mode} missing required arguments: {missing}")


if __name__ == "__main__":
    main()
