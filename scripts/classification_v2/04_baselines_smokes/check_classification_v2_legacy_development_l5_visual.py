"""Run pretrained-weight preparation or the bounded L5 VRAM probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.legacy_development_l5 import (
    load_legacy_l5_config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_visual import (
    prepare_legacy_l5_pretrained_weights,
    run_legacy_l5_vram_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run legacy-only L5 pretrained visual gates."
    )
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("prepare_weights", "vram_probe"),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--readiness-audit-json", type=Path)
    parser.add_argument("--full-cache-audit-json", type=Path)
    parser.add_argument("--weights-audit-json", type=Path)
    parser.add_argument("--weight-cache-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_json.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists; pass --overwrite explicitly: {args.output_json}"
        )
    config = load_legacy_l5_config(args.config_json)
    if args.mode == "prepare_weights":
        if args.readiness_audit_json is None or args.weight_cache_root is None:
            raise ValueError(
                "prepare_weights requires readiness audit and weight cache root"
            )
        if args.full_cache_audit_json is not None:
            raise ValueError("prepare_weights does not accept a full-cache audit")
        if args.weights_audit_json is not None:
            raise ValueError("prepare_weights does not accept a weights audit")
        result = prepare_legacy_l5_pretrained_weights(
            config,
            readiness_audit_path=args.readiness_audit_json,
            weight_cache_root=args.weight_cache_root,
            allow_download=args.allow_download,
        )
    else:
        if args.full_cache_audit_json is None or args.weights_audit_json is None:
            raise ValueError(
                "vram_probe requires full-cache and pretrained-weight audits"
            )
        if args.allow_download:
            raise ValueError("vram_probe forbids network download")
        if args.weight_cache_root is not None:
            raise ValueError("vram_probe reads the cache root from its audit")
        result = run_legacy_l5_vram_probe(
            config,
            full_cache_audit_path=args.full_cache_audit_json,
            weights_audit_path=args.weights_audit_json,
            device_name=args.device,
        )
    _write_json_atomic(args.output_json, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit(2)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
