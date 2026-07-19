"""Evaluate paired C6 temporal controls and freeze the modality base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_c6_temporal_freeze import (
    evaluate_c6_temporal_freeze,
)
from pig_behavior.classification_v2.training.lineage_hashing import file_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260719)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output, payload = evaluate_c6_temporal_freeze(
        args.config,
        output_path=args.output,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary = {
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "status": payload["status"],
        "decision": payload["decision"],
        "selected_base_mode": payload["selected_base_mode"],
        "family_decisions": payload["family_decisions"],
        "valid": payload["valid"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
