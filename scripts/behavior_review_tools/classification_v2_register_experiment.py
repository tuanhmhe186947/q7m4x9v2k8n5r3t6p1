from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.experiments.registry import ExperimentRecordConfig, write_experiment_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a classification_v2 experiment/smoke run.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_v2/experiment_registry"))
    parser.add_argument("--notes", default="")
    parser.add_argument("--max-hash-bytes", type=int, default=100_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = write_experiment_record(
        ExperimentRecordConfig(
            name=args.name,
            output_dir=args.output_dir,
            metrics_json=args.metrics_json,
            artifacts=tuple(args.artifact),
            notes=args.notes,
            max_hash_bytes=args.max_hash_bytes,
        )
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))
    missing = [artifact["path"] for artifact in record["artifacts"] if not artifact["exists"]]
    if missing:
        raise SystemExit(f"Missing registered artifacts: {missing}")


if __name__ == "__main__":
    main()
