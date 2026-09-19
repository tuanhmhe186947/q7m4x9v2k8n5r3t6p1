from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.runtime_benchmark import write_runtime_benchmark


def main() -> None:
    """Summarize matched CUDA pilots and emit an audited runtime recommendation."""

    parser = argparse.ArgumentParser(description="Summarize classification_v2 CUDA runtime pilots.")
    parser.add_argument("audit_json", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-reserved-memory-mb",
        type=float,
        default=3000.0,
        help="Reject candidates above this measured CUDA reserved-memory budget.",
    )
    args = parser.parse_args()
    result = write_runtime_benchmark(
        args.audit_json,
        args.output_dir,
        max_reserved_memory_mb=args.max_reserved_memory_mb,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
