"""Create reusable raw-byte identity evidence for the PRE-S1 RGB source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.pre_s1_rgb_binding import (
    build_rgb_source_integrity_evidence,
)


def parse_args() -> argparse.Namespace:
    """Accept only source locations and the exact source-hash authorization."""

    parser = argparse.ArgumentParser(
        description="Hash existing PRE-S1 RGB source artifacts without media rebuild."
    )
    parser.add_argument("--rgb-source-root", type=Path, required=True)
    parser.add_argument("--input-parity-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-authorization", required=True)
    return parser.parse_args()


def main() -> None:
    """Hash source artifacts once for a later inner-only binding materialization."""

    args = parse_args()
    if args.execution_authorization != "PRE_S1_RGB_SOURCE_HASH_AUTHORIZED":
        raise SystemExit("PRE-S1 RGB source hashing requires its exact authorization token")
    parity = json.loads(args.input_parity_evidence.read_text(encoding="utf-8"))
    report = build_rgb_source_integrity_evidence(
        rgb_source_root=args.rgb_source_root,
        output_path=args.output,
        input_parity_evidence=parity,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
