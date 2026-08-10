"""Resolve a registered remote input locator before any resolution-screen media read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.remote_input_resolution import (
    load_remote_input_authority,
    resolve_remote_input_root,
)


def main() -> None:
    """Emit the runtime locator separately from the scientific input identity."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binding = resolve_remote_input_root(
        load_remote_input_authority(args.authority),
        parity_report_path=args.parity_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(binding.as_dict(), indent=2), encoding="utf-8")
    print(json.dumps(binding.as_dict(), indent=2))


if __name__ == "__main__":
    main()
