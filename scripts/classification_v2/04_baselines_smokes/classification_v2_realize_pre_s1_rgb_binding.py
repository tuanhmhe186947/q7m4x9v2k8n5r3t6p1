"""Create a local or remote path realization for one scientific RGB binding."""

from __future__ import annotations

import argparse
from pathlib import Path

from pig_behavior.classification_v2.training.pre_s1_rgb_binding import (
    write_execution_path_realization,
)


def parse_args() -> argparse.Namespace:
    """Accept no scientific inputs; the binding identity is already frozen."""

    parser = argparse.ArgumentParser(
        description="Realize an existing PRE-S1 scientific RGB binding path."
    )
    parser.add_argument("--binding-dir", type=Path, required=True)
    parser.add_argument("--packed-cache-path", type=Path, required=True)
    parser.add_argument("--filename", default="pre_s1_calibration_data_bindings.json")
    parser.add_argument("--execution-authorization", required=True)
    return parser.parse_args()


def main() -> None:
    """Write only a machine-specific packed-cache location."""

    args = parse_args()
    if args.execution_authorization != "PRE_S1_RGB_REALIZATION_AUTHORIZED":
        raise SystemExit("PRE-S1 RGB realization requires its exact authorization token")
    binding_dir = args.binding_dir.resolve()
    result = write_execution_path_realization(
        output_dir=binding_dir,
        scientific_binding_path=binding_dir / "scientific_rgb_binding.json",
        packed_cache_path=args.packed_cache_path,
        filename=args.filename,
    )
    print(result)


if __name__ == "__main__":
    main()
