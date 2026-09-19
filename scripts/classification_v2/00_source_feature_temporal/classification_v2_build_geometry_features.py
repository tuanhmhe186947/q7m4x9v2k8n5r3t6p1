"""Build geometry features from policy-normalized frame objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.features.geometry import (
    build_geometry_features,
    validate_geometry_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build geometry features for classification_v2."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived geometry artifacts explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = [args.output_csv]
    if args.audit_json is not None:
        output_paths.append(args.audit_json)
    require_output_paths_available(
        output_paths,
        overwrite=args.overwrite,
    )

    print(f"reading: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)

    print("building geometry features...")
    out = build_geometry_features(df)

    audit = validate_geometry_features(out)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if audit["errors"]:
        raise ValueError(f"Geometry audit errors: {audit['errors']}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"saved csv: {args.output_csv}")

    if args.audit_json is not None:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved audit: {args.audit_json}")


if __name__ == "__main__":
    main()
