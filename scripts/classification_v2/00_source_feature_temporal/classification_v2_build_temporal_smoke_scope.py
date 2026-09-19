"""Build a bounded legacy+CVAT scope from complete temporal scene blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.datasets.temporal_smoke_scope import (
    SUPPORTED_SOURCES,
    TemporalSmokeScopeConfig,
    select_temporal_smoke_scope,
)


def _parse_required_sources(value: str) -> tuple[str, ...]:
    """Parse an explicit comma-separated source contract."""

    sources = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(sources).difference(SUPPORTED_SOURCES))
    if not sources or unknown:
        raise argparse.ArgumentTypeError(
            f"required sources must be from {SUPPORTED_SOURCES}; got {value!r}"
        )
    return sources


def _parse_strict_boolean(value: str) -> bool:
    """Accept only explicit true/false claim values."""

    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError(
        "human review completion must be exactly true or false"
    )


def parse_args() -> argparse.Namespace:
    """Parse versioned smoke paths and source-unit contract settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--blocks-per-source", type=int, default=4)
    parser.add_argument("--cvat-label-stride", type=int, default=6)
    parser.add_argument(
        "--legacy-expected-sequence-length",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--required-sources",
        type=_parse_required_sources,
        default=SUPPORTED_SOURCES,
        help=(
            "Comma-separated sources required in this smoke scope. "
            "Use legacy_recovered for the isolated legacy lane."
        ),
    )
    parser.add_argument(
        "--lineage-scope",
        default=None,
        help=(
            "Optional nonblank artifact claim. It must be paired with "
            "--human-review-complete."
        ),
    )
    parser.add_argument(
        "--human-review-complete",
        type=_parse_strict_boolean,
        default=None,
        help=(
            "Optional strict true/false claim. It must be paired with "
            "--lineage-scope."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Write only derived smoke artifacts and fail closed on scope errors."""

    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    frames = pd.read_csv(args.input_csv, low_memory=False)
    selected, audit = select_temporal_smoke_scope(
        frames,
        config=TemporalSmokeScopeConfig(
            blocks_per_source=args.blocks_per_source,
            cvat_label_stride=args.cvat_label_stride,
            legacy_expected_sequence_length=(
                args.legacy_expected_sequence_length
            ),
            required_sources=args.required_sources,
            lineage_scope=args.lineage_scope,
            human_review_complete=args.human_review_complete,
        ),
    )
    audit["input_csv"] = str(args.input_csv)
    audit["output_csv"] = str(args.output_csv)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_csv, index=False)
    args.audit_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
