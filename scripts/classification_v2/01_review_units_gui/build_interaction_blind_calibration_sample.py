"""Build frozen grouped interaction calibration and confirmation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.interaction_calibration_sampling import (
    InteractionCalibrationSamplingConfig,
    build_interaction_calibration_sample,
    calibration_sample_size_options,
)
from pig_behavior.classification_v2.review.safe_non_interaction_view import (
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-universe", type=Path, required=True)
    parser.add_argument("--expected-universe-sha256", required=True)
    parser.add_argument("--interaction-diagnostic", type=Path, required=True)
    parser.add_argument("--expected-diagnostic-sha256", required=True)
    parser.add_argument("--producer-sha", required=True)
    parser.add_argument("--presentation-version", required=True)
    parser.add_argument("--presentation-semantic-hash", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-count", type=int, default=300)
    parser.add_argument("--confirmation-count", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2026072901)
    return parser.parse_args()


def _forbidden_live_ledger_path(path: Path) -> bool:
    normalized = str(path).replace("/", "\\").casefold()
    return (
        "human_review_workspace\\classification_v2\\" in normalized
        and "\\human_decisions\\behavior\\" in normalized
    )


def main() -> None:
    args = parse_args()
    for path in (
        args.review_universe,
        args.interaction_diagnostic,
        args.output_dir,
    ):
        if _forbidden_live_ledger_path(path):
            raise SystemExit("live Behavior decision-ledger paths are forbidden")

    actual_hashes = {
        "review_universe": sha256_file(args.review_universe),
        "interaction_diagnostic": sha256_file(
            args.interaction_diagnostic
        ),
    }
    expected_hashes = {
        "review_universe": args.expected_universe_sha256,
        "interaction_diagnostic": args.expected_diagnostic_sha256,
    }
    if actual_hashes != expected_hashes:
        raise SystemExit(
            "immutable input hash mismatch: "
            + json.dumps(
                {"expected": expected_hashes, "actual": actual_hashes},
                sort_keys=True,
            )
        )

    config = InteractionCalibrationSamplingConfig(
        development_count=args.development_count,
        confirmation_count=args.confirmation_count,
        seed=args.seed,
    )
    result = build_interaction_calibration_sample(
        pd.read_csv(args.review_universe, low_memory=False),
        pd.read_csv(args.interaction_diagnostic, low_memory=False),
        producer_sha=args.producer_sha,
        input_hashes=actual_hashes,
        presentation_version=args.presentation_version,
        presentation_semantic_hash=args.presentation_semantic_hash,
        config=config,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "calibration_group_split_manifest.csv": result.group_split,
        "blinded_calibration_manifest.csv": result.blinded_manifest,
        "internal_calibration_trace.csv": result.internal_trace,
        "calibration_media_authority.csv": result.media_authority,
        "calibration_sample_size_options.csv": (
            calibration_sample_size_options()
        ),
    }
    output_hashes: dict[str, str] = {}
    for name, frame in paths.items():
        path = args.output_dir / name
        frame.to_csv(path, index=False, lineterminator="\n")
        output_hashes[name] = sha256_file(path)
    audit = {**result.audit, "output_hashes": output_hashes}
    write_json(
        args.output_dir / "interaction_calibration_sample_audit.json",
        audit,
    )
    if not audit["valid"]:
        raise SystemExit(
            "interaction calibration sample checker failed: "
            + ";".join(audit["checker"]["errors"])
        )
    print(
        "INTERACTION_CALIBRATION_SAMPLE_VALID "
        f"development={args.development_count} "
        f"confirmation={args.confirmation_count}"
    )


if __name__ == "__main__":
    main()
