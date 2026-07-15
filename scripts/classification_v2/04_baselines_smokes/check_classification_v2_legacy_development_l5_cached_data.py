"""Audit the legacy L5 cached-feature consumer in a fresh CPU process."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pig_behavior.classification_v2.contracts.temporal_tier_contract import (
    LEGACY_TEMPORAL_MODEL_VIEW_SPECS,
)
from pig_behavior.classification_v2.training.legacy_development_l5 import (
    load_legacy_l5_config,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    MAX_CACHED_AUDIT_BATCH_SIZE,
    MAX_CACHED_AUDIT_BATCHES_PER_ROLE,
    audit_legacy_l5_cached_feature_batches,
    build_legacy_l5_cached_feature_view,
    write_legacy_l5_cached_data_packet,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded CPU-only legacy L5 cached-feature consumer audit."
        )
    )
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--feature-result-json", type=Path, required=True)
    parser.add_argument(
        "--temporal-view-name",
        choices=tuple(sorted(LEGACY_TEMPORAL_MODEL_VIEW_SPECS)),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=f"CPU audit batch size, at most {MAX_CACHED_AUDIT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--max-batches-per-role",
        type=int,
        default=2,
        help=(
            "Bounded train/validation batches, at most "
            f"{MAX_CACHED_AUDIT_BATCHES_PER_ROLE} per role."
        ),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_legacy_l5_config(args.config_json)
    view = build_legacy_l5_cached_feature_view(
        config,
        feature_result_path=args.feature_result_json,
        temporal_view_name=args.temporal_view_name,
    )
    view = audit_legacy_l5_cached_feature_batches(
        view,
        batch_size=args.batch_size,
        max_batches_per_role=args.max_batches_per_role,
    )
    runtime_seconds = time.perf_counter() - started
    output_dir = config.l5_output_root / args.run_id
    manifest = write_legacy_l5_cached_data_packet(
        view,
        output_dir=output_dir,
        run_id=args.run_id,
        runtime_seconds=runtime_seconds,
    )
    result = {
        "status": "PASS_LEGACY_DEVELOPMENT_L5_CACHED_CONSUMER_AUDIT",
        "run_id": args.run_id,
        "output_dir": str(output_dir.resolve()),
        "run_manifest": manifest,
        "bounded_batch_audit": view.audit["bounded_batch_audit"],
        "gpu_execution_performed": False,
        "outer_holdout_predictions_created": 0,
        "source_media_reads": 0,
        "valid": True,
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
