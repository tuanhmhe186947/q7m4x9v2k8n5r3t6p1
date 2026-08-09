"""Materialize the hash-bound, inner-only RGB binding for PRE-S1 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.training.pre_s1_calibration import (
    EXPECTED_TRAIN_WINDOWS,
    EXPECTED_VALIDATION_WINDOWS,
    create_calibration_plan,
    load_canonical_inner_rows,
    preflight_calibration,
)
from pig_behavior.classification_v2.training.pre_s1_rgb_binding import (
    materialize_inner_rgb_binding,
)


def parse_args() -> argparse.Namespace:
    """Accept only path realization and the exact binding authorization."""

    parser = argparse.ArgumentParser(
        description="Materialize only the inner T6 RGB calibration binding."
    )
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--rgb-source-root", type=Path, required=True)
    parser.add_argument("--input-parity-evidence", type=Path, required=True)
    parser.add_argument("--source-integrity-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-authorization", required=True)
    return parser.parse_args()


def main() -> None:
    """Bind existing packed RGB media without decoding or regenerating it."""

    args = parse_args()
    if args.execution_authorization != "PRE_S1_RGB_BINDING_AUTHORIZED":
        raise SystemExit("PRE-S1 RGB binding requires its exact authorization token")
    repository_root = args.repository_root.resolve()
    plan = create_calibration_plan(
        args.authority,
        repository_root=repository_root,
        outputs_root=args.outputs_root.resolve(),
        device_name="cuda",
    )
    hashes = preflight_calibration(plan)
    train, validation, _ = load_canonical_inner_rows(plan, hashes)
    requested_roles = pd.concat(
        [
            train.loc[:, ["window_id", "primary_s1_role"]],
            validation.loc[:, ["window_id", "primary_s1_role"]],
        ],
        ignore_index=True,
    )
    parity = json.loads(args.input_parity_evidence.read_text(encoding="utf-8"))
    source_integrity = json.loads(
        args.source_integrity_evidence.read_text(encoding="utf-8")
    )
    report = materialize_inner_rgb_binding(
        output_dir=args.output_dir,
        rgb_source_root=args.rgb_source_root,
        requested_roles=requested_roles,
        authority_sha256=plan.authority_sha256,
        provenance_hashes=hashes,
        expected_train_windows=EXPECTED_TRAIN_WINDOWS,
        expected_validation_windows=EXPECTED_VALIDATION_WINDOWS,
        input_parity_evidence=parity,
        source_integrity_evidence=source_integrity,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
