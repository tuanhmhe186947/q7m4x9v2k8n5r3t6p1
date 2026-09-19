"""Freeze sequential mini-CVAT source corrections and final file hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.review.post_review_learning import (
    assert_not_active_behavior_ledger_path,
    build_corrected_source_authority,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--identity-apply-manifest-json",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source-target",
        action="append",
        type=Path,
        required=True,
        help="Repeat for every final CSV/XML target named by the manifests.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        *args.identity_apply_manifest_json,
        *args.source_target,
        args.output_json,
    ):
        assert_not_active_behavior_ledger_path(path)
    manifests = []
    for path in args.identity_apply_manifest_json:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["manifest_sha256"] = sha256_file(path)
        payload["manifest_path"] = str(path.resolve())
        manifests.append(payload)
    observed_hashes = {}
    for path in args.source_target:
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hashes[str(path.resolve())] = sha256_file(path)

    authority = build_corrected_source_authority(
        identity_apply_manifests=manifests,
        observed_target_hashes=observed_hashes,
    )
    write_json(args.output_json, authority)
    print("PASS: corrected source authority frozen")
    print(args.output_json)


if __name__ == "__main__":
    main()
