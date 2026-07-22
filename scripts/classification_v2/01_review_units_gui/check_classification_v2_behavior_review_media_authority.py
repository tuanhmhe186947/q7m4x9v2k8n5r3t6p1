"""Independently rebuild and check behavior-review media authority."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.media_authority import (
    build_behavior_review_media_authority,
    finalize_media_authority_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-units-csv", required=True, type=Path)
    parser.add_argument("--native-evidence-csv", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--legacy-crop-root", required=True, type=Path)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--code-authority-sha", required=True)
    parser.add_argument("--index-csv", required=True, type=Path)
    parser.add_argument("--authority-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    expected_index, expected_summary = build_behavior_review_media_authority(
        pd.read_csv(args.review_units_csv, low_memory=False),
        pd.read_csv(args.native_evidence_csv, low_memory=False),
        video_root=args.video_root,
        legacy_crop_root=args.legacy_crop_root,
        lineage_id=args.lineage_id,
        code_authority_sha=args.code_authority_sha,
    )
    observed_index = pd.read_csv(args.index_csv, low_memory=False)
    observed_summary = json.loads(args.authority_json.read_text(encoding="utf-8"))
    errors = list(expected_summary["errors"])
    try:
        pd.testing.assert_frame_equal(
            observed_index,
            expected_index,
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        errors.append(f"media_authority_index_drift={exc}")
    with tempfile.TemporaryDirectory() as temp_dir:
        expected_path = Path(temp_dir) / "index.csv"
        expected_index.to_csv(expected_path, index=False)
        expected = finalize_media_authority_summary(
            expected_summary,
            index_csv=expected_path,
            index_display_path=args.index_csv,
        )
    if observed_summary != expected:
        errors.append("media_authority_json_drift")
    audit = {
        "lineage_id": args.lineage_id,
        "code_authority_sha": args.code_authority_sha.lower(),
        "valid": not errors,
        "errors": errors,
        "index_rows": len(observed_index),
        "deterministic_digest_match": (
            observed_summary.get("authority_sha256")
            == expected.get("authority_sha256")
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
