"""Build exact per-unit behavior-review media authority."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
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
    parser.add_argument("--output-index-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    require_output_paths_available(
        [args.output_index_csv, args.output_json],
        overwrite=args.overwrite,
    )
    index, summary = build_behavior_review_media_authority(
        pd.read_csv(args.review_units_csv, low_memory=False),
        pd.read_csv(args.native_evidence_csv, low_memory=False),
        video_root=args.video_root,
        legacy_crop_root=args.legacy_crop_root,
        lineage_id=args.lineage_id,
        code_authority_sha=args.code_authority_sha,
    )
    _atomic_bundle(index, summary, args.output_index_csv, args.output_json)
    if summary["errors"]:
        raise SystemExit(2)
    print(json.dumps({"valid": True, "units": len(index)}, indent=2))


def _atomic_bundle(
    index: pd.DataFrame,
    summary: dict[str, object],
    index_path: Path,
    json_path: Path,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    index_temp = index_path.with_name(f".{index_path.name}.{token}.tmp")
    json_temp = json_path.with_name(f".{json_path.name}.{token}.tmp")
    try:
        index.to_csv(index_temp, index=False)
        final = finalize_media_authority_summary(
            summary,
            index_csv=index_temp,
            index_display_path=index_path,
        )
        json_temp.write_text(
            json.dumps(final, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        index_temp.replace(index_path)
        json_temp.replace(json_path)
    finally:
        index_temp.unlink(missing_ok=True)
        json_temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
