"""CLI for merging classification_v2 frame-object sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pig_behavior.classification_v2.merge_sources import (
    audit_merged_frame_objects,
    merge_frame_object_sources,
    save_merged_frame_objects,
)
from pig_behavior.classification_v2.sources import (
    load_cvat_tracking_xml,
    load_legacy_frame_objects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge legacy and CVAT tracking XML annotations into "
        "one canonical classification_v2 frame-object CSV."
    )

    parser.add_argument(
        "--cvat-tracking-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing CVAT 1.1 tracking XML files. Can be repeated.",
    )

    parser.add_argument(
    "--trust-hidden",
    action="store_true",
    help=(
        "Deprecated fail-closed flag. CVAT Hidden trust must come from "
        "completed human hidden review provenance."
    ),
)

    parser.add_argument(
        "--legacy-csv",
        type=Path,
        action="append",
        default=[],
        help="Path to legacy_frame_object_annotations.csv. Can be repeated.",
    )
    parser.add_argument(
        "--cvat-tracking-xml",
        type=Path,
        action="append",
        default=[],
        help="Path to CVAT 1.1 tracking XML. Can be repeated.",
    )
    parser.add_argument(
        "--cvat-video-key",
        action="append",
        default=[],
        help="Video key for each --cvat-tracking-xml. Optional.",
    )
    parser.add_argument(
        "--cvat-dataset-id",
        action="append",
        default=[],
        help="Dataset id for each --cvat-tracking-xml. Optional.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help=(
            "Canonical decoded-video FPS used for both legacy and CVAT "
            "timestamp_sec. Default: 30."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output merged canonical CSV.",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=None,
        help="Optional audit JSON output path.",
    )
    parser.add_argument(
        "--lineage-json",
        type=Path,
        required=True,
        help="Write input/output hashes and merged-source provenance.",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="Optional debug row limit per source.",
    )
    parser.add_argument(
        "--require-full-8-for-eval",
        action="store_true",
        help="Set use_for_main_eval=False for CVAT frames without 8 pigs.",
    )

    return parser.parse_args()


def _get_optional(values: list[str], index: int) -> str | None:
    if index >= len(values):
        return None
    value = values[index].strip()
    return value or None


def main() -> None:
    args = parse_args()

    frames = []
    names = []

    for legacy_csv in args.legacy_csv:
        print(f"reading legacy csv: {legacy_csv}")
        df = load_legacy_frame_objects(
            legacy_csv,
            max_rows=args.max_rows_per_source,
            source_fps=args.fps,
        )
        frames.append(df)
        names.append(str(legacy_csv))
        print(f"  rows={len(df)} frames={df['frame_uid'].nunique()}")

    xml_paths = list(args.cvat_tracking_xml)

    for xml_dir in args.cvat_tracking_dir:
        if not xml_dir.exists():
            raise FileNotFoundError(f"CVAT tracking directory not found: {xml_dir}")
        if not xml_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {xml_dir}")

        found = sorted(xml_dir.glob("*.xml"))
        if not found:
            raise FileNotFoundError(f"No XML files found in: {xml_dir}")

        xml_paths.extend(found)

    # Remove duplicate XML paths while keeping order.
    seen = set()
    unique_xml_paths = []
    for path in xml_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_xml_paths.append(path)

    xml_paths = unique_xml_paths

    for idx, xml_path in enumerate(xml_paths):
        video_key = _get_optional(args.cvat_video_key, idx)
        dataset_id = _get_optional(args.cvat_dataset_id, idx)

        print(f"reading cvat tracking xml: {xml_path}")
        df = load_cvat_tracking_xml(
            xml_path,
            video_key=video_key,
            dataset_id=dataset_id,
            fps=args.fps,
            require_full_8_for_eval=args.require_full_8_for_eval,
            max_rows=args.max_rows_per_source,
            trust_hidden=args.trust_hidden,
        )
        frames.append(df)
        names.append(str(xml_path))
        print(f"  rows={len(df)} frames={df['frame_uid'].nunique()}")

    if not frames:
        raise ValueError("No input sources were provided.")

    print("merging sources...")
    merged = merge_frame_object_sources(frames, source_names=names, strict_schema=True)

    audit = audit_merged_frame_objects(merged)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if audit["errors"]:
        raise ValueError(f"Merge audit has errors: {audit['errors']}")

    save_merged_frame_objects(
        merged,
        args.output_csv,
        audit_json=args.audit_json,
    )

    lineage = {
        "schema_version": "classification_v2.merged_source_lineage.v1",
        "source_files": [
            {
                "path": str(Path(name).resolve()),
                "size": Path(name).stat().st_size,
                "sha256": _sha256(Path(name)),
            }
            for name in names
        ],
        "output": {
            "path": str(args.output_csv.resolve()),
            "size": args.output_csv.stat().st_size,
            "sha256": _sha256(args.output_csv),
        },
        "audit_json": (
            None if args.audit_json is None else str(args.audit_json.resolve())
        ),
        "source_type_counts": audit.get("sources", {}),
        "canonical_source_fps": float(args.fps),
        "canonical_timestamp_formula": "source_frame_index/source_fps",
        "times_txt_role": "acquisition_audit_only_not_motion_clock",
        "rows": int(len(merged)),
        "errors": audit.get("errors", []),
    }
    args.lineage_json.parent.mkdir(parents=True, exist_ok=True)
    args.lineage_json.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"saved csv: {args.output_csv}")
    if args.audit_json is not None:
        print(f"saved audit: {args.audit_json}")
    print(f"saved lineage: {args.lineage_json}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
