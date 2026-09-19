"""Build ROI features from geometry-normalized frame features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.features.roi import (
    build_roi_features,
    load_scene_rois_from_coco,
    validate_roi_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build static scene ROI features for classification_v2."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--roi-coco", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--near-distance-n", type=float, default=0.08)
    parser.add_argument("--contact-distance-n", type=float, default=0.02)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived ROI artifacts explicitly.",
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

    print(f"reading frame features: {args.input_csv}")
    df = pd.read_csv(args.input_csv, low_memory=False)

    rois = load_scene_rois_from_coco(args.roi_coco)
    print("loaded ROIs:")
    for roi in rois:
        print(
            f"  id={roi.roi_id} class={roi.category} "
            f"bbox=({roi.x1:.1f},{roi.y1:.1f},{roi.x2:.1f},{roi.y2:.1f}) "
            f"image={int(roi.image_width)}x{int(roi.image_height)}"
        )

    print("building ROI features...")
    out = build_roi_features(
        df,
        roi_coco_path=args.roi_coco,
        near_distance_n=args.near_distance_n,
        contact_distance_n=args.contact_distance_n,
    )

    audit = validate_roi_features(out)
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    if audit["errors"]:
        raise ValueError(f"ROI audit errors: {audit['errors']}")

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
