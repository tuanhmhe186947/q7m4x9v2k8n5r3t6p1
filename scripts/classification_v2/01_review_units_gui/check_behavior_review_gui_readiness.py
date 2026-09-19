"""Validate Behavior Review GUI inputs without opening the GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.review.gui_readiness import (
    audit_behavior_gui_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-units-csv", required=True, type=Path)
    parser.add_argument("--native-evidence-csv", required=True, type=Path)
    parser.add_argument("--pig-strenet-artifact-dir", required=True, type=Path)
    parser.add_argument("--hidden-apply-manifest", required=True, type=Path)
    parser.add_argument("--legacy-crop-root", required=True, type=Path)
    parser.add_argument("--expected-hidden-reviewed-rows", type=int, default=5233)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    audit = audit_behavior_gui_readiness(
        review_units_csv=args.review_units_csv,
        native_evidence_csv=args.native_evidence_csv,
        pig_strenet_artifact_dir=args.pig_strenet_artifact_dir,
        hidden_apply_manifest=args.hidden_apply_manifest,
        legacy_crop_root=args.legacy_crop_root,
        expected_hidden_reviewed_rows=args.expected_hidden_reviewed_rows,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(f".{args.output_json.name}.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output_json)
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
    if not audit["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
