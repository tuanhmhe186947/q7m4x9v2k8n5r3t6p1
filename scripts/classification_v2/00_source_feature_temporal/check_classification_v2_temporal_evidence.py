from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.temporal_evidence import (
    audit_temporal_evidence_lineage,
)


def parse_args() -> argparse.Namespace:
    """Parse canonical paths while allowing bounded smoke artifacts."""

    parser = argparse.ArgumentParser(
        description="Audit classification_v2 temporal evidence lineage."
    )
    parser.add_argument(
        "--enhanced-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/frame_features/"
            "spatiotemporal_frame_features_enhanced.csv"
        ),
    )
    parser.add_argument(
        "--intervals-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/sequence_features/"
            "temporal_label_intervals.csv"
        ),
    )
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/sequence_features/"
            "sequence_window_features.csv"
        ),
    )
    parser.add_argument(
        "--review-units-csv",
        type=Path,
        default=Path(
            "outputs/classification_v2/review_units/review_unit_manifest.csv"
        ),
    )
    parser.add_argument(
        "--trainer-contract-json",
        type=Path,
        default=Path("configs/classification_v2/trainer_contract_v1.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/audits/temporal_evidence_lineage_audit.json"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing derived temporal audit explicitly.",
    )
    return parser.parse_args()


def main() -> None:
    """Read derived artifacts, write one audit JSON, and fail on violations."""

    args = parse_args()
    require_output_paths_available(
        [args.output_json],
        overwrite=args.overwrite,
    )
    for path in [
        args.enhanced_csv,
        args.intervals_csv,
        args.windows_csv,
        args.trainer_contract_json,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    enhanced = pd.read_csv(args.enhanced_csv, low_memory=False)
    intervals = pd.read_csv(args.intervals_csv, low_memory=False)
    windows = pd.read_csv(args.windows_csv, low_memory=False)
    review_units = (
        pd.read_csv(args.review_units_csv, low_memory=False)
        if args.review_units_csv.exists()
        else None
    )
    trainer_contract = json.loads(
        args.trainer_contract_json.read_text(encoding="utf-8")
    )
    audit = audit_temporal_evidence_lineage(
        enhanced,
        intervals,
        windows,
        review_units,
        trainer_contract,
    )
    audit["inputs"] = {
        "enhanced_csv": str(args.enhanced_csv),
        "intervals_csv": str(args.intervals_csv),
        "windows_csv": str(args.windows_csv),
        "review_units_csv": str(args.review_units_csv),
        "trainer_contract_json": str(args.trainer_contract_json),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
