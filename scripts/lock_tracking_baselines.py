#!/usr/bin/env python3
"""Hash-lock the accepted hybrid and five-mode tracking baselines."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.evaluation.tracking.baseline_lock import (  # noqa: E402
    lock_historical_baselines,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid-eval-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/eval/hybrid_bytetrack/20260707_230230"
            / "smooth_det020_loose/iou0_area0_condarea0_merge0"
        ),
    )
    parser.add_argument(
        "--mode-compare-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/eval/mode_compare/20260709_040751",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/eval/baseline_locks"
            / f"{datetime.now():%Y%m%d_%H%M%S}_tracking_baselines.json"
        ),
    )
    parser.add_argument("--expected-video-count", type=int, default=13)
    parser.add_argument(
        "--record-incomplete-evidence",
        action="store_true",
        help="Write an INCOMPLETE audit when historical prediction XML is missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = lock_historical_baselines(
        hybrid_eval_dir=args.hybrid_eval_dir,
        mode_compare_root=args.mode_compare_root,
        output_path=args.output,
        expected_video_count=args.expected_video_count,
        allow_missing_predictions=args.record_incomplete_evidence,
    )
    print(f"[WROTE] Tracking baseline lock: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
