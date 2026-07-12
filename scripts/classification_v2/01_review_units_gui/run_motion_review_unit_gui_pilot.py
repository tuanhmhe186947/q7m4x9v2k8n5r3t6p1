from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run motion/context review-unit GUI pilot.")
    parser.add_argument(
        "--review-units-csv",
        type=Path,
        default=Path(r"outputs/classification_v2/review_units/motion_review_unit_template.csv"),
    )
    parser.add_argument(
        "--frame-features-csv",
        type=Path,
        default=Path(r"outputs/classification_v2/frame_features/spatiotemporal_frame_features_enhanced.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"outputs/classification_v2/review_policy/motion_review_unit_gui_pilot"),
    )
    parser.add_argument("--video-root", type=Path, default=Path(r"data/videos"))
    parser.add_argument("--raw-root", type=Path, default=Path(r"data/raw/legacy_full_multigt_masked_nodup_16f/crops"))
    parser.add_argument("--source-type", default="")
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--padding", type=float, default=0.8)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--copy-contact-sheets", action="store_true", default=True)
    args = parser.parse_args()

    if args.fresh and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        r"scripts\classification_v2\01_review_units_gui\review_temporal_unit_gui.py",
        "--review-units-csv",
        str(args.review_units_csv),
        "--frame-features-csv",
        str(args.frame_features_csv),
        "--output-dir",
        str(args.output_dir),
        "--video-root",
        str(args.video_root),
        "--raw-root",
        str(args.raw_root),
        "--max-items",
        str(args.max_items),
        "--padding",
        str(args.padding),
        "--copy-contact-sheets",
    ]
    if args.source_type:
        cmd.extend(["--source-type", args.source_type])

    print("RUN MOTION GUI:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
