from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()

DEFAULT_TEMPLATE = Path(r"outputs\classification_v2\review_templates\interaction_review_template_shortlist.csv")
DEFAULT_OUTPUT_DIR = Path(r"outputs\classification_v2\review_policy\interaction_spatial_gui_pilot")
FRAME_FEATURES = Path(r"outputs\classification_v2\frame_features\spatiotemporal_frame_features_enhanced.csv")
RAW_ROOT = Path(r"data\raw\legacy_full_multigt_masked_nodup_16f\crops")
VIDEO_ROOT = Path(r"data\videos")
GUI_SCRIPT = Path(r"scripts\behavior_review_tools\review_spatial_behavior_gui.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-csv", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--source-type", default="cvat_tracking_xml")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    for p in [args.template_csv, FRAME_FEATURES, GUI_SCRIPT]:
        if not p.exists():
            raise SystemExit(f"Missing file: {p}")

    if args.fresh and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(GUI_SCRIPT),
        "--review-csv", str(args.template_csv),
        "--frame-features-csv", str(FRAME_FEATURES),
        "--video-root", str(VIDEO_ROOT),
        "--output-dir", str(args.output_dir),
        "--max-items", str(args.max_items),
        "--padding", "0.80",
        "--copy-contact-sheets",
    ]

    if RAW_ROOT.exists():
        cmd.extend(["--raw-root", str(RAW_ROOT)])

    if args.source_type:
        cmd.extend(["--source-type", args.source_type])

    print("RUN GUI:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    print("\nGUI closed. Output dir:")
    print(args.output_dir)


if __name__ == "__main__":
    main()
