"""Persist non-destructive source and matched-length control views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.evaluation.domain_controls import build_source_matched_views


def main() -> None:
    parser = argparse.ArgumentParser(description="Build classification_v2 source control views.")
    parser.add_argument(
        "--window-manifest",
        type=Path,
        default=Path("outputs/classification_v2/train_ready_windows/split_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/source_matched_views"),
    )
    args = parser.parse_args()
    views, audit = build_source_matched_views(pd.read_csv(args.window_manifest, low_memory=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views.to_csv(args.output_dir / "source_matched_view_manifest.csv", index=False)
    payload = {
        "input_csv": str(args.window_manifest),
        "view_manifest_csv": str(args.output_dir / "source_matched_view_manifest.csv"),
        **audit,
    }
    (args.output_dir / "source_matched_view_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not audit["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
