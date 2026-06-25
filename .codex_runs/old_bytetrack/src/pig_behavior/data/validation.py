"""Validate dataset schema and files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_REQUIRED_COLUMNS = {
    "img_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "behavior",
    "behavior_coarse",
    "hidden",
    "group_id",
    "in_feeder",
    "in_drinker",
    "in_toy",
    "speed_feat",
    "min_dist_other",
    "num_close_other",
}


def validate_project_outputs(df_feats: pd.DataFrame, raw_image_dir: Path) -> None:
    """Validate CSV schema and copied image files before finishing."""
    missing = sorted(PROJECT_REQUIRED_COLUMNS.difference(df_feats.columns))
    if missing:
        raise ValueError(f"Missing project training columns: {missing}")

    missing_images = [
        name
        for name in df_feats["img_name"].drop_duplicates()
        if not (raw_image_dir / name).exists()
    ]
    if missing_images:
        raise FileNotFoundError(
            "CSV references missing images under "
            f"{raw_image_dir}: {missing_images[:10]}"
        )
