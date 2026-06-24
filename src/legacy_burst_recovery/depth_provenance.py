from __future__ import annotations

from pathlib import Path

import pandas as pd

from .path_utils import SourceResources
from .video_utils import count_video_frames


def build_depth_provenance_audit(
    resources_by_video: dict[str, SourceResources],
    timestamp_audit: pd.DataFrame,
) -> pd.DataFrame:
    timestamp_by_video = {row["source_video_original"]: row for _, row in timestamp_audit.iterrows()}
    rows: list[dict[str, object]] = []
    for original, resources in resources_by_video.items():
        depth_path = resources.depth_video_path
        num_depth = count_video_frames(depth_path) if depth_path and Path(depth_path).exists() else None
        ts_row = timestamp_by_video.get(original, {})
        num_color = ts_row.get("num_color_frames")
        resource_paths = [
            resources.depth_video_path,
            resources.background_depth_path,
            resources.depth_scale_path,
            resources.inverse_intrinsic_path,
            resources.rot_path,
        ]
        complete = all(Path(path).exists() for path in resource_paths if path)
        match = bool(num_color is not None and num_depth is not None and int(num_color) == int(num_depth))
        rows.append(
            {
                "source_folder": resources.source_folder,
                "color_video_path": resources.color_video_path,
                "depth_video_path": resources.depth_video_path,
                "times_txt_path": resources.times_txt_path,
                "num_color_frames": num_color,
                "num_depth_frames": num_depth,
                "num_timestamps": ts_row.get("num_timestamps"),
                "color_depth_frame_count_match": match,
                "depth_resources_complete": complete,
                "depth_sync_status": "not_verified",
                "qa_status": "ok" if complete else "review",
                "qa_notes": "" if complete else "missing_depth_or_calibration_resource",
            }
        )
    return pd.DataFrame(rows)
