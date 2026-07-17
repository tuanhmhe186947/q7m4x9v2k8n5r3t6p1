from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from .timestamp_utils import timestamp_at


def sparse_support_frames(anchor: int) -> list[int]:
    return [anchor, anchor + 6, anchor + 12]


def build_anchor_records(
    df: pd.DataFrame,
    timestamps_by_video: dict[str, list[float]],
    track_end_mode: str,
    *,
    show_progress: bool = False,
    skip_tracklet_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    grouped = list(df.groupby(["group_id", "pig_id"], sort=False, dropna=False))
    iterator = tqdm(grouped, desc="Building legacy anchors", disable=not show_progress)
    for tracklet_idx, ((group_id, pig_id), group) in enumerate(iterator):
        tracklet_id = f"tracklet_{tracklet_idx:08d}"
        if skip_tracklet_ids and tracklet_id in skip_tracklet_ids:
            continue
        row = group.iloc[0]
        frames = list(row["parsed_frames"])
        anchor = int(frames[0])
        track_end = anchor + 12 if track_end_mode == "sample_0_6_12" else int(frames[-1])
        dense_frames = list(range(anchor, track_end + 1))
        timestamps = timestamps_by_video.get(str(row["video_final"]), [])
        legacy_times = [timestamp_at(timestamps, int(frame)) for frame in frames]
        support = [frame for frame in sparse_support_frames(anchor) if anchor <= frame <= track_end]
        records.append(
            {
                "tracklet_id": tracklet_id,
                "group_id": group_id,
                "sample_id": row["sample_id"],
                "pig_id": pig_id,
                "behavior": row["behavior"],
                "hidden": row["hidden"],
                "label_source": row.get(
                    "label_source",
                    "legacy_dense_behavior",
                ),
                "behavior_authority_slot": row.get(
                    "behavior_authority_slot",
                    pd.NA,
                ),
                "behavior_propagation_policy": row.get(
                    "behavior_propagation_policy",
                    "",
                ),
                "day_final": row["day_final"],
                "video_final": row["video_final"],
                "img_name": row["img_name"],
                "center_frame_from_img": row["center_frame_from_img"],
                "center_frame_final": row["center_frame_final"],
                "frame_mismatch": row["frame_mismatch"],
                "match_source": row["match_source"],
                "anchor_bbox": (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                "legacy_anchor_frame": anchor,
                "legacy_anchor_time_sec": timestamp_at(timestamps, anchor),
                "legacy_anchor_frame_mod_6": anchor % 6,
                "legacy_interval_frame_list": frames,
                "legacy_interval_timestamp_list": legacy_times,
                "legacy_interval_start_frame": int(frames[0]),
                "legacy_interval_end_frame": int(frames[-1]),
                "legacy_interval_start_time_sec": legacy_times[0] if legacy_times else None,
                "legacy_interval_end_time_sec": legacy_times[-1] if legacy_times else None,
                "dense_frame_indices": dense_frames,
                "gt_support_frames": support,
            }
        )
    return records
