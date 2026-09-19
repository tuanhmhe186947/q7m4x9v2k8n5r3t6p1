from __future__ import annotations

import json

import pandas as pd
from tqdm import tqdm


def evenly_sample(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    if count <= 1:
        return [values[0]]
    positions = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    return [values[pos] for pos in positions]


def build_sequence_views(
    dense_df: pd.DataFrame,
    views: list[str] | None = None,
    *,
    show_progress: bool = False,
    include_all_tracklets: bool = False,
) -> pd.DataFrame:
    if views is None:
        views = ["sparse_3_0_6_12"]
    rows: list[dict[str, object]] = []
    if dense_df.empty:
        return pd.DataFrame(rows)
    if not include_all_tracklets and "include_in_training" in dense_df.columns:
        dense_df = dense_df[dense_df["include_in_training"].fillna(False)].copy()
        if dense_df.empty:
            return pd.DataFrame(rows)

    grouped = list(dense_df.groupby("tracklet_id", sort=False))
    iterator = tqdm(grouped, desc="Building sequence views", disable=not show_progress)
    for tracklet_id, group in iterator:
        group = group.sort_values("frame_index")
        anchor = int(group["legacy_anchor_frame"].iloc[0])
        available = set(group["frame_index"].astype(int).tolist())
        dense_frames = group["frame_index"].astype(int).tolist()
        legacy_frames = [int(v) for v in str(group["legacy_interval_frame_list"].iloc[0]).split("|") if v]
        requested: dict[str, list[int]] = {
            "sparse_3_0_6_12": [anchor, anchor + 6, anchor + 12],
            "legacy_old_pattern_6": legacy_frames,
            "dense_6_same_span": evenly_sample([f for f in dense_frames if anchor <= f <= anchor + 12], 6),
            "dense_12_same_span": evenly_sample([f for f in dense_frames if anchor <= f <= anchor + 12], 12),
            "full_dense_0_to_12": [f for f in dense_frames if anchor <= f <= anchor + 12],
        }

        for window_type in views:
            frame_indices = [frame for frame in requested.get(window_type, []) if frame in available]
            if not frame_indices:
                continue
            selected = group[group["frame_index"].isin(frame_indices)].sort_values("frame_index")
            qa_notes = []
            if len(frame_indices) < len(requested.get(window_type, [])):
                qa_notes.append("some_requested_frames_missing_from_dense_tracklet")
            qa_status = "ok" if selected["qa_status"].eq("ok").all() and not qa_notes else "review"
            crop_series = selected["crop_path"] if "crop_path" in selected else pd.Series([""] * len(selected))
            full_series = (
                selected["full_frame_path"] if "full_frame_path" in selected else pd.Series([""] * len(selected))
            )
            rows.append(
                {
                    "sequence_id": f"{tracklet_id}_{window_type}",
                    "tracklet_id": tracklet_id,
                    "group_id": selected["group_id"].iloc[0],
                    "sample_id": selected["sample_id"].iloc[0],
                    "pig_id": selected["pig_id"].iloc[0],
                    "behavior": selected["behavior"].iloc[0],
                    "hidden": selected["hidden"].iloc[0],
                    "window_type": window_type,
                    "frame_indices": "|".join(map(str, selected["frame_index"].astype(int).tolist())),
                    "frame_timestamps_sec": "|".join(
                        "" if pd.isna(v) else str(v) for v in selected["timestamp_sec"].tolist()
                    ),
                    "bbox_list": json.dumps(selected[["x1", "y1", "x2", "y2"]].values.tolist()),
                    "crop_paths": "|".join(crop_series.fillna("").astype(str).tolist()),
                    "full_frame_paths": "|".join(full_series.fillna("").astype(str).tolist()),
                    "source_video_resolved": selected["source_video_resolved"].iloc[0],
                    "source_folder": selected["source_folder"].iloc[0],
                    "times_txt_path": selected["timestamp_file_resolved"].iloc[0],
                    "depth_video_path": selected["depth_video_path"].iloc[0],
                    "background_depth_path": selected["background_depth_path"].iloc[0],
                    "depth_scale_path": selected["depth_scale_path"].iloc[0],
                    "inverse_intrinsic_path": selected["inverse_intrinsic_path"].iloc[0],
                    "rot_path": selected["rot_path"].iloc[0],
                    "depth_sync_status": "not_verified",
                    "track_confidence_min": selected["track_confidence"].min(),
                    "track_confidence_mean": selected["track_confidence"].mean(),
                    "qa_status": qa_status,
                    "qa_notes": ";".join(qa_notes),
                    "auto_qa_status": selected["auto_qa_status"].iloc[0] if "auto_qa_status" in selected else qa_status,
                    "manual_decision": selected["manual_decision"].iloc[0] if "manual_decision" in selected else "",
                    "manual_reason": selected["manual_reason"].iloc[0] if "manual_reason" in selected else "",
                    "include_in_training": (
                        bool(selected["include_in_training"].iloc[0]) if "include_in_training" in selected else True
                    ),
                    "training_tier": selected["training_tier"].iloc[0] if "training_tier" in selected else "clean",
                }
            )
    return pd.DataFrame(rows)
