from pathlib import Path

import pandas as pd

VIDEO_KEY = "Pigs281119_000085_30fps"
PIG_ID = "ID_4"
FRAME = 1020

files = [
    r"outputs\classification_v2\frame_features\spatiotemporal_frame_features_roi.csv",
    r"outputs\classification_v2\frame_features\spatiotemporal_frame_features_enhanced.csv",
    r"outputs\classification_v2\sequence_features\temporal_label_intervals.csv",
    r"outputs\classification_v2\review_units\review_unit_manifest.csv",
]

for f in files:
    p = Path(f)
    print("\n===", f, "===")
    if not p.exists():
        print("MISSING")
        continue

    df = pd.read_csv(p, low_memory=False)

    if "frame_index" in df.columns:
        q = df[
            df["source_type"].astype(str).eq("cvat_tracking_xml")
            & df["video_key"].astype(str).eq(VIDEO_KEY)
            & df["pig_id"].astype(str).eq(PIG_ID)
            & pd.to_numeric(df["frame_index"], errors="coerce").eq(FRAME)
        ].copy()
        cols = [
            "source_type",
            "video_key",
            "pig_id",
            "track_id",
            "frame_index",
            "behavior",
            "label_anchor_frame_index",
            "label_window_start",
            "label_window_end",
            "temporal_unit_key",
        ]
    elif "label_window_start" in df.columns:
        q = df[
            df["source_type"].astype(str).eq("cvat_tracking_xml")
            & df["video_key"].astype(str).eq(VIDEO_KEY)
            & df["pig_id"].astype(str).eq(PIG_ID)
            & pd.to_numeric(df["label_window_start"], errors="coerce").eq(FRAME)
        ].copy()
        cols = [
            "source_type",
            "video_key",
            "pig_id",
            "track_id",
            "label_window_start",
            "label_window_end",
            "behavior_temporal_final",
            "temporal_consistency_status",
            "temporal_unit_key",
        ]
    elif "unit_start_frame" in df.columns:
        q = df[
            df["source_type"].astype(str).eq("cvat_tracking_xml")
            & df["video_key"].astype(str).eq(VIDEO_KEY)
            & df["pig_id"].astype(str).eq(PIG_ID)
            & pd.to_numeric(df["unit_start_frame"], errors="coerce").eq(FRAME)
        ].copy()
        cols = [
            "source_type",
            "video_key",
            "pig_id",
            "track_id",
            "unit_start_frame",
            "unit_end_frame",
            "behavior_label",
            "review_template",
            "review_reason",
            "review_unit_id",
        ]
    else:
        print("No known frame/unit columns")
        continue

    cols = [c for c in cols if c in q.columns]
    print(q[cols].to_string(index=False))
