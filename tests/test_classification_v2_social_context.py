from __future__ import annotations

import pandas as pd

from pig_behavior.classification_v2.features.spatiotemporal import (
    EnhancedFeatureConfig,
    _add_social_context_columns,
)


def _row(
    *,
    frame_index: int,
    frame_uid: str,
    pig_id: str,
    cx_n: float,
    x1: float,
    bbox_valid: bool = True,
) -> dict[str, object]:
    return {
        "source_type": "cvat_tracking_xml",
        "dataset_id": "fixture",
        "video_key": "video-a",
        "frame_uid": frame_uid,
        "frame_index": frame_index,
        "object_track_key": f"video-a|pig={pig_id}",
        "pig_id": pig_id,
        "track_id": pig_id.replace("ID_", ""),
        "bbox_valid": bbox_valid,
        "cx_n": cx_n,
        "cy_n": 0.5,
        "x1": x1,
        "y1": 10.0,
        "x2": x1 + 20.0,
        "y2": 30.0,
        "bbox_area": 400.0,
        "delta_frame_prev": 1.0,
        "speed_n_per_frame": 0.01,
    }


def test_invalid_bbox_cannot_be_selected_as_social_partner() -> None:
    rows = pd.DataFrame(
        [
            _row(
                frame_index=0,
                frame_uid="f0",
                pig_id="ID_1",
                cx_n=0.10,
                x1=10.0,
            ),
            _row(
                frame_index=0,
                frame_uid="f0",
                pig_id="ID_2",
                cx_n=0.16,
                x1=40.0,
            ),
            _row(
                frame_index=0,
                frame_uid="f0",
                pig_id="ID_bad",
                cx_n=0.101,
                x1=11.0,
                bbox_valid=False,
            ),
        ]
    )

    enriched = _add_social_context_columns(rows, EnhancedFeatureConfig())
    actor = enriched.loc[enriched["pig_id"].eq("ID_1")].iloc[0]
    invalid = enriched.loc[enriched["pig_id"].eq("ID_bad")].iloc[0]

    assert actor["nearest_pig_id"] == "ID_2"
    assert actor["social_density_near_count"] == 1
    assert invalid["nearest_pig_id"] == ""
    assert invalid["social_density_near_count"] == 0


def test_partial_missing_frame_uid_does_not_merge_different_frames() -> None:
    rows = pd.DataFrame(
        [
            _row(
                frame_index=0,
                frame_uid="f0",
                pig_id="ID_0",
                cx_n=0.10,
                x1=10.0,
            ),
            _row(
                frame_index=1,
                frame_uid="",
                pig_id="ID_1",
                cx_n=0.20,
                x1=20.0,
            ),
            _row(
                frame_index=2,
                frame_uid="",
                pig_id="ID_2",
                cx_n=0.21,
                x1=21.0,
            ),
        ]
    )

    enriched = _add_social_context_columns(rows, EnhancedFeatureConfig())

    assert enriched["nearest_pig_id"].eq("").all()
    assert enriched["social_density_near_count"].eq(0).all()
    assert "_social_frame_group_key" not in enriched.columns
