from collections import Counter

import numpy as np

from pig_behavior.tracking.association import lost_track_detection_is_plausible
from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.schemas import Detection, FixedTrack


def test_lost_track_does_not_reacquire_raw_id_owned_by_other_track() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    detection = Detection(
        box=np.array([104.0, 104.0, 204.0, 204.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=np.zeros(16, dtype=np.float32),
    )

    assert not lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1280,
        height=720,
        raw_owner={42: 4},
    )
    assert lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1280,
        height=720,
        raw_owner={42: 7},
    )
