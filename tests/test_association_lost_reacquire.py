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


def test_lost_track_allows_same_raw_fast_motion_owner_bypass() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    owner_track = FixedTrack(
        fixed_id=4,
        last_box=np.array([120.0, 100.0, 220.0, 200.0], dtype=np.float32),
        missed=0,
        last_source="detected",
        last_ambiguous=True,
        ever_detected=True,
    )
    hist = np.zeros(16, dtype=np.float32)
    hist[0] = 1.0
    track.hist_bank.append(hist)
    detection = Detection(
        box=np.array([400.0, 100.0, 500.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=hist.copy(),
    )

    assert lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 4},
        raw_owner_tracks={4: owner_track},
    )


def test_lost_track_owner_bypass_rejects_visible_stable_owner() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    owner_track = FixedTrack(
        fixed_id=4,
        last_box=np.array([120.0, 100.0, 220.0, 200.0], dtype=np.float32),
        missed=0,
        last_source="detected",
        ever_detected=True,
    )
    hist = np.zeros(16, dtype=np.float32)
    hist[0] = 1.0
    track.hist_bank.append(hist)
    detection = Detection(
        box=np.array([400.0, 100.0, 500.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=hist.copy(),
    )

    assert not lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 4},
        raw_owner_tracks={4: owner_track},
    )


def test_lost_track_owner_bypass_rejects_near_contact_jump() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    owner_track = FixedTrack(
        fixed_id=4,
        last_box=np.array([120.0, 100.0, 220.0, 200.0], dtype=np.float32),
        missed=0,
        last_source="detected",
        ever_detected=True,
    )
    hist = np.zeros(16, dtype=np.float32)
    hist[0] = 1.0
    track.hist_bank.append(hist)
    detection = Detection(
        box=np.array([135.0, 100.0, 235.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=hist.copy(),
    )

    assert not lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 4},
        raw_owner_tracks={4: owner_track},
    )


def test_lost_track_owner_bypass_rejects_distant_owner_conflict() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    owner_track = FixedTrack(
        fixed_id=4,
        last_box=np.array([500.0, 100.0, 600.0, 200.0], dtype=np.float32),
        missed=0,
        last_source="detected",
        ever_detected=True,
    )
    hist = np.zeros(16, dtype=np.float32)
    hist[0] = 1.0
    track.hist_bank.append(hist)
    detection = Detection(
        box=np.array([400.0, 100.0, 500.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=hist.copy(),
    )

    assert not lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 4},
        raw_owner_tracks={4: owner_track},
    )


def test_lost_track_allows_raw_owner_transfer_when_track_is_better_match() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    owner_track = FixedTrack(
        fixed_id=4,
        last_box=np.array([500.0, 100.0, 600.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 1}),
        missed=0,
        last_source="detected",
        ever_detected=True,
    )
    track_hist = np.zeros(16, dtype=np.float32)
    track_hist[0] = 1.0
    owner_hist = np.zeros(16, dtype=np.float32)
    owner_hist[1] = 1.0
    track.hist_bank.append(track_hist)
    owner_track.hist_bank.append(owner_hist)
    detection = Detection(
        box=np.array([120.0, 100.0, 220.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=track_hist.copy(),
    )

    assert lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 4},
        raw_owner_tracks={4: owner_track},
    )


def test_lost_track_raw_owner_transfer_rejects_when_owner_is_better_match() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    owner_track = FixedTrack(
        fixed_id=4,
        last_box=np.array([120.0, 100.0, 220.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 1}),
        missed=0,
        last_source="detected",
        ever_detected=True,
    )
    track_hist = np.zeros(16, dtype=np.float32)
    track_hist[0] = 1.0
    owner_hist = np.zeros(16, dtype=np.float32)
    owner_hist[1] = 1.0
    track.hist_bank.append(track_hist)
    owner_track.hist_bank.append(owner_hist)
    detection = Detection(
        box=np.array([120.0, 100.0, 220.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=owner_hist.copy(),
    )

    assert not lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 4},
        raw_owner_tracks={4: owner_track},
    )


def test_lost_track_allows_different_raw_when_owner_is_hidden() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({99: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    owner_track = FixedTrack(
        fixed_id=4,
        last_box=np.array([220.0, 100.0, 320.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=2,
        last_source="predicted",
        ever_detected=True,
    )
    hist = np.zeros(16, dtype=np.float32)
    hist[0] = 1.0
    track.hist_bank.append(hist)
    detection = Detection(
        box=np.array([130.0, 100.0, 230.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=hist.copy(),
    )

    assert lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 4},
        raw_owner_tracks={4: owner_track},
    )


def test_lost_track_rejects_different_raw_when_owner_is_visible() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({99: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    owner_track = FixedTrack(
        fixed_id=4,
        last_box=np.array([120.0, 100.0, 220.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=0,
        last_source="detected",
        ever_detected=True,
    )
    hist = np.zeros(16, dtype=np.float32)
    hist[0] = 1.0
    track.hist_bank.append(hist)
    detection = Detection(
        box=np.array([130.0, 100.0, 230.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=hist.copy(),
    )

    assert not lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 4},
        raw_owner_tracks={4: owner_track},
    )


def test_lost_track_allows_same_raw_far_jump_with_matching_appearance() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    hist = np.zeros(16, dtype=np.float32)
    hist[0] = 1.0
    track.hist_bank.append(hist)
    detection = Detection(
        box=np.array([700.0, 100.0, 800.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=hist.copy(),
    )

    assert lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 7},
    )


def test_lost_track_rejects_same_raw_far_jump_with_poor_appearance() -> None:
    cfg = TrackingConfig()
    track = FixedTrack(
        fixed_id=7,
        last_box=np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32),
        raw_id_counts=Counter({42: 5}),
        missed=1,
        last_source="predicted",
        ever_detected=True,
    )
    track_hist = np.zeros(16, dtype=np.float32)
    track_hist[0] = 1.0
    det_hist = np.zeros(16, dtype=np.float32)
    det_hist[1] = 1.0
    track.hist_bank.append(track_hist)
    detection = Detection(
        box=np.array([700.0, 100.0, 800.0, 200.0], dtype=np.float32),
        score=0.9,
        raw_id=42,
        class_id=0,
        hist=det_hist,
    )

    assert not lost_track_detection_is_plausible(
        track,
        detection,
        cfg,
        width=1000,
        height=1000,
        raw_owner={42: 7},
    )
