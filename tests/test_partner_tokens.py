"""Focused correctness tests for partner_tokens.py."""

from __future__ import annotations

import numpy as np

from pig_behavior.classification_v2.features.partner_tokens import (
    DEFAULT_K,
    DEFAULT_T,
    PARTNER_TOKEN_COLUMNS,
    PARTNER_TOKEN_DIM,
    FrameObservation,
    PartnerTokenIndex,
    extract_frame_partner_tokens,
)


def _make_obs(
    track_id: str,
    cx_n: float,
    cy_n: float,
    bw_n: float = 0.1,
    bh_n: float = 0.2,
    bbox_valid: bool = True,
    frame_index: int = 0,
) -> FrameObservation:
    x1 = (cx_n - bw_n / 2.0) * 1280.0
    x2 = (cx_n + bw_n / 2.0) * 1280.0
    y1 = (cy_n - bh_n / 2.0) * 720.0
    y2 = (cy_n + bh_n / 2.0) * 720.0
    return FrameObservation(
        source_type="test_source",
        dataset_id="test_dataset",
        video_key="test_video",
        scene_frame_uid="test_video::f000000",
        frame_index=frame_index,
        object_track_key=track_id,
        cx_n=cx_n,
        cy_n=cy_n,
        bw_n=bw_n,
        bh_n=bh_n,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        bbox_valid=bbox_valid,
    )


def test_token_columns_order():
    assert PARTNER_TOKEN_DIM == 6
    assert PARTNER_TOKEN_COLUMNS == (
        "relative_center_dx_n",
        "relative_center_dy_n",
        "relative_width_delta_n",
        "relative_height_delta_n",
        "axis_distance_n",
        "bbox_iou",
    )


def test_actor_excluded_and_deterministic_nearest_ordering():
    # Actor at (0.5, 0.5)
    actor = _make_obs("actor_track", 0.5, 0.5)
    # Partner 1 at (0.5, 0.6) -> dist = 0.1
    p1 = _make_obs("partner_near", 0.5, 0.6)
    # Partner 2 at (0.5, 0.8) -> dist = 0.3
    p2 = _make_obs("partner_far", 0.5, 0.8)

    objects = [p2, actor, p1]
    tokens, mask, partner_ids, partner_ranks = extract_frame_partner_tokens(actor, objects, k=2)

    assert mask[0] is True or mask[0] == 1
    assert mask[1] is True or mask[1] == 1
    assert partner_ids[0] == "partner_near"
    assert partner_ids[1] == "partner_far"
    assert partner_ranks == [0, 1]
    assert np.isclose(tokens[0, 4], 0.1)
    assert np.isclose(tokens[1, 4], 0.3)


def test_object_track_key_tie_break():
    # Actor at (0.5, 0.5)
    actor = _make_obs("actor_track", 0.5, 0.5)
    # Two partners at exact same distance (0.1)
    # p_b at (0.5, 0.6), p_a at (0.6, 0.5)
    p_b = _make_obs("track_B", 0.5, 0.6)
    p_a = _make_obs("track_A", 0.6, 0.5)

    objects = [p_b, p_a, actor]
    tokens, mask, partner_ids, partner_ranks = extract_frame_partner_tokens(actor, objects, k=2)

    assert partner_ids == ["track_A", "track_B"]
    assert partner_ranks == [0, 1]
    assert np.isclose(tokens[0, 4], 0.1)
    assert np.isclose(tokens[1, 4], 0.1)


def test_k2_truncation():
    actor = _make_obs("actor_track", 0.5, 0.5)
    p1 = _make_obs("p1", 0.5, 0.55)  # dist 0.05
    p2 = _make_obs("p2", 0.5, 0.60)  # dist 0.10
    p3 = _make_obs("p3", 0.5, 0.70)  # dist 0.20
    p4 = _make_obs("p4", 0.5, 0.80)  # dist 0.30

    tokens, mask, partner_ids, partner_ranks = extract_frame_partner_tokens(
        actor,
        [p4, p3, p2, p1],
        k=2,
    )

    assert tokens.shape == (2, 6)
    assert mask.shape == (2,)
    assert mask.tolist() == [True, True]
    assert partner_ids == ["p1", "p2"]
    assert partner_ranks == [0, 1]


def test_one_partner_padding():
    actor = _make_obs("actor_track", 0.5, 0.5)
    p1 = _make_obs("p1", 0.5, 0.6)  # dist 0.1

    tokens, mask, partner_ids, partner_ranks = extract_frame_partner_tokens(
        actor,
        [p1, actor],
        k=2,
    )

    assert mask.tolist() == [True, False]
    assert partner_ids == ["p1", ""]
    assert partner_ranks == [0, -1]
    assert np.isclose(tokens[0, 4], 0.1)
    # Padded slot is all zeros
    assert np.all(tokens[1] == 0.0)


def test_zero_partner_padding():
    actor = _make_obs("actor_track", 0.5, 0.5)

    tokens, mask, partner_ids, partner_ranks = extract_frame_partner_tokens(
        actor,
        [actor],
        k=2,
    )

    assert mask.tolist() == [False, False]
    assert partner_ids == ["", ""]
    assert partner_ranks == [-1, -1]
    assert np.all(tokens == 0.0)


def test_two_distinct_partners_remain_distinct():
    actor = _make_obs("actor", 0.5, 0.5, bw_n=0.1, bh_n=0.2)
    p1 = _make_obs("p1", 0.55, 0.5, bw_n=0.12, bh_n=0.18)
    p2 = _make_obs("p2", 0.5, 0.65, bw_n=0.08, bh_n=0.25)

    tokens, mask, partner_ids, partner_ranks = extract_frame_partner_tokens(
        actor,
        [p1, p2, actor],
        k=2,
    )

    # First row is p1 (dist 0.05), second row is p2 (dist 0.15)
    assert not np.allclose(tokens[0], tokens[1])
    assert partner_ids[0] != partner_ids[1]


def test_partner_token_index_batch_extraction():
    actor_f0 = _make_obs("actor_track", 0.5, 0.5, frame_index=0)
    actor_f1 = _make_obs("actor_track", 0.51, 0.5, frame_index=1)
    p1_f0 = _make_obs("p1", 0.55, 0.5, frame_index=0)
    p2_f0 = _make_obs("p2", 0.5, 0.65, frame_index=0)
    p1_f1 = _make_obs("p1", 0.56, 0.5, frame_index=1)

    index = PartnerTokenIndex([actor_f0, actor_f1, p1_f0, p2_f0, p1_f1])

    windows = [
        {
            "video_key": "test_video",
            "object_track_key": "actor_track",
            "selected_frame_indices": [0, 1],
        }
    ]

    tokens, mask, lineage = index.extract_batch_tokens(windows, k=DEFAULT_K, t=DEFAULT_T)

    assert tokens.shape == (1, 6, 2, 6)
    assert mask.shape == (1, 6, 2)
    assert tokens.dtype == np.float32
    assert mask.dtype == bool

    # Frame 0 has 2 partners
    assert mask[0, 0].tolist() == [True, True]
    # Frame 1 has 1 partner
    assert mask[0, 1].tolist() == [True, False]
    # Frames 2..5 are unobserved -> padded
    assert mask[0, 2:].sum() == 0
