from __future__ import annotations

from dataclasses import asdict

import pytest

from pig_behavior.evaluation.tracking.identity_episodes_v2 import (
    EXPLICIT_AUTHORITY_POLICY,
    FIRST_OBSERVATION_AUTHORITY_POLICY,
    IDENTITY_EPISODE_CONTRACT_ID,
    MatchedIdentityRow,
    build_identity_episode_result,
)


def _row(
    frame: int,
    gt_id: str,
    pred_id: str,
    *,
    sequence_key: str = "video-a",
    ambiguous: bool = False,
) -> MatchedIdentityRow:
    return MatchedIdentityRow(
        sequence_key=sequence_key,
        frame=frame,
        gt_id=gt_id,
        pred_id=pred_id,
        authority_ambiguous=ambiguous,
    )


def _correct_pair(frame: int, *, sequence_key: str = "video-a") -> list[MatchedIdentityRow]:
    return [
        _row(frame, "A", "track-a", sequence_key=sequence_key),
        _row(frame, "B", "track-b", sequence_key=sequence_key),
    ]


def _swapped_pair(frame: int, *, sequence_key: str = "video-a") -> list[MatchedIdentityRow]:
    return [
        _row(frame, "A", "track-b", sequence_key=sequence_key),
        _row(frame, "B", "track-a", sequence_key=sequence_key),
    ]


def test_clean_tracks_freeze_first_authority_without_episodes() -> None:
    rows = [
        *_correct_pair(0),
        *_correct_pair(1),
    ]

    result = build_identity_episode_result(rows)

    assert result.contract_id == IDENTITY_EPISODE_CONTRACT_ID
    assert result.authority_policy == FIRST_OBSERVATION_AUTHORITY_POLICY
    assert {(item.gt_id, item.pred_id) for item in result.authorities} == {
        ("A", "track-a"),
        ("B", "track-b"),
    }
    assert result.identity_error_episode_count == 0
    assert result.wrong_id_matched_frames == 0
    assert result.persistent_pairwise_identity_swap_count == 0


def test_one_frame_swap_then_recovery_creates_two_recovered_gt_episodes() -> None:
    rows = [
        *_correct_pair(0),
        *_swapped_pair(1),
        *_correct_pair(2),
    ]

    result = build_identity_episode_result(
        rows,
        fps_by_sequence={"video-a": 2.0},
    )

    assert result.identity_error_episode_count == 2
    assert result.recovered_identity_error_episode_count == 2
    assert result.terminal_identity_error_episode_count == 0
    assert result.wrong_id_matched_frames == 2
    assert result.wrong_id_matched_seconds == pytest.approx(1.0)
    assert {episode.duration_frames for episode in result.episodes} == {1}
    assert [
        episode.recovery_latency_seconds for episode in result.episodes
    ] == pytest.approx([0.5, 0.5])
    assert len(result.pairwise_events) == 1
    assert result.pairwise_events[0].persistent is False


def test_short_reciprocal_swap_is_persistent_when_terminal_to_both() -> None:
    result = build_identity_episode_result(
        [
            *_correct_pair(0),
            *_swapped_pair(1),
        ]
    )

    assert result.terminal_identity_error_episode_count == 2
    assert result.persistent_pairwise_identity_swap_count == 1
    pair_event = result.pairwise_events[0]
    assert pair_event.gt_ids == ("A", "B")
    assert pair_event.direct_joint_observations == 1
    assert pair_event.persistence_basis == "terminal_to_both"


def test_sixty_direct_pair_observations_are_persistent_after_recovery() -> None:
    rows = [*_correct_pair(0)]
    for frame in range(1, 61):
        rows.extend(_swapped_pair(frame))
    rows.extend(_correct_pair(61))

    result = build_identity_episode_result(rows)

    assert result.recovered_identity_error_episode_count == 2
    assert result.terminal_identity_error_episode_count == 0
    assert result.persistent_pairwise_identity_swap_count == 1
    pair_event = result.pairwise_events[0]
    assert pair_event.direct_joint_observations == 60
    assert pair_event.persistence_basis == "direct_observation_horizon"


def test_fifty_nine_recovered_pair_observations_are_not_persistent() -> None:
    rows = [*_correct_pair(0)]
    for frame in range(1, 60):
        rows.extend(_swapped_pair(frame))
    rows.extend(_correct_pair(60))

    result = build_identity_episode_result(rows)

    assert len(result.pairwise_events) == 1
    assert result.pairwise_events[0].direct_joint_observations == 59
    assert result.pairwise_events[0].persistent is False
    assert result.persistent_pairwise_identity_swap_count == 0


def test_large_wrong_row_gap_splits_censored_and_terminal_episodes() -> None:
    result = build_identity_episode_result(
        [
            *_correct_pair(0),
            _row(1, "A", "track-b"),
            _row(20, "A", "track-b"),
        ]
    )

    assert result.identity_error_episode_count == 2
    assert [episode.status for episode in result.episodes] == [
        "censored",
        "terminal",
    ]
    assert [episode.duration_frames for episode in result.episodes] == [1, 1]
    assert result.wrong_id_rows_classified == 2


def test_episode_duration_is_animal_frames_not_frame_span() -> None:
    result = build_identity_episode_result(
        [
            *_correct_pair(0),
            _row(2, "A", "track-b"),
            _row(10, "A", "track-b"),
            _row(11, "A", "track-a"),
        ],
        fps_by_sequence={"video-a": 4.0},
    )

    episode = result.episodes[0]
    assert episode.start_frame == 2
    assert episode.end_frame == 10
    assert episode.duration_frames == 2
    assert episode.duration_seconds == pytest.approx(0.5)
    assert episode.recovery_latency_seconds == pytest.approx(0.25)


def test_ambiguous_row_is_preserved_then_later_authority_can_freeze() -> None:
    result = build_identity_episode_result(
        [
            _row(0, "A", "track-x", ambiguous=True),
            _row(1, "A", "track-a"),
            _row(2, "A", "track-x"),
        ]
    )

    assert len(result.ambiguous_rows) == 1
    assert result.ambiguous_rows[0].row.frame == 0
    assert result.ambiguous_rows[0].reason == "input_authority_ambiguous"
    assert result.authorities[0].established_frame == 1
    assert result.wrong_id_matched_frames == 1
    assert result.episodes[0].target_gt_ids == (None,)


def test_explicit_authority_is_one_to_one_and_missing_gt_stays_ambiguous() -> None:
    result = build_identity_episode_result(
        [
            _row(0, "A", "track-a"),
            _row(0, "B", "track-b"),
        ],
        explicit_authority={("video-a", "A"): "track-a"},
    )

    assert result.authority_policy == EXPLICIT_AUTHORITY_POLICY
    assert len(result.authorities) == 1
    assert len(result.authoritative_correct_rows) == 1
    assert len(result.ambiguous_rows) == 1
    assert result.ambiguous_rows[0].reason == "explicit_authority_missing_gt"

    with pytest.raises(ValueError, match="one-to-one"):
        build_identity_episode_result(
            [],
            explicit_authority={
                ("video-a", "A"): "same-track",
                ("video-a", "B"): "same-track",
            },
        )


def test_prediction_owned_by_another_gt_cannot_seed_new_authority() -> None:
    result = build_identity_episode_result(
        [
            _row(0, "A", "track-a"),
            _row(1, "B", "track-a"),
        ]
    )

    assert {(item.gt_id, item.pred_id) for item in result.authorities} == {
        ("A", "track-a")
    }
    assert len(result.ambiguous_rows) == 1
    assert (
        result.ambiguous_rows[0].reason
        == "prediction_authority_belongs_to_other_gt"
    )


def test_unmapped_wrong_prediction_has_no_pairwise_target() -> None:
    result = build_identity_episode_result(
        [
            _row(0, "A", "track-a"),
            _row(1, "A", "new-track"),
        ]
    )

    assert result.episodes[0].target_gt_ids == (None,)
    assert result.pairwise_events == ()


def test_reciprocal_exchange_is_counted_once_not_once_per_gt() -> None:
    result = build_identity_episode_result(
        [
            *_correct_pair(0),
            *_swapped_pair(1),
        ]
    )

    assert result.identity_error_episode_count == 2
    assert len(result.pairwise_events) == 1
    assert result.pairwise_events[0].gt_ids == ("A", "B")
    assert len(set(result.pairwise_events[0].linked_episode_ids)) == 2


def test_three_way_cycle_is_not_a_pairwise_swap() -> None:
    result = build_identity_episode_result(
        [
            _row(0, "A", "track-a"),
            _row(0, "B", "track-b"),
            _row(0, "C", "track-c"),
            _row(1, "A", "track-b"),
            _row(1, "B", "track-c"),
            _row(1, "C", "track-a"),
        ]
    )

    assert result.identity_error_episode_count == 3
    assert result.pairwise_events == ()


def test_sequence_boundaries_never_connect_episodes_or_authority() -> None:
    rows = [
        *_correct_pair(0, sequence_key="video-a"),
        *_swapped_pair(1, sequence_key="video-a"),
        *_correct_pair(0, sequence_key="video-b"),
        *_swapped_pair(1, sequence_key="video-b"),
    ]

    result = build_identity_episode_result(rows)

    assert result.identity_error_episode_count == 4
    assert len(result.pairwise_events) == 2
    assert {episode.sequence_key for episode in result.episodes} == {
        "video-a",
        "video-b",
    }
    assert all(
        len({row_key[0] for row_key in episode.row_keys}) == 1
        for episode in result.episodes
    )


def test_canonical_order_makes_results_and_event_ids_deterministic() -> None:
    rows = [
        *_correct_pair(0),
        *_swapped_pair(1),
        *_correct_pair(2),
    ]

    forward = build_identity_episode_result(rows)
    reversed_result = build_identity_episode_result(reversed(rows))

    assert asdict(forward) == asdict(reversed_result)
    assert [item.event_id for item in forward.episodes] == [
        item.event_id for item in reversed_result.episodes
    ]


def test_wrong_row_conservation_is_exact() -> None:
    result = build_identity_episode_result(
        [
            *_correct_pair(0),
            *_swapped_pair(1),
            *_swapped_pair(2),
            *_correct_pair(3),
        ]
    )
    episode_keys = [
        key
        for episode in result.episodes
        for key in episode.row_keys
    ]

    assert sorted(episode_keys) == sorted(
        row.key for row in result.authoritative_wrong_rows
    )
    assert result.wrong_id_rows_input == 4
    assert result.wrong_id_rows_classified == 4
    assert result.wrong_id_rows_unclassified == 0
    assert result.wrong_id_rows_double_counted == 0


@pytest.mark.parametrize(
    "rows, expected_message",
    [
        (
            [
                _row(0, "A", "track-a"),
                _row(0, "A", "track-b"),
            ],
            "duplicate GT",
        ),
        (
            [
                _row(0, "A", "track-a"),
                _row(0, "B", "track-a"),
            ],
            "duplicate prediction",
        ),
    ],
)
def test_duplicate_frame_identity_rows_fail_closed(
    rows: list[MatchedIdentityRow],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        build_identity_episode_result(rows)


@pytest.mark.parametrize("fps", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_fps_fails_closed(fps: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        build_identity_episode_result(
            [_row(0, "A", "track-a")],
            fps_by_sequence={"video-a": fps},
        )


def test_seconds_are_null_without_complete_time_authority() -> None:
    rows = [
        _row(0, "A", "track-a", sequence_key="video-a"),
        _row(1, "A", "track-x", sequence_key="video-a"),
        _row(0, "B", "track-b", sequence_key="video-b"),
    ]

    result = build_identity_episode_result(
        rows,
        fps_by_sequence={"video-a": 30.0},
    )

    assert result.wrong_id_matched_frames == 1
    assert result.wrong_id_matched_seconds is None
    assert result.episodes[0].duration_seconds == pytest.approx(1 / 30)
