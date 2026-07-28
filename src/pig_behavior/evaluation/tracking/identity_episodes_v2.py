"""Versioned identity-error episode construction for tracking evaluator V2.

This module consumes already matched observations.  It does not perform spatial
matching, association, or identity remapping.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

IDENTITY_EPISODE_CONTRACT_ID = "IDENTITY_ERROR_EPISODES_V2"
FIRST_OBSERVATION_AUTHORITY_POLICY = "IDENTITY_AUTHORITY_FIRST_OBSERVATION_V2"
EXPLICIT_AUTHORITY_POLICY = "IDENTITY_AUTHORITY_EXPLICIT_V2"
MAX_EPISODE_FRAME_DELTA = 15
PAIRWISE_PERSISTENCE_OBSERVATIONS = 60

EpisodeStatus = Literal["recovered", "terminal", "censored"]
AuthoritySource = Literal["explicit", "first_unambiguous_observation"]
PersistenceBasis = Literal[
    "not_persistent",
    "direct_observation_horizon",
    "terminal_to_both",
    "direct_observation_horizon_and_terminal_to_both",
]
RowKey = tuple[str, int, str, str]


@dataclass(frozen=True, slots=True)
class MatchedIdentityRow:
    """One eligible, one-to-one GT/prediction match from a single sequence."""

    sequence_key: str
    frame: int
    gt_id: str
    pred_id: str
    authority_ambiguous: bool = False

    @property
    def key(self) -> RowKey:
        """Return the stable conservation key for this matched row."""
        return (self.sequence_key, self.frame, self.gt_id, self.pred_id)


@dataclass(frozen=True, slots=True)
class IdentityAuthority:
    """One frozen GT-to-prediction identity authority relation."""

    sequence_key: str
    gt_id: str
    pred_id: str
    source: AuthoritySource
    established_frame: int | None


@dataclass(frozen=True, slots=True)
class AmbiguousIdentityRow:
    """A matched row retained outside authoritative severity totals."""

    row: MatchedIdentityRow
    reason: str


@dataclass(frozen=True, slots=True)
class IdentityErrorEpisode:
    """A GT-primary connected component of authoritative wrong-ID rows."""

    event_id: str
    sequence_key: str
    gt_id: str
    expected_pred_id: str
    start_frame: int
    end_frame: int
    row_keys: tuple[RowKey, ...]
    observed_pred_ids: tuple[str, ...]
    target_gt_ids: tuple[str | None, ...]
    duration_frames: int
    duration_seconds: float | None
    status: EpisodeStatus
    recovery_frame: int | None
    recovery_latency_seconds: float | None


@dataclass(frozen=True, slots=True)
class PairwiseIdentitySwapEvent:
    """A reciprocal pair cross-link; primary GT episodes remain unchanged."""

    event_id: str
    sequence_key: str
    gt_ids: tuple[str, str]
    start_frame: int
    end_frame: int
    direct_joint_frames: tuple[int, ...]
    direct_joint_observations: int
    linked_episode_ids: tuple[str, str]
    persistent: bool
    persistence_basis: PersistenceBasis


@dataclass(frozen=True, slots=True)
class IdentityEpisodeResult:
    """Deterministic result and conservation evidence for one or more sequences."""

    contract_id: str
    authority_policy: str
    authorities: tuple[IdentityAuthority, ...]
    authoritative_correct_rows: tuple[MatchedIdentityRow, ...]
    authoritative_wrong_rows: tuple[MatchedIdentityRow, ...]
    ambiguous_rows: tuple[AmbiguousIdentityRow, ...]
    episodes: tuple[IdentityErrorEpisode, ...]
    pairwise_events: tuple[PairwiseIdentitySwapEvent, ...]
    wrong_id_matched_seconds: float | None
    wrong_id_rows_input: int
    wrong_id_rows_classified: int
    wrong_id_rows_unclassified: int
    wrong_id_rows_double_counted: int

    @property
    def identity_error_episode_count(self) -> int:
        return len(self.episodes)

    @property
    def recovered_identity_error_episode_count(self) -> int:
        return sum(episode.status == "recovered" for episode in self.episodes)

    @property
    def terminal_identity_error_episode_count(self) -> int:
        return sum(episode.status == "terminal" for episode in self.episodes)

    @property
    def censored_identity_error_episode_count(self) -> int:
        return sum(episode.status == "censored" for episode in self.episodes)

    @property
    def persistent_pairwise_identity_swap_count(self) -> int:
        return sum(event.persistent for event in self.pairwise_events)

    @property
    def wrong_id_matched_frames(self) -> int:
        return len(self.authoritative_wrong_rows)


@dataclass(frozen=True, slots=True)
class _ResolvedRow:
    row: MatchedIdentityRow
    expected_pred_id: str
    is_correct: bool
    target_gt_id: str | None


def _stable_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:20]}"


def _validate_rows(rows: tuple[MatchedIdentityRow, ...]) -> None:
    seen_gt: set[tuple[str, int, str]] = set()
    seen_pred: set[tuple[str, int, str]] = set()
    for row in rows:
        if not isinstance(row, MatchedIdentityRow):
            raise TypeError("rows must contain MatchedIdentityRow values")
        if not row.sequence_key or not row.gt_id or not row.pred_id:
            raise ValueError("sequence_key, gt_id, and pred_id must be non-empty")
        if isinstance(row.frame, bool) or not isinstance(row.frame, int):
            raise TypeError("frame must be an integer")
        if row.frame < 0:
            raise ValueError("frame must be non-negative")
        if not isinstance(row.authority_ambiguous, bool):
            raise TypeError("authority_ambiguous must be boolean")
        gt_key = (row.sequence_key, row.frame, row.gt_id)
        pred_key = (row.sequence_key, row.frame, row.pred_id)
        if gt_key in seen_gt:
            raise ValueError(f"duplicate GT identity in frame: {gt_key!r}")
        if pred_key in seen_pred:
            raise ValueError(f"duplicate prediction identity in frame: {pred_key!r}")
        seen_gt.add(gt_key)
        seen_pred.add(pred_key)


def _validate_fps(
    fps_by_sequence: Mapping[str, float] | None,
) -> dict[str, float]:
    if fps_by_sequence is None:
        return {}
    result: dict[str, float] = {}
    for sequence_key, fps_value in fps_by_sequence.items():
        if not isinstance(sequence_key, str) or not sequence_key:
            raise ValueError("FPS sequence keys must be non-empty strings")
        if isinstance(fps_value, bool):
            raise TypeError("FPS must be numeric")
        fps = float(fps_value)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be finite and positive")
        result[sequence_key] = fps
    return result


def _load_explicit_authority(
    explicit_authority: Mapping[tuple[str, str], str] | None,
) -> tuple[
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
    list[IdentityAuthority],
]:
    by_gt: dict[tuple[str, str], str] = {}
    by_pred: dict[tuple[str, str], str] = {}
    authorities: list[IdentityAuthority] = []
    if explicit_authority is None:
        return by_gt, by_pred, authorities

    validated_items: list[tuple[tuple[str, str], str]] = []
    for raw_key, pred_id in explicit_authority.items():
        if (
            not isinstance(raw_key, tuple)
            or len(raw_key) != 2
            or not all(isinstance(value, str) and value for value in raw_key)
        ):
            raise ValueError("authority keys must be (sequence_key, gt_id)")
        if not isinstance(pred_id, str) or not pred_id:
            raise ValueError("authority prediction IDs must be non-empty strings")
        validated_items.append((raw_key, pred_id))

    for raw_key, pred_id in sorted(validated_items):
        sequence_key, gt_id = raw_key
        pred_key = (sequence_key, pred_id)
        previous_gt = by_pred.get(pred_key)
        if previous_gt is not None and previous_gt != gt_id:
            raise ValueError("explicit authority must be one-to-one per sequence")
        by_gt[(sequence_key, gt_id)] = pred_id
        by_pred[pred_key] = gt_id
        authorities.append(
            IdentityAuthority(
                sequence_key=sequence_key,
                gt_id=gt_id,
                pred_id=pred_id,
                source="explicit",
                established_frame=None,
            )
        )
    return by_gt, by_pred, authorities


def _resolve_rows(
    rows: tuple[MatchedIdentityRow, ...],
    explicit_authority: Mapping[tuple[str, str], str] | None,
) -> tuple[
    tuple[_ResolvedRow, ...],
    tuple[AmbiguousIdentityRow, ...],
    tuple[IdentityAuthority, ...],
]:
    by_gt, by_pred, authorities = _load_explicit_authority(explicit_authority)
    resolved: list[_ResolvedRow] = []
    ambiguous: list[AmbiguousIdentityRow] = []
    infer_authority = explicit_authority is None

    for row in rows:
        gt_key = (row.sequence_key, row.gt_id)
        pred_key = (row.sequence_key, row.pred_id)
        if row.authority_ambiguous:
            ambiguous.append(
                AmbiguousIdentityRow(
                    row=row,
                    reason="input_authority_ambiguous",
                )
            )
            continue

        expected_pred_id = by_gt.get(gt_key)
        if expected_pred_id is None:
            if not infer_authority:
                ambiguous.append(
                    AmbiguousIdentityRow(
                        row=row,
                        reason="explicit_authority_missing_gt",
                    )
                )
                continue
            existing_gt = by_pred.get(pred_key)
            if existing_gt is not None:
                ambiguous.append(
                    AmbiguousIdentityRow(
                        row=row,
                        reason="prediction_authority_belongs_to_other_gt",
                    )
                )
                continue
            expected_pred_id = row.pred_id
            by_gt[gt_key] = expected_pred_id
            by_pred[pred_key] = row.gt_id
            authorities.append(
                IdentityAuthority(
                    sequence_key=row.sequence_key,
                    gt_id=row.gt_id,
                    pred_id=row.pred_id,
                    source="first_unambiguous_observation",
                    established_frame=row.frame,
                )
            )

        is_correct = row.pred_id == expected_pred_id
        target_gt_id = None if is_correct else by_pred.get(pred_key)
        resolved.append(
            _ResolvedRow(
                row=row,
                expected_pred_id=expected_pred_id,
                is_correct=is_correct,
                target_gt_id=target_gt_id,
            )
        )

    authorities.sort(
        key=lambda item: (
            item.sequence_key,
            item.gt_id,
            item.pred_id,
            item.established_frame if item.established_frame is not None else -1,
        )
    )
    return tuple(resolved), tuple(ambiguous), tuple(authorities)


def _episode_from_component(
    component: list[_ResolvedRow],
    all_gt_rows: list[_ResolvedRow],
    fps_by_sequence: Mapping[str, float],
) -> IdentityErrorEpisode:
    first = component[0]
    last = component[-1]
    later_correct = next(
        (
            item
            for item in all_gt_rows
            if item.row.frame > last.row.frame and item.is_correct
        ),
        None,
    )
    final_authoritative = all_gt_rows[-1]
    if later_correct is not None:
        status: EpisodeStatus = "recovered"
    elif final_authoritative.row.key == last.row.key:
        status = "terminal"
    else:
        status = "censored"

    fps = fps_by_sequence.get(first.row.sequence_key)
    duration_seconds = len(component) / fps if fps is not None else None
    recovery_frame = later_correct.row.frame if later_correct is not None else None
    recovery_latency_seconds = None
    if recovery_frame is not None and fps is not None:
        recovery_latency_seconds = (recovery_frame - last.row.frame) / fps

    row_keys = tuple(item.row.key for item in component)
    event_id = _stable_id(
        "identity-episode-v2",
        {
            "sequence_key": first.row.sequence_key,
            "gt_id": first.row.gt_id,
            "row_keys": row_keys,
        },
    )
    return IdentityErrorEpisode(
        event_id=event_id,
        sequence_key=first.row.sequence_key,
        gt_id=first.row.gt_id,
        expected_pred_id=first.expected_pred_id,
        start_frame=first.row.frame,
        end_frame=last.row.frame,
        row_keys=row_keys,
        observed_pred_ids=tuple(item.row.pred_id for item in component),
        target_gt_ids=tuple(item.target_gt_id for item in component),
        duration_frames=len(component),
        duration_seconds=duration_seconds,
        status=status,
        recovery_frame=recovery_frame,
        recovery_latency_seconds=recovery_latency_seconds,
    )


def _build_episodes(
    resolved_rows: tuple[_ResolvedRow, ...],
    fps_by_sequence: Mapping[str, float],
) -> tuple[IdentityErrorEpisode, ...]:
    rows_by_gt: dict[tuple[str, str], list[_ResolvedRow]] = defaultdict(list)
    for item in resolved_rows:
        rows_by_gt[(item.row.sequence_key, item.row.gt_id)].append(item)

    episodes: list[IdentityErrorEpisode] = []
    for gt_rows in rows_by_gt.values():
        gt_rows.sort(key=lambda item: (item.row.frame, item.row.pred_id))
        component: list[_ResolvedRow] = []
        for item in gt_rows:
            if item.is_correct:
                if component:
                    episodes.append(
                        _episode_from_component(
                            component,
                            gt_rows,
                            fps_by_sequence,
                        )
                    )
                    component = []
                continue
            if (
                component
                and item.row.frame - component[-1].row.frame
                > MAX_EPISODE_FRAME_DELTA
            ):
                episodes.append(
                    _episode_from_component(
                        component,
                        gt_rows,
                        fps_by_sequence,
                    )
                )
                component = []
            component.append(item)
        if component:
            episodes.append(
                _episode_from_component(
                    component,
                    gt_rows,
                    fps_by_sequence,
                )
            )

    episodes.sort(
        key=lambda item: (
            item.sequence_key,
            item.gt_id,
            item.start_frame,
            item.end_frame,
            item.event_id,
        )
    )
    return tuple(episodes)


def _has_correct_between(
    correct_frames: Mapping[tuple[str, str], set[int]],
    sequence_key: str,
    gt_ids: tuple[str, str],
    previous_frame: int,
    current_frame: int,
) -> bool:
    return any(
        any(
            previous_frame < frame < current_frame
            for frame in correct_frames.get((sequence_key, gt_id), set())
        )
        for gt_id in gt_ids
    )


def _pair_event_from_component(
    sequence_key: str,
    gt_ids: tuple[str, str],
    component: list[tuple[int, _ResolvedRow, _ResolvedRow]],
    episode_by_row: Mapping[RowKey, IdentityErrorEpisode],
) -> PairwiseIdentitySwapEvent:
    frames = tuple(item[0] for item in component)
    first_a = component[0][1]
    first_b = component[0][2]
    episode_a = episode_by_row[first_a.row.key]
    episode_b = episode_by_row[first_b.row.key]
    linked_ids = (episode_a.event_id, episode_b.event_id)

    if any(
        episode_by_row[row.row.key].event_id != linked_event_id
        for _, row_a, row_b in component
        for row, linked_event_id in (
            (row_a, linked_ids[0]),
            (row_b, linked_ids[1]),
        )
    ):
        raise RuntimeError("one reciprocal component crossed a primary episode")

    terminal_to_both = (
        episode_a.status == "terminal"
        and episode_b.status == "terminal"
        and episode_a.target_gt_ids[-1] == gt_ids[1]
        and episode_b.target_gt_ids[-1] == gt_ids[0]
    )
    horizon_met = len(frames) >= PAIRWISE_PERSISTENCE_OBSERVATIONS
    persistent = horizon_met or terminal_to_both
    if horizon_met and terminal_to_both:
        basis: PersistenceBasis = (
            "direct_observation_horizon_and_terminal_to_both"
        )
    elif horizon_met:
        basis = "direct_observation_horizon"
    elif terminal_to_both:
        basis = "terminal_to_both"
    else:
        basis = "not_persistent"

    event_id = _stable_id(
        "pairwise-swap-v2",
        {
            "sequence_key": sequence_key,
            "gt_ids": gt_ids,
            "frames": frames,
            "linked_episode_ids": linked_ids,
        },
    )
    return PairwiseIdentitySwapEvent(
        event_id=event_id,
        sequence_key=sequence_key,
        gt_ids=gt_ids,
        start_frame=frames[0],
        end_frame=frames[-1],
        direct_joint_frames=frames,
        direct_joint_observations=len(frames),
        linked_episode_ids=linked_ids,
        persistent=persistent,
        persistence_basis=basis,
    )


def _build_pairwise_events(
    resolved_rows: tuple[_ResolvedRow, ...],
    episodes: tuple[IdentityErrorEpisode, ...],
) -> tuple[PairwiseIdentitySwapEvent, ...]:
    wrong_by_frame: dict[tuple[str, int], dict[str, _ResolvedRow]] = defaultdict(dict)
    correct_frames: dict[tuple[str, str], set[int]] = defaultdict(set)
    for item in resolved_rows:
        gt_key = (item.row.sequence_key, item.row.gt_id)
        if item.is_correct:
            correct_frames[gt_key].add(item.row.frame)
        else:
            wrong_by_frame[(item.row.sequence_key, item.row.frame)][
                item.row.gt_id
            ] = item

    direct_by_pair: dict[
        tuple[str, tuple[str, str]],
        list[tuple[int, _ResolvedRow, _ResolvedRow]],
    ] = defaultdict(list)
    for (sequence_key, frame), by_gt in wrong_by_frame.items():
        for gt_id, item in by_gt.items():
            target_gt_id = item.target_gt_id
            if target_gt_id is None or gt_id >= target_gt_id:
                continue
            counterpart = by_gt.get(target_gt_id)
            if counterpart is None or counterpart.target_gt_id != gt_id:
                continue
            gt_ids = (gt_id, target_gt_id)
            direct_by_pair[(sequence_key, gt_ids)].append(
                (frame, item, counterpart)
            )

    episode_by_row = {
        row_key: episode
        for episode in episodes
        for row_key in episode.row_keys
    }
    events: list[PairwiseIdentitySwapEvent] = []
    for (sequence_key, gt_ids), observations in direct_by_pair.items():
        observations.sort(key=lambda item: item[0])
        component: list[tuple[int, _ResolvedRow, _ResolvedRow]] = []
        for observation in observations:
            frame = observation[0]
            if component:
                previous_frame = component[-1][0]
                disconnected = (
                    frame - previous_frame > MAX_EPISODE_FRAME_DELTA
                    or _has_correct_between(
                        correct_frames,
                        sequence_key,
                        gt_ids,
                        previous_frame,
                        frame,
                    )
                )
                if disconnected:
                    events.append(
                        _pair_event_from_component(
                            sequence_key,
                            gt_ids,
                            component,
                            episode_by_row,
                        )
                    )
                    component = []
            component.append(observation)
        if component:
            events.append(
                _pair_event_from_component(
                    sequence_key,
                    gt_ids,
                    component,
                    episode_by_row,
                )
            )

    events.sort(
        key=lambda item: (
            item.sequence_key,
            item.gt_ids,
            item.start_frame,
            item.end_frame,
            item.event_id,
        )
    )
    return tuple(events)


def _validate_conservation(
    wrong_rows: tuple[MatchedIdentityRow, ...],
    episodes: tuple[IdentityErrorEpisode, ...],
    pairwise_events: tuple[PairwiseIdentitySwapEvent, ...],
) -> None:
    input_keys = Counter(row.key for row in wrong_rows)
    classified_keys = Counter(
        row_key
        for episode in episodes
        for row_key in episode.row_keys
    )
    if input_keys != classified_keys:
        raise RuntimeError("wrong-ID row conservation failed")
    if any(episode.duration_frames != len(episode.row_keys) for episode in episodes):
        raise RuntimeError("episode animal-frame duration is inconsistent")
    if any(
        len(set(event.gt_ids)) != 2
        or event.gt_ids != tuple(sorted(event.gt_ids))
        for event in pairwise_events
    ):
        raise RuntimeError("pairwise event identity conservation failed")
    pair_keys = [
        (
            event.sequence_key,
            event.gt_ids,
            event.direct_joint_frames,
        )
        for event in pairwise_events
    ]
    if len(pair_keys) != len(set(pair_keys)):
        raise RuntimeError("pairwise event was counted twice")


def build_identity_episode_result(
    rows: Iterable[MatchedIdentityRow],
    *,
    explicit_authority: Mapping[tuple[str, str], str] | None = None,
    fps_by_sequence: Mapping[str, float] | None = None,
) -> IdentityEpisodeResult:
    """Build V2 identity episodes from pre-matched rows.

    ``explicit_authority`` is all-or-explicit: a GT omitted from a supplied map
    remains audit-ambiguous.  Without a map, authority is frozen at the first
    unambiguous matched observation.
    """

    input_rows = tuple(rows)
    _validate_rows(input_rows)
    canonical_rows = tuple(
        sorted(
            input_rows,
            key=lambda row: (
                row.sequence_key,
                row.frame,
                row.gt_id,
                row.pred_id,
            ),
        )
    )
    canonical_fps = _validate_fps(fps_by_sequence)
    resolved_rows, ambiguous_rows, authorities = _resolve_rows(
        canonical_rows,
        explicit_authority,
    )
    correct_rows = tuple(item.row for item in resolved_rows if item.is_correct)
    wrong_rows = tuple(item.row for item in resolved_rows if not item.is_correct)
    episodes = _build_episodes(resolved_rows, canonical_fps)
    pairwise_events = _build_pairwise_events(resolved_rows, episodes)
    _validate_conservation(wrong_rows, episodes, pairwise_events)

    sequence_keys = {row.sequence_key for row in canonical_rows}
    seconds_available = bool(sequence_keys) and sequence_keys <= canonical_fps.keys()
    wrong_seconds = None
    if seconds_available:
        wrong_seconds = sum(
            1.0 / canonical_fps[row.sequence_key]
            for row in wrong_rows
        )

    return IdentityEpisodeResult(
        contract_id=IDENTITY_EPISODE_CONTRACT_ID,
        authority_policy=(
            FIRST_OBSERVATION_AUTHORITY_POLICY
            if explicit_authority is None
            else EXPLICIT_AUTHORITY_POLICY
        ),
        authorities=authorities,
        authoritative_correct_rows=correct_rows,
        authoritative_wrong_rows=wrong_rows,
        ambiguous_rows=ambiguous_rows,
        episodes=episodes,
        pairwise_events=pairwise_events,
        wrong_id_matched_seconds=wrong_seconds,
        wrong_id_rows_input=len(wrong_rows),
        wrong_id_rows_classified=sum(
            episode.duration_frames for episode in episodes
        ),
        wrong_id_rows_unclassified=0,
        wrong_id_rows_double_counted=0,
    )
