"""Side-effect-free H2-CDSP current-main shadow diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from pig_behavior.tracking.config import TrackingConfig
from pig_behavior.tracking.schemas import FixedTrack, TrackingRuntimeState

CONFIDENCE_HALF_LIFE_FRAMES = 6.0
APPEARANCE_HALF_LIFE_FRAMES = 8.0
MOTION_HALF_LIFE_FRAMES = 4.0
BASE_UNCERTAINTY_GROWTH = 0.05
WEAK_MOTION_UNCERTAINTY_GROWTH = 0.10
BOUNDARY_PENALTY = 0.15
MAXIMUM_PRESERVATION_AGE = 10
MINIMUM_USABLE_CONFIDENCE = 0.30
MAXIMUM_USABLE_UNCERTAINTY = 0.75

H2_SHADOW_COUNTERS = (
    "h2_shadow_stage_calls",
    "h2_shadow_visible_confirmed_tracks",
    "h2_shadow_dropout_entries",
    "h2_shadow_baseline_state_loss_points",
    "h2_shadow_preservation_candidates",
    "h2_shadow_preservable_states",
    "h2_shadow_unpreservable_missing_core",
    "h2_shadow_unpreservable_low_initial_quality",
    "h2_shadow_states_expired",
    "h2_shadow_states_invalidated",
    "h2_shadow_states_surviving_to_reentry",
    "h2_shadow_reentry_opportunities",
    "h2_shadow_extra_usable_state_at_reentry",
    "h2_shadow_control_preservation_events",
    "h2_shadow_control_overpreservation",
    "h2_shadow_invalid_numeric",
    "h2_shadow_terminal_revival_blocked",
)


@dataclass(frozen=True, slots=True)
class H2FormulaResult:
    """Frozen formula output; never an assignment or mutation command."""

    state: str
    confidence: float
    uncertainty: float
    appearance_reliability: float
    motion_reliability: float
    usable: bool
    invalidation_reason: str
    direct_assignment: bool = False
    reserves_detection: bool = False


@dataclass(frozen=True, slots=True)
class H2TrustedSnapshot:
    """Immutable causal copy of state accepted by the ordinary baseline."""

    sequence_token: object
    last_trusted_frame_index: int
    bbox: tuple[float, float, float, float]
    normalized_velocity: tuple[float, float]
    appearance_available: bool
    appearance_quality: float
    motion_available: bool
    motion_quality: float
    boundary_seen: bool = False


@dataclass(frozen=True, slots=True)
class H2TrackShadowState:
    """Per-track diagnostic state stored outside the production track."""

    state: str
    snapshot: H2TrustedSnapshot | None
    last_observed_frame_index: int
    confidence: float
    uncertainty: float
    usable: bool
    baseline_track_state: str
    baseline_state_loss_seen: bool = False
    terminal: bool = False


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def evaluate_h2_cdsp_formula(
    *,
    age: int,
    initial_confidence: float = 1.0,
    initial_uncertainty: float = 0.10,
    appearance_available: bool,
    appearance_quality: float,
    motion_available: bool,
    motion_quality: float,
    occlusion_support: bool,
    boundary_seen: bool = False,
    trusted_match: bool = False,
    sequence_changed: bool = False,
    baseline_terminated: bool = False,
    source_state: str | None = None,
) -> H2FormulaResult:
    """Evaluate the frozen preservation formulas independently of association."""
    numeric = (
        initial_confidence,
        initial_uncertainty,
        appearance_quality,
        motion_quality,
    )
    if baseline_terminated or source_state == "TERMINATED":
        return H2FormulaResult("TERMINATED", 0.0, 1.0, 0.0, 0.0, False, "terminal")
    if sequence_changed:
        return H2FormulaResult(
            "INVALIDATED", 0.0, 1.0, 0.0, 0.0, False, "sequence_boundary"
        )
    if age < 0 or not all(math.isfinite(float(value)) for value in numeric):
        return H2FormulaResult(
            "INVALIDATED", 0.0, 1.0, 0.0, 0.0, False, "invalid_numeric"
        )
    if trusted_match:
        return H2FormulaResult(
            "VISIBLE_CONFIRMED",
            1.0,
            0.10,
            _clip(appearance_quality if appearance_available else 0.0, 0.0, 1.0),
            _clip(motion_quality if motion_available else 0.0, 0.0, 1.0),
            False,
            "",
        )
    confidence = _clip(
        initial_confidence * 2.0 ** (-age / CONFIDENCE_HALF_LIFE_FRAMES),
        0.0,
        1.0,
    )
    appearance_reliability = _clip(
        (appearance_quality if appearance_available else 0.0)
        * 2.0 ** (-age / APPEARANCE_HALF_LIFE_FRAMES),
        0.0,
        1.0,
    )
    motion_reliability = _clip(
        (motion_quality if motion_available else 0.0)
        * 2.0 ** (-age / MOTION_HALF_LIFE_FRAMES),
        0.0,
        1.0,
    )
    uncertainty = _clip(
        initial_uncertainty
        + age
        * (
            BASE_UNCERTAINTY_GROWTH
            + WEAK_MOTION_UNCERTAINTY_GROWTH * (1.0 - motion_reliability)
        )
        + (BOUNDARY_PENALTY if boundary_seen else 0.0),
        0.0,
        1.0,
    )
    if age <= 2:
        state = "DROPOUT_GRACE"
    elif occlusion_support and age <= 6:
        state = "OCCLUSION_PRESERVED"
    else:
        state = "STALE_PRESERVED"
    reason = ""
    if age > MAXIMUM_PRESERVATION_AGE:
        reason = "maximum_age"
    elif confidence < MINIMUM_USABLE_CONFIDENCE:
        reason = "low_confidence"
    elif uncertainty > MAXIMUM_USABLE_UNCERTAINTY:
        reason = "high_uncertainty"
    if reason:
        state = "INVALIDATED"
        uncertainty = 1.0
    return H2FormulaResult(
        state,
        confidence,
        uncertainty,
        appearance_reliability,
        motion_reliability,
        not reason and state != "VISIBLE_CONFIRMED",
        reason,
    )


def _finite_bbox(track: FixedTrack) -> tuple[float, float, float, float] | None:
    box = np.asarray(track.last_box, dtype=np.float64)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        return None
    if float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
        return None
    return tuple(float(value) for value in box)


def _trusted_match(track: FixedTrack, cfg: TrackingConfig) -> bool:
    return (
        track.state == "VISIBLE"
        and track.last_source == "detected"
        and not track.last_ambiguous
        and track.hits >= 4
        and math.isfinite(float(track.last_score))
        and float(track.last_score) >= float(cfg.track_high_conf)
        and _finite_bbox(track) is not None
    )


def _snapshot(
    track: FixedTrack,
    frame_index: int,
    sequence_token: object,
) -> H2TrustedSnapshot | None:
    bbox = _finite_bbox(track)
    if bbox is None:
        return None
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    diagonal = math.hypot(width, height)
    velocity = np.asarray(track.reliable_velocity_xy, dtype=np.float64)
    motion_available = (
        velocity.shape == (2,)
        and np.all(np.isfinite(velocity))
        and len(track.reliable_center_history) >= 2
        and diagonal > 0.0
    )
    normalized_velocity_array = (
        velocity / diagonal
        if motion_available
        else np.zeros(2, dtype=np.float64)
    )
    velocity_norm = float(np.linalg.norm(normalized_velocity_array))
    if velocity_norm > 0.25:
        normalized_velocity_array *= 0.25 / velocity_norm
    normalized_velocity = tuple(
        float(value) for value in normalized_velocity_array
    )
    appearance = track.mean_hist()
    appearance_available = (
        appearance is not None
        and np.asarray(appearance).size > 0
        and np.all(np.isfinite(np.asarray(appearance)))
    )
    return H2TrustedSnapshot(
        sequence_token=sequence_token,
        last_trusted_frame_index=frame_index,
        bbox=bbox,
        normalized_velocity=normalized_velocity,
        appearance_available=appearance_available,
        appearance_quality=_clip(float(track.last_score), 0.0, 1.0)
        if appearance_available
        else 0.0,
        motion_available=motion_available,
        motion_quality=1.0 if motion_available else 0.0,
    )


def _propagated_geometry(
    snapshot: H2TrustedSnapshot,
    age: int,
    frame_dimensions: tuple[int, int] | None,
) -> tuple[tuple[float, float, float, float], bool]:
    x1, y1, x2, y2 = snapshot.bbox
    width = x2 - x1
    height = y2 - y1
    diagonal = math.hypot(width, height)
    dx = age * snapshot.normalized_velocity[0] * diagonal
    dy = age * snapshot.normalized_velocity[1] * diagonal
    propagated = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
    if frame_dimensions is None:
        return propagated, snapshot.boundary_seen
    frame_height, frame_width = frame_dimensions
    boundary_seen = snapshot.boundary_seen or (
        propagated[0] < 0.0
        or propagated[1] < 0.0
        or propagated[2] > float(frame_width)
        or propagated[3] > float(frame_height)
    )
    return propagated, boundary_seen


def _counter(
    runtime: TrackingRuntimeState,
    key: str,
    increment: int = 1,
) -> None:
    if key not in H2_SHADOW_COUNTERS or key not in runtime.telemetry:
        raise KeyError(f"non-canonical H2 shadow telemetry key: {key}")
    runtime.telemetry[key] = int(runtime.telemetry[key]) + int(increment)


def observe_h2_cdsp_shadow(
    tracks: Mapping[int, FixedTrack],
    *,
    frame_index: int,
    cfg: TrackingConfig,
    runtime: TrackingRuntimeState | None,
    sequence_token: object,
    detector_frame: bool = True,
    frame_dimensions: tuple[int, int] | None = None,
) -> None:
    """Observe immutable track copies and append diagnostic records only."""
    if runtime is None or not runtime.h2_shadow_enabled:
        return
    _counter(runtime, "h2_shadow_stage_calls")
    active_ids = set(tracks)
    for track_id in tuple(runtime.h2_shadow_track_states):
        if track_id in active_ids:
            continue
        prior = runtime.h2_shadow_track_states[track_id]
        runtime.h2_shadow_track_states[track_id] = H2TrackShadowState(
            "TERMINATED",
            prior.snapshot,
            frame_index,
            0.0,
            1.0,
            False,
            prior.baseline_track_state,
            prior.baseline_state_loss_seen,
            True,
        )
    for track_id in sorted(active_ids):
        track = tracks[track_id]
        prior = runtime.h2_shadow_track_states.get(track_id)
        if prior is not None and prior.terminal:
            _counter(runtime, "h2_shadow_terminal_revival_blocked")
            continue
        trusted = detector_frame and _trusted_match(track, cfg)
        reentry = bool(
            trusted
            and prior is not None
            and prior.snapshot is not None
            and prior.state != "VISIBLE_CONFIRMED"
        )
        reentry_survival = bool(reentry and prior and prior.usable)
        extra_at_reentry = bool(
            reentry_survival
            and prior
            and prior.baseline_state_loss_seen
        )
        if reentry:
            _counter(runtime, "h2_shadow_reentry_opportunities")
        if reentry_survival:
            _counter(runtime, "h2_shadow_states_surviving_to_reentry")
        if extra_at_reentry:
            _counter(runtime, "h2_shadow_extra_usable_state_at_reentry")
        if trusted:
            snapshot = _snapshot(track, frame_index, sequence_token)
            if snapshot is None:
                _counter(runtime, "h2_shadow_invalid_numeric")
                continue
            result = evaluate_h2_cdsp_formula(
                age=0,
                appearance_available=snapshot.appearance_available,
                appearance_quality=snapshot.appearance_quality,
                motion_available=snapshot.motion_available,
                motion_quality=snapshot.motion_quality,
                occlusion_support=False,
                trusted_match=True,
            )
            _counter(runtime, "h2_shadow_visible_confirmed_tracks")
        else:
            snapshot = prior.snapshot if prior is not None else None
            if snapshot is None or snapshot.sequence_token is not sequence_token:
                result = H2FormulaResult(
                    "INVALIDATED",
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    False,
                    "missing_core_or_sequence",
                )
                _counter(runtime, "h2_shadow_unpreservable_missing_core")
            else:
                age = frame_index - snapshot.last_trusted_frame_index
                propagated_geometry, boundary_seen = _propagated_geometry(
                    snapshot,
                    age,
                    frame_dimensions,
                )
                if boundary_seen and not snapshot.boundary_seen:
                    snapshot = replace(snapshot, boundary_seen=True)
                continuous = (
                    prior is not None
                    and frame_index == prior.last_observed_frame_index + 1
                )
                if not continuous:
                    result = H2FormulaResult(
                        "INVALIDATED",
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        False,
                        "frame_continuity",
                    )
                else:
                    occlusion_support = bool(
                        track.state == "OCCLUDED"
                        and (
                            track.last_source == "occlusion_hold"
                            or track.is_area_occluded
                            or track.last_merged_split
                        )
                    )
                    result = evaluate_h2_cdsp_formula(
                        age=age,
                        appearance_available=snapshot.appearance_available,
                        appearance_quality=snapshot.appearance_quality,
                        motion_available=snapshot.motion_available,
                        motion_quality=snapshot.motion_quality,
                        occlusion_support=occlusion_support,
                        boundary_seen=boundary_seen,
                    )
                    _counter(runtime, "h2_shadow_preservation_candidates")
                    if prior.state == "VISIBLE_CONFIRMED":
                        _counter(runtime, "h2_shadow_dropout_entries")
                    if result.usable:
                        _counter(runtime, "h2_shadow_preservable_states")
                    elif result.invalidation_reason == "maximum_age":
                        _counter(runtime, "h2_shadow_states_expired")
                    else:
                        _counter(runtime, "h2_shadow_states_invalidated")
        age = (
            frame_index - snapshot.last_trusted_frame_index
            if snapshot is not None
            else -1
        )
        bbox = snapshot.bbox if snapshot is not None else None
        propagated_geometry = (
            _propagated_geometry(snapshot, max(age, 0), frame_dimensions)[0]
            if snapshot is not None
            else None
        )
        baseline_bbox_available = _finite_bbox(track) is not None
        baseline_appearance_available = track.mean_hist() is not None
        baseline_motion_available = bool(track.reliable_center_history)
        baseline_state_loss = (
            track.ever_detected
            and not baseline_bbox_available
            and not baseline_appearance_available
            and not baseline_motion_available
        )
        if baseline_state_loss and not bool(
            prior and prior.baseline_state_loss_seen
        ):
            _counter(runtime, "h2_shadow_baseline_state_loss_points")
        baseline_state_loss_seen = bool(
            not trusted
            and (
                baseline_state_loss
                or (prior and prior.baseline_state_loss_seen)
            )
        )
        runtime.h2_shadow_track_states[track_id] = H2TrackShadowState(
            result.state,
            snapshot,
            frame_index,
            result.confidence,
            result.uncertainty,
            result.usable,
            track.state,
            baseline_state_loss_seen,
        )
        runtime.h2_shadow_transition_rows.append(
            {
                "frame_index": frame_index,
                "track_id": track_id,
                "baseline_state_before": (
                    prior.baseline_track_state if prior else "UNOBSERVED"
                ),
                "baseline_state_after": track.state,
                "shadow_state_before": prior.state if prior else "UNOBSERVED",
                "state_loss_reason": (
                    track.state_reason if baseline_state_loss else ""
                ),
                "last_trusted_detection_frame": (
                    snapshot.last_trusted_frame_index if snapshot else None
                ),
                "dropout_age": age,
                "last_trusted_bbox": bbox,
                "normalized_geometry": propagated_geometry,
                "causal_velocity_estimate": (
                    snapshot.normalized_velocity if snapshot else None
                ),
                "motion_available": (
                    snapshot.motion_available if snapshot else False
                ),
                "motion_quality": (
                    snapshot.motion_quality if snapshot else 0.0
                ),
                "appearance_available": (
                    snapshot.appearance_available if snapshot else False
                ),
                "appearance_quality": (
                    snapshot.appearance_quality if snapshot else 0.0
                ),
                "appearance_reliability": result.appearance_reliability,
                "initial_state_confidence": 1.0 if snapshot else 0.0,
                "shadow_preserved_confidence": result.confidence,
                "shadow_uncertainty": result.uncertainty,
                "motion_reliability": result.motion_reliability,
                "preservation_state": result.state,
                "expiry_frame": (
                    snapshot.last_trusted_frame_index
                    + MAXIMUM_PRESERVATION_AGE
                    if snapshot
                    else None
                ),
                "invalidation_reason": result.invalidation_reason,
                "preserved_state_available": result.usable,
                "baseline_state_loss": baseline_state_loss,
                "reentry_frame": frame_index if reentry else None,
                "preserved_state_available_at_reentry": reentry_survival,
                "extra_usable_evidence_relative_to_baseline": (
                    extra_at_reentry
                ),
                "direct_assignment": False,
                "reserves_detection": False,
            }
        )


__all__ = [
    "H2FormulaResult",
    "H2TrackShadowState",
    "H2TrustedSnapshot",
    "H2_SHADOW_COUNTERS",
    "evaluate_h2_cdsp_formula",
    "observe_h2_cdsp_shadow",
]
