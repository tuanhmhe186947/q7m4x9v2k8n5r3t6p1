"""Side-effect-free H1-r3 shadow support-density observer.

The bounded score is uncalibrated and is not a probability.  This module
returns diagnostic records only; it has no API that can reserve a detection or
modify an association.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pig_behavior.tracking.schemas import Detection, FixedTrack

H1_R3_SCORE_NAME = "owner_preference_lower_bound"
H1_R3_SCORE_THRESHOLD = 0.625
H1_R3_SUPPORT_MARGIN = 0.25
H1_R3_MIN_RELATIVE_OVERLAP = 0.10
H1_R3_MIN_DETECTION_CONFIDENCE = 0.25
H1_R3_MAX_DETECTION_OPPORTUNITIES = 8
H1_R3_NUMERIC_TOLERANCE = 1e-12

H1_R3_WEIGHTS: dict[str, float] = {
    "overlap_similarity": 0.60,
    "normalized_center_similarity": 0.0,
    "scale_similarity": 0.0,
    "appearance_similarity": 0.15,
    "motion_consistency": 0.10,
    "track_freshness": 0.15,
    "appearance_available": 0.0,
    "motion_available": 0.0,
}


@dataclass(frozen=True, slots=True)
class H1R3Evidence:
    """Symmetric evidence for one candidate track and one detection."""

    overlap_similarity: float | None
    normalized_center_similarity: float | None
    scale_similarity: float | None
    appearance_similarity: float | None
    motion_consistency: float | None
    track_freshness: float | None
    appearance_available: int
    motion_available: int
    overlap_valid: int
    freshness_valid: int
    appearance_quality: float
    motion_quality: float
    reference_source: str

    def export(self, prefix: str) -> dict[str, object]:
        """Return stable candidate-row fields with a declared side prefix."""
        return {
            f"{prefix}_{key}": value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True, slots=True)
class H1R3ShadowDecision:
    """Pure H1-r3 diagnostic result; never an association instruction."""

    hidden: H1R3Evidence
    visible: H1R3Evidence
    core_eligible: bool
    relative_overlap: float | None
    relative_freshness: float | None
    overlap_contribution: float | None
    freshness_contribution: float | None
    appearance_contribution: float | None
    motion_contribution: float | None
    appearance_lower: float | None
    motion_lower: float | None
    relative_owner_support_lower: float | None
    relative_owner_support_upper: float | None
    owner_preference_lower_bound: float | None
    activation_margin: float | None
    would_activate: bool
    abstention_reason: str


def _clip_unit(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _valid_box(box: np.ndarray | None) -> bool:
    if box is None:
        return False
    values = np.asarray(box, dtype=np.float64)
    return bool(
        values.shape == (4,)
        and np.all(np.isfinite(values))
        and values[2] > values[0]
        and values[3] > values[1]
    )


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            0.5 * (float(box[0]) + float(box[2])),
            0.5 * (float(box[1]) + float(box[3])),
        ],
        dtype=np.float64,
    )


def _box_diagonal(box: np.ndarray) -> float:
    return math.hypot(
        float(box[2]) - float(box[0]),
        float(box[3]) - float(box[1]),
    )


def _overlap_similarity(
    reference_box: np.ndarray,
    detection_box: np.ndarray,
) -> float:
    x1 = max(float(reference_box[0]), float(detection_box[0]))
    y1 = max(float(reference_box[1]), float(detection_box[1]))
    x2 = min(float(reference_box[2]), float(detection_box[2]))
    y2 = min(float(reference_box[3]), float(detection_box[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    reference_area = (
        (float(reference_box[2]) - float(reference_box[0]))
        * (float(reference_box[3]) - float(reference_box[1]))
    )
    detection_area = (
        (float(detection_box[2]) - float(detection_box[0]))
        * (float(detection_box[3]) - float(detection_box[1]))
    )
    union = reference_area + detection_area - intersection
    return _clip_unit(intersection / max(union, np.finfo(float).eps))


def _center_similarity(
    reference_box: np.ndarray,
    detection_box: np.ndarray,
) -> float:
    distance = float(
        np.linalg.norm(
            _box_center(reference_box) - _box_center(detection_box)
        )
    )
    denominator = _box_diagonal(reference_box) + _box_diagonal(detection_box)
    return 1.0 - _clip_unit(
        2.0 * distance / max(denominator, np.finfo(float).eps)
    )


def _scale_similarity(
    reference_box: np.ndarray,
    detection_box: np.ndarray,
) -> float:
    reference_area = (
        (float(reference_box[2]) - float(reference_box[0]))
        * (float(reference_box[3]) - float(reference_box[1]))
    )
    detection_area = (
        (float(detection_box[2]) - float(detection_box[0]))
        * (float(detection_box[3]) - float(detection_box[1]))
    )
    residual = abs(math.log(detection_area / reference_area)) / math.log(4.0)
    return 1.0 - _clip_unit(residual)


def _normalized_descriptor(
    descriptor: np.ndarray | None,
) -> np.ndarray | None:
    if descriptor is None:
        return None
    values = np.asarray(descriptor, dtype=np.float64).reshape(-1)
    if (
        values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        return None
    total = float(values.sum())
    return values / total if total > 0.0 else None


def _appearance_similarity(
    track_descriptor: np.ndarray | None,
    detection_descriptor: np.ndarray | None,
) -> tuple[float | None, int]:
    first = _normalized_descriptor(track_descriptor)
    second = _normalized_descriptor(detection_descriptor)
    if first is None or second is None or first.shape != second.shape:
        return None, 0
    coefficient = _clip_unit(float(np.sum(np.sqrt(first * second))))
    return _clip_unit(1.0 - math.sqrt(max(0.0, 1.0 - coefficient))), 1


def _motion_similarity(
    predicted_box: np.ndarray | None,
    detection_box: np.ndarray,
) -> tuple[float | None, int]:
    if not _valid_box(predicted_box) or not _valid_box(detection_box):
        return None, 0
    assert predicted_box is not None
    distance = float(
        np.linalg.norm(
            _box_center(predicted_box) - _box_center(detection_box)
        )
    )
    denominator = 0.5 * (
        _box_diagonal(predicted_box) + _box_diagonal(detection_box)
    )
    return (
        1.0
        - _clip_unit(distance / max(denominator, np.finfo(float).eps)),
        1,
    )


def _reference_box(
    track: FixedTrack,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, str]:
    """Use the common causal cascade without claiming unavailable LK quality."""
    if track.ever_detected:
        prediction = track.predicted_box(width, height)
        if _valid_box(prediction):
            return prediction, "causal_prediction"
    if _valid_box(track.reliable_box):
        assert track.reliable_box is not None
        return track.reliable_box.copy(), "last_confirmed"
    if track.ever_detected and _valid_box(track.last_box):
        return track.last_box.copy(), "last_confirmed"
    return None, "unavailable"


def build_h1_r3_evidence(
    track: FixedTrack,
    detection: Detection,
    width: int,
    height: int,
) -> H1R3Evidence:
    """Build the identical frozen H1-r3 feature map for either candidate."""
    reference_box, reference_source = _reference_box(track, width, height)
    box_valid = int(
        _valid_box(reference_box) and _valid_box(detection.box)
    )
    age = int(track.missed)
    freshness_valid = int(
        0 <= age <= H1_R3_MAX_DETECTION_OPPORTUNITIES
    )
    overlap = None
    center = None
    scale = None
    if box_valid:
        assert reference_box is not None
        overlap = _overlap_similarity(reference_box, detection.box)
        center = _center_similarity(reference_box, detection.box)
        scale = _scale_similarity(reference_box, detection.box)

    appearance, appearance_available = _appearance_similarity(
        track.mean_hist(),
        detection.hist,
    )
    appearance_quality = (
        2.0 ** (-float(age) / 4.0) if appearance_available else 0.0
    )

    motion_box = (
        track.predicted_box(width, height)
        if track.ever_detected and track.hits >= 2
        else None
    )
    motion, motion_available = _motion_similarity(
        motion_box,
        detection.box,
    )
    motion_quality = 0.5 if motion_available else 0.0
    freshness = (
        1.0 - _clip_unit(float(age) / H1_R3_MAX_DETECTION_OPPORTUNITIES)
        if freshness_valid
        else None
    )
    return H1R3Evidence(
        overlap_similarity=overlap,
        normalized_center_similarity=center,
        scale_similarity=scale,
        appearance_similarity=appearance,
        motion_consistency=motion,
        track_freshness=freshness,
        appearance_available=appearance_available,
        motion_available=motion_available,
        overlap_valid=box_valid,
        freshness_valid=freshness_valid,
        appearance_quality=appearance_quality,
        motion_quality=motion_quality,
        reference_source=reference_source,
    )


def decide_h1_r3_shadow(
    hidden: H1R3Evidence,
    visible: H1R3Evidence,
    *,
    detection_confidence: float,
) -> H1R3ShadowDecision:
    """Apply the frozen H1-r3 gate without returning an assignment command."""
    core_eligible = bool(
        math.isfinite(float(detection_confidence))
        and detection_confidence >= H1_R3_MIN_DETECTION_CONFIDENCE
        and hidden.overlap_valid
        and visible.overlap_valid
        and hidden.freshness_valid
        and visible.freshness_valid
    )
    if not core_eligible:
        reason = (
            "missing_core_overlap"
            if not hidden.overlap_valid or not visible.overlap_valid
            else "missing_core_freshness"
        )
        return H1R3ShadowDecision(
            hidden,
            visible,
            False,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            False,
            reason,
        )

    assert hidden.overlap_similarity is not None
    assert visible.overlap_similarity is not None
    assert hidden.track_freshness is not None
    assert visible.track_freshness is not None
    relative_overlap = (
        hidden.overlap_similarity - visible.overlap_similarity
    )
    relative_freshness = (
        hidden.track_freshness - visible.track_freshness
    )
    overlap_contribution = 0.60 * relative_overlap
    freshness_contribution = 0.15 * relative_freshness
    core_support = overlap_contribution + freshness_contribution

    appearance_contribution: float | None = None
    if hidden.appearance_available and visible.appearance_available:
        assert hidden.appearance_similarity is not None
        assert visible.appearance_similarity is not None
        appearance_contribution = (
            0.15
            * min(hidden.appearance_quality, visible.appearance_quality)
            * (
                hidden.appearance_similarity
                - visible.appearance_similarity
            )
        )
        appearance_lower = appearance_contribution
        appearance_upper = appearance_contribution
    else:
        appearance_lower = -0.15
        appearance_upper = 0.15

    motion_contribution: float | None = None
    if hidden.motion_available and visible.motion_available:
        assert hidden.motion_consistency is not None
        assert visible.motion_consistency is not None
        motion_contribution = (
            0.10
            * min(hidden.motion_quality, visible.motion_quality)
            * (hidden.motion_consistency - visible.motion_consistency)
        )
        motion_lower = motion_contribution
        motion_upper = motion_contribution
    else:
        motion_lower = -0.10
        motion_upper = 0.10

    lower = core_support + appearance_lower + motion_lower
    upper = core_support + appearance_upper + motion_upper
    score = _clip_unit(0.5 + 0.5 * lower)
    activation_margin = lower - H1_R3_SUPPORT_MARGIN
    overlap_passes = (
        relative_overlap + H1_R3_NUMERIC_TOLERANCE
        >= H1_R3_MIN_RELATIVE_OVERLAP
    )
    score_passes = (
        score + H1_R3_NUMERIC_TOLERANCE >= H1_R3_SCORE_THRESHOLD
    )
    support_passes = (
        lower + H1_R3_NUMERIC_TOLERANCE >= H1_R3_SUPPORT_MARGIN
    )
    would_activate = overlap_passes and score_passes and support_passes
    if would_activate:
        reason = ""
    elif not overlap_passes:
        reason = "relative_overlap_margin"
    else:
        reason = "below_threshold"
    return H1R3ShadowDecision(
        hidden,
        visible,
        True,
        relative_overlap,
        relative_freshness,
        overlap_contribution,
        freshness_contribution,
        appearance_contribution,
        motion_contribution,
        appearance_lower,
        motion_lower,
        lower,
        upper,
        score,
        activation_margin,
        would_activate,
        reason,
    )


def shadow_decision_row(
    decision: H1R3ShadowDecision,
    *,
    frame_index: int | None,
    hidden_track_id: int,
    visible_track_id: int,
    detection_index: int,
    selected_cost: float,
) -> dict[str, object]:
    """Serialize one deterministic row without development or GT metadata."""
    row: dict[str, object] = {
        "frame_index": frame_index,
        "hidden_track_id": hidden_track_id,
        "visible_track_id": visible_track_id,
        "detection_index": detection_index,
        **decision.hidden.export("hidden"),
        **decision.visible.export("visible"),
        "relative_overlap": decision.relative_overlap,
        "relative_freshness": decision.relative_freshness,
        "overlap_contribution": decision.overlap_contribution,
        "freshness_contribution": decision.freshness_contribution,
        "appearance_contribution": decision.appearance_contribution,
        "motion_contribution": decision.motion_contribution,
        "appearance_lower": decision.appearance_lower,
        "motion_lower": decision.motion_lower,
        "relative_owner_support_lower": (
            decision.relative_owner_support_lower
        ),
        "relative_owner_support_upper": (
            decision.relative_owner_support_upper
        ),
        H1_R3_SCORE_NAME: decision.owner_preference_lower_bound,
        "threshold": H1_R3_SCORE_THRESHOLD,
        "visible_competitor_margin": H1_R3_SUPPORT_MARGIN,
        "activation_margin": decision.activation_margin,
        "core_eligible": int(decision.core_eligible),
        "would_activate": int(decision.would_activate),
        "abstention_reason": decision.abstention_reason,
        "selected_cost": selected_cost,
        "actual_baseline_assignment": visible_track_id,
        "shadow_activation_would_disagree_with_baseline": int(
            decision.would_activate
        ),
    }
    return row


def record_h1_r3_shadow_counter(
    runtime: Any,
    key: str,
    amount: int = 1,
) -> None:
    """Record optional telemetry without participating in the decision."""
    if runtime is None:
        return
    runtime.telemetry[key] = int(runtime.telemetry.get(key, 0)) + int(amount)


__all__ = [
    "H1_R3_MAX_DETECTION_OPPORTUNITIES",
    "H1_R3_MIN_RELATIVE_OVERLAP",
    "H1_R3_SCORE_NAME",
    "H1_R3_SCORE_THRESHOLD",
    "H1_R3_SUPPORT_MARGIN",
    "H1_R3_WEIGHTS",
    "H1R3Evidence",
    "H1R3ShadowDecision",
    "build_h1_r3_evidence",
    "decide_h1_r3_shadow",
    "record_h1_r3_shadow_counter",
    "shadow_decision_row",
]
