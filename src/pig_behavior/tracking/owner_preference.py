"""Frozen symmetric H1-r2 hidden-owner preference mathematics.

The bounded score is uncalibrated and is not a probability. This module uses
only the current detection and causal track state supplied by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

OWNER_PREFERENCE_SCORE_NAME = "owner_preference_score"
OWNER_PREFERENCE_THRESHOLD = 0.60
OWNER_PREFERENCE_MIN_QUALITY_MARGIN = 0.20
OWNER_PREFERENCE_MIN_DETECTION_CONFIDENCE = 0.25
OWNER_PREFERENCE_MIN_HIDDEN_OVERLAP = 0.50
OWNER_PREFERENCE_MAX_DETECTION_OPPORTUNITIES = 5
OWNER_PREFERENCE_NUMERIC_TOLERANCE = 1e-12

OWNER_PREFERENCE_WEIGHTS: dict[str, float] = {
    "overlap_similarity": 0.250,
    "normalized_center_similarity": 0.200,
    "scale_similarity": 0.100,
    "appearance_similarity": 0.200,
    "motion_consistency": 0.100,
    "track_freshness": 0.100,
    "appearance_available": 0.025,
    "motion_available": 0.025,
}


@dataclass(frozen=True, slots=True)
class OwnerPreferenceFeatures:
    """The eight common-scale features used identically for either track."""

    overlap_similarity: float
    normalized_center_similarity: float
    scale_similarity: float
    appearance_similarity: float
    motion_consistency: float
    track_freshness: float
    appearance_available: float
    motion_available: float

    def as_dict(self) -> dict[str, float]:
        """Return features in the frozen coefficient order."""
        return {
            name: float(getattr(self, name))
            for name in OWNER_PREFERENCE_WEIGHTS
        }


@dataclass(frozen=True, slots=True)
class OwnerPreferenceDecision:
    """A deterministic H1-r2 apply/abstain decision."""

    owner_preference_score: float | None
    hidden_quality: float | None
    visible_quality: float | None
    apply: bool
    reason: str


def _clip_unit(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _valid_box(box: np.ndarray) -> bool:
    values = np.asarray(box, dtype=np.float64)
    return bool(
        values.shape == (4,)
        and np.all(np.isfinite(values))
        and values[2] > values[0]
        and values[3] > values[1]
    )


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.array(
        [(float(box[0]) + float(box[2])) * 0.5,
         (float(box[1]) + float(box[3])) * 0.5],
        dtype=np.float64,
    )


def _box_diagonal(box: np.ndarray) -> float:
    return math.hypot(
        float(box[2]) - float(box[0]),
        float(box[3]) - float(box[1]),
    )


def overlap_similarity(
    reference_box: np.ndarray,
    detection_box: np.ndarray,
) -> float:
    """Return clipped intersection-over-union for two valid boxes."""
    if not _valid_box(reference_box) or not _valid_box(detection_box):
        raise ValueError("overlap_similarity requires valid boxes")
    x1 = max(float(reference_box[0]), float(detection_box[0]))
    y1 = max(float(reference_box[1]), float(detection_box[1]))
    x2 = min(float(reference_box[2]), float(detection_box[2]))
    y2 = min(float(reference_box[3]), float(detection_box[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    reference_area = max(
        0.0,
        (float(reference_box[2]) - float(reference_box[0]))
        * (float(reference_box[3]) - float(reference_box[1])),
    )
    detection_area = max(
        0.0,
        (float(detection_box[2]) - float(detection_box[0]))
        * (float(detection_box[3]) - float(detection_box[1])),
    )
    union = reference_area + detection_area - intersection
    return _clip_unit(intersection / max(union, np.finfo(float).eps))


def normalized_center_similarity(
    reference_box: np.ndarray,
    detection_box: np.ndarray,
) -> float:
    """Return object-scale-normalized center similarity."""
    if not _valid_box(reference_box) or not _valid_box(detection_box):
        raise ValueError("normalized_center_similarity requires valid boxes")
    distance = float(
        np.linalg.norm(_box_center(reference_box) - _box_center(detection_box))
    )
    denominator = _box_diagonal(reference_box) + _box_diagonal(detection_box)
    residual = 2.0 * distance / max(denominator, np.finfo(float).eps)
    return 1.0 - _clip_unit(residual)


def scale_similarity(
    reference_box: np.ndarray,
    detection_box: np.ndarray,
) -> float:
    """Return the frozen symmetric log-area similarity."""
    if not _valid_box(reference_box) or not _valid_box(detection_box):
        raise ValueError("scale_similarity requires valid boxes")
    reference_area = max(
        1.0,
        (float(reference_box[2]) - float(reference_box[0]))
        * (float(reference_box[3]) - float(reference_box[1])),
    )
    detection_area = max(
        1.0,
        (float(detection_box[2]) - float(detection_box[0]))
        * (float(detection_box[3]) - float(detection_box[1])),
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
    if total <= 0.0:
        return None
    return values / total


def appearance_similarity(
    track_descriptor: np.ndarray | None,
    detection_descriptor: np.ndarray | None,
) -> tuple[float, float]:
    """Return 1-Hellinger and its explicit validity mask."""
    first = _normalized_descriptor(track_descriptor)
    second = _normalized_descriptor(detection_descriptor)
    if first is None or second is None or first.shape != second.shape:
        return 0.5, 0.0
    coefficient = _clip_unit(float(np.sum(np.sqrt(first * second))))
    hellinger = math.sqrt(max(0.0, 1.0 - coefficient))
    return _clip_unit(1.0 - hellinger), 1.0


def motion_consistency(
    predicted_box: np.ndarray | None,
    detection_box: np.ndarray,
    *,
    available: bool,
) -> tuple[float, float]:
    """Return causal object-scale motion consistency and validity mask."""
    if (
        not available
        or predicted_box is None
        or not _valid_box(predicted_box)
        or not _valid_box(detection_box)
    ):
        return 0.5, 0.0
    distance = float(
        np.linalg.norm(_box_center(predicted_box) - _box_center(detection_box))
    )
    denominator = 0.5 * (
        _box_diagonal(predicted_box) + _box_diagonal(detection_box)
    )
    residual = distance / max(denominator, np.finfo(float).eps)
    return 1.0 - _clip_unit(residual), 1.0


def track_freshness(
    detection_opportunities_since_confirmed: int,
    max_opportunities: int = OWNER_PREFERENCE_MAX_DETECTION_OPPORTUNITIES,
) -> float:
    """Return freshness in detector opportunities, not raw frames."""
    if max_opportunities <= 0:
        raise ValueError("max_opportunities must be positive")
    opportunities = max(0, int(detection_opportunities_since_confirmed))
    return 1.0 - _clip_unit(opportunities / max_opportunities)


def build_owner_preference_features(
    *,
    reference_box: np.ndarray,
    detection_box: np.ndarray,
    track_descriptor: np.ndarray | None,
    detection_descriptor: np.ndarray | None,
    predicted_box: np.ndarray | None,
    motion_is_available: bool,
    detection_opportunities_since_confirmed: int,
) -> OwnerPreferenceFeatures:
    """Build the identical feature vector used for hidden or visible tracks."""
    appearance, appearance_mask = appearance_similarity(
        track_descriptor,
        detection_descriptor,
    )
    motion, motion_mask = motion_consistency(
        predicted_box,
        detection_box,
        available=motion_is_available,
    )
    return OwnerPreferenceFeatures(
        overlap_similarity=overlap_similarity(reference_box, detection_box),
        normalized_center_similarity=normalized_center_similarity(
            reference_box,
            detection_box,
        ),
        scale_similarity=scale_similarity(reference_box, detection_box),
        appearance_similarity=appearance,
        motion_consistency=motion,
        track_freshness=track_freshness(
            detection_opportunities_since_confirmed
        ),
        appearance_available=appearance_mask,
        motion_available=motion_mask,
    )


def common_scale_quality(features: OwnerPreferenceFeatures) -> float:
    """Return Q(track, detection) from the frozen nonnegative weights."""
    values = features.as_dict()
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("owner preference features must be finite")
    if any(value < 0.0 or value > 1.0 for value in values.values()):
        raise ValueError("owner preference features must be in [0, 1]")
    return _clip_unit(
        sum(
            OWNER_PREFERENCE_WEIGHTS[name] * values[name]
            for name in OWNER_PREFERENCE_WEIGHTS
        )
    )


def owner_preference_score(
    hidden: OwnerPreferenceFeatures,
    visible: OwnerPreferenceFeatures,
) -> float:
    """Return the frozen bounded, uncalibrated owner preference score."""
    hidden_quality = common_scale_quality(hidden)
    visible_quality = common_scale_quality(visible)
    return _clip_unit(0.5 + 0.5 * (hidden_quality - visible_quality))


def decide_owner_preference(
    hidden: OwnerPreferenceFeatures,
    visible: OwnerPreferenceFeatures,
    *,
    detection_confidence: float,
    hidden_detection_opportunities: int,
    visible_detection_opportunities: int,
) -> OwnerPreferenceDecision:
    """Apply every frozen H1-r2 eligibility and abstention rule."""
    if not math.isfinite(float(detection_confidence)):
        return OwnerPreferenceDecision(None, None, None, False, "score_invalid")
    if detection_confidence < OWNER_PREFERENCE_MIN_DETECTION_CONFIDENCE:
        return OwnerPreferenceDecision(
            None,
            None,
            None,
            False,
            "missing_evidence",
        )
    if (
        hidden_detection_opportunities
        > OWNER_PREFERENCE_MAX_DETECTION_OPPORTUNITIES
        or visible_detection_opportunities
        > OWNER_PREFERENCE_MAX_DETECTION_OPPORTUNITIES
    ):
        return OwnerPreferenceDecision(
            None,
            None,
            None,
            False,
            "missing_evidence",
        )
    if hidden.overlap_similarity < OWNER_PREFERENCE_MIN_HIDDEN_OVERLAP:
        return OwnerPreferenceDecision(
            None,
            None,
            None,
            False,
            "missing_evidence",
        )
    if (
        hidden.appearance_available + hidden.motion_available < 1.0
        or visible.appearance_available + visible.motion_available < 1.0
    ):
        return OwnerPreferenceDecision(
            None,
            None,
            None,
            False,
            "missing_evidence",
        )
    try:
        hidden_quality = common_scale_quality(hidden)
        visible_quality = common_scale_quality(visible)
    except ValueError:
        return OwnerPreferenceDecision(None, None, None, False, "score_invalid")
    score = _clip_unit(0.5 + 0.5 * (hidden_quality - visible_quality))
    margin = hidden_quality - visible_quality
    if abs(margin) <= OWNER_PREFERENCE_NUMERIC_TOLERANCE:
        return OwnerPreferenceDecision(
            score,
            hidden_quality,
            visible_quality,
            False,
            "tie_or_margin",
        )
    if (
        score + OWNER_PREFERENCE_NUMERIC_TOLERANCE
        < OWNER_PREFERENCE_THRESHOLD
    ):
        return OwnerPreferenceDecision(
            score,
            hidden_quality,
            visible_quality,
            False,
            "below_threshold",
        )
    if (
        margin + OWNER_PREFERENCE_NUMERIC_TOLERANCE
        < OWNER_PREFERENCE_MIN_QUALITY_MARGIN
    ):
        return OwnerPreferenceDecision(
            score,
            hidden_quality,
            visible_quality,
            False,
            "tie_or_margin",
        )
    return OwnerPreferenceDecision(
        score,
        hidden_quality,
        visible_quality,
        True,
        "applied",
    )


__all__ = [
    "OWNER_PREFERENCE_MAX_DETECTION_OPPORTUNITIES",
    "OWNER_PREFERENCE_MIN_DETECTION_CONFIDENCE",
    "OWNER_PREFERENCE_MIN_HIDDEN_OVERLAP",
    "OWNER_PREFERENCE_MIN_QUALITY_MARGIN",
    "OWNER_PREFERENCE_SCORE_NAME",
    "OWNER_PREFERENCE_THRESHOLD",
    "OWNER_PREFERENCE_WEIGHTS",
    "OwnerPreferenceDecision",
    "OwnerPreferenceFeatures",
    "appearance_similarity",
    "build_owner_preference_features",
    "common_scale_quality",
    "decide_owner_preference",
    "motion_consistency",
    "normalized_center_similarity",
    "overlap_similarity",
    "owner_preference_score",
    "scale_similarity",
    "track_freshness",
]
