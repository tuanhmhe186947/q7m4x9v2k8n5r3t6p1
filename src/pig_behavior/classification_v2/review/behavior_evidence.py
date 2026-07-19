"""Review-only behavior consistency scores from temporal evidence.

These scores help a human reviewer find likely confusion cases. They are not
class probabilities, training targets, sample weights, or automatic label
corrections. Every output column starts with ``review_`` so model-input guards
exclude it by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

INTERACTION_BEHAVIORS = {"fight", "social-nose"}
ROI_BEHAVIOR_TO_CLASS = {
    "eat": "feeder",
    "drink": "drinker",
    "playwithtoy": "toy",
}
MOTION_BEHAVIORS = {"move", "explore", "stand"}
POSTURE_BEHAVIORS = {"lying", "sitting"}

REVIEW_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "review_evidence_available",
    "review_motion_evidence_available",
    "review_roi_evidence_available",
    "review_social_evidence_available",
    "review_posture_evidence_available",
    "review_relevant_evidence_available",
    "review_evidence_quality_score",
    "review_evidence_insufficiency_score",
    "review_motion_support_score",
    "review_roi_support_score",
    "review_social_support_score",
    "review_posture_transition_score",
    "review_temporal_phase_support_score",
    "review_difference_motion_support_score",
    "review_social_phase_support_score",
    "review_pig_strenet_conflict_score",
    "review_evidence_conflict_score",
    "review_evidence_priority_auto",
    "review_confusion_pairs_auto",
    "review_evidence_reason_auto",
    "review_evidence_status_auto",
)

REQUIRED_EVIDENCE_COLUMNS: tuple[str, ...] = (
    "temporal_observation_ratio_unit",
    "temporal_pair_coverage_ratio_unit",
    "motion_active_ratio_unit",
    "motion_stationary_ratio_unit",
    "motion_speed_p90_unit",
    "trajectory_straightness_unit",
    "bbox_shape_change_p90_unit",
    "social_pair_contact_ratio_unit",
    "social_partner_persistence_ratio_unit",
    "social_aggression_proxy_p90_unit",
)


@dataclass(frozen=True, slots=True)
class BehaviorEvidenceConfig:
    """Fixed review-queue thresholds; values are not learned from labels."""

    low_motion_support: float = 0.25
    strong_motion_support: float = 0.60
    low_roi_support: float = 0.25
    strong_roi_support: float = 0.65
    low_social_support: float = 0.25
    strong_social_support: float = 0.60
    conflict_review_threshold: float = 0.45
    aggression_reference: float = 0.02
    shape_transition_reference: float = 0.20

    def validate(self) -> None:
        """Reject invalid review-score ranges before building a queue."""

        bounded = [
            self.low_motion_support,
            self.strong_motion_support,
            self.low_roi_support,
            self.strong_roi_support,
            self.low_social_support,
            self.strong_social_support,
            self.conflict_review_threshold,
        ]
        if any(not 0 <= value <= 1 for value in bounded):
            raise ValueError("behavior evidence ratio thresholds must be in [0, 1]")
        if self.low_motion_support >= self.strong_motion_support:
            raise ValueError("motion support thresholds are not ordered")
        if self.low_roi_support >= self.strong_roi_support:
            raise ValueError("ROI support thresholds are not ordered")
        if self.low_social_support >= self.strong_social_support:
            raise ValueError("social support thresholds are not ordered")
        if self.aggression_reference <= 0:
            raise ValueError("aggression_reference must be > 0")
        if self.shape_transition_reference <= 0:
            raise ValueError("shape_transition_reference must be > 0")


def add_behavior_review_evidence(
    temporal_units: pd.DataFrame,
    *,
    behavior_col: str = "behavior_temporal_final",
    config: BehaviorEvidenceConfig | None = None,
) -> pd.DataFrame:
    """Attach deterministic review evidence without changing unit labels.

    Older interval files remain readable. If raw evidence columns are absent,
    the output marks evidence unavailable but does not invent a conflict or
    silently expand the review queue.
    """

    if behavior_col not in temporal_units.columns:
        raise ValueError(f"missing behavior column for review evidence: {behavior_col}")
    config = config or BehaviorEvidenceConfig()
    config.validate()
    out = temporal_units.copy()
    original_labels = out[behavior_col].copy(deep=True)
    evidence_available = set(REQUIRED_EVIDENCE_COLUMNS).issubset(out.columns)

    scored = [
        _score_behavior_row(
            row,
            behavior_col=behavior_col,
            evidence_available=evidence_available,
            config=config,
        )
        for _, row in out.iterrows()
    ]
    score_table = pd.DataFrame(scored, index=out.index)
    for column in REVIEW_EVIDENCE_COLUMNS:
        out[column] = score_table[column]

    if len(out) != len(temporal_units):
        raise RuntimeError("behavior review evidence changed row count")
    if not out[behavior_col].equals(original_labels):
        raise RuntimeError("behavior review evidence changed a label")
    return out


def audit_behavior_review_evidence(
    temporal_units: pd.DataFrame,
) -> dict[str, Any]:
    """Report coverage and score distributions for the review-only layer."""

    missing = [
        column for column in REVIEW_EVIDENCE_COLUMNS if column not in temporal_units
    ]
    errors = [f"missing_review_evidence_columns={missing}"] if missing else []
    available = _bool_series(
        temporal_units.get(
            "review_evidence_available",
            pd.Series(False, index=temporal_units.index),
        )
    )
    return {
        "rows": int(len(temporal_units)),
        "evidence_available_rows": int(available.sum()),
        "evidence_unavailable_rows": int((~available).sum()),
        "relevant_evidence_available": _counts(
            temporal_units,
            "review_relevant_evidence_available",
        ),
        "evidence_status": _counts(
            temporal_units,
            "review_evidence_status_auto",
        ),
        "insufficiency_score": _numeric_summary(
            temporal_units,
            "review_evidence_insufficiency_score",
        ),
        "conflict_score": _numeric_summary(
            temporal_units,
            "review_evidence_conflict_score",
        ),
        "priority_score": _numeric_summary(
            temporal_units,
            "review_evidence_priority_auto",
        ),
        "confusion_pairs": _counts(
            temporal_units,
            "review_confusion_pairs_auto",
        ),
        "reasons": _counts(
            temporal_units,
            "review_evidence_reason_auto",
        ),
        "errors": errors,
        "warnings": (
            ["some_temporal_units_lack_review_evidence"]
            if (~available).any()
            else []
        ),
    }


def _score_behavior_row(
    row: pd.Series,
    *,
    behavior_col: str,
    evidence_available: bool,
    config: BehaviorEvidenceConfig,
) -> dict[str, float | bool | str]:
    """Score one unit using behavior-specific, transparent rules."""

    behavior = str(row.get(behavior_col, "")).strip().lower()
    pig = _pig_strenet_support(row, behavior)
    if not evidence_available and not pig["evidence_available"]:
        return _empty_score()
    quality = _quality_score(row)
    availability = _branch_availability(row, behavior)
    relevant_available = _relevant_evidence_available(
        behavior,
        availability,
    )
    relevant_available = bool(
        relevant_available or pig["relevant_available"]
    )
    insufficiency = 0.0 if relevant_available else 1.0
    motion = max(_motion_support(row), pig["target_motion"])
    roi_support = max(_roi_support(row, behavior), pig["target_roi"])
    social = max(_social_support(row, config), pig["social_contact"])
    posture_transition = _scaled(
        _number(row, "bbox_shape_change_p90_unit"),
        config.shape_transition_reference,
    )
    posture_transition = max(posture_transition, pig["shape_transition"])
    if relevant_available:
        conflict, reasons, pairs = _behavior_conflict(
            row,
            behavior=behavior,
            motion=motion,
            roi_support=roi_support,
            social=social,
            posture_transition=posture_transition,
            pig=pig,
            config=config,
        )
        evidence_status = "sufficient"
    else:
        conflict = 0.0
        reasons = [_missing_evidence_reason(behavior)]
        pairs = [_confusion_pair(behavior)]
        evidence_status = "missing_relevant_modality"
    quality_penalty = 1.0 - quality
    priority = (
        60.0 * conflict
        + 60.0 * insufficiency
        + 20.0 * quality_penalty
    )
    if conflict < config.conflict_review_threshold and insufficiency == 0.0:
        reasons = []
        pairs = []
    elif conflict >= config.conflict_review_threshold and not reasons:
        reasons = ["behavior_evidence_conflict"]
    return {
        "review_evidence_available": bool(
            evidence_available or pig["evidence_available"]
        ),
        "review_motion_evidence_available": availability["motion"],
        "review_roi_evidence_available": availability["roi"],
        "review_social_evidence_available": availability["social"],
        "review_posture_evidence_available": availability["posture"],
        "review_relevant_evidence_available": relevant_available,
        "review_evidence_quality_score": quality,
        "review_evidence_insufficiency_score": insufficiency,
        "review_motion_support_score": motion,
        "review_roi_support_score": roi_support,
        "review_social_support_score": social,
        "review_posture_transition_score": posture_transition,
        "review_temporal_phase_support_score": pig["temporal_phase"],
        "review_difference_motion_support_score": pig["target_motion"],
        "review_social_phase_support_score": pig["social_phase"],
        "review_pig_strenet_conflict_score": pig["conflict_hint"],
        "review_evidence_conflict_score": conflict,
        "review_evidence_priority_auto": float(np.clip(priority, 0.0, 100.0)),
        "review_confusion_pairs_auto": "|".join(_unique(pairs)),
        "review_evidence_reason_auto": ";".join(_unique(reasons)),
        "review_evidence_status_auto": evidence_status,
    }


def _behavior_conflict(
    row: pd.Series,
    *,
    behavior: str,
    motion: float,
    roi_support: float,
    social: float,
    posture_transition: float,
    pig: dict[str, float | bool],
    config: BehaviorEvidenceConfig,
) -> tuple[float, list[str], list[str]]:
    """Return conflict strength plus explicit reason/confusion tokens."""

    reasons: list[str] = []
    pairs: list[str] = []
    conflict = 0.0
    if behavior == "move":
        move_support = max(
            motion,
            float(pig["stationary_to_motion"]),
            float(pig["target_motion"]),
        )
        conflict = 1.0 - move_support
        if move_support < config.low_motion_support:
            reasons.append("move_with_weak_motion_evidence")
        pairs.append("move_vs_explore_vs_stand")
    elif behavior == "stand":
        stand_conflict = max(
            motion,
            float(pig["stationary_to_motion"]),
            float(pig["target_motion"]),
        )
        conflict = stand_conflict
        if stand_conflict > config.strong_motion_support:
            reasons.append("stand_with_strong_motion_evidence")
        pairs.append("move_vs_explore_vs_stand")
    elif behavior == "explore":
        roi_max = _maximum_roi_support(row)
        roi_persistence = max(roi_max, float(pig["maximum_roi"]))
        feeding_like = roi_persistence * (1.0 - motion)
        move_like = max(0.0, motion - config.strong_motion_support)
        conflict = max(feeding_like, move_like)
        if feeding_like >= config.conflict_review_threshold:
            reasons.append("explore_with_stationary_persistent_roi_contact")
            pairs.append("explore_vs_eat_drink_playwithtoy")
        if move_like >= config.conflict_review_threshold:
            reasons.append("explore_with_move_like_motion")
            pairs.append("move_vs_explore_vs_stand")
    elif behavior in ROI_BEHAVIOR_TO_CLASS:
        best_other = _maximum_roi_support(
            row,
            exclude=ROI_BEHAVIOR_TO_CLASS[behavior],
        )
        best_other = max(best_other, float(pig["maximum_other_roi"]))
        conflict = max(1.0 - roi_support, best_other)
        if roi_support < config.low_roi_support:
            reasons.append("roi_label_without_persistent_target_support")
        if best_other > config.strong_roi_support:
            reasons.append("different_roi_has_stronger_support")
        pairs.append(f"{behavior}_vs_stand_explore")
    elif behavior == "fight":
        aggression = _scaled(
            _number(row, "social_aggression_proxy_p90_unit"),
            config.aggression_reference,
        )
        fight_support = float(
            np.clip(
                0.35 * social
                + 0.25 * aggression
                + 0.20 * float(pig["social_phase"])
                + 0.20 * float(pig["pair_motion"]),
                0.0,
                1.0,
            )
        )
        conflict = 1.0 - fight_support
        if fight_support < config.low_social_support:
            reasons.append("fight_without_persistent_contact_or_aggression")
        pairs.append("fight_vs_social-nose_stand_move")
    elif behavior == "social-nose":
        aggression = _scaled(
            _number(row, "social_aggression_proxy_p90_unit"),
            config.aggression_reference,
        )
        social_support = max(
            social,
            float(pig["social_contact"]),
            float(pig["contact_persistence"]),
        )
        fight_like = max(
            aggression,
            float(pig["pair_motion"]) * float(pig["social_phase"]),
        )
        conflict = max(1.0 - social_support, fight_like)
        if social_support < config.low_social_support:
            reasons.append("social_nose_without_persistent_partner_contact")
        if fight_like > config.strong_social_support:
            reasons.append("social_nose_with_fight_like_motion")
        pairs.append("social-nose_vs_fight_stand")
    elif behavior in POSTURE_BEHAVIORS:
        conflict = max(
            posture_transition,
            0.50 * float(pig["target_motion"]),
        )
        if posture_transition >= config.conflict_review_threshold:
            reasons.append("posture_label_during_strong_shape_transition")
        elif conflict >= config.conflict_review_threshold:
            reasons.append("posture_label_with_strong_pixel_motion")
        pairs.append("lying_vs_sitting")
    return float(np.clip(conflict, 0.0, 1.0)), reasons, pairs


def _quality_score(row: pd.Series) -> float:
    """Combine observation, motion-pair, and bbox validity coverage."""

    observation = _number(row, "temporal_observation_ratio_unit")
    pair = _number(row, "temporal_pair_coverage_ratio_unit")
    bbox = _number(row, "bbox_valid_ratio_interval", default=1.0)
    return float(np.clip(0.45 * observation + 0.35 * pair + 0.20 * bbox, 0.0, 1.0))


def _branch_availability(
    row: pd.Series,
    behavior: str,
) -> dict[str, bool]:
    """Report per-row modality availability without using target decisions."""

    observation = _number(row, "temporal_observation_ratio_unit")
    pair = _number(row, "temporal_pair_coverage_ratio_unit")
    bbox = _number(row, "bbox_valid_ratio_interval", default=1.0)
    target_roi = ROI_BEHAVIOR_TO_CLASS.get(behavior)
    if target_roi:
        roi_available = _number(
            row,
            f"roi_{target_roi}_availability_ratio_unit",
        ) > 0
    else:
        roi_available = any(
            _number(row, f"roi_{roi_class}_availability_ratio_unit") > 0
            for roi_class in ("feeder", "drinker", "toy")
        )
    return {
        "motion": bool(observation > 0 and pair > 0 and bbox > 0),
        "roi": bool(roi_available and bbox > 0),
        "social": bool(
            _number(row, "social_neighbor_availability_ratio_unit") > 0
            and bbox > 0
        ),
        "posture": bool(observation > 0 and bbox > 0),
    }


def _relevant_evidence_available(
    behavior: str,
    availability: dict[str, bool],
) -> bool:
    if behavior in MOTION_BEHAVIORS:
        return availability["motion"]
    if behavior in ROI_BEHAVIOR_TO_CLASS:
        return availability["roi"]
    if behavior in INTERACTION_BEHAVIORS:
        return availability["social"]
    if behavior in POSTURE_BEHAVIORS:
        return availability["posture"]
    return False


def _missing_evidence_reason(behavior: str) -> str:
    if behavior in MOTION_BEHAVIORS:
        return "motion_evidence_unavailable"
    if behavior in ROI_BEHAVIOR_TO_CLASS:
        return "target_roi_evidence_unavailable"
    if behavior in INTERACTION_BEHAVIORS:
        return "social_evidence_unavailable"
    if behavior in POSTURE_BEHAVIORS:
        return "posture_evidence_unavailable"
    return "behavior_evidence_unavailable"


def _confusion_pair(behavior: str) -> str:
    if behavior in MOTION_BEHAVIORS:
        return "move_vs_explore_vs_stand"
    if behavior in ROI_BEHAVIOR_TO_CLASS:
        return f"{behavior}_vs_stand_explore"
    if behavior == "fight":
        return "fight_vs_social-nose_stand_move"
    if behavior == "social-nose":
        return "social-nose_vs_fight_stand"
    if behavior in POSTURE_BEHAVIORS:
        return "lying_vs_sitting"
    return ""


def _motion_support(row: pd.Series) -> float:
    """Combine active duration, high speed, and net trajectory evidence."""

    active = _number(row, "motion_active_ratio_unit")
    stationary = _number(row, "motion_stationary_ratio_unit")
    speed = _scaled(_number(row, "motion_speed_p90_unit"), 0.012)
    straightness = _number(row, "trajectory_straightness_unit")
    support = 0.45 * active + 0.25 * (1.0 - stationary)
    support += 0.20 * speed + 0.10 * straightness
    return float(np.clip(support, 0.0, 1.0))


def _roi_support(row: pd.Series, behavior: str) -> float:
    """Return target-ROI support only on the review surface."""

    roi_class = ROI_BEHAVIOR_TO_CLASS.get(behavior)
    if not roi_class:
        return 0.0
    return _roi_class_support(row, roi_class)


def _roi_class_support(row: pd.Series, roi_class: str) -> float:
    contact = _number(row, f"roi_{roi_class}_contact_ratio_unit")
    near = _number(row, f"roi_{roi_class}_near_ratio_unit")
    persistence = _number(
        row,
        f"roi_{roi_class}_contact_longest_run_ratio_unit",
    )
    return float(np.clip(0.55 * contact + 0.20 * near + 0.25 * persistence, 0.0, 1.0))


def _maximum_roi_support(
    row: pd.Series,
    *,
    exclude: str | None = None,
) -> float:
    values = [
        _roi_class_support(row, roi_class)
        for roi_class in ("feeder", "drinker", "toy")
        if roi_class != exclude
    ]
    return max(values, default=0.0)


def _social_support(row: pd.Series, config: BehaviorEvidenceConfig) -> float:
    """Combine contact duration, partner persistence, and proximity."""

    contact = _number(row, "social_pair_contact_ratio_unit")
    persistence = _number(row, "social_partner_persistence_ratio_unit")
    distance = _number(row, "social_nearest_dist_p50_unit", default=1.0)
    proximity = float(np.clip(1.0 - distance / 0.12, 0.0, 1.0))
    support = 0.50 * contact + 0.30 * persistence + 0.20 * proximity
    return float(np.clip(support, 0.0, 1.0))


def _pig_strenet_support(
    row: pd.Series,
    behavior: str,
) -> dict[str, float | bool]:
    evidence_available = _truth(row.get("review_pig_evidence_available", False))
    transition_available = _truth(
        row.get("review_pig_history_transition_available", False)
    )
    diff_valid = _number(row, "review_pig_diff_valid_ratio") > 0
    social_valid = _number(row, "review_pig_social_valid_ratio") > 0
    target_motion = max(
        _number(row, "review_pig_diff_active_pixel_ratio"),
        _scaled(_number(row, "review_pig_diff_inner_mean"), 0.20),
    ) if diff_valid else 0.0
    stationary_to_motion = (
        _number(row, "review_pig_stationary_to_motion_score")
        if transition_available
        else 0.0
    )
    motion_to_stationary = (
        _number(row, "review_pig_motion_to_stationary_score")
        if transition_available
        else 0.0
    )
    roi_class = ROI_BEHAVIOR_TO_CLASS.get(behavior)
    roi_values = {
        name: (
            _number(row, f"review_pig_roi_{name}_phase_score")
            if _number(row, f"review_pig_roi_{name}_valid_ratio") > 0
            and transition_available
            else 0.0
        )
        for name in ("feeder", "drinker", "toy")
    }
    target_roi = roi_values.get(roi_class or "", 0.0)
    other_roi = max(
        (value for name, value in roi_values.items() if name != roi_class),
        default=0.0,
    )
    social_phase = (
        _number(row, "review_pig_social_phase_score")
        if transition_available
        else 0.0
    )
    social_contact = (
        max(
            _number(row, "review_pig_topk_contact_ratio"),
            _number(row, "review_pig_contact_persistence_score"),
        )
        if social_valid
        else 0.0
    )
    pair_motion = (
        _number(row, "review_pig_pair_motion_source_percentile")
        if social_valid
        else 0.0
    )
    shape_transition = (
        _number(row, "review_pig_shape_transition_score")
        if transition_available
        else 0.0
    )
    relevant_available = False
    if behavior in MOTION_BEHAVIORS:
        relevant_available = diff_valid or transition_available
    elif behavior in ROI_BEHAVIOR_TO_CLASS:
        relevant_available = bool(
            roi_class
            and _number(row, f"review_pig_roi_{roi_class}_valid_ratio") > 0
        )
    elif behavior in INTERACTION_BEHAVIORS:
        relevant_available = social_valid
    elif behavior in POSTURE_BEHAVIORS:
        relevant_available = diff_valid or transition_available
    return {
        "evidence_available": evidence_available,
        "relevant_available": relevant_available,
        "temporal_phase": max(stationary_to_motion, motion_to_stationary),
        "target_motion": float(np.clip(target_motion, 0.0, 1.0)),
        "stationary_to_motion": stationary_to_motion,
        "motion_to_stationary": motion_to_stationary,
        "target_roi": target_roi,
        "maximum_roi": max(roi_values.values(), default=0.0),
        "maximum_other_roi": other_roi,
        "social_phase": social_phase,
        "social_contact": social_contact,
        "contact_persistence": _number(
            row,
            "review_pig_contact_persistence_score",
        ) if transition_available else 0.0,
        "pair_motion": pair_motion,
        "shape_transition": shape_transition,
        "conflict_hint": max(
            target_motion,
            social_phase,
            shape_transition,
            stationary_to_motion,
            motion_to_stationary,
        ),
    }


def _truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _empty_score() -> dict[str, float | bool | str]:
    """Return backward-compatible unavailable evidence without a decision."""

    return {
        "review_evidence_available": False,
        "review_motion_evidence_available": False,
        "review_roi_evidence_available": False,
        "review_social_evidence_available": False,
        "review_posture_evidence_available": False,
        "review_relevant_evidence_available": False,
        "review_evidence_quality_score": 0.0,
        "review_evidence_insufficiency_score": 0.0,
        "review_motion_support_score": 0.0,
        "review_roi_support_score": 0.0,
        "review_social_support_score": 0.0,
        "review_posture_transition_score": 0.0,
        "review_temporal_phase_support_score": 0.0,
        "review_difference_motion_support_score": 0.0,
        "review_social_phase_support_score": 0.0,
        "review_pig_strenet_conflict_score": 0.0,
        "review_evidence_conflict_score": 0.0,
        "review_evidence_priority_auto": 0.0,
        "review_confusion_pairs_auto": "",
        "review_evidence_reason_auto": "",
        "review_evidence_status_auto": "base_evidence_unavailable",
    }


def _number(row: pd.Series, column: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(column, default))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _scaled(value: float, reference: float) -> float:
    return float(np.clip(value / reference, 0.0, 1.0))


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    counts = df[column].fillna("").astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _numeric_summary(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    if column not in df.columns:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }
