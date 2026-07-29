from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.review.source_specific_blinded_presentation_v2 import (
    PRESENTATION_SEMANTIC_HASH,
    PRESENTATION_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "analyze_interaction_calibration_development.py"
)
SPEC = importlib.util.spec_from_file_location(
    "interaction_calibration_development_analysis",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _outcomes(
    *,
    positives: int,
    negatives: int,
    current_false_negatives: int = 0,
    static_false_negatives: int = 0,
) -> pd.DataFrame:
    total = positives + negatives
    positive = pd.Series(
        [True] * positives + [False] * negatives,
        dtype=bool,
    )
    current = positive.copy()
    static = positive.copy()
    if current_false_negatives:
        current.iloc[:current_false_negatives] = False
    if static_false_negatives:
        static.iloc[:static_false_negatives] = False
    return pd.DataFrame(
        {
            "combined_review_needed": positive,
            "current_interaction_candidate": current,
            "static_set_95_diagnostic": static,
        },
        index=range(total),
    )


def test_perfect_screen_passes_frozen_development_gates() -> None:
    outcomes = _outcomes(positives=40, negatives=1000)
    result = MODULE.evaluate_screen(
        outcomes,
        rule_id="current_991_screen",
        screen_column="current_interaction_candidate",
    )
    assert result["recall_point"] == 1.0
    assert result["false_negative"] == 0
    assert result["all_development_gates_pass"] is True


def test_missed_positive_fails_auto_carry_safety_gate() -> None:
    outcomes = _outcomes(
        positives=40,
        negatives=40,
        current_false_negatives=2,
    )
    result = MODULE.evaluate_screen(
        outcomes,
        rule_id="current_991_screen",
        screen_column="current_interaction_candidate",
    )
    assert result["false_negative"] == 2
    assert result["all_development_gates_pass"] is False


def test_decision_precedence_is_frozen() -> None:
    passing = {
        "all_development_gates_pass": True,
        "review_needed_count": 40,
    }
    failing = {
        "all_development_gates_pass": False,
        "review_needed_count": 40,
    }
    decision = MODULE.choose_post_calibration_decision(
        [
            {"rule_id": "current_991_screen", **passing},
            {"rule_id": "static_95_diagnostic_screen", **passing},
        ]
    )
    assert decision["post_calibration_decision"] == "DECISION_A_KEEP_CURRENT_991_WITH_REVALIDATION"

    decision = MODULE.choose_post_calibration_decision(
        [
            {"rule_id": "current_991_screen", **failing},
            {"rule_id": "static_95_diagnostic_screen", **passing},
        ]
    )
    assert decision["post_calibration_decision"] == "DECISION_C_NEW_CALIBRATED_SELECTIVE_PARTITION"


def test_failed_rules_route_to_census_or_inconclusive() -> None:
    decision = MODULE.choose_post_calibration_decision(
        [
            {
                "rule_id": "current_991_screen",
                "all_development_gates_pass": False,
                "review_needed_count": 40,
            },
            {
                "rule_id": "static_95_diagnostic_screen",
                "all_development_gates_pass": False,
                "review_needed_count": 40,
            },
        ]
    )
    assert decision["post_calibration_decision"] == "DECISION_B_FULL_INTERACTION_CENSUS"

    decision = MODULE.choose_post_calibration_decision(
        [
            {
                "rule_id": "current_991_screen",
                "all_development_gates_pass": False,
                "review_needed_count": 12,
            },
            {
                "rule_id": "static_95_diagnostic_screen",
                "all_development_gates_pass": False,
                "review_needed_count": 12,
            },
        ]
    )
    assert decision["post_calibration_decision"] == "DECISION_E_REMAIN_INCONCLUSIVE"


def test_complete_development_join_derives_all_outcomes() -> None:
    item_ids = [f"calibration_item_{index:06d}" for index in range(1, 301)]
    review_keys = [f"review_{index:06d}" for index in range(1, 301)]
    provisional = ["fight"] * 156 + ["social-nose"] * 144
    reviewed = provisional.copy()
    reviewability = ["reviewable"] * 300
    confidence = ["high"] * 300
    reviewed[0] = "social-nose"
    reviewed[1] = "unclear"
    reviewability[1] = "visually_unresolved"
    confidence[1] = "low"
    reviewed[2] = "unreviewable"
    reviewability[2] = "technical_authority_defect"
    confidence[2] = "low"

    decisions = pd.DataFrame(
        {
            "review_key": review_keys,
            "calibration_item_id": item_ids,
            "reviewed_behavior": reviewed,
            "visual_reviewability": reviewability,
            "review_confidence": confidence,
            "optional_short_note": [""] * 300,
            "presentation_version": [PRESENTATION_VERSION] * 300,
            "presentation_semantic_hash": [PRESENTATION_SEMANTIC_HASH] * 300,
            "reviewer": ["reviewer01"] * 300,
            "decision_timestamp": ["2026-07-30T00:00:00+00:00"] * 300,
        }
    )
    internal = pd.DataFrame(
        {
            "calibration_item_id": item_ids,
            "review_unit_id": review_keys,
            "frozen_subset": [MODULE.DEVELOPMENT_SUBSET] * 300,
            "behavior_label": provisional,
            "source_type": ["cvat_tracking_xml"] * 300,
            "dataset_id": ["dataset"] * 300,
            "video_key": ["video"] * 300,
            "recording_date": ["2026-07-30"] * 300,
            "current_interaction_candidate": [True] * 300,
            "static_set_95_diagnostic": [True] * 300,
            "removed_by_static_diagnostic": [False] * 300,
            "high_crowding": [False] * 300,
            "lower_crowding": [True] * 300,
            "contact_proxy_present": [True] * 300,
            "contact_proxy_absent": [False] * 300,
            "social_evidence_available": [True] * 300,
            "social_evidence_unavailable_or_low_quality": [False] * 300,
            "authority_risk_control": [False] * 300,
        }
    )
    media = pd.DataFrame(
        {
            "calibration_item_id": item_ids,
            "review_key": review_keys,
            "split": [MODULE.DEVELOPMENT_SUBSET] * 300,
            "source_type": ["cvat_tracking_xml"] * 300,
            "dataset_id": ["dataset"] * 300,
            "video_key": ["video"] * 300,
            "recording_date": ["2026-07-30"] * 300,
            "presentation_version": [PRESENTATION_VERSION] * 300,
            "presentation_semantic_hash": [PRESENTATION_SEMANTIC_HASH] * 300,
        }
    )

    outcomes, audit = MODULE.prepare_joined_outcomes(
        decisions,
        internal,
        media,
    )
    assert audit["valid"] is True
    assert outcomes["calibration_outcome"].value_counts().to_dict() == {
        "LABEL_SUPPORTED": 297,
        "CORRECTION_REQUIRED": 1,
        "VISUALLY_UNRESOLVED": 1,
        "TECHNICAL_AUTHORITY_DEFECT": 1,
    }
