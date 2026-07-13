from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from pig_behavior.classification_v2.review.behavior_evidence import (
    REVIEW_EVIDENCE_COLUMNS,
    add_behavior_review_evidence,
)
from pig_behavior.classification_v2.train_ready_features import (
    select_window_feature_columns,
)


def _unit(behavior: str, **overrides: float) -> dict[str, object]:
    row: dict[str, object] = {
        "temporal_unit_key": "unit-1",
        "behavior_temporal_final": behavior,
        "temporal_observation_ratio_unit": 1.0,
        "temporal_pair_coverage_ratio_unit": 1.0,
        "bbox_valid_ratio_interval": 1.0,
        "motion_active_ratio_unit": 0.0,
        "motion_stationary_ratio_unit": 1.0,
        "motion_speed_p90_unit": 0.0,
        "trajectory_straightness_unit": 0.0,
        "bbox_shape_change_p90_unit": 0.0,
        "roi_feeder_near_ratio_unit": 0.0,
        "roi_feeder_contact_ratio_unit": 0.0,
        "roi_feeder_contact_longest_run_ratio_unit": 0.0,
        "roi_drinker_near_ratio_unit": 0.0,
        "roi_drinker_contact_ratio_unit": 0.0,
        "roi_drinker_contact_longest_run_ratio_unit": 0.0,
        "roi_toy_near_ratio_unit": 0.0,
        "roi_toy_contact_ratio_unit": 0.0,
        "roi_toy_contact_longest_run_ratio_unit": 0.0,
        "social_pair_contact_ratio_unit": 0.0,
        "social_partner_persistence_ratio_unit": 0.0,
        "social_nearest_dist_p50_unit": 1.0,
        "social_aggression_proxy_p90_unit": 0.0,
    }
    row.update(overrides)
    return row


def test_review_evidence_flags_stationary_move_without_changing_label() -> None:
    units = pd.DataFrame([_unit("move")])
    labels_before = units["behavior_temporal_final"].copy()

    scored = add_behavior_review_evidence(units)

    pdt.assert_series_equal(scored["behavior_temporal_final"], labels_before)
    assert len(scored) == len(units)
    assert scored.iloc[0]["review_evidence_conflict_score"] > 0.9
    assert "move_with_weak_motion_evidence" in scored.iloc[0][
        "review_evidence_reason_auto"
    ]


def test_target_roi_persistence_supports_eat_but_not_drink() -> None:
    feeder_support = {
        "roi_feeder_near_ratio_unit": 1.0,
        "roi_feeder_contact_ratio_unit": 1.0,
        "roi_feeder_contact_longest_run_ratio_unit": 1.0,
    }
    units = pd.DataFrame(
        [
            _unit("eat", **feeder_support),
            _unit("drink", **feeder_support),
        ]
    )

    scored = add_behavior_review_evidence(units)

    assert scored.iloc[0]["review_evidence_conflict_score"] == 0.0
    assert scored.iloc[1]["review_evidence_conflict_score"] == 1.0
    assert "different_roi_has_stronger_support" in scored.iloc[1][
        "review_evidence_reason_auto"
    ]


def test_fight_without_partner_evidence_is_high_priority() -> None:
    scored = add_behavior_review_evidence(pd.DataFrame([_unit("fight")]))

    assert scored.iloc[0]["review_evidence_conflict_score"] == 1.0
    assert scored.iloc[0]["review_evidence_priority_auto"] >= 75.0
    assert "fight_vs_social-nose_stand_move" in scored.iloc[0][
        "review_confusion_pairs_auto"
    ]


def test_review_scores_are_never_selected_for_model_x() -> None:
    scored = add_behavior_review_evidence(pd.DataFrame([_unit("move")]))
    scored["speed_mean_window"] = 0.0

    selected = select_window_feature_columns(scored)

    assert "speed_mean_window" in selected
    assert not set(REVIEW_EVIDENCE_COLUMNS).intersection(selected)
