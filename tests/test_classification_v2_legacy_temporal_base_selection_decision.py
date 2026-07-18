from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation import (
    legacy_development_temporal_base_selection_decision as base_decision,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def _prediction_frame(mode_id: str, predicted: list[str]) -> pd.DataFrame:
    true_labels = list(VALID_BEHAVIORS) * 2
    rows = []
    for index, (true_label, predicted_label) in enumerate(
        zip(true_labels, predicted, strict=True)
    ):
        probabilities = np.full(len(VALID_BEHAVIORS), 0.01 / 9.0)
        predicted_index = list(VALID_BEHAVIORS).index(predicted_label)
        probabilities[predicted_index] = 0.99
        row = {
            "temporal_unit_key": f"unit-{index:02d}",
            "recording_group_id": f"date-{index // 10}",
            "video_key": f"video-{index // 10}",
            "source_type": "legacy_recovered",
            "dataset_id": "legacy_recovered_16f",
            "behavior_label": true_label,
            "target_index": list(VALID_BEHAVIORS).index(true_label),
            "predicted_index": predicted_index,
            "predicted_label": predicted_label,
            "training_scope": "full_development_confirmation",
            "lineage_scope": "legacy-only-unreviewed-development",
            "human_review_complete": False,
            "temporal_base_mode_id": mode_id,
        }
        for label, probability in zip(
            VALID_BEHAVIORS,
            probabilities,
            strict=True,
        ):
            row[f"prob_{label.replace('-', '_')}"] = probability
        rows.append(row)
    return pd.DataFrame(rows)


def _matrix() -> dict[str, pd.DataFrame]:
    correct = list(VALID_BEHAVIORS) * 2
    shifted_once = list(VALID_BEHAVIORS)[1:] + list(VALID_BEHAVIORS)[:1]
    wrong = shifted_once * 2
    predictions = {
        "SF128": wrong,
        "M128": correct,
        "A128": correct,
        "MW317": correct,
        "TCN128": wrong,
        "MW381": wrong,
        "TR128": correct,
    }
    return {
        mode_id: _prediction_frame(mode_id, predicted)
        for mode_id, predicted in predictions.items()
    }


def test_evaluator_emits_transfer_actions_and_bounded_candidate_packet() -> None:
    result, per_class, groups, confusion, modes = (
        base_decision.evaluate_temporal_base_predictions(
            _matrix(),
            run_summaries=None,
            iterations=1_000,
            seed=17,
            material_negative_ci_limit=0.02,
            maximum_group_macro_f1_drop=0.05,
            enforce_project_counts=False,
        )
    )

    assert result["valid"] is True
    comparisons = result["paired_comparisons"]
    assert comparisons["multiple_frames"]["transfer_decision"][
        "screening_action"
    ] == "CARRY"
    assert comparisons["content_weighting"]["transfer_decision"][
        "screening_action"
    ] == "DROP"
    assert comparisons["ordered_tcn"]["transfer_decision"][
        "screening_action"
    ] == "DROP"
    assert comparisons["timed_transformer"]["transfer_decision"][
        "screening_action"
    ] == "CARRY"
    assert comparisons["timed_transformer"]["transfer_decision"][
        "timing_claim_action"
    ] == "RETEST_ON_MIXED_REVIEWED_OBSERVED_TIME"
    packet = result["full_data_candidate_packet"]
    assert packet["legacy_can_set_final_base"] is False
    assert len(packet["carried_finalists"]) <= 3
    assert packet["carried_finalists"][0]["mode_id"] == "M128"
    assert packet["carried_finalists"][1]["mode_id"] == "TR128"
    expected_pairs = len(base_decision.PAIR_SPECS) + len(
        base_decision.OPERATIONAL_PAIR_SPECS
    )
    assert len(per_class) == expected_pairs * len(VALID_BEHAVIORS)
    assert len(groups) == expected_pairs * 5
    assert len(confusion) == len(base_decision.MODE_IDS) * 100
    assert len(modes) == len(base_decision.MODE_IDS)
    json.dumps(result)


def test_evaluator_rejects_native_universe_drift() -> None:
    matrix = _matrix()
    matrix["TR128"].loc[0, "temporal_unit_key"] = "unknown-unit"

    with pytest.raises(ValueError, match="native metadata universe differs"):
        base_decision.evaluate_temporal_base_predictions(
            matrix,
            run_summaries=None,
            iterations=1_000,
            seed=17,
            material_negative_ci_limit=0.02,
            maximum_group_macro_f1_drop=0.05,
            enforce_project_counts=False,
        )


def test_evaluator_rejects_duplicate_native_unit() -> None:
    matrix = _matrix()
    matrix["M128"].loc[1, "temporal_unit_key"] = "unit-00"

    with pytest.raises(ValueError, match="duplicate native units"):
        base_decision.evaluate_temporal_base_predictions(
            matrix,
            run_summaries=None,
            iterations=1_000,
            seed=17,
            material_negative_ci_limit=0.02,
            maximum_group_macro_f1_drop=0.05,
            enforce_project_counts=False,
        )


def test_operational_group_regression_is_retest_not_silent_carry() -> None:
    groups = {
        "interaction": 0.5,
        "locomotion_context": 0.1,
        "posture": 0.5,
        "rare": 0.5,
        "roi_behavior": 0.5,
    }
    baseline_groups = {key: 0.2 for key in groups}
    comparison = {
        "pair_id": "operational_attention_vs_single",
        "delta_candidate_minus_baseline": {
            "macro_f1_global_10_class": 0.01,
        },
        "video_cluster_bootstrap": {
            "macro_f1_delta_ci_low": -0.03,
            "macro_f1_delta_ci_high": 0.05,
        },
        "candidate_metrics": {"group_macro_f1": groups},
        "baseline_metrics": {"group_macro_f1": baseline_groups},
    }

    decision = base_decision._pair_transfer_decision(
        comparison,
        target_groups=("locomotion_context", "rare"),
        material_negative_ci_limit=0.02,
        maximum_group_macro_f1_drop=0.05,
        operational_check=True,
    )

    assert decision["screening_action"] == "RETEST"
    assert decision["operational_check"] is True
