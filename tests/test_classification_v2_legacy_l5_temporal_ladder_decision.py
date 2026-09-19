from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation import (
    legacy_development_l5_temporal_ladder_decision as ladder_decision,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    CANONICAL_VIEWS,
    FULL_SCOPE,
    LINEAGE_SCOPE,
)


def test_native_prediction_audit_recomputes_fixed_ten_class_metrics() -> None:
    predictions, metrics, per_class = _perfect_predictions()

    result = ladder_decision._validate_predictions(
        predictions,
        metrics,
        per_class,
        expected_windows_per_native=4,
    )

    assert result["native_units"] == 245
    assert result["video_clusters"] == 33
    assert result["macro_f1_global_10_class"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["nll"] == pytest.approx(-np.log(0.99))

    drifted = predictions.copy()
    probability_columns = [
        column for column in drifted if column.startswith("prob_")
    ]
    drifted.loc[:, probability_columns] *= 0.9
    with pytest.raises(ValueError, match="do not sum to one"):
        ladder_decision._validate_predictions(
            drifted,
            metrics,
            per_class,
            expected_windows_per_native=4,
        )


def test_common_native_universe_rejects_metadata_drift() -> None:
    predictions, _, _ = _perfect_predictions()
    packets = {
        "t6_centered": {"predictions": predictions},
        "t8_centered": {"predictions": predictions.copy()},
    }

    audit = ladder_decision._validate_common_native_universe(packets)

    assert audit["native_units"] == 245
    assert audit["video_clusters"] == 33
    assert audit["valid"] is True

    packets["t8_centered"]["predictions"].loc[0, "video_key"] = "drifted"
    with pytest.raises(ValueError, match="universe differs"):
        ladder_decision._validate_common_native_universe(packets)


def test_working_decision_preserves_pairwise_uncertainty() -> None:
    ranking = [
        {"view_id": "t6_sliding", "macro_f1_global_10_class": 0.53},
        {"view_id": "t8_sliding", "macro_f1_global_10_class": 0.41},
    ]
    comparisons = {
        view_id: _comparison(ci_low=0.02)
        for view_id in CANONICAL_VIEWS
        if view_id != "t6_sliding"
    }
    comparisons["t8_centered"] = _comparison(ci_low=-0.01)

    decision = ladder_decision._make_decision(
        ranking,
        comparisons,
        {
            "candidate_view": "t6_sliding",
            "matched_centered_reference": "t6_centered",
            "established_reference": "t16_centered",
            "minimum_macro_f1_gain": 0.01,
            "maximum_rare_group_recall_drop": 0.1,
            "maximum_runtime_ratio": 3.0,
        },
    )

    assert decision["working_baseline_retained"] is True
    assert decision["selected_working_view"] == "t6_sliding"
    assert decision["universal_pairwise_superiority_established"] is False
    assert decision["views_with_ci_crossing_zero_vs_candidate"] == [
        "t8_centered"
    ]
    assert decision["causal_temporal_length_claim_allowed"] is False
    assert decision["applies_to_merged_reviewed_data"] is False


def test_t16_equivalence_requires_probabilities_and_parameters() -> None:
    predictions, _, _ = _perfect_predictions()
    packets = {
        "t16_centered": {
            "predictions": predictions,
            "result": {"parameter_sha256": "a" * 64},
        },
        "t16_sliding": {
            "predictions": predictions.copy(),
            "result": {"parameter_sha256": "a" * 64},
        },
    }
    spec = {
        "left_view": "t16_centered",
        "right_view": "t16_sliding",
        "reason": "T16_has_one_complete_window_under_both_protocols",
        "require_exact_equivalence": True,
    }

    assert ladder_decision._validate_expected_equivalence(packets, spec)[
        "valid"
    ]

    drifted = copy.deepcopy(packets)
    drifted["t16_sliding"]["result"]["parameter_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="equivalence failed"):
        ladder_decision._validate_expected_equivalence(drifted, spec)


def _comparison(*, ci_low: float) -> dict[str, object]:
    return {
        "delta_candidate_minus_baseline": {
            "macro_f1_global_10_class": 0.1,
        },
        "video_cluster_bootstrap": {"ci_low": ci_low},
        "rare_group": {"recall_drop_candidate_vs_baseline": 0.0},
        "resource_comparison": {
            "runtime_ratio_candidate_to_baseline": 1.5,
            "parameter_count_delta": 0,
        },
    }


def _perfect_predictions() -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    rows = []
    for index in range(245):
        label_index = index % len(VALID_BEHAVIORS)
        label = VALID_BEHAVIORS[label_index]
        probabilities = np.full(len(VALID_BEHAVIORS), 0.01 / 9.0)
        probabilities[label_index] = 0.99
        row: dict[str, object] = {
            "prediction_order": index,
            "temporal_unit_key": f"unit-{index:03d}",
            "recording_group_id": f"recording-{index // 50}",
            "video_key": f"video-{index % 33:02d}",
            "source_type": "legacy_recovered",
            "dataset_id": "legacy_recovered_16f",
            "behavior_label": label,
            "target_index": label_index,
            "predicted_index": label_index,
            "predicted_label": label,
            "aggregated_window_count": 4,
            "training_scope": FULL_SCOPE,
            "lineage_scope": LINEAGE_SCOPE,
            "human_review_complete": False,
            "reviewed_or_final_claim_allowed": False,
            "q2_claim_allowed": False,
        }
        row.update(
            {
                "prob_" + behavior.replace("-", "_"): probabilities[position]
                for position, behavior in enumerate(VALID_BEHAVIORS)
            }
        )
        rows.append(row)
    predictions = pd.DataFrame.from_records(rows)
    supports = predictions["behavior_label"].value_counts().to_dict()
    per_class = pd.DataFrame.from_records(
        [
            {
                "behavior_label": label,
                "class_index": index,
                "support": supports[label],
                "true_positive": supports[label],
                "false_positive": 0,
                "false_negative": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
            for index, label in enumerate(VALID_BEHAVIORS)
        ]
    )
    metrics: dict[str, object] = {
        "native_unit_rows": len(predictions),
        "macro_f1_global_10_class": 1.0,
        "accuracy": 1.0,
        "nll": -np.log(0.99),
    }
    return predictions, metrics, per_class
