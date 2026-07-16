from __future__ import annotations

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.evaluation import (
    legacy_development_l7_imbalance_decision as decision,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def test_l7_decision_metrics_cover_rare_calibration_and_collapse() -> None:
    frame = _predictions()

    metrics = decision._validate_and_measure_predictions(
        frame,
        "event_balanced_ce",
    )

    assert metrics["native_units"] == 245
    assert metrics["video_clusters"] == 33
    assert metrics["macro_f1_global_10_class"] == 1.0
    assert metrics["rare_group_macro_f1"] == 1.0
    assert metrics["nll"] == 0.0
    assert metrics["top_label_ece"] == 0.0
    assert metrics["maximum_predicted_class_share"] < 0.11


def test_l7_decision_retains_baseline_without_promoted_alternative() -> None:
    packets = {
        policy: {"metrics": {"macro_f1_global_10_class": value, "nll": 1.0}}
        for policy, value in {
            "event_balanced_ce": 0.27,
            "effective_number_ce": 0.10,
            "balanced_softmax": 0.14,
        }.items()
    }
    comparisons = {
        policy: {"promotion_gate_passes": False}
        for policy in ("effective_number_ce", "balanced_softmax")
    }

    result = decision._select_policy(
        packets,
        comparisons,
        {"minimum_macro_f1_gain": 0.01},
    )

    assert result["selected_loss_policy"] == "event_balanced_ce"
    assert result["full_confirmation_authorized"] is False
    assert result["l8_candidate_lock_authorized"] is True


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(245):
        target_index = index % len(VALID_BEHAVIORS)
        label = VALID_BEHAVIORS[target_index]
        row: dict[str, object] = {
            "temporal_unit_key": f"unit-{index:03d}",
            "video_key": f"video-{index % 33:02d}",
            "behavior_label": label,
            "target_index": target_index,
            "predicted_index": target_index,
            "predicted_label": label,
            "loss_policy": "event_balanced_ce",
            "lineage_scope": "legacy-only-unreviewed-development",
            "human_review_complete": False,
            "reviewed_or_final_claim_allowed": False,
            "q2_claim_allowed": False,
        }
        row.update(
            {
                "prob_" + behavior.replace("-", "_"): float(
                    behavior == label
                )
                for behavior in VALID_BEHAVIORS
            }
        )
        rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    assert np.allclose(
        frame[
            ["prob_" + label.replace("-", "_") for label in VALID_BEHAVIORS]
        ].sum(axis=1),
        1.0,
    )
    return frame
