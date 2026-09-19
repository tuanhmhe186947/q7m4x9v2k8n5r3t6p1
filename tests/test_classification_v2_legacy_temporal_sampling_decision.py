from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation import (
    legacy_development_temporal_sampling_decision as temporal_decision,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def _prediction_frame(view_id: str, predicted: list[str]) -> pd.DataFrame:
    labels = list(VALID_BEHAVIORS) * 2
    rows = []
    for index, (true_label, predicted_label) in enumerate(zip(labels, predicted, strict=True)):
        probabilities = np.full(len(labels) // 2, 0.01 / 9.0, dtype=float)
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
            "temporal_sampling_view_id": view_id,
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
    labels = list(VALID_BEHAVIORS) * 2
    shifted = labels[1:10] + labels[:1]
    c6_predictions = shifted + shifted
    s6_predictions = labels
    return {
        "c6_contiguous_centered": _prediction_frame(
            "c6_contiguous_centered",
            c6_predictions,
        ),
        "c8_contiguous_centered": _prediction_frame(
            "c8_contiguous_centered",
            c6_predictions,
        ),
        "s6_uniform_span16": _prediction_frame(
            "s6_uniform_span16",
            s6_predictions,
        ),
    }


def test_evaluates_exact_native_matrix_and_per_class_effects() -> None:
    result, per_class, groups, confusion = temporal_decision.evaluate_temporal_sampling_predictions(
        _matrix(),
        iterations=1000,
        seed=17,
        maximum_rare_macro_f1_drop=0.05,
        enforce_project_counts=False,
    )

    assert result["valid"] is True
    assert result["common_native_universe"]["native_units"] == 20
    assert result["common_native_universe"]["video_clusters"] == 2
    assert result["decision"]["selected_working_view"] == "s6_uniform_span16"
    primary = result["paired_comparisons"]["primary_s6_vs_c6"]
    assert primary["delta_candidate_minus_baseline"]["macro_f1_global_10_class"] > 0.0
    assert primary["video_cluster_bootstrap"]["macro_f1_delta"] == pytest.approx(
        primary["delta_candidate_minus_baseline"]["macro_f1_global_10_class"]
    )
    assert primary["video_cluster_bootstrap"]["macro_f1_delta_ci_low"] > 0.0
    assert len(per_class) == 2 * len(VALID_BEHAVIORS)
    assert len(groups) == 2 * 5
    assert len(confusion) == len(temporal_decision.VIEW_IDS) * len(VALID_BEHAVIORS) ** 2
    assert not per_class[["pair_id", "behavior_label"]].duplicated().any()
    assert not groups[["pair_id", "group"]].duplicated().any()


def test_rejects_native_unit_universe_mismatch() -> None:
    matrix = _matrix()
    matrix["s6_uniform_span16"].loc[0, "temporal_unit_key"] = "unknown-unit"

    with pytest.raises(ValueError, match="native metadata universe differs"):
        temporal_decision.evaluate_temporal_sampling_predictions(
            matrix,
            iterations=1000,
            seed=17,
            maximum_rare_macro_f1_drop=0.05,
            enforce_project_counts=False,
        )


def test_rejects_duplicate_native_unit() -> None:
    matrix = _matrix()
    matrix["c8_contiguous_centered"].loc[1, "temporal_unit_key"] = "unit-00"

    with pytest.raises(ValueError, match="duplicate native units"):
        temporal_decision.evaluate_temporal_sampling_predictions(
            matrix,
            iterations=1000,
            seed=17,
            maximum_rare_macro_f1_drop=0.05,
            enforce_project_counts=False,
        )


def test_rejects_probability_label_disagreement() -> None:
    matrix = _matrix()
    matrix["c6_contiguous_centered"].loc[0, "predicted_label"] = "move"

    with pytest.raises(ValueError, match="predicted label differs"):
        temporal_decision.evaluate_temporal_sampling_predictions(
            matrix,
            iterations=1000,
            seed=17,
            maximum_rare_macro_f1_drop=0.05,
            enforce_project_counts=False,
        )
