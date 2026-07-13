from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_predictions,
)
from pig_behavior.classification_v2.evaluation.native_unit_metrics import (
    evaluate_native_oof,
)
from pig_behavior.classification_v2.evaluation.statistics import (
    paired_cluster_bootstrap,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def test_native_oof_preserves_authority_universe_and_global_metrics() -> None:
    """Every expected unit remains visible and primary macro-F1 uses 10 classes."""

    units, audit = evaluate_native_oof(_window_predictions(), _assignments())

    assert audit["valid"] is True
    assert len(units) == 4
    assert audit["expected_valid_native_units"] == 4
    assert audit["predicted_valid_native_units"] == 4
    assert audit["missing_valid_native_unit_count"] == 0
    assert audit["extra_native_prediction_count"] == 0
    assert audit["pooled_metrics"]["macro_f1"] == pytest.approx(0.4)
    assert audit["pooled_metrics"]["macro_f1_supported"] == pytest.approx(1.0)
    assert audit["pooled_metrics"]["weighted_f1"] == pytest.approx(1.0)
    assert len(audit["class_fold_support"]) == 20
    assert set(units["true_label"]) == set(units["behavior_label"])


@pytest.mark.parametrize("failure", ["missing", "extra", "non_evaluable"])
def test_native_oof_rejects_prediction_universe_drift(failure: str) -> None:
    assignments = _assignments()
    predictions = _window_predictions()
    if failure == "missing":
        predictions = predictions.loc[
            ~predictions["temporal_unit_key"].eq("u4")
        ].copy()
    elif failure == "extra":
        predictions = pd.concat(
            [predictions, _prediction_rows("u-extra", "drink", "fold_a")],
            ignore_index=True,
        )
    else:
        assignments.loc[
            assignments["temporal_unit_key"].eq("u4"),
            "native_unit_valid_for_main_eval",
        ] = False

    _, audit = evaluate_native_oof(predictions, assignments)

    assert audit["valid"] is False
    if failure == "missing":
        assert audit["missing_valid_native_unit_count"] == 1
    elif failure == "extra":
        assert audit["extra_native_prediction_count"] == 1
    else:
        assert audit["non_evaluable_native_prediction_count"] == 1


@pytest.mark.parametrize("failure", ["target", "fold"])
def test_native_oof_rejects_target_or_fold_payload_drift(failure: str) -> None:
    predictions = _window_predictions()
    unit_mask = predictions["temporal_unit_key"].eq("u1")
    if failure == "target":
        predictions.loc[unit_mask, "true_label"] = "eat"
    else:
        predictions.loc[unit_mask, "oof_fold_id"] = "fold_b"

    _, audit = evaluate_native_oof(predictions, _assignments())

    assert audit["valid"] is False
    key = (
        "prediction_true_label_mismatch_count"
        if failure == "target"
        else "prediction_outer_fold_mismatch_count"
    )
    assert audit[key] == 1


def test_native_oof_rejects_recording_group_crossing_outer_folds() -> None:
    assignments = _assignments()
    assignments.loc[assignments["temporal_unit_key"].eq("u2"), "outer_fold_id"] = (
        "fold_b"
    )

    _, audit = evaluate_native_oof(_window_predictions(), assignments)

    assert audit["valid"] is False
    assert "recording_group_id_crosses_outer_folds=1" in audit["errors"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("duplicate_window", "duplicate_window_id_rows=2"),
        ("blank_unit", "blank_temporal_unit_key_rows=1"),
        ("nan_probability", "invalid_probability_values=1"),
        (
            "argmax_mismatch",
            "predicted_label_probability_argmax_mismatch=1",
        ),
    ],
)
def test_native_collapse_fails_closed_on_malformed_rows(
    mutation: str,
    expected_error: str,
) -> None:
    predictions = _window_predictions()
    if mutation == "duplicate_window":
        predictions.loc[1, "window_id"] = predictions.loc[0, "window_id"]
    elif mutation == "blank_unit":
        predictions.loc[0, "temporal_unit_key"] = ""
    elif mutation == "nan_probability":
        predictions.loc[0, "prob_drink"] = np.nan
    else:
        predictions.loc[0, "predicted_label"] = "eat"

    units, audit = build_native_temporal_predictions(
        predictions,
        _strict_config(),
    )

    assert units.empty
    assert audit["valid"] is False
    assert expected_error in audit["errors"]


def test_native_collapse_rejects_true_label_and_fold_conflicts() -> None:
    predictions = _window_predictions()
    unit_rows = predictions.index[
        predictions["temporal_unit_key"].eq("u1")
    ].tolist()
    predictions.loc[unit_rows[1], "true_label"] = "eat"
    predictions.loc[unit_rows[1], "oof_fold_id"] = "fold_b"

    units, audit = build_native_temporal_predictions(
        predictions,
        _strict_config(),
    )

    u1 = units.loc[units["temporal_unit_key"].eq("u1")].iloc[0]
    assert bool(u1["true_label_conflict"])
    assert bool(u1["oof_fold_conflict"])
    assert bool(u1["native_metric_include"]) is False
    assert audit["valid"] is False
    assert audit["true_label_conflict_units"] == 1
    assert audit["oof_fold_conflict_units"] == 1


def test_paired_cluster_bootstrap_binds_unit_cluster_and_fold_mapping() -> None:
    candidate, _ = evaluate_native_oof(_window_predictions(), _assignments())
    candidate = candidate.loc[candidate["native_metric_include"]].copy()
    baseline = candidate.copy()
    baseline.loc[baseline["temporal_unit_key"].eq("u3"), "native_predicted_behavior"] = (
        "social-nose"
    )
    baseline.loc[baseline["temporal_unit_key"].eq("u4"), "native_predicted_behavior"] = (
        "fight"
    )

    result = paired_cluster_bootstrap(
        candidate,
        baseline,
        iterations=50,
        seed=17,
    )

    assert result["paired_native_units"] == 4
    assert result["cluster_count"] == 2
    assert result["macro_f1_delta"] > 0.0
    assert len(result["paired_unit_ids_sha256"]) == 64
    assert len(result["paired_fold_mapping_sha256"]) == 64
    assert result["two_sided_bootstrap_p"] is None
    assert result["p_value_status"].startswith("not_reported")
    assert result["outer_predictions_used_for_model_selection"] is False


@pytest.mark.parametrize("mutation", ["cluster", "fold", "unit_set"])
def test_paired_cluster_bootstrap_rejects_unpaired_lineage(mutation: str) -> None:
    candidate, _ = evaluate_native_oof(_window_predictions(), _assignments())
    candidate = candidate.loc[candidate["native_metric_include"]].copy()
    baseline = candidate.copy()
    if mutation == "cluster":
        baseline.loc[0, "recording_group_id"] = "recording-z"
    elif mutation == "fold":
        baseline.loc[
            baseline["recording_group_id"].eq("recording-a"),
            "outer_fold_id",
        ] = "fold-z"
    else:
        baseline = baseline.iloc[:-1].copy()

    message = {
        "cluster": "recording clusters disagree",
        "fold": "outer folds disagree",
        "unit_set": "native-unit sets differ",
    }[mutation]
    with pytest.raises(ValueError, match=message):
        paired_cluster_bootstrap(
            candidate,
            baseline,
            iterations=10,
        )


def test_paired_cluster_bootstrap_blocks_outer_model_selection() -> None:
    candidate, _ = evaluate_native_oof(_window_predictions(), _assignments())
    candidate = candidate.loc[candidate["native_metric_include"]].copy()

    with pytest.raises(ValueError, match="cannot select or tune"):
        paired_cluster_bootstrap(
            candidate,
            candidate.copy(),
            iterations=10,
            outer_predictions_used_for_model_selection=True,
        )


def _strict_config() -> NativeTemporalMetricsConfig:
    return NativeTemporalMetricsConfig(
        true_col="true_label",
        pred_col="predicted_label",
        weight_col=None,
        valid_col=None,
        label_order=tuple(VALID_BEHAVIORS),
        require_complete_probability_vector=True,
        require_oof_fold=True,
        bootstrap_iterations=0,
    )


def _assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": ["u1", "u2", "u3", "u4"],
            "recording_group_id": [
                "recording-a",
                "recording-a",
                "recording-b",
                "recording-b",
            ],
            "outer_fold_id": ["fold_a", "fold_a", "fold_b", "fold_b"],
            "behavior_label": ["drink", "eat", "fight", "social-nose"],
            "source_type": [
                "legacy_recovered",
                "legacy_recovered",
                "cvat_tracking_xml",
                "cvat_tracking_xml",
            ],
            "video_key": ["video-a", "video-a", "video-b", "video-b"],
            "native_unit_valid_for_main_eval": [True, True, True, True],
        }
    )


def _window_predictions() -> pd.DataFrame:
    rows = []
    for unit_id, label, fold_id in [
        ("u1", "drink", "fold_a"),
        ("u2", "eat", "fold_a"),
        ("u3", "fight", "fold_b"),
        ("u4", "social-nose", "fold_b"),
    ]:
        rows.extend(_prediction_rows(unit_id, label, fold_id).to_dict("records"))
    return pd.DataFrame(rows)


def _prediction_rows(
    unit_id: str,
    label: str,
    fold_id: str,
) -> pd.DataFrame:
    rows = []
    for index, confidence in enumerate([0.90, 0.80]):
        row = {
            "window_id": f"{unit_id}-window-{index}",
            "temporal_unit_key": unit_id,
            "oof_fold_id": fold_id,
            "true_label": label,
            "predicted_label": label,
        }
        row.update(_probabilities(label, confidence))
        rows.append(row)
    return pd.DataFrame(rows)


def _probabilities(label: str, confidence: float) -> dict[str, float]:
    remainder = (1.0 - confidence) / (len(VALID_BEHAVIORS) - 1)
    return {
        f"prob_{candidate}": confidence if candidate == label else remainder
        for candidate in VALID_BEHAVIORS
    }
