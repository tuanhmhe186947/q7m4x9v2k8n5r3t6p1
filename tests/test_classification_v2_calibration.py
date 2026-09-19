import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation.calibration import cross_fit_temperature_scaling
from pig_behavior.classification_v2.evaluation.native_temporal_metrics import build_native_temporal_predictions


def test_cross_fitted_temperature_excludes_evaluation_fold_labels() -> None:
    """Changing held-out labels cannot alter the temperature fitted for that fold."""

    frame = _prediction_frame()
    changed = frame.copy()
    changed.loc[changed["oof_fold_id"].eq("fold_a"), "behavior_true"] = ["eat", "drink"]

    calibrated, audit = cross_fit_temperature_scaling(frame, ece_bins=5, expected_fold_count=3)
    _, changed_audit = cross_fit_temperature_scaling(changed, ece_bins=5, expected_fold_count=3)

    temperature = {row["oof_fold_id"]: row["temperature"] for row in audit["fold_audits"]}
    changed_temperature = {row["oof_fold_id"]: row["temperature"] for row in changed_audit["fold_audits"]}
    assert temperature["fold_a"] == changed_temperature["fold_a"]
    np.testing.assert_allclose(
        calibrated[["cal_prob_drink", "cal_prob_eat"]].sum(axis=1).to_numpy(),
        np.ones(len(calibrated)),
    )
    assert audit["valid"] is True
    assert audit["complete_oof_fold_coverage"] is True


def test_cross_fitted_temperature_rejects_single_fold() -> None:
    """A one-fold pilot cannot be presented as leakage-safe calibrated evaluation."""

    frame = _prediction_frame()
    frame["oof_fold_id"] = "only_fold"

    with pytest.raises(ValueError, match="at least two"):
        cross_fit_temperature_scaling(frame)


def test_native_aggregation_preserves_mean_probabilities_and_fold_lineage() -> None:
    """Calibration must receive one weighted probability vector and one fold per native unit."""

    windows = pd.DataFrame(
        {
            "temporal_unit_key": ["u1", "u1"],
            "window_id": ["w1", "w2"],
            "behavior_true": ["drink", "drink"],
            "behavior_pred": ["drink", "eat"],
            "window_sample_weight": [1.0, 3.0],
            "window_valid_for_main_train": [True, True],
            "oof_fold_id": ["fold_a", "fold_a"],
            "prob_drink": [0.8, 0.4],
            "prob_eat": [0.2, 0.6],
        }
    )

    units, audit = build_native_temporal_predictions(windows)

    assert audit["valid"] is True
    assert units.loc[0, "prob_drink"] == pytest.approx(0.5)
    assert units.loc[0, "prob_eat"] == pytest.approx(0.5)
    assert units.loc[0, "oof_fold_id"] == "fold_a"
    assert bool(units.loc[0, "oof_fold_conflict"]) is False


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "oof_fold_id": ["fold_a", "fold_a", "fold_b", "fold_b", "fold_c", "fold_c"],
            "behavior_true": ["drink", "eat", "drink", "eat", "drink", "eat"],
            "prob_drink": [0.8, 0.7, 0.6, 0.2, 0.9, 0.4],
            "prob_eat": [0.2, 0.3, 0.4, 0.8, 0.1, 0.6],
            "oof_fold_conflict": False,
        }
    )
