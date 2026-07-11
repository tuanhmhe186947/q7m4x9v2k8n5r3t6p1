import pandas as pd

from pig_behavior.classification_v2.evaluation.native_temporal_metrics import (
    NativeTemporalMetricsConfig,
    build_native_temporal_metrics,
)


def test_native_metrics_bootstrap_resamples_complete_oof_folds() -> None:
    """Session-safe OOF metrics must cluster uncertainty by held-out fold."""

    predictions = _predictions()
    config = NativeTemporalMetricsConfig(bootstrap_iterations=20, bootstrap_seed=4)

    _, payload = build_native_temporal_metrics(predictions, config)

    for interval in payload["confidence_intervals"].values():
        assert interval["method"] == "oof_fold_cluster_bootstrap_percentile"
        assert interval["resample_unit"] == "oof_fold_id"
        assert interval["n_bootstrap"] == 20


def test_single_fold_pilot_uses_unit_bootstrap_without_paper_claim() -> None:
    """One-fold engineering pilots retain a deterministic fallback uncertainty method."""

    predictions = _predictions()
    predictions["oof_fold_id"] = "fold_a"

    _, payload = build_native_temporal_metrics(
        predictions,
        NativeTemporalMetricsConfig(bootstrap_iterations=5, bootstrap_seed=4),
    )

    for interval in payload["confidence_intervals"].values():
        assert interval["method"] == "unit_bootstrap_percentile"
        assert interval["resample_unit"] == "native_temporal_unit"


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": ["u1", "u2", "u3", "u4"],
            "window_id": ["w1", "w2", "w3", "w4"],
            "behavior_true": ["drink", "eat", "drink", "eat"],
            "behavior_pred": ["drink", "drink", "eat", "eat"],
            "window_sample_weight": [1.0, 1.0, 1.0, 1.0],
            "window_valid_for_main_train": [True, True, True, True],
            "oof_fold_id": ["fold_a", "fold_a", "fold_b", "fold_b"],
            "prob_drink": [0.8, 0.7, 0.4, 0.2],
            "prob_eat": [0.2, 0.3, 0.6, 0.8],
        }
    )
