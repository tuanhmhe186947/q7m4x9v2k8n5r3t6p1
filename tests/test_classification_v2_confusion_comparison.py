import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation.confusion_comparison import compare_confusion_focus


def test_confusion_comparison_is_paired_and_fold_clustered() -> None:
    """A proposed correction must appear as a paired focus error-rate and macro-F1 improvement."""

    baseline = _baseline_frame()
    proposed = baseline.rename(columns={"prediction": "proposed_prediction"}).copy()
    proposed["proposed_prediction"] = proposed["behavior_true"]
    proposed["calibrated_confidence"] = [0.9, 0.8, 0.95, 0.85]

    report, hard_errors = compare_confusion_focus(
        proposed,
        baseline,
        proposed_pred_col="proposed_prediction",
        baseline_pred_col="prediction",
        expected_fold_count=2,
        bootstrap_iterations=100,
        bootstrap_seed=3,
    )

    pair = report["focus_pairs"]["fight__vs__social-nose"]
    assert pair["baseline_pair_errors"] == 2
    assert pair["proposed_pair_errors"] == 0
    assert report["macro_f1_supported_delta"] > 0.0
    assert report["complete_oof_fold_coverage"] is True
    assert report["paper_facing_ready"] is False
    assert hard_errors.empty


def test_confusion_comparison_exports_high_confidence_focus_errors() -> None:
    """High-confidence errors must retain only unit/fold/label evidence, not model inputs."""

    baseline = _baseline_frame()
    proposed = baseline.rename(columns={"prediction": "proposed_prediction"}).copy()
    proposed["calibrated_confidence"] = [0.9, 0.8, 0.95, 0.85]

    _, hard_errors = compare_confusion_focus(
        proposed,
        baseline,
        proposed_pred_col="proposed_prediction",
        baseline_pred_col="prediction",
        expected_fold_count=2,
        bootstrap_iterations=10,
    )

    assert set(hard_errors["focus_pair"]) == {"fight__vs__social-nose"}
    assert len(hard_errors) == 2


def test_confusion_comparison_rejects_unpaired_native_units() -> None:
    """Different native-unit sets cannot produce a paired model comparison."""

    baseline = _baseline_frame()
    proposed = baseline.rename(columns={"prediction": "proposed_prediction"}).iloc[:-1].copy()

    with pytest.raises(ValueError, match="alignment mismatch"):
        compare_confusion_focus(
            proposed,
            baseline,
            proposed_pred_col="proposed_prediction",
            baseline_pred_col="prediction",
            bootstrap_iterations=10,
        )


def _baseline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": ["u1", "u2", "u3", "u4"],
            "oof_fold_id": ["fold_a", "fold_a", "fold_b", "fold_b"],
            "behavior_true": ["fight", "social-nose", "lying", "sitting"],
            "prediction": ["social-nose", "fight", "lying", "sitting"],
        }
    )
