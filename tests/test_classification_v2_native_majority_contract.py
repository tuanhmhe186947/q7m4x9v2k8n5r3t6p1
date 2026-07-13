import pandas as pd
import pytest

from pig_behavior.classification_v2.experiments.native_majority_baseline import (
    build_native_majority_predictions,
)


def _native_units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": ["unit-a", "unit-b", "unit-c"],
            "behavior_label": ["lying", "stand", "fight"],
            "native_unit_valid_for_main_eval": [True, True, False],
            "native_unit_sample_weight": [1.0, 2.0, 0.0],
        }
    )


def _folds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temporal_unit_key": ["unit-a", "unit-b", "unit-c"],
            "oof_fold_id": ["fold-a", "fold-b", "fold-a"],
        }
    )


def test_native_majority_audits_intentional_eval_exclusion() -> None:
    predictions, audit = build_native_majority_predictions(
        _native_units(),
        _folds(),
    )

    assert predictions["temporal_unit_key"].tolist() == ["unit-a", "unit-b"]
    assert predictions["behavior_pred"].tolist() == ["stand", "lying"]
    assert audit["excluded_invalid_main_eval_rows"] == 1
    assert audit["prediction_row_loss"] == 0
    assert audit["errors"] == []


def test_native_majority_rejects_duplicate_fold_key() -> None:
    folds = pd.concat([_folds(), _folds().iloc[[0]]], ignore_index=True)

    with pytest.raises(
        ValueError,
        match="duplicate_fold_temporal_unit_key_rows=2",
    ):
        build_native_majority_predictions(_native_units(), folds)


def test_native_majority_rejects_missing_fold_key() -> None:
    folds = _folds().loc[lambda frame: frame["temporal_unit_key"].ne("unit-c")]

    with pytest.raises(ValueError, match="missing_fold_key_count=1"):
        build_native_majority_predictions(_native_units(), folds)


def test_native_majority_rejects_invalid_validity_value() -> None:
    native = _native_units()
    native["native_unit_valid_for_main_eval"] = native[
        "native_unit_valid_for_main_eval"
    ].astype(object)
    native.loc[0, "native_unit_valid_for_main_eval"] = "unknown"

    with pytest.raises(ValueError, match="invalid_native_validity_values=1"):
        build_native_majority_predictions(native, _folds())


def test_native_majority_rejects_unknown_eligible_label() -> None:
    native = _native_units()
    native.loc[0, "behavior_label"] = "not-a-behavior"

    with pytest.raises(ValueError, match="invalid_label=1"):
        build_native_majority_predictions(native, _folds())
