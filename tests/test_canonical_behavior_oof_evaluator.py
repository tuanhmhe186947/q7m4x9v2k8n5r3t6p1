from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pig_behavior.classification_v2.evaluation import (
    evaluate_behavior_oof,
    per_class_metric_delta,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

M2_PREDICTIONS = Path(
    "outputs/classification_v2/m2_vft_primary_5fold_oof_analysis/"
    "m2_vft_primary_5fold_earlystop_concatenated_oof_predictions.csv"
)
CANONICAL_MANIFEST = Path(
    "outputs/classification_v2/full_t6_canonical_46d_20260816/"
    "full_t6_row_manifest.csv"
)


def test_evaluator_joins_exact_identity_population_and_uses_pooled_f1() -> None:
    native, report = evaluate_behavior_oof(_predictions(), _manifest())

    assert report["valid"] is True
    assert report["integrity"] == {
        "prediction_rows": 4,
        "manifest_rows": 4,
        "joined_rows": 4,
        "duplicate_prediction_identity_count": 0,
        "duplicate_manifest_identity_count": 0,
        "missing_prediction_identity_count": 0,
        "extra_prediction_identity_count": 0,
        "each_manifest_row_predicted_exactly_once": True,
    }
    assert len(native) == 4
    assert report["macro_f1"] == pytest.approx(0.4)
    fold_values = [metrics["macro_f1"] for metrics in report["fold_metrics"].values()]
    assert np.mean(fold_values) == pytest.approx(0.2)
    assert report["macro_f1"] != pytest.approx(np.mean(fold_values))
    assert report["headline_definition"] == "pooled_concatenated_native_oof_macro_f1"


def test_shuffled_prediction_rows_produce_identical_metrics() -> None:
    _, expected = evaluate_behavior_oof(_predictions(), _manifest())
    shuffled = _predictions().iloc[::-1].reset_index(drop=True)

    _, actual = evaluate_behavior_oof(shuffled, _manifest())

    assert actual["macro_f1"] == expected["macro_f1"]
    assert actual["nll"] == expected["nll"]
    assert actual["per_class"] == expected["per_class"]
    assert actual["confusion_matrix"] == expected["confusion_matrix"]


@pytest.mark.parametrize("problem", ["duplicate", "missing", "extra"])
def test_evaluator_rejects_duplicate_or_incomplete_prediction_identity(
    problem: str,
) -> None:
    predictions = _predictions()
    if problem == "duplicate":
        predictions.loc[1, "window_id"] = predictions.loc[0, "window_id"]
        message = "duplicate prediction window_id"
    elif problem == "missing":
        predictions = predictions.iloc[:-1].copy()
        message = "missing prediction identities"
    else:
        extra = predictions.iloc[[0]].copy()
        extra["window_id"] = "unexpected-target"
        predictions = pd.concat([predictions, extra], ignore_index=True)
        message = "extra prediction identities"

    with pytest.raises(ValueError, match=message):
        evaluate_behavior_oof(predictions, _manifest())


@pytest.mark.parametrize("problem", ["nonfinite", "negative", "row_sum"])
def test_evaluator_rejects_invalid_probability_rows(problem: str) -> None:
    predictions = _predictions()
    if problem == "nonfinite":
        predictions.loc[0, "prob_drink"] = np.nan
        message = "finite"
    elif problem == "negative":
        predictions.loc[0, "prob_drink"] = -0.1
        message = "non-negative"
    else:
        probability_columns = [f"prob_{label}" for label in VALID_BEHAVIORS]
        predictions.loc[0, probability_columns] *= 0.5
        message = "row sums"

    with pytest.raises(ValueError, match=message):
        evaluate_behavior_oof(predictions, _manifest())


def test_evaluator_rejects_probability_column_or_class_order_drift() -> None:
    predictions = _predictions()
    columns = list(predictions.columns)
    drink_index = columns.index("prob_drink")
    eat_index = columns.index("prob_eat")
    columns[drink_index], columns[eat_index] = columns[eat_index], columns[drink_index]

    with pytest.raises(ValueError, match="probability column order"):
        evaluate_behavior_oof(predictions[columns], _manifest())
    with pytest.raises(ValueError, match="class_order must equal VALID_BEHAVIORS"):
        evaluate_behavior_oof(
            predictions,
            _manifest(),
            class_order=sorted(VALID_BEHAVIORS),
        )


def test_m5_regression_nll_uses_locked_label_to_probability_mapping() -> None:
    manifest = _manifest().iloc[[0]].copy()
    manifest["behavior"] = "social-nose"
    predictions = _predictions().iloc[[0]].copy()
    predictions["true_label"] = "social-nose"
    predictions["predicted_label"] = "social-nose"
    predictions["y_true"] = "social-nose"
    predictions["y_pred"] = "social-nose"
    probabilities = _probabilities("social-nose", 0.8)
    for column, value in probabilities.items():
        predictions[column] = value

    _, report = evaluate_behavior_oof(predictions, manifest)

    assert report["nll"] == pytest.approx(-np.log(0.8), abs=1e-15)
    assert report["nll"] != pytest.approx(-np.log(probabilities["prob_fight"]))


def test_m7_regression_per_class_delta_is_keyed_by_class_name() -> None:
    baseline = {
        label: {"f1": float(index) / 100.0}
        for index, label in enumerate(reversed(VALID_BEHAVIORS))
    }
    candidate = {
        label: {"f1": baseline[label]["f1"] + 0.125}
        for label in VALID_BEHAVIORS
    }

    delta = per_class_metric_delta(candidate, baseline, metric="f1")

    assert list(delta) == list(VALID_BEHAVIORS)
    assert all(value == pytest.approx(0.125) for value in delta.values())


def test_per_class_delta_rejects_incomplete_class_mapping() -> None:
    baseline = {label: {"f1": 0.1} for label in VALID_BEHAVIORS[:-1]}
    candidate = {label: {"f1": 0.2} for label in VALID_BEHAVIORS}

    with pytest.raises(ValueError, match="baseline class keys"):
        per_class_metric_delta(candidate, baseline, metric="f1")


def test_retained_m2_oof_reproduction() -> None:
    root = Path(__file__).resolve().parents[1]
    prediction_path = root / M2_PREDICTIONS
    manifest_path = root / CANONICAL_MANIFEST
    if not prediction_path.exists() or not manifest_path.exists():
        pytest.skip("retained M2 OOF authority is not available")

    _, report = evaluate_behavior_oof(prediction_path, manifest_path)

    assert report["integrity"]["joined_rows"] == 33_287
    assert report["macro_f1"] == pytest.approx(0.612580356035, abs=5e-12)
    # Stored rows deviate from unit mass by at most 2.4e-7. The canonical
    # pre-normalization check and normalization shift historical raw NLL by
    # 5.41e-10, so 1e-9 is the narrow stored-precision tolerance.
    assert report["nll"] == pytest.approx(0.955128588645, abs=1e-9)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_id": ["target-1", "target-2", "target-3", "target-4"],
            "source_type": [
                "legacy_recovered",
                "legacy_recovered",
                "cvat_tracking_xml",
                "cvat_tracking_xml",
            ],
            "dataset_id": ["legacy", "legacy", "cvat-a", "cvat-b"],
            "video_key": ["video-a", "video-a", "video-b", "video-c"],
            "object_track_key": ["track-1", "track-2", "track-3", "track-4"],
            "behavior": ["drink", "eat", "fight", "social-nose"],
            "outer_fold_id": ["FOLD_1", "FOLD_1", "FOLD_2", "FOLD_2"],
            "split": ["test", "test", "test", "test"],
            "native_unit_id": ["native-1", "native-2", np.nan, np.nan],
        }
    )


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_id, source, label, fold_id, outer_fold_id in [
        ("target-1", "legacy_recovered", "drink", "VG1", "FOLD_1"),
        ("target-2", "legacy_recovered", "eat", "VG1", "FOLD_1"),
        ("target-3", "cvat_tracking_xml", "fight", "VG2", "FOLD_2"),
        ("target-4", "cvat_tracking_xml", "social-nose", "VG2", "FOLD_2"),
    ]:
        row: dict[str, object] = {
            "window_id": target_id,
            "fold_id": fold_id,
            "oof_fold_id": outer_fold_id,
            "source_type": source,
            "true_label": label,
            "predicted_label": label,
            "y_true": label,
            "y_pred": label,
        }
        row.update(_probabilities(label, 0.8))
        rows.append(row)
    return pd.DataFrame(rows)


def _probabilities(label: str, confidence: float) -> dict[str, float]:
    other = (1.0 - confidence) / (len(VALID_BEHAVIORS) - 1)
    return {
        f"prob_{candidate}": confidence if candidate == label else other
        for candidate in VALID_BEHAVIORS
    }
