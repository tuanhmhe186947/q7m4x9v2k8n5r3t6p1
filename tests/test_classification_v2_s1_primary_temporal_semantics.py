from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.datasets.fold_event_weights import (
    build_fold_event_weight_manifest,
)
from pig_behavior.classification_v2.datasets.primary_temporal_eligibility import (
    PrimaryTemporalEligibilityError,
    build_primary_s1_temporal_eligibility,
    build_primary_s1_validation_native_population,
    build_primary_s1_view_role_overlay,
    load_primary_s1_temporal_eligibility,
)
from pig_behavior.classification_v2.evaluation.s1_primary_native_evaluator import (
    S1PrimaryEvaluationError,
    evaluate_primary_s1_validation,
)


def test_primary_eligibility_excludes_mixed_shared_windows_and_preserves_rows() -> None:
    result = build_primary_s1_temporal_eligibility(_windows(), _roles())
    frame = result.windows.set_index("window_id")

    assert result.audit["valid"] is True
    assert result.audit["retained_window_feature_reuse"] == "PASS"
    assert result.audit["window_row_index_preserved"] == "PASS"
    assert frame.loc["legacy-t6", "primary_s1_eligibility_status"] == "VALID_SINGLE_LABEL"
    assert frame.loc["shared-t8", "primary_s1_eligibility_status"] == "VALID_SINGLE_LABEL"
    assert frame.loc["mixed-t8", "primary_s1_eligibility_status"] == "MIXED_LABEL"
    assert frame.loc["mixed-t8", "primary_s1_effective_sample_weight"] == 0.0
    assert frame.loc["outer-t16", "primary_s1_eligibility_status"] == "OUTER_ROLE_REJECTED"
    assert not bool(frame.loc["outer-t16", "primary_s1_eligible"])
    assert frame["window_row_index"].tolist() == _windows()["window_row_index"].tolist()


def test_view_local_roles_allow_train_only_event_weight_recomputation() -> None:
    eligibility = build_primary_s1_temporal_eligibility(_windows(), _roles()).windows
    role_overlay, audit = build_primary_s1_view_role_overlay(
        eligibility,
        _roles(),
        view_type="T8_contiguous",
    )
    windows = eligibility.loc[
        eligibility["view_type"].eq("T8_contiguous")
    ].copy()
    windows["window_valid_for_main_train"] = windows["primary_s1_eligible"]
    windows["window_sample_weight"] = windows["primary_s1_effective_sample_weight"]

    weights = build_fold_event_weight_manifest(windows, role_overlay)

    assert audit["valid"] is True
    assert audit["event_weight_train_only"] == "PASS"
    assert weights.audit["valid"] is True
    mixed = weights.weights.loc[weights.weights["window_id"].eq("mixed-t8")]
    assert not mixed["window_valid_for_fold_training_weight"].any()
    assert mixed["fold_event_sample_weight"].eq(0.0).all()


def test_primary_validation_evaluator_explodes_shared_windows_and_tie_breaks() -> None:
    eligibility = build_primary_s1_temporal_eligibility(_windows(), _roles()).windows
    native_units, population_audit = build_primary_s1_validation_native_population(
        eligibility,
        _roles(),
    )
    predictions = pd.DataFrame(
        [
            {"window_id": "validation-t12", "y_pred": "fight", "confidence": 0.5},
            {"window_id": "validation-t12-b", "y_pred": "drink", "confidence": 0.5},
        ]
    )
    extra_window = eligibility.loc[
        eligibility["window_id"].eq("validation-t12")
    ].copy()
    extra_window.loc[:, "window_id"] = "validation-t12-b"
    evaluation_windows = pd.concat([eligibility, extra_window], ignore_index=True)
    evaluation_windows.loc[
        evaluation_windows["window_id"].eq("validation-t12-b"),
        "primary_s1_eligible",
    ] = True

    first = evaluate_primary_s1_validation(
        predictions,
        evaluation_windows,
        native_units,
    )
    second = evaluate_primary_s1_validation(
        predictions,
        evaluation_windows,
        native_units,
    )

    assert population_audit["expected_native_units"] == 2
    assert first.audit["valid"] is True
    assert first.audit["native_units_unpredicted"] == 0
    assert first.audit["duplicate_collapsed_native_predictions"] == 0
    assert first.predictions["y_pred"].tolist() == ["drink", "drink"]
    assert first.predictions.to_csv(index=False) == second.predictions.to_csv(index=False)


def test_primary_validation_refuses_direct_composite_and_missing_native_coverage() -> None:
    eligibility = build_primary_s1_temporal_eligibility(_windows(), _roles()).windows
    native_units, _ = build_primary_s1_validation_native_population(eligibility, _roles())
    predictions = pd.DataFrame(
        [
            {
                "window_id": "validation-t12",
                "temporal_unit_key": "composite-not-allowed",
                "y_pred": "fight",
                "confidence": 1.0,
            }
        ]
    )
    with pytest.raises(S1PrimaryEvaluationError, match="direct temporal_unit_key"):
        evaluate_primary_s1_validation(predictions, eligibility, native_units)

    incomplete = native_units.iloc[[0]].copy()
    result = evaluate_primary_s1_validation(
        predictions.drop(columns=["temporal_unit_key"]),
        eligibility,
        incomplete,
    )
    assert result.audit["valid"] is False
    assert result.audit["unexpected_native_prediction_examples"]

    missing_population = pd.concat(
        [
            native_units,
            pd.DataFrame(
                {"temporal_unit_key": ["missing-native"], "behavior_label": ["fight"]}
            ),
        ],
        ignore_index=True,
    )
    missing_result = evaluate_primary_s1_validation(
        predictions.drop(columns=["temporal_unit_key"]),
        eligibility,
        missing_population,
    )
    assert missing_result.audit["valid"] is False
    assert "native_units_unpredicted=1" in missing_result.audit["errors"]


@pytest.mark.parametrize("forbidden_role", ["test", "outer", "q2_outer_00"])
def test_forbidden_role_is_rejected_before_metadata_open(
    monkeypatch: pytest.MonkeyPatch,
    forbidden_role: str,
) -> None:
    import pig_behavior.classification_v2.datasets.primary_temporal_eligibility as module

    def fail_if_opened(*args, **kwargs):
        raise AssertionError("metadata was opened after a forbidden role request")

    monkeypatch.setattr(module.pd, "read_csv", fail_if_opened)
    with pytest.raises(PrimaryTemporalEligibilityError, match="rejected before metadata open"):
        load_primary_s1_temporal_eligibility(
            Path("effective.csv"),
            Path("roles.csv"),
            requested_roles=[forbidden_role],
        )


def test_hash_mismatch_is_rejected_before_metadata_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import pig_behavior.classification_v2.datasets.primary_temporal_eligibility as module

    index = tmp_path / "effective.csv"
    roles = tmp_path / "roles.csv"
    index.write_text("placeholder", encoding="utf-8")
    roles.write_text("placeholder", encoding="utf-8")

    def fail_if_opened(*args, **kwargs):
        raise AssertionError("metadata was opened after a hash mismatch")

    monkeypatch.setattr(module.pd, "read_csv", fail_if_opened)
    with pytest.raises(PrimaryTemporalEligibilityError, match="hash mismatch"):
        load_primary_s1_temporal_eligibility(
            index,
            roles,
            expected_window_index_sha256="0" * 64,
        )


def _windows() -> pd.DataFrame:
    rows = [
        ("legacy-t6", "T6_contiguous", ["n1"], "drink"),
        ("shared-t8", "T8_contiguous", ["n2", "n3"], "eat"),
        ("mixed-t8", "T8_contiguous", ["n3", "n4"], "eat"),
        ("validation-t12", "T12_contiguous", ["n5", "n6"], "fight"),
        ("outer-t16", "T16_contiguous", ["n7"], "stand"),
    ]
    return pd.DataFrame(
        {
            "window_id": [row[0] for row in rows],
            "window_row_index": list(range(len(rows))),
            "view_type": [row[1] for row in rows],
            "window_length_frames": [int(row[1][1:].split("_")[0]) for row in rows],
            "source_type": ["cvat_tracking_xml"] * len(rows),
            "temporal_unit_keys_json": [json.dumps(row[2]) for row in rows],
            "behavior_window_label": [row[3] for row in rows],
            "window_valid_for_main_train": [True] * len(rows),
            "window_sample_weight": [1.0] * len(rows),
        }
    )


def _roles() -> pd.DataFrame:
    rows = [
        ("n1", "train", "drink"),
        ("n2", "train", "eat"),
        ("n3", "train", "eat"),
        ("n4", "train", "move"),
        ("n5", "validation", "fight"),
        ("n6", "validation", "fight"),
        ("n7", "test", "stand"),
    ]
    return pd.DataFrame(
        {
            "outer_fold_id": ["FOLD_3"] * len(rows),
            "temporal_unit_key": [row[0] for row in rows],
            "role": [row[1] for row in rows],
            "behavior_label": [row[2] for row in rows],
            "native_unit_valid_for_main_train": [True] * len(rows),
            "native_unit_valid_for_main_eval": [True] * len(rows),
        }
    )
