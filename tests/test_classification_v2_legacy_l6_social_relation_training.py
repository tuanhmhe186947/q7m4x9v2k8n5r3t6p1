from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderSelection,
)
from pig_behavior.classification_v2.training.legacy_development_l6_social_relation import (
    EXPECTED_PARAMETER_COUNT,
    FEATURE_DIM,
    MODEL_INPUT_DIM,
    SEQUENCE_LENGTH,
    SOCIAL_RELATION_DIM,
    SOCIAL_RELATION_FEATURE_NAMES,
    _rename_geometry_surfaces,
    build_social_relation_model,
    build_social_relation_view,
    fit_social_relation_normalization,
    l6_social_relation_feature_whitelist,
)
from pig_behavior.classification_v2.training.legacy_development_l6_social_relation_runtime import (
    SOCIAL_RELATION_RUNTIME_SPEC,
)


@dataclass
class _FakeBase:
    windows: pd.DataFrame
    sequences: np.ndarray
    observed_mask: np.ndarray
    time_delta: np.ndarray
    targets: np.ndarray
    sample_weights: np.ndarray

    def load_sequences(self, positions: np.ndarray) -> np.ndarray:
        return self.sequences[np.asarray(positions, dtype=np.int64)].copy()


@dataclass
class _FakeCache:
    social_relation: np.ndarray
    availability: np.ndarray
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    audit: dict[str, Any]

    def load_social_relation(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        if rows is None:
            return self.social_relation.copy()
        return self.social_relation[np.asarray(rows, dtype=np.int64)].copy()

    def load_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        if rows is None:
            return self.availability.copy()
        return self.availability[np.asarray(rows, dtype=np.int64)].copy()


def _synthetic_inputs() -> tuple[
    _FakeBase,
    _FakeCache,
    TemporalLadderSelection,
]:
    windows = pd.DataFrame(
        {
            "window_id": ["w0", "w1", "w2"],
            "temporal_unit_key": ["u0", "u1", "u2"],
            "l5_role": ["train", "train", "validation"],
            "source_type": ["legacy_recovered"] * 3,
            "dataset_id": ["legacy_recovered_16f"] * 3,
        }
    )
    base = _FakeBase(
        windows=windows,
        sequences=np.zeros(
            (3, SEQUENCE_LENGTH, FEATURE_DIM),
            dtype=np.float32,
        ),
        observed_mask=np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_),
        time_delta=np.tile(
            np.arange(SEQUENCE_LENGTH, dtype=np.float32),
            (3, 1),
        ),
        targets=np.asarray([0, 1, 2], dtype=np.int64),
        sample_weights=np.full(3, 0.25, dtype=np.float32),
    )
    base_values = np.arange(1, SEQUENCE_LENGTH + 1, dtype=np.float32)
    social = np.stack(
        [
            np.stack(
                [base_values * (feature + 1) + row for feature in range(
                    SOCIAL_RELATION_DIM
                )],
                axis=1,
            )
            for row in range(3)
        ],
        axis=0,
    )
    availability = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    availability[1] = False
    social[1] = 0.0
    slot_rows: list[dict[str, Any]] = []
    for cache_row in range(3):
        for slot_index in range(SEQUENCE_LENGTH):
            slot_rows.append(
                {
                    "cache_row": cache_row,
                    "window_id": f"w{cache_row}",
                    "slot_index": slot_index,
                    "frame_uid": f"f-{cache_row}-{slot_index}",
                }
            )
    cache = _FakeCache(
        social_relation=social,
        availability=availability,
        window_index=windows.assign(
            cache_row=np.arange(3, dtype=np.int64),
            lineage_scope="legacy-only-unreviewed-development",
            human_review_complete=False,
        ),
        slot_index=pd.DataFrame.from_records(slot_rows),
        audit={
            "manifest_sha256": "a" * 64,
            "outer_holdout_slots_materialized": 0,
            "source_media_reads": 0,
        },
    )
    selection = TemporalLadderSelection(
        manifest=pd.DataFrame({"window_id": ["w0", "w1", "w2"]}),
        train_positions=np.asarray([0, 1], dtype=np.int64),
        validation_positions=np.asarray([2], dtype=np.int64),
        audit={"selection_content_sha256": "b" * 64},
    )
    return base, cache, selection


def test_social_normalization_uses_available_train_window_slots() -> None:
    _, cache, selection = _synthetic_inputs()

    state = fit_social_relation_normalization(cache, selection)

    assert state.modality_name == "social_relation"
    assert state.feature_names == SOCIAL_RELATION_FEATURE_NAMES
    assert state.identity_field == "social_relation_window_slot_uid"
    assert state.train_window_rows == 2
    assert state.train_slot_exposures == SEQUENCE_LENGTH
    assert state.unique_train_identity_rows == SEQUENCE_LENGTH
    assert state.duplicate_train_slot_exposures == 0
    assert state.validation_rows_read_for_fit == 0
    assert state.outer_holdout_rows_read_for_fit == 0


def test_social_modes_are_parameter_matched_and_missing_safe() -> None:
    base, cache, selection = _synthetic_inputs()
    normalization = fit_social_relation_normalization(cache, selection)
    positions = np.asarray([0, 1], dtype=np.int64)

    zero = build_social_relation_view(
        base,
        cache,
        mode="parameter_matched_zero",
        normalization=normalization,
    ).load_sequences(positions)
    availability = build_social_relation_view(
        base,
        cache,
        mode="availability_only",
        normalization=normalization,
    ).load_sequences(positions)
    social_view = build_social_relation_view(
        base,
        cache,
        mode="social_relation",
        normalization=normalization,
    )
    social = social_view.load_sequences(positions)
    missing = social_view.with_missing_modality().load_sequences(positions)

    assert zero.shape == (2, SEQUENCE_LENGTH, MODEL_INPUT_DIM)
    assert availability.shape == zero.shape
    assert social.shape == zero.shape
    assert np.count_nonzero(zero[..., FEATURE_DIM:]) == 0
    assert np.count_nonzero(availability[..., FEATURE_DIM:-1]) == 0
    assert np.all(availability[0, :, -1] == 1.0)
    assert np.all(availability[1, :, -1] == 0.0)
    assert np.count_nonzero(social[0, :, FEATURE_DIM:-1]) > 0
    assert np.count_nonzero(social[1, :, FEATURE_DIM:]) == 0
    assert np.count_nonzero(missing[..., FEATURE_DIM:]) == 0


def test_social_classifier_width_parameter_count_and_mask_safety() -> None:
    class _Config:
        payload = {
            "model": {
                "temporal_encoder_name": "masked_mean",
                "hidden_dim": 128,
                "dropout": 0.1,
                "transformer_layers": 1,
                "transformer_heads": 4,
            }
        }

    model = build_social_relation_model(_Config()).eval()
    observed_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert observed_parameters == EXPECTED_PARAMETER_COUNT
    features = torch.zeros((2, SEQUENCE_LENGTH, MODEL_INPUT_DIM))
    observed = torch.ones((2, SEQUENCE_LENGTH))
    observed[:, -1] = 0.0
    features[:, -1] = torch.nan
    time_delta = torch.arange(SEQUENCE_LENGTH).repeat(2, 1).float()

    with torch.inference_mode():
        first = model(features, observed, time_delta=time_delta)
        features[:, -1] = 999.0
        second = model(features, observed, time_delta=time_delta)

    assert first.shape == (2, len(VALID_BEHAVIORS))
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_social_whitelist_and_surface_rename() -> None:
    whitelist = l6_social_relation_feature_whitelist("social_relation")

    assert whitelist["feature_count"] == MODEL_INPUT_DIM
    assert whitelist["modality_feature_count"] == SOCIAL_RELATION_DIM
    assert all("roi_relation_" not in name for name in whitelist["features"])
    assert all("motion_" not in name for name in whitelist["features"])
    result: dict[str, Any] = {
        "frame": pd.DataFrame({"geometry_mode": ["social_relation"]}),
        "metrics": {"geometry_mode": "social_relation"},
    }
    _rename_geometry_surfaces(result)
    assert "social_relation_mode" in result["frame"]
    assert result["metrics"]["social_relation_mode"] == "social_relation"


def test_social_runtime_spec_has_exact_three_modes() -> None:
    assert SOCIAL_RELATION_RUNTIME_SPEC.modality_name == "social_relation"
    assert SOCIAL_RELATION_RUNTIME_SPEC.modes == (
        "parameter_matched_zero",
        "availability_only",
        "social_relation",
    )
