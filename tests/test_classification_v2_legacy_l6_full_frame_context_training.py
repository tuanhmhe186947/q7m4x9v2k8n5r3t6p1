from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training import (
    legacy_development_l6_full_frame_context_runtime as full_frame_runtime,
)
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM as ACTOR_FEATURE_DIM,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderSelection,
)
from pig_behavior.classification_v2.training.legacy_development_l6_full_frame_context import (
    EXPECTED_PARAMETER_COUNT,
    FEATURE_DIM,
    FEATURE_NAMES,
    MODEL_INPUT_DIM,
    MODES,
    SEQUENCE_LENGTH,
    _rename_geometry_surfaces,
    build_full_frame_context_model,
    build_full_frame_context_view,
    fit_full_frame_context_normalization,
    full_frame_context_feature_whitelist,
)
from pig_behavior.classification_v2.training.legacy_development_l6_full_frame_context_cache import (
    _scene_id_sequence,
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
    full_frame_context: np.ndarray
    availability: np.ndarray
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    audit: dict[str, Any]

    def load_full_frame_context(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        if rows is None:
            return self.full_frame_context.copy()
        return self.full_frame_context[
            np.asarray(rows, dtype=np.int64)
        ].copy()

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
            (3, SEQUENCE_LENGTH, ACTOR_FEATURE_DIM),
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
    values = np.arange(1, SEQUENCE_LENGTH + 1, dtype=np.float32)
    full_frame = np.stack(
        [
            np.stack(
                [
                    values * (feature + 1) + row
                    for feature in range(FEATURE_DIM)
                ],
                axis=1,
            )
            for row in range(3)
        ],
        axis=0,
    )
    availability = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    availability[1] = False
    full_frame[1] = 0.0
    slot_rows: list[dict[str, Any]] = []
    for cache_row in range(3):
        for slot_index in range(SEQUENCE_LENGTH):
            slot_rows.append(
                {
                    "cache_row": cache_row,
                    "window_id": f"w{cache_row}",
                    "slot_index": slot_index,
                    "scene_frame_uid": f"scene-{slot_index}",
                }
            )
    cache = _FakeCache(
        full_frame_context=full_frame,
        availability=availability,
        window_index=windows.assign(
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


def test_full_frame_normalization_uses_unique_train_scene_frames() -> None:
    _, cache, selection = _synthetic_inputs()

    state = fit_full_frame_context_normalization(cache, selection)

    assert state.modality_name == "full_frame_context"
    assert state.feature_names == FEATURE_NAMES
    assert state.identity_field == "scene_frame_uid"
    assert state.train_window_rows == 2
    assert state.train_slot_exposures == SEQUENCE_LENGTH
    assert state.unique_train_identity_rows == SEQUENCE_LENGTH
    assert state.validation_rows_read_for_fit == 0
    assert state.outer_holdout_rows_read_for_fit == 0


def test_full_frame_modes_are_parameter_matched_and_missing_safe() -> None:
    base, cache, selection = _synthetic_inputs()
    normalization = fit_full_frame_context_normalization(cache, selection)
    positions = np.asarray([0, 1], dtype=np.int64)

    zero = build_full_frame_context_view(
        base,
        cache,
        mode="parameter_matched_zero",
        normalization=normalization,
    ).load_sequences(positions)
    availability = build_full_frame_context_view(
        base,
        cache,
        mode="availability_only",
        normalization=normalization,
    ).load_sequences(positions)
    context_view = build_full_frame_context_view(
        base,
        cache,
        mode="full_frame_context",
        normalization=normalization,
    )
    context = context_view.load_sequences(positions)
    missing = context_view.with_missing_modality().load_sequences(positions)

    assert zero.shape == (2, SEQUENCE_LENGTH, MODEL_INPUT_DIM)
    assert availability.shape == zero.shape
    assert context.shape == zero.shape
    assert np.count_nonzero(zero[..., ACTOR_FEATURE_DIM:]) == 0
    assert np.count_nonzero(availability[..., ACTOR_FEATURE_DIM:-1]) == 0
    assert np.all(availability[0, :, -1] == 1.0)
    assert np.all(availability[1, :, -1] == 0.0)
    assert np.count_nonzero(context[0, :, ACTOR_FEATURE_DIM:-1]) > 0
    assert np.count_nonzero(context[1, :, ACTOR_FEATURE_DIM:]) == 0
    assert np.count_nonzero(missing[..., ACTOR_FEATURE_DIM:]) == 0


def test_full_frame_classifier_width_and_mask_safety() -> None:
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

    model = build_full_frame_context_model(_Config()).eval()
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


def test_full_frame_whitelist_runtime_and_surface_names() -> None:
    whitelist = full_frame_context_feature_whitelist("full_frame_context")

    assert whitelist["feature_count"] == MODEL_INPUT_DIM
    assert whitelist["modality_feature_count"] == FEATURE_DIM
    assert all("roi_relation_" not in name for name in whitelist["features"])
    assert all("social_relation_" not in name for name in whitelist["features"])
    assert all("union_context_" not in name for name in whitelist["features"])
    assert full_frame_runtime.FULL_FRAME_CONTEXT_RUNTIME_SPEC.modes == MODES
    result: dict[str, Any] = {
        "frame": pd.DataFrame({"geometry_mode": ["full_frame_context"]}),
        "metrics": {"geometry_mode": "full_frame_context"},
    }
    _rename_geometry_surfaces(result)
    assert "full_frame_context_mode" in result["frame"]
    assert result["metrics"]["full_frame_context_mode"] == (
        "full_frame_context"
    )


def test_full_frame_scene_sequence_parser() -> None:
    scene_ids = [
        "legacy::scene=a::f000001",
        "legacy::scene=a::f000002",
    ]

    assert _scene_id_sequence("|".join(scene_ids)) == scene_ids
