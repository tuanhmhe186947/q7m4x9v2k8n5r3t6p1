from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5_cached_data import (
    FEATURE_DIM as ACTOR_FEATURE_DIM,
)
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderSelection,
)
from pig_behavior.classification_v2.training.legacy_development_l6_union_context import (
    EXPECTED_PARAMETER_COUNT,
    FEATURE_DIM,
    FEATURE_NAMES,
    MODEL_INPUT_DIM,
    MODES,
    SEQUENCE_LENGTH,
    _rename_geometry_surfaces,
    build_union_context_model,
    build_union_context_view,
    fit_union_context_normalization,
    union_context_feature_whitelist,
)
from pig_behavior.classification_v2.training.legacy_development_l6_union_context_cache import (
    _context_id_sequence,
    _frame_uid_sequence,
)
from pig_behavior.classification_v2.training.legacy_development_l6_union_context_runtime import (
    UNION_CONTEXT_RUNTIME_SPEC,
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
    union_context: np.ndarray
    availability: np.ndarray
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    audit: dict[str, Any]

    def load_union_context(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        if rows is None:
            return self.union_context.copy()
        return self.union_context[np.asarray(rows, dtype=np.int64)].copy()

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
    union = np.stack(
        [
            np.stack(
                [values * (feature + 1) + row for feature in range(FEATURE_DIM)],
                axis=1,
            )
            for row in range(3)
        ],
        axis=0,
    )
    availability = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    availability[1] = False
    union[1] = 0.0
    slot_rows: list[dict[str, Any]] = []
    for cache_row in range(3):
        for slot_index in range(SEQUENCE_LENGTH):
            slot_rows.append(
                {
                    "cache_row": cache_row,
                    "window_id": f"w{cache_row}",
                    "slot_index": slot_index,
                    "union_context_window_slot_uid": (
                        f"w{cache_row}::slot={slot_index}"
                    ),
                }
            )
    cache = _FakeCache(
        union_context=union,
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


def test_union_normalization_uses_available_train_slots() -> None:
    _, cache, selection = _synthetic_inputs()

    state = fit_union_context_normalization(cache, selection)

    assert state.modality_name == "union_context"
    assert state.feature_names == FEATURE_NAMES
    assert state.identity_field == "union_context_window_slot_uid"
    assert state.train_window_rows == 2
    assert state.train_slot_exposures == SEQUENCE_LENGTH
    assert state.unique_train_identity_rows == SEQUENCE_LENGTH
    assert state.validation_rows_read_for_fit == 0
    assert state.outer_holdout_rows_read_for_fit == 0


def test_union_modes_are_parameter_matched_and_missing_safe() -> None:
    base, cache, selection = _synthetic_inputs()
    normalization = fit_union_context_normalization(cache, selection)
    positions = np.asarray([0, 1], dtype=np.int64)

    zero = build_union_context_view(
        base,
        cache,
        mode="parameter_matched_zero",
        normalization=normalization,
    ).load_sequences(positions)
    availability = build_union_context_view(
        base,
        cache,
        mode="availability_only",
        normalization=normalization,
    ).load_sequences(positions)
    union_view = build_union_context_view(
        base,
        cache,
        mode="union_context",
        normalization=normalization,
    )
    union = union_view.load_sequences(positions)
    missing = union_view.with_missing_modality().load_sequences(positions)

    assert zero.shape == (2, SEQUENCE_LENGTH, MODEL_INPUT_DIM)
    assert availability.shape == zero.shape
    assert union.shape == zero.shape
    assert np.count_nonzero(zero[..., ACTOR_FEATURE_DIM:]) == 0
    assert np.count_nonzero(availability[..., ACTOR_FEATURE_DIM:-1]) == 0
    assert np.all(availability[0, :, -1] == 1.0)
    assert np.all(availability[1, :, -1] == 0.0)
    assert np.count_nonzero(union[0, :, ACTOR_FEATURE_DIM:-1]) > 0
    assert np.count_nonzero(union[1, :, ACTOR_FEATURE_DIM:]) == 0
    assert np.count_nonzero(missing[..., ACTOR_FEATURE_DIM:]) == 0


def test_union_classifier_width_and_mask_safety() -> None:
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

    model = build_union_context_model(_Config()).eval()
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


def test_union_whitelist_and_runtime_spec() -> None:
    whitelist = union_context_feature_whitelist("union_context")

    assert whitelist["feature_count"] == MODEL_INPUT_DIM
    assert whitelist["modality_feature_count"] == FEATURE_DIM
    assert all("roi_relation_" not in name for name in whitelist["features"])
    assert all("social_relation_" not in name for name in whitelist["features"])
    assert UNION_CONTEXT_RUNTIME_SPEC.modes == MODES
    result: dict[str, Any] = {
        "frame": pd.DataFrame({"geometry_mode": ["union_context"]}),
        "metrics": {"geometry_mode": "union_context"},
    }
    _rename_geometry_surfaces(result)
    assert "union_context_mode" in result["frame"]
    assert result["metrics"]["union_context_mode"] == "union_context"


def test_union_context_and_frame_sequences_use_distinct_delimiters() -> None:
    context_ids = [
        "legacy|scene=a|track=t|f000001",
        "legacy|scene=a|track=t|f000002",
    ]
    frame_uids = [
        "legacy::scene=a::f000001::track=t",
        "legacy::scene=a::f000002::track=t",
    ]

    assert _context_id_sequence(";;".join(context_ids)) == context_ids
    assert _frame_uid_sequence("|".join(frame_uids)) == frame_uids
