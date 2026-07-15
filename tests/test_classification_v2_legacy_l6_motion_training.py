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
from pig_behavior.classification_v2.training.legacy_development_l6_motion import (
    EXPECTED_PARAMETER_COUNT,
    FEATURE_DIM,
    MODEL_INPUT_DIM,
    MOTION_DIM,
    MOTION_FEATURE_NAMES,
    SEQUENCE_LENGTH,
    _rename_geometry_surfaces,
    build_motion_model,
    build_motion_view,
    fit_motion_normalization,
    l6_motion_feature_whitelist,
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
    motion: np.ndarray
    availability: np.ndarray
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    audit: dict[str, Any]

    def load_motion(self, rows: np.ndarray | None = None) -> np.ndarray:
        if rows is None:
            return self.motion.copy()
        return self.motion[np.asarray(rows, dtype=np.int64)].copy()

    def load_availability(self, rows: np.ndarray | None = None) -> np.ndarray:
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
    current_frames = np.asarray(
        [
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8],
            [100, 101, 102, 103, 104, 105],
        ],
        dtype=np.float64,
    )
    motion = np.stack(
        [current_frames * (index + 1) for index in range(MOTION_DIM)],
        axis=2,
    ).astype(np.float32)
    availability = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    availability[:, 0] = False
    motion[:, 0] = 0.0
    slot_rows: list[dict[str, Any]] = []
    for cache_row, frame_numbers in enumerate(current_frames.astype(int)):
        for slot_index, frame_number in enumerate(frame_numbers):
            pair = "" if slot_index == 0 else f"pair-{frame_number}"
            slot_rows.append(
                {
                    "cache_row": cache_row,
                    "slot_index": slot_index,
                    "frame_uid": f"frame-{frame_number}",
                    "motion_pair_uid": pair,
                }
            )
    cache = _FakeCache(
        motion=motion,
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


def test_motion_normalization_uses_unique_available_pairs() -> None:
    _, cache, selection = _synthetic_inputs()

    state = fit_motion_normalization(cache, selection)

    assert state.feature_names == MOTION_FEATURE_NAMES
    assert state.identity_field == "motion_pair_uid"
    assert state.train_window_rows == 2
    assert state.train_slot_exposures == 10
    assert state.unique_train_identity_rows == 8
    assert state.duplicate_train_slot_exposures == 2
    assert state.validation_rows_read_for_fit == 0
    assert state.outer_holdout_rows_read_for_fit == 0


def test_motion_modes_are_parameter_matched_and_missing_safe() -> None:
    base, cache, selection = _synthetic_inputs()
    normalization = fit_motion_normalization(cache, selection)
    positions = np.asarray([0, 1], dtype=np.int64)

    zero = build_motion_view(
        base,
        cache,
        mode="parameter_matched_zero",
        normalization=normalization,
    ).load_sequences(positions)
    availability = build_motion_view(
        base,
        cache,
        mode="availability_only",
        normalization=normalization,
    ).load_sequences(positions)
    motion_view = build_motion_view(
        base,
        cache,
        mode="motion",
        normalization=normalization,
    )
    motion = motion_view.load_sequences(positions)
    missing = motion_view.with_missing_modality().load_sequences(positions)

    assert zero.shape == (2, SEQUENCE_LENGTH, MODEL_INPUT_DIM)
    assert availability.shape == zero.shape
    assert motion.shape == zero.shape
    assert np.count_nonzero(zero[..., FEATURE_DIM:]) == 0
    assert np.count_nonzero(availability[..., FEATURE_DIM:-1]) == 0
    assert np.all(availability[:, 0, -1] == 0.0)
    assert np.all(availability[:, 1:, -1] == 1.0)
    assert np.isfinite(motion).all()
    assert np.count_nonzero(motion[..., FEATURE_DIM:-1]) > 0
    assert np.count_nonzero(missing[..., FEATURE_DIM:]) == 0


def test_motion_classifier_width_parameter_count_and_mask_safety() -> None:
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

    model = build_motion_model(_Config()).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        EXPECTED_PARAMETER_COUNT
    )
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


def test_motion_whitelist_and_geometry_surface_rename() -> None:
    whitelist = l6_motion_feature_whitelist("motion")

    assert whitelist["feature_count"] == MODEL_INPUT_DIM
    assert whitelist["modality_feature_count"] == MOTION_DIM
    assert all("geometry_" not in name for name in whitelist["features"])
    result: dict[str, Any] = {
        "frame": pd.DataFrame({"geometry_mode": ["motion"]}),
        "metrics": {"geometry_mode": "motion"},
    }
    _rename_geometry_surfaces(result)
    assert "motion_mode" in result["frame"]
    assert result["metrics"]["motion_mode"] == "motion"
