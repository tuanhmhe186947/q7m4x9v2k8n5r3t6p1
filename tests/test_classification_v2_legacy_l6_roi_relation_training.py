from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS
from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderSelection,
)
from pig_behavior.classification_v2.training.legacy_development_l6_roi_relation import (
    EXPECTED_PARAMETER_COUNT,
    FEATURE_DIM,
    MODALITY_NAME,
    MODEL_INPUT_DIM,
    ROI_RELATION_DIM,
    ROI_RELATION_FEATURE_NAMES,
    SEQUENCE_LENGTH,
    _rename_geometry_surfaces,
    _validate_config_payload,
    build_roi_relation_model,
    build_roi_relation_view,
    fit_roi_relation_normalization,
    l6_roi_relation_feature_whitelist,
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
    roi_relation: np.ndarray
    availability: np.ndarray
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    audit: dict[str, Any]

    def load_roi_relation(self, rows: np.ndarray | None = None) -> np.ndarray:
        if rows is None:
            return self.roi_relation.copy()
        return self.roi_relation[np.asarray(rows, dtype=np.int64)].copy()

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
    frame_numbers = np.asarray(
        [
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8],
            [100, 101, 102, 103, 104, 105],
        ],
        dtype=np.float64,
    )
    roi_relation = np.stack(
        [frame_numbers * (index + 1) for index in range(ROI_RELATION_DIM)],
        axis=2,
    ).astype(np.float32)
    roi_relation[..., 0] = 0.0
    availability = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    slot_rows: list[dict[str, Any]] = []
    for cache_row, numbers in enumerate(frame_numbers.astype(int)):
        for slot_index, frame_number in enumerate(numbers):
            slot_rows.append(
                {
                    "cache_row": cache_row,
                    "window_id": f"w{cache_row}",
                    "slot_index": slot_index,
                    "frame_uid": f"frame-{frame_number}",
                }
            )
    cache = _FakeCache(
        roi_relation=roi_relation,
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


def test_roi_normalization_deduplicates_frame_uids_and_allows_constants() -> None:
    _, cache, selection = _synthetic_inputs()

    state = fit_roi_relation_normalization(cache, selection)

    assert state.modality_name == MODALITY_NAME
    assert state.feature_names == ROI_RELATION_FEATURE_NAMES
    assert state.identity_field == "frame_uid"
    assert state.train_window_rows == 2
    assert state.train_slot_exposures == 12
    assert state.unique_train_identity_rows == 9
    assert state.duplicate_train_slot_exposures == 3
    assert state.constant_feature_names == (ROI_RELATION_FEATURE_NAMES[0],)
    assert state.scale[0] == 1.0
    assert state.validation_rows_read_for_fit == 0
    assert state.outer_holdout_rows_read_for_fit == 0


def test_roi_relation_modes_are_parameter_matched_and_missing_safe() -> None:
    base, cache, selection = _synthetic_inputs()
    normalization = fit_roi_relation_normalization(cache, selection)
    positions = np.asarray([0, 1], dtype=np.int64)

    zero = build_roi_relation_view(
        base,
        cache,
        mode="parameter_matched_zero",
        normalization=normalization,
    ).load_sequences(positions)
    availability = build_roi_relation_view(
        base,
        cache,
        mode="availability_only",
        normalization=normalization,
    ).load_sequences(positions)
    relation_view = build_roi_relation_view(
        base,
        cache,
        mode=MODALITY_NAME,
        normalization=normalization,
    )
    relation = relation_view.load_sequences(positions)
    missing = relation_view.with_missing_modality().load_sequences(positions)

    assert zero.shape == (2, SEQUENCE_LENGTH, MODEL_INPUT_DIM)
    assert availability.shape == zero.shape
    assert relation.shape == zero.shape
    assert np.count_nonzero(zero[..., FEATURE_DIM:]) == 0
    assert np.count_nonzero(availability[..., FEATURE_DIM:-1]) == 0
    assert np.all(availability[..., -1] == 1.0)
    assert np.isfinite(relation).all()
    assert np.count_nonzero(relation[..., FEATURE_DIM:-1]) > 0
    assert np.count_nonzero(missing[..., FEATURE_DIM:]) == 0


def test_roi_relation_model_width_parameter_count_and_mask_safety() -> None:
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

    model = build_roi_relation_model(_Config()).eval()
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


def test_roi_relation_whitelist_and_surface_rename() -> None:
    whitelist = l6_roi_relation_feature_whitelist(MODALITY_NAME)

    assert whitelist["feature_count"] == MODEL_INPUT_DIM
    assert whitelist["modality_feature_count"] == ROI_RELATION_DIM
    assert all("geometry_" not in name for name in whitelist["features"])
    assert all("motion_" not in name for name in whitelist["features"])
    result: dict[str, Any] = {
        "frame": pd.DataFrame({"geometry_mode": [MODALITY_NAME]}),
        "metrics": {"geometry_mode": MODALITY_NAME},
    }
    _rename_geometry_surfaces(result)
    assert "roi_relation_mode" in result["frame"]
    assert result["metrics"]["roi_relation_mode"] == MODALITY_NAME


def test_roi_relation_short_config_locks_width_and_family() -> None:
    path = Path(
        "configs/classification_v2/"
        "legacy_development_l6_roi_relation_short_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    _validate_config_payload(payload)

    assert payload["canonical_source_name"] == "legacy_16f"
    assert payload["model"]["model_input_dim"] == 531
    assert payload["model"]["parameter_count"] == 70_704
    assert payload["experiment_contract"]["changed_family"] == (
        "all_class_roi_relation_only"
    )
    assert payload["experiment_contract"]["motion_feature_values_in_model_x"] is False
