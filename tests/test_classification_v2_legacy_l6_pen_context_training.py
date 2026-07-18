from __future__ import annotations

import copy
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
from pig_behavior.classification_v2.training.legacy_development_l6_geometry_cache import (
    GEOMETRY_DIM,
    SEQUENCE_LENGTH,
)
from pig_behavior.classification_v2.training.legacy_development_l6_motion_cache import (
    MOTION_DIM,
)
from pig_behavior.classification_v2.training.legacy_development_l6_pen_context import (
    EXPECTED_PARAMETER_COUNT,
    FEATURE_DIM,
    MODEL_INPUT_DIM,
    LegacyL6CompositeCache,
    build_pen_context_model,
    build_pen_context_view,
    fit_pen_context_normalization,
    fit_pen_feature_normalization,
    pen_context_feature_whitelist,
)
from pig_behavior.classification_v2.training.legacy_development_l6_pen_context_cache import (
    PEN_DIM,
    PEN_STATIC_FEATURE_COUNT,
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
    values: np.ndarray
    availability: np.ndarray
    window_index: pd.DataFrame
    slot_index: pd.DataFrame
    audit: dict[str, Any]
    feature_availability: np.ndarray | None = None
    quality: np.ndarray | None = None
    motion_availability: np.ndarray | None = None

    def _rows(self, rows: np.ndarray | None) -> np.ndarray:
        if rows is None:
            return np.arange(len(self.values), dtype=np.int64)
        return np.asarray(rows, dtype=np.int64)

    def load_geometry(self, rows: np.ndarray | None = None) -> np.ndarray:
        return self.values[self._rows(rows)].copy()

    def load_motion(self, rows: np.ndarray | None = None) -> np.ndarray:
        return self.values[self._rows(rows)].copy()

    def load_pen(self, rows: np.ndarray | None = None) -> np.ndarray:
        return self.values[self._rows(rows)].copy()

    def load_availability(self, rows: np.ndarray | None = None) -> np.ndarray:
        return self.availability[self._rows(rows)].copy()

    def load_feature_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        assert self.feature_availability is not None
        return self.feature_availability[self._rows(rows)].copy()

    def load_quality(self, rows: np.ndarray | None = None) -> np.ndarray:
        assert self.quality is not None
        return self.quality[self._rows(rows)].copy()

    def load_motion_availability(
        self,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        assert self.motion_availability is not None
        return self.motion_availability[self._rows(rows)].copy()


def _synthetic_inputs() -> tuple[
    _FakeBase,
    LegacyL6CompositeCache,
    TemporalLadderSelection,
]:
    windows = pd.DataFrame(
        {
            "cache_row": np.arange(3),
            "window_id": ["w0", "w1", "w2"],
            "temporal_unit_key": ["u0", "u1", "u2"],
            "l5_role": ["train", "train", "validation"],
            "source_type": ["legacy_recovered"] * 3,
            "dataset_id": ["legacy_recovered_16f"] * 3,
            "lineage_scope": ["legacy-only-unreviewed-development"] * 3,
            "human_review_complete": [False] * 3,
        }
    )
    base = _FakeBase(
        windows=windows,
        sequences=np.zeros((3, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32),
        observed_mask=np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_),
        time_delta=np.tile(np.arange(SEQUENCE_LENGTH), (3, 1)).astype(np.float32),
        targets=np.asarray([0, 1, 2], dtype=np.int64),
        sample_weights=np.ones(3, dtype=np.float32),
    )
    slot_rows: list[dict[str, Any]] = []
    frame_numbers = [np.arange(0, 6), np.arange(3, 9), np.arange(20, 26)]
    for cache_row, frames in enumerate(frame_numbers):
        for slot_index, frame in enumerate(frames):
            slot_rows.append(
                {
                    "cache_row": cache_row,
                    "window_id": f"w{cache_row}",
                    "slot_index": slot_index,
                    "frame_uid": f"frame-{int(frame)}",
                    "pen_pair_uid": (
                        "" if slot_index == 0 else f"pair-{int(frame - 1)}-{int(frame)}"
                    ),
                }
            )
    slots = pd.DataFrame.from_records(slot_rows)
    base_values = np.stack(frame_numbers).astype(np.float32)
    geometry_values = np.stack(
        [base_values * (index + 1) + index for index in range(GEOMETRY_DIM)],
        axis=2,
    )
    geometry_available = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    motion_values = np.stack(
        [base_values * (index + 1) for index in range(MOTION_DIM)],
        axis=2,
    )
    motion_available = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    motion_available[:, 0] = False
    motion_values[:, 0] = 0.0
    pen_values = np.stack(
        [base_values * (index + 1) + 0.25 for index in range(PEN_DIM)],
        axis=2,
    )
    pen_feature_available = np.ones_like(pen_values, dtype=np.bool_)
    pen_feature_available[:, 0, PEN_STATIC_FEATURE_COUNT:] = False
    pen_values[~pen_feature_available] = 0.0
    pen_available = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    pen_motion = np.ones((3, SEQUENCE_LENGTH), dtype=np.bool_)
    pen_motion[:, 0] = False
    audits = [{"manifest_sha256": char * 64} for char in "abc"]
    geometry = _FakeCache(
        values=geometry_values,
        availability=geometry_available,
        window_index=windows,
        slot_index=slots,
        audit=audits[0],
    )
    motion = _FakeCache(
        values=motion_values,
        availability=motion_available,
        window_index=windows,
        slot_index=slots.assign(
            motion_window_slot_uid=(
                slots["window_id"] + "::slot=" + slots["slot_index"].astype(str)
            )
        ),
        audit=audits[1],
    )
    pen = _FakeCache(
        values=pen_values,
        availability=pen_available,
        feature_availability=pen_feature_available,
        quality=pen_available,
        motion_availability=pen_motion,
        window_index=windows,
        slot_index=slots.assign(
            object_track_key="track-a",
            frame_index=np.concatenate(frame_numbers),
        ),
        audit=audits[2],
    )
    selection = TemporalLadderSelection(
        manifest=pd.DataFrame({"window_id": ["w0", "w1", "w2"]}),
        train_positions=np.asarray([0, 1], dtype=np.int64),
        validation_positions=np.asarray([2], dtype=np.int64),
        audit={"selection_content_sha256": "d" * 64},
    )
    return base, LegacyL6CompositeCache(geometry, motion, pen), selection


def test_pen_normalization_uses_train_frames_and_pairs_only() -> None:
    _, cache, selection = _synthetic_inputs()

    state = fit_pen_feature_normalization(cache.pen, selection)
    first = state.to_payload()
    cache.pen.values[2] = 1_000_000.0
    second = fit_pen_feature_normalization(cache.pen, selection).to_payload()

    assert first == second
    assert state.identity_kinds[:PEN_STATIC_FEATURE_COUNT] == (
        "frame_uid",
    ) * PEN_STATIC_FEATURE_COUNT
    assert set(state.identity_kinds[PEN_STATIC_FEATURE_COUNT:]) == {
        "pen_pair_uid"
    }
    assert state.validation_rows_read_for_fit == 0
    assert state.outer_holdout_rows_read_for_fit == 0


def test_pen_modes_keep_geometry_motion_fixed_and_parameter_matched() -> None:
    base, cache, selection = _synthetic_inputs()
    normalization = fit_pen_context_normalization(cache, selection)
    positions = np.asarray([0, 1], dtype=np.int64)
    views = {
        mode: build_pen_context_view(
            base,
            cache,
            mode=mode,
            normalization=normalization,
        ).load_sequences(positions)
        for mode in (
            "parameter_matched_zero",
            "availability_only",
            "pen_context",
        )
    }
    fixed_end = FEATURE_DIM + GEOMETRY_DIM + 1 + MOTION_DIM + 1
    assert all(value.shape == (2, SEQUENCE_LENGTH, MODEL_INPUT_DIM) for value in views.values())
    np.testing.assert_array_equal(
        views["parameter_matched_zero"][..., :fixed_end],
        views["availability_only"][..., :fixed_end],
    )
    np.testing.assert_array_equal(
        views["parameter_matched_zero"][..., :fixed_end],
        views["pen_context"][..., :fixed_end],
    )
    assert np.count_nonzero(views["parameter_matched_zero"][..., fixed_end:]) == 0
    assert np.count_nonzero(views["availability_only"][..., fixed_end:-1]) == 0
    assert np.count_nonzero(views["availability_only"][..., -1]) > 0
    assert np.count_nonzero(views["pen_context"][..., fixed_end:-1]) > 0
    missing = build_pen_context_view(
        base,
        cache,
        mode="pen_context",
        normalization=normalization,
    ).with_missing_modality().load_sequences(positions)
    np.testing.assert_array_equal(missing[..., :fixed_end], views["pen_context"][..., :fixed_end])
    assert np.count_nonzero(missing[..., fixed_end:]) == 0


def test_pen_model_forward_backward_parameter_count_and_whitelist() -> None:
    class _Config:
        payload = {
            "model": {
                "temporal_encoder_name": "masked_mean",
                "hidden_dim": 128,
                "dropout": 0.0,
                "transformer_layers": 1,
                "transformer_heads": 4,
            }
        }

    model = build_pen_context_model(_Config())
    assert sum(value.numel() for value in model.parameters()) == (
        EXPECTED_PARAMETER_COUNT
    )
    features = torch.randn(8, SEQUENCE_LENGTH, MODEL_INPUT_DIM)
    observed = torch.ones(8, SEQUENCE_LENGTH)
    timing = torch.arange(SEQUENCE_LENGTH).repeat(8, 1).float()
    targets = torch.arange(8) % len(VALID_BEHAVIORS)
    logits = model(features, observed, time_delta=timing)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    assert logits.shape == (8, len(VALID_BEHAVIORS))
    assert all(
        value.grad is None or torch.isfinite(value.grad).all()
        for value in model.parameters()
    )
    whitelists = [
        pen_context_feature_whitelist(mode)
        for mode in (
            "parameter_matched_zero",
            "availability_only",
            "pen_context",
        )
    ]
    assert {item["feature_count"] for item in whitelists} == {MODEL_INPUT_DIM}
    assert all(item["geometry_and_motion_fixed_active"] for item in whitelists)
    forbidden = ("behavior", "label", "review", "path", "fold", "frame_uid")
    assert not any(
        token in feature.lower()
        for item in whitelists
        for feature in item["features"]
        for token in forbidden
    )


def test_pen_modes_tiny_overfit_and_checkpoint_resume_equivalence(
    tmp_path: Path,
) -> None:
    config = _SyntheticModelConfig()
    modes = (
        "parameter_matched_zero",
        "availability_only",
        "pen_context",
    )
    for mode in modes:
        batch = _synthetic_model_batch(mode, seed=20260717)
        torch.manual_seed(20260717)
        model = build_pen_context_model(config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.02,
            weight_decay=0.0,
        )
        initial_loss, _ = _model_loss_and_accuracy(model, batch)
        losses = [_model_step(model, optimizer, batch) for _ in range(60)]
        final_loss, final_accuracy = _model_loss_and_accuracy(model, batch)
        assert np.isfinite([initial_loss, final_loss, *losses]).all()
        assert final_accuracy == 1.0
        assert final_loss / initial_loss < 0.1

        torch.manual_seed(20260717)
        original = build_pen_context_model(config)
        original_optimizer = torch.optim.AdamW(
            original.parameters(),
            lr=0.003,
            weight_decay=0.0001,
        )
        _model_step(original, original_optimizer, batch)
        checkpoint = {
            "mode": mode,
            "model_state": copy.deepcopy(original.state_dict()),
            "optimizer_state": copy.deepcopy(original_optimizer.state_dict()),
            "torch_rng_state": torch.get_rng_state().clone(),
        }
        checkpoint_path = tmp_path / f"{mode}.pt"
        torch.save(checkpoint, checkpoint_path)
        loaded = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        resumed = build_pen_context_model(config)
        resumed_optimizer = torch.optim.AdamW(
            resumed.parameters(),
            lr=0.003,
            weight_decay=0.0001,
        )
        resumed.load_state_dict(loaded["model_state"])
        resumed_optimizer.load_state_dict(loaded["optimizer_state"])
        assert loaded["mode"] == mode
        assert _state_equal(original.state_dict(), resumed.state_dict())
        assert _state_equal(
            original_optimizer.state_dict(),
            resumed_optimizer.state_dict(),
        )

        rng_state = loaded["torch_rng_state"]
        torch.set_rng_state(rng_state.clone())
        original_next_loss = _model_step(
            original,
            original_optimizer,
            batch,
        )
        torch.set_rng_state(rng_state.clone())
        resumed_next_loss = _model_step(
            resumed,
            resumed_optimizer,
            batch,
        )
        assert original_next_loss == resumed_next_loss
        assert _state_equal(original.state_dict(), resumed.state_dict())
        assert _state_equal(
            original_optimizer.state_dict(),
            resumed_optimizer.state_dict(),
        )


class _SyntheticModelConfig:
    payload = {
        "model": {
            "temporal_encoder_name": "masked_mean",
            "hidden_dim": 128,
            "dropout": 0.1,
            "transformer_layers": 1,
            "transformer_heads": 4,
        }
    }


def _synthetic_model_batch(
    mode: str,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = np.random.default_rng(seed)
    targets = np.tile(np.arange(len(VALID_BEHAVIORS)), 2)
    features = generator.normal(
        size=(len(targets), SEQUENCE_LENGTH, MODEL_INPUT_DIM)
    ).astype(np.float32)
    pen_start = FEATURE_DIM + GEOMETRY_DIM + 1 + MOTION_DIM + 1
    features[..., pen_start:] = 0.0
    if mode == "availability_only":
        features[..., -1] = 1.0
    elif mode == "pen_context":
        features[..., pen_start:-1] = generator.normal(
            size=(len(targets), SEQUENCE_LENGTH, PEN_DIM)
        )
        features[..., -1] = 1.0
    elif mode != "parameter_matched_zero":
        raise ValueError(f"unknown synthetic pen mode={mode}")
    return {
        "features": torch.from_numpy(features),
        "observed_mask": torch.ones(len(targets), SEQUENCE_LENGTH),
        "time_delta": torch.arange(SEQUENCE_LENGTH).repeat(len(targets), 1).float(),
        "targets": torch.from_numpy(targets).long(),
    }


def _model_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        batch["features"],
        batch["observed_mask"],
        time_delta=batch["time_delta"],
    )
    loss = torch.nn.functional.cross_entropy(logits, batch["targets"])
    loss.backward()
    assert all(
        value.grad is None or torch.isfinite(value.grad).all()
        for value in model.parameters()
    )
    optimizer.step()
    return float(loss.detach())


def _model_loss_and_accuracy(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(
            batch["features"],
            batch["observed_mask"],
            time_delta=batch["time_delta"],
        )
        loss = torch.nn.functional.cross_entropy(logits, batch["targets"])
    accuracy = logits.argmax(dim=1).eq(batch["targets"]).float().mean()
    return float(loss), float(accuracy)


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right
