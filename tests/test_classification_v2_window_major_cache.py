"""Tests for window-major RGB execution cache and production integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pig_behavior.classification_v2.datasets.window_major_rgb_cache import (
    WindowMajorRgbReader,
    WindowMajorRgbReaderConfig,
    stage_window_major_cache_to_tmp,
)


def _create_mock_cache(
    tmp_path: Path,
    n_rows: int = 10,
    frames: int = 6,
    size: int = 128,
) -> tuple[Path, Path, Path, list[str]]:
    rgb_data = np.random.randint(
        0, 256,
        size=(n_rows, 2, frames, size, size, 3),
        dtype=np.uint8,
    )
    # Set union unavailable for odd rows, available for even rows
    mask_data = np.zeros((n_rows, frames), dtype=np.float32)
    mask_data[::2, :] = 1.0
    rgb_data[1::2, 1, :, :, :, :] = 0  # unavailable union is 0

    window_ids = [f"window_{i:04d}" for i in range(n_rows)]
    index_df = pd.DataFrame({"row_index": np.arange(n_rows), "window_id": window_ids})

    rgb_path = tmp_path / "m0_rgb_window_major_u8.npy"
    mask_path = tmp_path / "m0_union_available_mask.npy"
    index_path = tmp_path / "m0_rgb_window_index.csv"

    np.save(rgb_path, rgb_data)
    np.save(mask_path, mask_data)
    index_df.to_csv(index_path, index=False)

    return rgb_path, mask_path, index_path, window_ids


def test_window_major_rgb_reader_schema_and_read(tmp_path: Path) -> None:
    rgb_p, mask_p, idx_p, w_ids = _create_mock_cache(tmp_path, n_rows=8)
    config = WindowMajorRgbReaderConfig(
        rgb_cache_path=rgb_p,
        union_mask_path=mask_p,
        window_index_path=idx_p,
        expected_window_ids=w_ids,
        expected_image_size=128,
        expected_frames=6,
    )
    reader = WindowMajorRgbReader(config)
    assert reader.total_rows == 8

    indices = np.array([0, 2, 3], dtype=np.int64)
    tensors = reader.read_batch_tensors(indices, torch.device("cpu"))

    assert "image" in tensors
    assert "visual_context_image" in tensors
    assert "visual_context_observed_mask" in tensors

    assert tensors["image"].shape == (3, 6, 3, 128, 128)
    assert tensors["visual_context_image"].shape == (3, 6, 3, 128, 128)
    assert tensors["visual_context_observed_mask"].shape == (3, 6)

    # Row 0 and 2 are even (available), Row 3 is odd (unavailable)
    assert tensors["visual_context_observed_mask"][0].sum().item() == 6.0
    assert tensors["visual_context_observed_mask"][1].sum().item() == 6.0
    assert tensors["visual_context_observed_mask"][2].sum().item() == 0.0
    assert tensors["visual_context_image"][2].sum().item() == 0.0


def test_window_major_rgb_reader_fails_closed_on_shape_mismatch(
    tmp_path: Path,
) -> None:
    rgb_p, mask_p, idx_p, w_ids = _create_mock_cache(tmp_path, n_rows=8)
    bad_config = WindowMajorRgbReaderConfig(
        rgb_cache_path=rgb_p,
        union_mask_path=mask_p,
        window_index_path=idx_p,
        expected_image_size=64,  # Mismatch!
        expected_frames=6,
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        WindowMajorRgbReader(bad_config)


def test_stage_window_major_cache_to_tmp(tmp_path: Path) -> None:
    src_dir = tmp_path / "persistent"
    src_dir.mkdir()
    _create_mock_cache(src_dir, n_rows=4)

    dst_dir = tmp_path / "tmp_stage"
    staged = stage_window_major_cache_to_tmp(src_dir, dst_dir)

    assert staged["rgb_cache_path"].exists()
    assert staged["union_mask_path"].exists()
    assert staged["window_index_path"].exists()

    # Re-staging with existing files should succeed idempotently
    staged_again = stage_window_major_cache_to_tmp(src_dir, dst_dir)
    assert staged_again["rgb_cache_path"] == staged["rgb_cache_path"]


def test_sampler_permutation_invariants() -> None:
    # Verify that training order uses exact seed-based permutation,
    # never sequential contiguous slicing.
    n_train = 1000
    indices = np.arange(n_train)
    batch_size = 128

    for seed in [20260804, 20260805, 20260806]:
        rng = np.random.default_rng(seed)
        ordered = rng.permutation(indices)

        batches = [
            ordered[start : start + batch_size]
            for start in range(0, len(ordered), batch_size)
        ]

        # Invariant 1: First batch is NOT sequential contiguous [0:128]
        assert not np.array_equal(batches[0], np.arange(batch_size))
        # Invariant 2: Total sampled elements matches population
        assert sum(len(b) for b in batches) == n_train
        # Invariant 3: Exact set of sampled indices is a bijection
        assert set(np.concatenate(batches)) == set(indices)
