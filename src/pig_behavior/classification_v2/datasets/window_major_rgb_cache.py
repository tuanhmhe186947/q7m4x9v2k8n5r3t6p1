"""Fast window-major reader and staging utilities for packed RGB tensors."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True, slots=True)
class WindowMajorRgbReaderConfig:
    rgb_cache_path: Path
    union_mask_path: Path
    window_index_path: Path
    expected_window_ids: pd.Series | list[str] | None = None
    expected_image_size: int = 128
    expected_frames: int = 6


class WindowMajorRgbReader:
    """Zero-overhead window-major contiguous reader for M0 training and eval."""

    def __init__(self, config: WindowMajorRgbReaderConfig) -> None:
        self.config = config
        if not config.rgb_cache_path.exists():
            raise FileNotFoundError(
                f"window-major RGB cache missing: {config.rgb_cache_path}"
            )
        if not config.union_mask_path.exists():
            raise FileNotFoundError(
                f"window-major union mask missing: {config.union_mask_path}"
            )
        if not config.window_index_path.exists():
            raise FileNotFoundError(
                f"window-major index missing: {config.window_index_path}"
            )

        self.index_df = pd.read_csv(
            config.window_index_path,
            low_memory=False,
        )
        self.total_rows = len(self.index_df)

        self.rgb_tensor = np.load(
            config.rgb_cache_path,
            mmap_mode="r",
        )
        self.mask_tensor = np.load(
            config.union_mask_path,
            mmap_mode="r",
        )

        self._validate_contracts()

    def _validate_contracts(self) -> None:
        expected_shape = (
            self.total_rows,
            2,
            self.config.expected_frames,
            self.config.expected_image_size,
            self.config.expected_image_size,
            3,
        )
        if self.rgb_tensor.shape != expected_shape:
            raise ValueError(
                "window-major RGB shape mismatch: "
                f"observed={self.rgb_tensor.shape}, expected={expected_shape}"
            )
        if self.rgb_tensor.dtype != np.uint8:
            raise ValueError(
                f"window-major RGB dtype must be uint8, got {self.rgb_tensor.dtype}"
            )

        expected_mask_shape = (self.total_rows, self.config.expected_frames)
        if self.mask_tensor.shape != expected_mask_shape:
            raise ValueError(
                "window-major union mask shape mismatch: "
                f"observed={self.mask_tensor.shape}, "
                f"expected={expected_mask_shape}"
            )

        if self.config.expected_window_ids is not None:
            expected = list(self.config.expected_window_ids)
            id_col = (
                "window_id"
                if "window_id" in self.index_df.columns
                else "target_id"
            )
            if id_col not in self.index_df.columns:
                raise ValueError("window-major index missing window_id/target_id column")
            observed = self.index_df[id_col].astype(str).tolist()
            if observed != expected:
                raise ValueError("window-major index does not match expected window IDs")

    def read_batch_tensors(
        self,
        indices: np.ndarray,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Read contiguous window-major batch and convert vectorially to float."""
        rgb_u8 = self.rgb_tensor[indices]
        mask_raw = self.mask_tensor[indices]

        rgb_t = torch.from_numpy(rgb_u8)
        rgb_float = rgb_t.permute(0, 1, 2, 5, 3, 4).float().div_(255.0)

        image = rgb_float[:, 0].to(device)
        visual_context_image = rgb_float[:, 1].to(device)
        visual_context_observed_mask = (
            torch.from_numpy(mask_raw).float().to(device)
        )

        return {
            "image": image,
            "visual_context_image": visual_context_image,
            "visual_context_observed_mask": visual_context_observed_mask,
        }


def file_sha256(path: Path) -> str:
    """Compute SHA256 of one file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def stage_window_major_cache_to_tmp(
    persistent_dir: Path,
    target_dir: Path = Path("/tmp/m0_window_major_r128_t6"),
    verify_hashes: bool = False,
) -> dict[str, Path]:
    """Stage persistent window-major cache files to fast runtime /tmp storage."""
    target_dir.mkdir(parents=True, exist_ok=True)
    required_files = [
        "m0_rgb_window_major_u8.npy",
        "m0_union_available_mask.npy",
        "m0_rgb_window_index.csv",
    ]

    staged_paths: dict[str, Path] = {}
    for filename in required_files:
        src = persistent_dir / filename
        dst = target_dir / filename
        if not src.exists():
            raise FileNotFoundError(
                f"persistent cache source missing: {src}"
            )

        need_copy = True
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                if not verify_hashes or file_sha256(dst) == file_sha256(src):
                    need_copy = False

        if need_copy:
            shutil.copyfile(src, dst)
            if dst.stat().st_size != src.stat().st_size:
                raise OSError(
                    f"failed to stage {src} -> {dst}: size mismatch"
                )

        staged_paths[filename] = dst

    return {
        "rgb_cache_path": staged_paths["m0_rgb_window_major_u8.npy"],
        "union_mask_path": staged_paths["m0_union_available_mask.npy"],
        "window_index_path": staged_paths["m0_rgb_window_index.csv"],
    }
