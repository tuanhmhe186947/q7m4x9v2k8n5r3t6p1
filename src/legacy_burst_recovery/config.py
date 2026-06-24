from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VALID_BEHAVIORS = {
    "drink",
    "eat",
    "fight",
    "social-nose",
    "explore",
    "lying",
    "stand",
    "move",
    "sitting",
    "playwithtoy",
}

RESOURCE_NAMES = {
    "color_video_path": "color.mp4",
    "depth_video_path": "depth.mp4",
    "times_txt_path": "times.txt",
    "background_path": "background.png",
    "background_depth_path": "background_depth.png",
    "mask_path": "mask.png",
    "depth_scale_path": "depth_scale.npy",
    "inverse_intrinsic_path": "inverse_intrinsic.npy",
    "rot_path": "rot.npy",
}


@dataclass(frozen=True)
class RecoveryConfig:
    input_csv: Path
    drive_root: Path
    output_root: Path
    detector_weights: Path | None
    manifest_only: bool
    extract_crops: bool
    extract_full_frames: bool
    track_end_mode: str
    save_debug_visuals: bool
    no_detect_manifest_only: bool
    max_rows: int | None
    max_videos: int | None
    filter_group_id: str | None
    filter_video: str | None
    log_file: Path | None
    flush_every: int
    resume: bool
    progress: bool


def ensure_output_dirs(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "crops").mkdir(exist_ok=True)
    (output_root / "full_frames").mkdir(exist_ok=True)
    (output_root / "debug_visuals").mkdir(exist_ok=True)
