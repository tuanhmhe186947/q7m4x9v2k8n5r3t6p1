from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import RESOURCE_NAMES

COLAB_PREFIXES = (
    "/content/drive/MyDrive/",
    "/content/drive/My Drive/",
)


@dataclass(frozen=True)
class SourceResources:
    source_video_original: str
    source_video_resolved: str
    source_folder: str
    color_video_path: str
    depth_video_path: str
    times_txt_path: str
    background_path: str
    background_depth_path: str
    mask_path: str
    depth_scale_path: str
    inverse_intrinsic_path: str
    rot_path: str


def map_drive_path(raw_path: object, drive_root: Path) -> Path | None:
    if raw_path is None or pd.isna(raw_path):
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    normalized = text.replace("\\", "/")
    for prefix in COLAB_PREFIXES:
        if normalized.startswith(prefix):
            return drive_root / normalized[len(prefix) :]
    return Path(text)


def resolve_existing_path(raw_path: object, drive_root: Path) -> tuple[str, str, bool, str]:
    candidate = map_drive_path(raw_path, drive_root)
    original = "" if raw_path is None or pd.isna(raw_path) else str(raw_path)
    if candidate is None:
        return original, "", False, "empty_path"
    exists = candidate.exists()
    return original, str(candidate), exists, "ok" if exists else "missing"


def source_folder_from_video(video_path: Path) -> Path:
    return video_path.parent if video_path.name.lower() == "color.mp4" else video_path


def build_source_resources(original_video: str, resolved_video: str) -> SourceResources:
    color = Path(resolved_video) if resolved_video else Path()
    folder = source_folder_from_video(color) if resolved_video else Path()
    values: dict[str, str] = {
        "source_video_original": original_video,
        "source_video_resolved": str(color) if resolved_video else "",
        "source_folder": str(folder) if resolved_video else "",
    }
    for field_name, resource_name in RESOURCE_NAMES.items():
        values[field_name] = str(folder / resource_name) if resolved_video else ""
    return SourceResources(**values)


def collect_path_resolution(
    df: pd.DataFrame,
    drive_root: Path,
    *,
    show_progress: bool = False,
    max_videos: int | None = None,
) -> tuple[dict[str, SourceResources], pd.DataFrame]:
    reports: list[dict[str, object]] = []
    resources_by_original: dict[str, SourceResources] = {}
    videos = sorted({str(v) for v in df["video_final"].dropna().unique()})
    if max_videos is not None:
        videos = videos[:max_videos]
    iterator = tqdm(videos, desc="Resolving source video paths", disable=not show_progress)
    for raw_video in iterator:
        original, resolved, exists, status = resolve_existing_path(raw_video, drive_root)
        resources = build_source_resources(original, resolved)
        resources_by_original[raw_video] = resources
        reports.append(
            {
                "source_video_original": original,
                "source_video_resolved": resolved,
                "exists": exists,
                "status": status,
                "source_folder": resources.source_folder,
                "times_txt_resolved": resources.times_txt_path,
                "times_txt_exists": Path(resources.times_txt_path).exists() if resources.times_txt_path else False,
            }
        )
    return resources_by_original, pd.DataFrame(reports)


def write_path_resolution_report(report_df: pd.DataFrame, output_root: Path) -> None:
    report_df.to_csv(output_root / "path_resolution_report.csv", index=False)


def resource_completeness(paths: Iterable[str]) -> bool:
    return all(Path(p).exists() for p in paths if p)
