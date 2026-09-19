"""Path profile helpers for tracking annotation and evaluation commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_PATH_CONFIG = PROJECT_ROOT / "configs" / "tracking_paths.json"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def resolve_project_path(value: str | Path | None) -> Path | None:
    """Resolve a user path relative to the project root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_tracking_path_profile(
    config_path: Path | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Load one path profile from configs/tracking_paths.json."""
    path = config_path or DEFAULT_TRACKING_PATH_CONFIG
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    profiles = payload.get("profiles", {})
    selected = profile_name or payload.get("active_profile") or "default"
    if selected not in profiles:
        known = ", ".join(sorted(profiles)) or "<none>"
        raise KeyError(f"Unknown tracking path profile '{selected}'. Known: {known}")
    profile = dict(profiles[selected])
    profile["_profile_name"] = selected
    profile["_config_path"] = str(path)
    return profile


def profile_path(
    profile: dict[str, Any],
    key: str,
    fallback: Path | None = None,
) -> Path | None:
    """Resolve a simple path field from a profile with optional fallback."""
    value = profile.get(key)
    if value in (None, ""):
        return fallback
    return resolve_project_path(value)


def profile_video_path(
    profile: dict[str, Any],
    video_key: str | None = None,
    fallback: Path | None = None,
) -> Path | None:
    """Resolve a video from explicit path, configured alias, or video_dir stem."""
    videos = profile.get("videos") or {}
    key = video_key or profile.get("active_video")
    if key and key in videos:
        return resolve_project_path(videos[key])
    if key:
        direct = resolve_project_path(key)
        if direct is not None and direct.exists():
            return direct
        video_dir = profile_path(profile, "video_dir", PROJECT_ROOT / "data" / "videos")
        if video_dir is not None:
            candidates = []
            key_path = Path(key)
            names = [key_path.name]
            if key_path.suffix:
                names.append(key_path.stem)
            for name in dict.fromkeys(names):
                candidate = video_dir / name
                if candidate.exists() and candidate.is_file():
                    candidates.append(candidate)
                if not Path(name).suffix:
                    candidates.extend(
                        video_dir / f"{name}{suffix}" for suffix in VIDEO_EXTENSIONS
                    )
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    return candidate
        known = ", ".join(sorted(videos)) or "<none>"
        searched = str(video_dir) if video_dir is not None else "<none>"
        raise FileNotFoundError(
            f"Video '{key}' was not found as a path, configured alias, or file "
            f"stem in video_dir={searched}. Configured aliases: {known}"
        )
    return fallback


def profile_video_paths(
    profile: dict[str, Any],
    video_keys: list[str] | None = None,
) -> list[Path]:
    """Resolve multiple videos from configured aliases or all files in video_dir."""
    videos = profile.get("videos") or {}
    if video_keys:
        return [
            path
            for key in video_keys
            if (path := profile_video_path(profile, key)) is not None
        ]
    video_dir = profile_path(profile, "video_dir", PROJECT_ROOT / "data" / "videos")
    if video_dir is not None and video_dir.exists():
        return sorted(
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    return [
        path
        for key in videos
        if (path := profile_video_path(profile, key)) is not None
    ]
