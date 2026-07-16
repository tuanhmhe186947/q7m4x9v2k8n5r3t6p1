"""Fail-closed artifact checks for tracking experiments."""

from __future__ import annotations

from pathlib import Path


def find_mp4_artifacts(root: Path) -> list[Path]:
    """Return every MP4 file recursively below an existing root."""
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )


def assert_no_mp4_artifacts(root: Path, *, context: str) -> None:
    """Raise when an analysis output contains an MP4 artifact."""
    artifacts = find_mp4_artifacts(root)
    if not artifacts:
        return
    relative_paths = ", ".join(
        str(path.relative_to(root)) for path in artifacts[:10]
    )
    if len(artifacts) > 10:
        relative_paths += f", ... ({len(artifacts)} total)"
    raise RuntimeError(f"{context} contains forbidden MP4: {relative_paths}")


__all__ = ["assert_no_mp4_artifacts", "find_mp4_artifacts"]
