from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tracking" / "generate_current_main_baseline_caches.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("baseline_cache_tool", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_video_cadence_is_exact_and_finite() -> None:
    module = _load_module()
    video = module.VideoAuthority(
        video_key="example",
        video_path=Path("example.mp4"),
        video_sha256="a" * 64,
        gt_path=Path("example.xml"),
        gt_sha256="b" * 64,
        frame_count=1800,
        width=1920,
        height=1080,
        gt_authority="AUTHORITATIVE_FOR_MECHANISTIC_CONCLUSIONS",
    )

    indices = module.frame_indices(video, 2)

    assert len(indices) == 900
    assert indices[0] == 0
    assert indices[-1] == 1798
    assert all(
        right - left == 2
        for left, right in zip(indices, indices[1:], strict=False)
    )


def test_thirteen_video_population_requests_11700_frames() -> None:
    module = _load_module()
    videos = [
        module.VideoAuthority(
            video_key=f"video_{index:02d}",
            video_path=Path(f"video_{index:02d}.mp4"),
            video_sha256="a" * 64,
            gt_path=Path(f"video_{index:02d}.xml"),
            gt_sha256="b" * 64,
            frame_count=1800,
            width=1920,
            height=1080,
            gt_authority="AUTHORITATIVE_FOR_MECHANISTIC_CONCLUSIONS",
        )
        for index in range(13)
    ]

    assert sum(len(module.frame_indices(video, 2)) for video in videos) == 11700


def test_cache_path_is_partitioned_and_deterministic(tmp_path: Path) -> None:
    module = _load_module()

    first = module.cache_path(tmp_path, "Pigs281119_000216_30fps")
    second = module.cache_path(tmp_path, "Pigs281119_000216_30fps")

    assert first == second
    assert first.relative_to(tmp_path).as_posix() == (
        "partitions/Pigs281119_000216_30fps/detector_evidence.npz"
    )


def test_preflight_refuses_existing_cache_root(tmp_path: Path) -> None:
    module = _load_module()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    with pytest.raises(module.BaselineCacheError, match="refusing existing"):
        module.preflight(
            tmp_path,
            tmp_path / "lineage.json",
            cache_root,
            tmp_path / "baseline",
        )


def test_locked_unresolved_gt_video_is_named_explicitly() -> None:
    module = _load_module()

    assert module.STARTING_MAIN_SHA == (
        "64d835cbf1b25ecdef3a777a50f0b46db6c93f61"
    )
    assert module.SOURCE_LINEAGE_SHA256 == (
        "0cfb26acc7766e05c497d9efdfafa40dc92f2d5c527e0338b89602eef0838dfc"
    )


def test_lineage_authority_is_payload_hash_not_self_referential_file_hash() -> None:
    module = _load_module()
    payload = {
        "schema_version": "example",
    }
    payload["manifest_sha256"] = module.canonical_hash(payload)

    original = module.SOURCE_LINEAGE_SHA256
    module.SOURCE_LINEAGE_SHA256 = payload["manifest_sha256"]
    try:
        assert module.locked_lineage_payload_hash(payload) == payload[
            "manifest_sha256"
        ]
    finally:
        module.SOURCE_LINEAGE_SHA256 = original
