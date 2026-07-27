"""Focused deterministic checks for the H1-r3 shadow audit harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "tracking"
        / "run_h1_r3_shadow_support_density.py"
    )
    spec = importlib.util.spec_from_file_location(
        "h1_r3_shadow_support_density",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_development_population_is_loaded_without_validation() -> None:
    module = _module()
    manifest = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "tracking"
        / "h1_r3"
        / "H1_R3_DEVELOPMENT_MANIFEST.csv"
    )

    episodes = module.load_development_episodes(manifest)

    assert len(episodes) == 6
    assert sum(item["role"] == "positive" for item in episodes) == 4
    assert sum(item["role"] == "control" for item in episodes) == 2
    assert all(not item["episode_id"].startswith("V") for item in episodes)


def test_distribution_has_deterministic_quantiles() -> None:
    module = _module()
    rows = [{"score": value} for value in (0.1, 0.2, 0.3, 0.4)]

    summary = module.summarize_numeric(rows, "score")

    assert summary["count"] == 4
    assert summary["minimum"] == pytest.approx(0.1)
    assert summary["median"] == pytest.approx(0.25)
    assert summary["maximum"] == pytest.approx(0.4)


def test_empty_distribution_is_not_reported_as_zero() -> None:
    module = _module()

    summary = module.summarize_numeric([], "score")

    assert summary["count"] == 0
    assert summary["minimum"] == "NOT_MEASURED"
    assert summary["median"] == "NOT_MEASURED"
    assert summary["maximum"] == "NOT_MEASURED"


def test_candidate_export_fields_are_stably_ordered() -> None:
    module = _module()
    fields = module.candidate_fieldnames(
        [
            {
                "episode_id": "E01",
                "development_role": "positive",
                "video_key": "video",
                "frame_index": 1,
                "hidden_track_id": 1,
                "visible_track_id": 2,
                "detection_index": 0,
                "z_field": 1,
                "a_field": 2,
            }
        ]
    )

    assert fields[:7] == [
        "episode_id",
        "development_role",
        "video_key",
        "frame_index",
        "hidden_track_id",
        "visible_track_id",
        "detection_index",
    ]
    assert fields[7:] == ["a_field", "z_field"]
