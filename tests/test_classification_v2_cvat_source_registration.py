from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pig_behavior.classification_v2.training.cvat_source_registration import (
    CvatSourceRegistrationError,
    audit_cvat_source_path_enrichment,
    enrich_cvat_source_video_paths,
    load_cvat_source_registration,
)
from pig_behavior.classification_v2.training.post_s1_resolution_screening import (
    PostS1ResolutionError,
    _validate_cvat_source_registration,
)

REGISTRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/classification_v2/corrected_pooled_route_20260806"
    / "next_phase_20260806_r2/cvat_source_registration_authority_20260811.json"
)


def _frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_context_id": ["cvat-1", "cvat-2", "legacy-1"],
            "source_type": [
                "cvat_tracking_xml",
                "cvat_tracking_xml",
                "legacy_recovered",
            ],
            "source_video_key": [
                "Pigs291119_000216_30fps",
                "Pigs291119_000225_30fps.mp4",
                "legacy-crop-a",
            ],
            "source_video_path": ["", "", "crops/legacy-a.jpg"],
            "frame_index": [2, 5, 9],
            "object_track_key": ["pig-1", "pig-2", "pig-3"],
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
            "behavior_label": ["move", "fight", "stand"],
            "hidden_review_decision": ["No", "Yes", "No"],
            "include_in_training": [True, True, True],
            "harmonization_status": ["accepted", "accepted", "accepted"],
            "corrected_source_lineage": ["review-a", "review-b", "legacy-a"],
            "primary_s1_role": ["train", "validation", "train"],
            "outer_membership": [False, False, False],
            "temporal_unit_key": ["t6-a", "t6-b", "legacy-16"],
        }
    )


def _write_authority(path: Path, registrations: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "classification_v2.cvat_source_registration.v1",
                "status": "ACTIVE_PATH_ONLY_ENRICHMENT",
                "registrations": registrations,
            }
        ),
        encoding="utf-8",
    )


def _registration(key: str, filename: str) -> dict[str, object]:
    return {
        "source_video_key": key,
        "registered_relative_media_path": f"data/videos/{filename}.mp4",
        "source_provenance": {"tracking_video_key": filename},
    }


def test_current_authority_resolves_the_exact_twelve_cvat_keys_once() -> None:
    mapping, authority_sha256 = load_cvat_source_registration(REGISTRATION_PATH)

    assert len(mapping) == 12
    assert len(set(mapping.values())) == 12
    assert authority_sha256 == "891a7bbe28ca33fc6fb1f264d9ea3bc90476376d8d7f4735b9eeedb5a7752526"
    assert mapping["Pigs291119_000225_30fps.mp4"] == (
        "data/videos/Pigs291119_000225_30fps.mp4"
    )
    assert mapping["Pigs291119_000231"] == "data/videos/Pigs291119_000231_30fps.mp4"
    assert mapping["Pigs291119_000233"] == "data/videos/Pigs291119_000233_30fps.mp4"
    assert mapping["test video Pigs291119_000302_30fps"] == (
        "data/videos/Pigs291119_000302_30fps.mp4"
    )


def test_registration_enriches_only_blank_cvat_paths_and_preserves_science() -> None:
    before = _frames()
    after, _ = enrich_cvat_source_video_paths(
        before,
        registration_path=REGISTRATION_PATH,
    )
    audit = audit_cvat_source_path_enrichment(before, after)

    assert before["source_video_path"].tolist() == ["", "", "crops/legacy-a.jpg"]
    assert after["source_video_path"].tolist() == [
        "data/videos/Pigs291119_000216_30fps.mp4",
        "data/videos/Pigs291119_000225_30fps.mp4",
        "crops/legacy-a.jpg",
    ]
    assert audit["scientific_projection_sha256_before"] == (
        audit["scientific_projection_sha256_after"]
    )
    assert audit["review_projection_sha256_before"] == audit["review_projection_sha256_after"]
    assert audit["label_change_count"] == 0
    assert audit["review_row_change_count"] == 0
    assert audit["trainability_change_count"] == 0
    assert audit["harmonization_change_count"] == 0


def test_unknown_or_ambiguous_cvat_registration_fails_closed(tmp_path: Path) -> None:
    unknown = _frames()
    unknown.loc[0, "source_video_key"] = "C:/opaque/not-a-media-path.mp4"
    with pytest.raises(CvatSourceRegistrationError, match="unregistered"):
        enrich_cvat_source_video_paths(unknown, registration_path=REGISTRATION_PATH)

    duplicate = tmp_path / "duplicate.json"
    _write_authority(
        duplicate,
        [
            _registration("key-a", "one"),
            _registration("key-a", "two"),
        ],
    )
    with pytest.raises(CvatSourceRegistrationError, match="duplicate"):
        load_cvat_source_registration(duplicate)


def test_scientific_projection_rejects_a_review_or_label_change() -> None:
    before = _frames()
    after, _ = enrich_cvat_source_video_paths(
        before,
        registration_path=REGISTRATION_PATH,
    )
    after.loc[0, "behavior_label"] = "fight"

    with pytest.raises(CvatSourceRegistrationError, match="scientific fields"):
        audit_cvat_source_path_enrichment(before, after)


def test_post_s1_authority_rejects_registration_hash_drift(tmp_path: Path) -> None:
    registration_dir = tmp_path / "registrations"
    registration_dir.mkdir()
    registration = registration_dir / "registration.json"
    _write_authority(registration, [_registration("key-a", "one")])
    authority = {
        "cvat_source_registration": {
            "relative_segments": ["registrations"],
            "filename": registration.name,
            "sha256": "0" * 64,
        }
    }

    with pytest.raises(PostS1ResolutionError, match="hash drifted"):
        _validate_cvat_source_registration(authority, repository_root=tmp_path)
