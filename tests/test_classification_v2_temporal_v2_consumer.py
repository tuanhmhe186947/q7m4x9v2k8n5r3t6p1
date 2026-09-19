"""Focused contracts for the final canonical Temporal-v2 consumer routes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from pig_behavior.classification_v2.training.post_s1_resolution_screening import (
    load_canonical_resolution_temporal_target,
)
from pig_behavior.classification_v2.training.pre_s1_rgb_binding import (
    load_canonical_s1_temporal_target,
)
from pig_behavior.classification_v2.training.temporal_v2_consumer import (
    FULL,
    MATCHED,
    TemporalV2ConsumerError,
    audit_matched_support,
    audit_release_counts,
    audit_resolution_parity,
    build_target_frame_offset_index,
    load_resolution_temporal_v2_target,
    load_temporal_v2_target,
    verify_registered_canonical_authority,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(
    target_id: str,
    corpus: str,
    view: str,
    frames: list[int],
    source: str,
    support: str = "",
) -> dict[str, str]:
    return {
        "target_id": target_id,
        "pool": corpus,
        "view_id": view,
        "target_length": str(len(frames)),
        "selected_frame_indices": json.dumps(frames),
        "source_type": source,
        "dataset_id": f"dataset-{source}",
        "video_key": f"video-{source}",
        "object_track_key": f"actor-{source}",
        "behavior": "eat",
        "matched_support_id": support,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _root(tmp_path: Path) -> dict[str, dict[str, str]]:
    views = {
        "T6": [5, 6, 7, 8, 9, 10],
        "T8": [4, 5, 6, 7, 8, 9, 10, 11],
        "T12": list(range(2, 14)),
        "T16": list(range(16)),
    }
    targets = {
        f"legacy-{view}": _target(
            f"legacy-{view}", FULL, view, frames, "legacy"
        )
        for view, frames in views.items()
    }
    targets.update(
        {
            f"cvat-{view}": _target(
                f"cvat-{view}",
                MATCHED,
                view,
                [frame + 100 for frame in frames],
                "cvat",
                "cvat-support",
            )
            for view, frames in views.items()
        }
    )
    _write_csv(
        tmp_path / "full_temporal_window_manifest_release.csv",
        [value for key, value in targets.items() if key.startswith("legacy-")],
    )
    _write_csv(
        tmp_path / "matched_temporal_window_manifest_release.csv",
        [value for key, value in targets.items() if key.startswith("cvat-")],
    )
    split_rows = [
        {"target_id": target_id, "outer_fold_id": "FOLD_1", "split": "validation"}
        for target_id in targets
    ]
    _write_csv(tmp_path / "target_split_roles.csv", split_rows)
    frame_rows: list[dict[str, str]] = []
    for target_id, target in targets.items():
        for index, frame in enumerate(json.loads(target["selected_frame_indices"])):
            frame_rows.append(
                {
                    "target_id": target_id,
                    "frame_index": str(frame),
                    "observed_mask": "True",
                    "window_first_step_reset": str(index == 0),
                    "source_type": target["source_type"],
                    "dataset_id": target["dataset_id"],
                    "video_key": target["video_key"],
                    "object_track_key": target["object_track_key"],
                    "behavior": target["behavior"],
                    "pool": target["pool"],
                    "view_id": target["view_id"],
                    "split": "validation",
                    "outer_fold_id": "FOLD_1",
                }
            )
    _write_csv(tmp_path / "sequence_frame_features.csv", frame_rows)
    _write_csv(
        tmp_path / "window_temporal_feature_summary.csv",
        [
            {
                "target_id": target_id,
                "temporal_pair_coverage_ratio_window": "1.0",
            }
            for target_id in targets
        ],
    )
    (tmp_path / "temporal_semantics_authority_v2.json").write_text(
        json.dumps(
            {
                "schema_version": "classification_v2.temporal_semantics_authority.v2",
                "gate": {"status": "PASS"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "temporal_v2_publication_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "classification_v2.temporal_v2_publication_receipt.v1",
                "status": "PASS",
                "release_counts": {
                    FULL: {view: 1 for view in views},
                    MATCHED: {view: 1 for view in views},
                },
            }
        ),
        encoding="utf-8",
    )
    names = (
        "full_temporal_window_manifest_release.csv",
        "matched_temporal_window_manifest_release.csv",
        "sequence_frame_features.csv",
        "target_split_roles.csv",
        "window_temporal_feature_summary.csv",
    )
    (tmp_path / "temporal_v2_artifact_hash_manifest.json").write_text(
        json.dumps({"artifacts": {name: {"sha256": _sha256(tmp_path / name)} for name in names}}),
        encoding="utf-8",
    )
    (tmp_path / "canonical_mapping.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "classification_v2.temporal_v2_canonical_authority_mapping.v1"
                ),
                "verification": {"status": "PASS"},
                "source": {
                    "authority_sha256": _sha256(
                        tmp_path / "temporal_semantics_authority_v2.json"
                    ),
                    "artifact_manifest_sha256": _sha256(
                        tmp_path / "temporal_v2_artifact_hash_manifest.json"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    return targets


def test_legacy_membership_is_emitted_for_all_views(tmp_path: Path) -> None:
    targets = _root(tmp_path)
    expected = {"T6": range(5, 11), "T8": range(4, 12), "T12": range(2, 14), "T16": range(16)}
    for view, frames in expected.items():
        target = load_temporal_v2_target(
            tmp_path,
            corpus=FULL,
            view=view,
            target_id=targets[f"legacy-{view}"]["target_id"],
        )
        assert target.frames == tuple(frames)
        assert target.boundary_reset is True


def test_cvat_emitted_membership_and_matched_support_are_preserved(tmp_path: Path) -> None:
    targets = _root(tmp_path)
    values = [
        load_temporal_v2_target(
            tmp_path,
            corpus=MATCHED,
            view=view,
            target_id=targets[f"cvat-{view}"]["target_id"],
        )
        for view in ("T6", "T8", "T12", "T16")
    ]
    assert values[0].frames == (105, 106, 107, 108, 109, 110)
    assert {value.matched_support_id for value in values} == {"cvat-support"}
    assert audit_matched_support(values)["status"] == "PASS"


def test_release_counts_are_bound_to_the_publication_receipt(tmp_path: Path) -> None:
    _root(tmp_path)
    assert audit_release_counts(tmp_path) == {
        FULL: {"T6": 1, "T8": 1, "T12": 1, "T16": 1},
        MATCHED: {"T6": 1, "T8": 1, "T12": 1, "T16": 1},
    }


def test_registered_mapping_allows_bounded_runtime_validation(tmp_path: Path) -> None:
    _root(tmp_path)
    receipt = verify_registered_canonical_authority(
        tmp_path,
        mapping_path=tmp_path / "canonical_mapping.json",
    )
    assert set(receipt) == {"authority_sha256", "artifact_manifest_sha256"}
    mapping = tmp_path / "canonical_mapping.json"
    mapping.write_text("{}", encoding="utf-8")
    with pytest.raises(TemporalV2ConsumerError, match="mapping is not verified"):
        verify_registered_canonical_authority(tmp_path, mapping_path=mapping)


def test_frame_offset_index_reuses_existing_rows_and_fails_on_source_drift(
    tmp_path: Path,
) -> None:
    targets = _root(tmp_path)
    index = tmp_path / "frame_offsets.json"
    result = build_target_frame_offset_index(tmp_path, output_path=index)
    assert result["target_count"] == len(targets)
    value = load_temporal_v2_target(
        tmp_path,
        corpus=FULL,
        view="T6",
        target_id=targets["legacy-T6"]["target_id"],
        verify_hashes=False,
        frame_offset_index=index,
    )
    assert value.frames == (5, 6, 7, 8, 9, 10)
    path = tmp_path / "sequence_frame_features.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TemporalV2ConsumerError, match="source identity mismatch"):
        load_temporal_v2_target(
            tmp_path,
            corpus=FULL,
            view="T6",
            target_id=targets["legacy-T6"]["target_id"],
            verify_hashes=False,
            frame_offset_index=index,
        )


def test_final_s1_and_resolution_routes_share_one_target(tmp_path: Path) -> None:
    targets = _root(tmp_path)
    common = {
        "authority_root": tmp_path,
        "corpus": FULL,
        "view": "T6",
        "target_id": targets["legacy-T6"]["target_id"],
    }
    s1 = load_canonical_s1_temporal_target(**common)
    resolutions = [
        load_canonical_resolution_temporal_target(input_resolution=value, **common)
        for value in (64, 128, 160)
    ]
    assert s1.membership_source == "emitted:selected_frame_indices"
    assert s1.historical_selectors_reachable is False
    assert {value.target.frames for value in resolutions} == {(5, 6, 7, 8, 9, 10)}
    assert audit_resolution_parity([value.target for value in resolutions])["status"] == "PASS"


def test_resolution_cannot_expand_or_resample_a_canonical_target(tmp_path: Path) -> None:
    targets = _root(tmp_path)
    for resolution in (64, 128, 160):
        value = load_resolution_temporal_v2_target(
            authority_root=tmp_path,
            corpus=FULL,
            view="T6",
            target_id=targets["legacy-T6"]["target_id"],
            input_resolution=resolution,
        )
        assert value.frames == (5, 6, 7, 8, 9, 10)
    with pytest.raises(TemporalV2ConsumerError, match="unsupported resolution"):
        load_resolution_temporal_v2_target(
            authority_root=tmp_path,
            corpus=FULL,
            view="T6",
            target_id=targets["legacy-T6"]["target_id"],
            input_resolution=1600,
        )


@pytest.mark.parametrize(
    ("corpus", "view", "target_id", "message"),
    [
        ("UNKNOWN", "T6", "legacy-T6", "unsupported corpus"),
        (FULL, "T99", "legacy-T6", "unsupported view"),
        (MATCHED, "T6", "legacy-T6", "not unique"),
        (FULL, "T6", "missing", "not unique"),
    ],
)
def test_wrong_corpus_view_or_target_fails_closed(
    tmp_path: Path,
    corpus: str,
    view: str,
    target_id: str,
    message: str,
) -> None:
    _root(tmp_path)
    with pytest.raises(TemporalV2ConsumerError, match=message):
        load_temporal_v2_target(
            tmp_path,
            corpus=corpus,
            view=view,
            target_id=target_id,
        )


def test_hash_and_publication_version_fail_closed(tmp_path: Path) -> None:
    targets = _root(tmp_path)
    manifest = tmp_path / "full_temporal_window_manifest_release.csv"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TemporalV2ConsumerError, match="artifact hash mismatch"):
        load_temporal_v2_target(
            tmp_path,
            corpus=FULL,
            view="T6",
            target_id=targets["legacy-T6"]["target_id"],
        )
    _root(tmp_path)
    authority = tmp_path / "temporal_semantics_authority_v2.json"
    authority.write_text(json.dumps({"schema_version": "wrong", "gate": {"status": "PASS"}}))
    with pytest.raises(TemporalV2ConsumerError, match="version mismatch"):
        load_temporal_v2_target(
            tmp_path,
            corpus=FULL,
            view="T6",
            target_id=targets["legacy-T6"]["target_id"],
            verify_hashes=False,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("frame_index", "99", "order differs"),
        ("behavior", "move", "behavior disagreement"),
        ("split", "train", "split disagreement"),
        ("observed_mask", "MAYBE", "invalid authority boolean"),
        ("window_first_step_reset", "False", "does not reset"),
    ],
)
def test_frame_label_split_mask_order_and_boundary_disagreement_fails_closed(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    targets = _root(tmp_path)
    path = tmp_path / "sequence_frame_features.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0][column] = value
    _write_csv(path, rows)
    with pytest.raises(TemporalV2ConsumerError, match=message):
        load_temporal_v2_target(
            tmp_path,
            corpus=FULL,
            view="T6",
            target_id=targets["legacy-T6"]["target_id"],
            verify_hashes=False,
        )
