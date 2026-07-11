from __future__ import annotations

from pig_behavior.classification_v2.contracts.training_snapshot import _key_coverage, _snapshot_id


def test_snapshot_id_excludes_git_provenance() -> None:
    """Code-only commits must not rename an unchanged artifact snapshot."""

    snapshot = {"contract_version": "v1", "artifacts": {"x": {"sha256": "abc"}}, "git_commit": "old"}
    changed_commit = {**snapshot, "git_commit": "new"}

    assert _snapshot_id(snapshot) == _snapshot_id(changed_commit)


def test_multiple_key_coverage_groups_report_the_failed_lineage() -> None:
    """Window and image cache key spaces are validated independently."""

    contract = {
        "key_coverage_groups": [
            {"source_artifact": "windows", "artifacts": ["window_context"]},
            {"source_artifact": "frames", "artifacts": ["image_cache"]},
        ]
    }
    artifacts = {
        "windows": {"key_set_sha256": "window-keys"},
        "window_context": {"key_set_sha256": "window-keys"},
        "frames": {"key_set_sha256": "frame-keys"},
        "image_cache": {"key_set_sha256": "wrong-keys"},
    }

    coverage = _key_coverage(contract, artifacts)

    assert coverage["covered"] is False
    assert coverage["mismatched"] == ["frames->image_cache"]
