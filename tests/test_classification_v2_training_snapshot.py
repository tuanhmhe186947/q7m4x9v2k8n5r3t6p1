from __future__ import annotations

import json

import pandas as pd
import pytest

from pig_behavior.classification_v2.contracts.training_snapshot import (
    _artifact_hash_bindings,
    _key_alignment,
    _key_coverage,
    _ordered_key_digest,
    _snapshot_id,
    _validate_contract_profiles,
    freeze_training_snapshot,
)
from pig_behavior.classification_v2.contracts.window_alignment import (
    ordered_window_id_sha256,
)


def test_snapshot_id_excludes_git_provenance() -> None:
    """Code-only commits must not rename an unchanged artifact snapshot."""

    snapshot = {
        "contract_version": "v1",
        "artifacts": {"x": {"sha256": "abc"}},
        "git_commit": "old",
    }
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


def test_snapshot_ordered_hash_matches_exporter_contract(tmp_path) -> None:
    """Snapshot and exporter hashes must use identical ordered-key bytes."""

    path = tmp_path / "windows.csv"
    window_ids = pd.Series(["window-2", "window-0", "window-1"])
    pd.DataFrame({"window_id": window_ids}).to_csv(path, index=False)

    profile = _ordered_key_digest(path, "window_id")

    assert profile["ordered_key_hash_version"] == "newline_join_v1"
    assert profile["ordered_key_sha256"] == ordered_window_id_sha256(
        window_ids
    )


def test_key_alignment_rejects_same_keys_in_different_order() -> None:
    """Set equality cannot substitute for positional multimodal alignment."""

    contract = {
        "window_id_source_artifact": "split",
        "key_alignment_group": ["split", "images", "interaction"],
    }
    artifacts = {
        "split": {"ordered_key_sha256": "ordered-a"},
        "images": {"ordered_key_sha256": "ordered-a"},
        "interaction": {"ordered_key_sha256": "ordered-b"},
    }

    alignment = _key_alignment(contract, artifacts)

    assert alignment["aligned"] is False
    assert alignment["mismatched"] == ["interaction"]


def test_key_alignment_fails_when_source_digest_is_missing() -> None:
    """A missing reference digest must not make an empty group look aligned."""

    alignment = _key_alignment(
        {
            "window_id_source_artifact": "split",
            "key_alignment_group": ["split", "images"],
        },
        {"split": {}, "images": {}},
    )

    assert alignment["aligned"] is False
    assert alignment["mismatched"] == ["images", "split"]


def test_contract_profiles_reject_blank_keys() -> None:
    """Blank stable keys are contract violations, not joinable rows."""

    contract = {
        "window_id_source_artifact": "split",
        "key_alignment_group": ["split"],
        "artifacts": {"split": {}},
    }
    artifacts = {
        "split": {
            "ordered_key_sha256": "ordered",
            "null_key_count": 1,
        }
    }

    errors = _validate_contract_profiles(contract, artifacts)

    assert "blank_key:split=1" in errors


def test_freeze_refuses_to_persist_invalid_snapshot(tmp_path) -> None:
    """A missing required artifact must not create an immutable snapshot."""

    output_path = tmp_path / "snapshot.json"
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "contract_version": "fixture-v1",
                "snapshot_name": "fixture",
                "root": str(tmp_path),
                "snapshot_output_dir": ".",
                "artifacts": {
                    "missing": {
                        "path": "missing.csv",
                        "type": "csv",
                        "required": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Cannot freeze an invalid"):
        freeze_training_snapshot(
            contract_path,
            output_path=output_path,
        )

    assert output_path.exists() is False


def test_snapshot_requires_passing_scientific_json_gate(tmp_path) -> None:
    """An existing gate file cannot pass while its status remains blocked."""

    gate_path = tmp_path / "hidden_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "status": "BLOCKED_INCOMPLETE_OR_INSUFFICIENT_REVIEW",
                "training_snapshot_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "contract_version": "fixture-json-gate-v1",
                "snapshot_name": "fixture",
                "root": str(tmp_path),
                "snapshot_output_dir": ".",
                "artifacts": {
                    "hidden_gate": {
                        "path": "hidden_gate.json",
                        "type": "json",
                        "required": True,
                        "required_json_values": {
                            "status": "PASS",
                            "training_snapshot_allowed": True,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_json_value_mismatch"):
        freeze_training_snapshot(
            contract_path,
            output_path=tmp_path / "snapshot.json",
        )


def test_snapshot_accepts_nested_scientific_json_assertions(tmp_path) -> None:
    gate_path = tmp_path / "hidden_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "training_snapshot_allowed": True,
                "random_hidden_no_prevalence": {"final_estimate": True},
            }
        ),
        encoding="utf-8",
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "contract_version": "fixture-json-gate-v1",
                "snapshot_name": "fixture",
                "root": str(tmp_path),
                "snapshot_output_dir": ".",
                "artifacts": {
                    "hidden_gate": {
                        "path": "hidden_gate.json",
                        "type": "json",
                        "required": True,
                        "required_json_values": {
                            "status": "PASS",
                            "training_snapshot_allowed": True,
                            "random_hidden_no_prevalence.final_estimate": True,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = freeze_training_snapshot(
        contract_path,
        output_path=tmp_path / "snapshot.json",
    )

    profile = snapshot["artifacts"]["hidden_gate"]
    assert profile["required_json_value_mismatches"] == {}


def test_snapshot_rejects_stale_scientific_gate_hash_binding() -> None:
    contract = {
        "artifact_hash_bindings": [
            {
                "source_artifact": "policy",
                "consumer_artifact": "gate",
                "consumer_json_field": "policy_sha256",
            }
        ]
    }
    artifacts = {
        "policy": {"sha256": "current-policy"},
        "gate": {
            "observed_json_hash_fields": {
                "policy_sha256": "stale-policy",
            }
        },
    }

    binding = _artifact_hash_bindings(contract, artifacts)
    errors = _validate_contract_profiles(contract, artifacts)

    assert binding["aligned"] is False
    assert errors == ["artifact_hash_binding_mismatch:policy->gate.policy_sha256"]

    artifacts["gate"]["observed_json_hash_fields"][
        "policy_sha256"
    ] = "current-policy"
    assert _artifact_hash_bindings(contract, artifacts)["aligned"] is True
    assert _validate_contract_profiles(contract, artifacts) == []
