from __future__ import annotations

from pig_behavior.classification_v2.contracts.training_lineage import (
    audit_training_lineage_packet,
)

ORDERED_HASH = "ordered-window-sha"
LINEAGE_FILE_HASH = "lineage-file-sha"
CODE_SHA = "code-sha"


def test_training_lineage_accepts_valid_full_packet() -> None:
    result = _audit()

    assert result["technical_valid"] is True
    assert result["training_authorized"] is True
    assert result["valid"] is True


def test_training_lineage_rejects_same_keys_in_wrong_interaction_order() -> None:
    snapshot = _snapshot_check()
    snapshot["current"]["artifacts"][
        "interaction_window_context_manifest"
    ]["ordered_key_sha256"] = "different-order"

    result = _audit(snapshot=snapshot)

    assert result["valid"] is False
    assert any(
        "snapshot_ordered_window_hash_mismatch="
        "interaction_window_context_manifest" in error
        for error in result["errors"]
    )


def test_training_lineage_rejects_wrong_exporter_ordered_hash() -> None:
    lineage = _lineage_packet()
    lineage["exported_window_hashes"]["artifacts"]["spatial"][
        "matches_sequence"
    ] = False

    result = _audit(lineage=lineage)

    assert result["valid"] is False
    assert "lineage_exporter_hash_mismatch=spatial" in result["errors"]


def test_training_lineage_rejects_stale_code_sha() -> None:
    lineage = _lineage_packet()
    lineage["code_state"]["git_sha"] = "stale-code"

    result = _audit(lineage=lineage)

    assert result["valid"] is False
    assert any(
        "lineage_git_commit_mismatch" in error
        for error in result["errors"]
    )


def test_training_lineage_rejects_dataset_hash_drift() -> None:
    lineage = _lineage_packet()
    lineage["artifact_sha256"][
        "train_ready_tables.X_window_features"
    ] = "stale-data"

    result = _audit(lineage=lineage)

    assert result["valid"] is False
    assert any(
        "train_ready_tables.X_window_features->tabular_X" in error
        for error in result["errors"]
    )


def test_bounded_packet_can_pass_technical_without_training_authority() -> None:
    lineage = _lineage_packet(authorized=False)
    lineage["full_multimodal_lineage_complete"] = False

    bounded = _audit(
        lineage=lineage,
        require_full_multimodal=False,
        require_clean_code=False,
        require_training_authorization=False,
    )
    training = _audit(
        lineage=lineage,
        require_full_multimodal=True,
        require_training_authorization=True,
    )

    assert bounded["technical_valid"] is True
    assert bounded["training_authorized"] is False
    assert bounded["valid"] is True
    assert training["valid"] is False
    assert "full_multimodal_lineage_incomplete" in training["errors"]
    assert any(
        "reviewed_dataset_authorized" in error
        for error in training["errors"]
    )


def _audit(
    *,
    lineage: dict | None = None,
    snapshot: dict | None = None,
    require_full_multimodal: bool = True,
    require_clean_code: bool = True,
    require_training_authorization: bool = True,
) -> dict:
    return audit_training_lineage_packet(
        lineage or _lineage_packet(),
        snapshot or _snapshot_check(),
        lineage_file_sha256=LINEAGE_FILE_HASH,
        expected_git_commit=CODE_SHA,
        require_full_multimodal=require_full_multimodal,
        require_clean_code=require_clean_code,
        require_training_authorization=require_training_authorization,
    )


def _snapshot_check() -> dict:
    hashes = _snapshot_artifact_hashes()
    artifacts = {
        name: {"sha256": value}
        for name, value in hashes.items()
    }
    for name in (
        "split_manifest",
        "image_window_context_manifest",
        "interaction_window_context_manifest",
    ):
        artifacts[name]["ordered_key_sha256"] = ORDERED_HASH
    artifacts["identifier_v2_lineage_audit"] = {
        "sha256": LINEAGE_FILE_HASH
    }
    return {
        "valid": True,
        "errors": [],
        "expected_snapshot_id": "snapshot-id",
        "current": {
            "contract_digest": "contract-sha",
            "lineage_audit_artifact": "identifier_v2_lineage_audit",
            "required_ordered_window_artifacts": [
                "split_manifest",
                "image_window_context_manifest",
                "interaction_window_context_manifest",
            ],
            "artifacts": artifacts,
            "key_alignment": {
                "aligned": True,
                "mismatched": [],
            },
        },
    }


def _lineage_packet(*, authorized: bool = True) -> dict:
    exporter_names = (
        "train_ready",
        "spatial",
        "image_context_input",
        "image_context_output",
        "interaction_context_input",
        "interaction_context_output",
    )
    return {
        "schema_version": "classification_v2.source_to_window_lineage.v1",
        "technical_pass": True,
        "full_multimodal_lineage_complete": True,
        "errors": [],
        "window_lineage": {
            "ordered_window_id_sha256": ORDERED_HASH,
        },
        "exported_window_hashes": {
            "expected_sha256": ORDERED_HASH,
            "artifacts": {
                name: {
                    "sha256": ORDERED_HASH,
                    "matches_sequence": True,
                    "audited": True,
                }
                for name in exporter_names
            },
        },
        "artifact_sha256": _lineage_artifact_hashes(),
        "code_state": {
            "git_sha": CODE_SHA,
            "dirty_worktree": False,
        },
        "authorization": {
            "reviewed_dataset_authorized": authorized,
            "model_training_authorized": authorized,
            "full_oof_authorized": False,
            "q2_claim_authorized": False,
        },
        "human_review_blockers": [] if authorized else ["review incomplete"],
    }


def _lineage_artifact_hashes() -> dict[str, str]:
    snapshot_hashes = _snapshot_artifact_hashes()
    bindings = {
        "tables.image_frame_manifest": "image_frame_context_manifest",
        "tables.image_window_manifest": "image_window_context_manifest",
        "tables.interaction_window_manifest": (
            "interaction_window_context_manifest"
        ),
        "train_ready_tables.X_window_features": "tabular_X",
        "train_ready_tables.y_behavior": "y_behavior",
        "train_ready_tables.train_mask": "train_mask",
        "train_ready_tables.sample_weight": "sample_weight",
        "audits.train_ready": "train_ready_audit",
        "audits.spatial": "spatial_sequence_audit",
        "audits.image_context": "image_context_index_audit",
        "audits.interaction_context": "interaction_context_audit",
        "spatial_npz": "spatial_sequences",
    }
    return {
        audit_name: snapshot_hashes[snapshot_name]
        for audit_name, snapshot_name in bindings.items()
    }


def _snapshot_artifact_hashes() -> dict[str, str]:
    names = (
        "split_manifest",
        "image_window_context_manifest",
        "interaction_window_context_manifest",
        "image_frame_context_manifest",
        "tabular_X",
        "y_behavior",
        "train_mask",
        "sample_weight",
        "train_ready_audit",
        "spatial_sequence_audit",
        "image_context_index_audit",
        "interaction_context_audit",
        "spatial_sequences",
    )
    return {name: f"sha-{name}" for name in names}
