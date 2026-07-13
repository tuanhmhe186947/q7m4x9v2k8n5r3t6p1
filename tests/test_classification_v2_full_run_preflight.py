from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    full_run_config_fingerprint,
)
from pig_behavior.classification_v2.training.full_run_contract import (
    FULL_RUN_AUTHORIZATION_PURPOSE,
    FULL_RUN_AUTHORIZATION_SCHEMA_VERSION,
)
from pig_behavior.classification_v2.training.full_run_contract import (
    validate_full_run_authorization as _validate_full_run_authorization,
)
from pig_behavior.classification_v2.training.full_run_preflight import (
    _feature_whitelist_audit_errors,
    _runtime_match_errors,
    validate_preflight_for_execution,
)


def test_runtime_preflight_requires_exact_recommended_precision_and_batch() -> None:
    """A measured runtime recommendation cannot silently change a full run."""

    matching = FullMultimodalOofConfig(precision="amp", train_batch_size=128)
    changed = FullMultimodalOofConfig(precision="fp32", train_batch_size=64)
    runtime = {
        "valid": True,
        "errors": [],
        "recommended_runtime_config": {
            "precision": "amp",
            "train_batch_size": 128,
            "model_architecture_version": matching.model_architecture_version,
        },
    }
    assert _runtime_match_errors(matching, runtime) == []
    errors = _runtime_match_errors(changed, runtime)
    assert any("runtime_precision_mismatch" in error for error in errors)
    assert any("runtime_batch_size_mismatch" in error for error in errors)


def test_execution_gate_rejects_preflight_from_another_commit() -> None:
    """A clean preflight cannot authorize code after an unreviewed commit."""

    config = FullMultimodalOofConfig()
    preflight = {
        "valid": True,
        "errors": [],
        "config_sha256": full_run_config_fingerprint(config),
        "git_commit": "old",
        "git_dirty": False,
    }
    errors = validate_preflight_for_execution(
        config,
        preflight,
        git_state={"commit": "new", "dirty": False},
    )
    assert any("preflight_schema_version_mismatch" in error for error in errors)
    assert any("preflight_git_commit_mismatch" in error for error in errors)


def test_execution_gate_rejects_missing_lineage_files() -> None:
    """A saved valid flag cannot replace execution-time artifact checks."""

    config = FullMultimodalOofConfig()
    preflight = {
        "schema_version": "classification_v2_full_run_preflight_v2",
        "valid": True,
        "errors": [],
        "lineage_binding_valid": True,
        "lineage_training_authorized": True,
        "config_sha256": full_run_config_fingerprint(config),
        "git_commit": "same",
        "git_dirty": False,
        "snapshot_json": "missing-snapshot.json",
        "lineage_audit_json": "missing-lineage.json",
    }

    errors = validate_preflight_for_execution(
        config,
        preflight,
        git_state={"commit": "same", "dirty": False},
    )

    assert any("execution_missing_snapshot_json" in error for error in errors)


def test_full_preflight_requires_valid_feature_whitelist_audit() -> None:
    """Full OOF cannot fall back to unsafe feature selection."""

    valid = {
        "valid": True,
        "errors": [],
        "never_use_all_numeric_columns": True,
        "fail_closed_on_unknown_columns": True,
        "forbidden_probe_columns_not_blocked": [],
    }
    assert _feature_whitelist_audit_errors(valid) == []

    invalid = {
        "valid": False,
        "errors": ["forbidden_probe_columns_not_blocked=['manual_review_decision']"],
        "never_use_all_numeric_columns": False,
        "fail_closed_on_unknown_columns": False,
        "forbidden_probe_columns_not_blocked": ["manual_review_decision"],
    }
    errors = _feature_whitelist_audit_errors(invalid)
    assert any("invalid_feature_whitelist_audit" in error for error in errors)
    assert any("feature_whitelist_must_block_all_numeric_columns" in error for error in errors)
    assert any("feature_whitelist_probe_leakage" in error for error in errors)


def test_full_run_authorization_binds_preflight_hash_and_commit() -> None:
    """Full OOF needs explicit approval for the exact preflight artifact."""

    config = FullMultimodalOofConfig(run_mode="full")
    config_hash = full_run_config_fingerprint(config)
    preflight = {
        "config_sha256": config_hash,
        "git_commit": "abc123",
        "snapshot_id": "snapshot-123",
        "snapshot_file_sha256": "snapshot-file-123",
        "lineage_audit_sha256": "lineage-123",
        "lineage_binding_audit": {
            "expected_ordered_window_id_sha256": "ordered-window-123",
        },
    }
    authorization = {
        "schema_version": FULL_RUN_AUTHORIZATION_SCHEMA_VERSION,
        "authorized": True,
        "purpose": FULL_RUN_AUTHORIZATION_PURPOSE,
        "acknowledges_long_run": True,
        "acknowledges_no_q2_claim_until_verified": True,
        "reviewer": "pytest_reviewer",
        "reviewed_at": "2026-07-13T00:00:00+07:00",
        "preflight_config_sha256": config_hash,
        "git_commit": "abc123",
        "snapshot_id": "snapshot-123",
        "snapshot_file_sha256": "snapshot-file-123",
        "lineage_audit_sha256": "lineage-123",
        "ordered_window_id_sha256": "ordered-window-123",
    }
    assert _validate_full_run_authorization(config, preflight, authorization) == []

    bad_authorization = dict(authorization)
    bad_authorization.update(
        {
            "acknowledges_long_run": False,
            "preflight_config_sha256": "different",
            "git_commit": "stale",
        }
    )
    errors = _validate_full_run_authorization(config, preflight, bad_authorization)
    assert any("acknowledge_long_run" in error for error in errors)
    assert any("preflight_hash_mismatch" in error for error in errors)
    assert any("git_commit_mismatch" in error for error in errors)

    stale_lineage = dict(authorization)
    stale_lineage["snapshot_id"] = "another-snapshot"
    stale_lineage["lineage_audit_sha256"] = "another-lineage"
    errors = _validate_full_run_authorization(
        config,
        preflight,
        stale_lineage,
    )
    assert any("snapshot_id" in error for error in errors)
    assert any("lineage_audit_sha256" in error for error in errors)

    missing_provenance = dict(authorization)
    missing_provenance.pop("reviewer")
    missing_provenance.pop("reviewed_at")
    errors = _validate_full_run_authorization(
        config,
        preflight,
        missing_provenance,
    )
    assert any("requires_reviewer" in error for error in errors)
    assert any("requires_reviewed_at" in error for error in errors)
