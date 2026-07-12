from scripts.behavior_review_tools.classification_v2_run_full_multimodal_oof import (
    FULL_RUN_AUTHORIZATION_PURPOSE,
    _validate_full_run_authorization,
)

from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    full_run_config_fingerprint,
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
    assert any("preflight_git_commit_mismatch" in error for error in errors)


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
    }
    authorization = {
        "authorized": True,
        "purpose": FULL_RUN_AUTHORIZATION_PURPOSE,
        "acknowledges_long_run": True,
        "acknowledges_no_q2_claim_until_verified": True,
        "preflight_config_sha256": config_hash,
        "git_commit": "abc123",
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
