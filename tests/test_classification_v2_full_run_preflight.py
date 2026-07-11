from pig_behavior.classification_v2.training.full_multimodal_oof import (
    FullMultimodalOofConfig,
    full_run_config_fingerprint,
)
from pig_behavior.classification_v2.training.full_run_preflight import (
    _runtime_match_errors,
    validate_preflight_for_execution,
)


def test_runtime_preflight_requires_exact_recommended_precision_and_batch() -> None:
    """A measured runtime recommendation cannot be silently changed for the full run."""

    runtime = {
        "valid": True,
        "errors": [],
        "recommended_runtime_config": {"precision": "amp", "train_batch_size": 128},
    }
    matching = FullMultimodalOofConfig(precision="amp", train_batch_size=128)
    changed = FullMultimodalOofConfig(precision="fp32", train_batch_size=64)

    assert _runtime_match_errors(matching, runtime) == []
    errors = _runtime_match_errors(changed, runtime)
    assert any("runtime_precision_mismatch" in error for error in errors)
    assert any("runtime_batch_size_mismatch" in error for error in errors)


def test_execution_gate_rejects_preflight_from_another_commit() -> None:
    """A clean preflight cannot authorize code from a later unreviewed commit."""

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
