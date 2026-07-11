from pig_behavior.classification_v2.training.full_multimodal_oof import FullMultimodalOofConfig
from pig_behavior.classification_v2.training.full_run_preflight import _runtime_match_errors


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
