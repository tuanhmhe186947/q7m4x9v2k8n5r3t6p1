from __future__ import annotations

from pathlib import Path

import pytest

from pig_behavior.classification_v2.training.legacy_development_l5_temporal_ladder import (
    TemporalLadderConfig,
)
from pig_behavior.classification_v2.training.legacy_development_l7_synthetic import (
    run_l7_synthetic_gate,
)


def _config(tmp_path: Path) -> TemporalLadderConfig:
    payload = {
        "model": {
            "temporal_encoder_name": "masked_mean",
            "hidden_dim": 128,
            "dropout": 0.1,
            "transformer_layers": 1,
            "transformer_heads": 4,
        },
        "optimization": {
            "seed": 20260714,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
        },
    }
    return TemporalLadderConfig(
        path=tmp_path / "synthetic.json",
        payload=payload,
        repo_root=tmp_path,
    )


def test_l7_synthetic_gate_passes_all_policies(tmp_path: Path) -> None:
    result = run_l7_synthetic_gate(_config(tmp_path), tiny_steps=30)

    assert result["valid"] is True
    assert result["errors"] == []
    assert set(result["policies"]) == {
        "event_balanced_ce",
        "effective_number_ce",
        "balanced_softmax",
    }
    for audit in result["policies"].values():
        assert audit["one_batch"]["valid"] is True
        assert audit["tiny_overfit"]["valid"] is True
        assert audit["resume"]["valid"] is True
        assert audit["resume"]["next_loss_abs_delta"] == 0.0
        assert audit["resume"]["next_logit_max_abs_delta"] == 0.0


def test_l7_synthetic_gate_rejects_nonpositive_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tiny steps must be positive"):
        run_l7_synthetic_gate(_config(tmp_path), tiny_steps=0)
