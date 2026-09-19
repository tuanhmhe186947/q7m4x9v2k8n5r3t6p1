from __future__ import annotations

import json
from pathlib import Path

from pig_behavior.classification_v2.evaluation.legacy_c6_temporal_freeze import (
    BASE_MODE_IDS,
    FAMILY_REQUIREMENTS,
    _family_decisions,
    freeze_c6_base_from_full_development_decision,
)


def _comparison(passes: bool) -> dict[str, object]:
    return {"passes_control": passes}


def test_temporal_family_requires_every_declared_control() -> None:
    comparisons = {
        pair_id: _comparison(True)
        for pair_ids in FAMILY_REQUIREMENTS.values()
        for pair_id in pair_ids
    }
    comparisons["transformer_timing_constant"] = _comparison(False)

    decisions = _family_decisions(comparisons)

    assert decisions["TCN128"]["screening_action"] == "RETEST"
    assert decisions["TR128_REAL_DELTA"]["screening_action"] == "DROP"
    assert decisions["TR128_REAL_DELTA"]["failed_pairs"] == [
        "transformer_timing_constant"
    ]


def test_temporal_family_fails_closed_on_missing_comparison() -> None:
    decisions = _family_decisions({})

    assert all(
        not decision["passes_all_required_controls"]
        for decision in decisions.values()
    )
    assert decisions["TCN128"]["failed_pairs"] == [
        "tcn_capacity",
        "tcn_order",
    ]


def test_freezes_measured_a128_from_full_development_decision(
    tmp_path: Path,
) -> None:
    metrics = {
        mode_id: {
            "macro_f1_global_10_class": 0.2,
            "nll": 1.0,
        }
        for mode_id in BASE_MODE_IDS
    }
    metrics["A128"] = {
        "macro_f1_global_10_class": 0.4,
        "nll": 0.9,
    }
    decision = {
        "schema_version": (
            "classification_v2.legacy_development.temporal_base_decision.v1"
        ),
        "status": "PASS_LEGACY_TEMPORAL_BASE_PAIRED_DECISION",
        "lineage_scope": "legacy-only-unreviewed-development",
        "human_review_complete": False,
        "legacy_sets_final_full_data_base": False,
        "common_native_universe": {
            "native_units": 241,
            "video_clusters": 32,
            "outer_holdout_rows": 0,
        },
        "mode_metrics": metrics,
        "paired_comparisons": {
            pair_id: {
                "transfer_decision": {"screening_action": "CARRY"}
            }
            for pair_id in (
                "content_weighting",
                "operational_attention_vs_single",
            )
        },
        "valid": True,
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    _, freeze = freeze_c6_base_from_full_development_decision(
        decision_path,
        tmp_path / "freeze.json",
        project_root=tmp_path,
    )

    assert freeze["valid"] is True
    assert freeze["selected_base_mode"] == "A128"
    assert freeze["mode_ranking"][0] == "A128"
    assert (
        freeze["selected_base_is_carried_prior_not_tested_in_this_matrix"]
        is False
    )
