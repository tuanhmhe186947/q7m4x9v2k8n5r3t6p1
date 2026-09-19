from __future__ import annotations

from copy import deepcopy

from pig_behavior.classification_v2.contracts.technical_smoke_gate import (
    audit_technical_smoke_gate,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS


def _valid_payloads() -> dict[str, dict[str, object]]:
    """Return the smallest cross-stage PASS fixture."""

    common = {"errors": [], "warnings": []}
    return {
        "scope": {
            **common,
            "selected_rows": 10,
            "selected_source_counts": {
                "cvat_tracking_xml": 5,
                "legacy_recovered": 5,
            },
            "selected_behavior_counts": {
                behavior: 1 for behavior in VALID_BEHAVIORS
            },
        },
        "enhanced": {**common, "rows": 10},
        "sequence": {
            **common,
            "temporal_harmonization": {
                "rows": 10,
                "temporal_intervals": 2,
            },
            "sequence_windows": {"window_rows": 4},
        },
        "review_units": {
            **common,
            "rows": {"review_units": 2},
        },
        "temporal_evidence": {
            **common,
            "valid": True,
            "rows": {"intervals": 2},
            "keys": {
                "duplicate_temporal_unit_key": 0,
                "duplicate_window_id": 0,
            },
            "review_units": {"duplicate_review_unit_id": 0},
            "native_lengths": {"cvat_invalid": 0, "legacy_invalid": 0},
            "missing_temporal_evidence_from_whitelist": [],
            "review_evidence_in_model_whitelist": [],
            "evidence_column_counts": {"window": 73},
        },
        "train_ready": {
            **common,
            "rows": {"input": 4, "X": 4, "y": 4, "mask_true": 3, "mask_false": 1},
            "feature_selection": {
                "explicit_whitelist_used": True,
                "feature_whitelist_match": True,
                "forbidden_selected": [],
            },
        },
        "feature_semantics": {
            **common,
            "valid": True,
            "tabular_contract_match": True,
            "tabular_feature_count": 110,
            "spatial_model_input_array_count": 6,
            "forbidden_tabular_features": [],
            "undeclared_spatial_arrays": [],
            "spatial_model_input_role_errors": [],
        },
        "spatial_validation": {
            **common,
            "rows": 4,
            "train_mask_completeness": {
                "available": True,
                "trainable_rows_with_missing_slots": 0,
                "trainable_missing_slots": 0,
            },
        },
    }


def test_technical_smoke_gate_passes_without_authorizing_training() -> None:
    gate = audit_technical_smoke_gate(
        _valid_payloads(),
        repeatability={"all_match": True, "pair_count": 5},
        decision_files=[],
    )

    assert gate["technical_pass"] is True
    assert gate["status"] == "PASS_TECHNICAL_SMOKE_HUMAN_REVIEW_BLOCKED"
    assert gate["authorization"]["reviewed_dataset_authorized"] is False
    assert gate["authorization"]["full_oof_authorized"] is False
    assert len(gate["human_gate_blockers"]) == 2


def test_technical_smoke_gate_rejects_cross_stage_row_drift() -> None:
    payloads = deepcopy(_valid_payloads())
    payloads["enhanced"]["rows"] = 9

    gate = audit_technical_smoke_gate(
        payloads,
        repeatability={"all_match": True},
        decision_files=[],
    )

    assert gate["technical_pass"] is False
    assert any("frame_row_lineage_mismatch" in error for error in gate["errors"])


def test_technical_smoke_gate_rejects_synthetic_decision_files() -> None:
    gate = audit_technical_smoke_gate(
        _valid_payloads(),
        repeatability={"all_match": True},
        decision_files=["behavior_unit_review_decisions.csv"],
    )

    assert gate["technical_pass"] is False
    assert any(
        "unexpected_human_decision_files" in error for error in gate["errors"]
    )
