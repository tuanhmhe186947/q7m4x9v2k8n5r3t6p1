"""Forensic tests for the superseded 2x2 grounding authority."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

DATE = "20260729"
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_ROOT = ROOT / "docs" / "tracking" / "scientific_grounding"


def _json(name: str) -> dict[str, object]:
    return json.loads((AUTHORITY_ROOT / name).read_text(encoding="utf-8"))


def _csv(name: str) -> list[dict[str, str]]:
    with (AUTHORITY_ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scientific_grounding_authority_hashes_and_counts() -> None:
    authority = _json(f"TRACKING_SCIENTIFIC_GROUNDING_AUTHORITY_{DATE}.json")
    methods = _csv(f"TRACKING_METHOD_IDENTITY_REGISTRY_{DATE}.csv")
    parameters = _csv(f"TRACKING_PARAMETER_SCIENTIFIC_SUPPORT_{DATE}.csv")
    metrics = _csv(f"TRACKING_METRIC_PROVENANCE_LEDGER_{DATE}.csv")
    claims = _csv(f"TRACKING_SCIENTIFIC_CLAIM_AUTHORIZATION_{DATE}.csv")
    contradictions = _csv(f"TRACKING_AUTHORITY_CONTRADICTIONS_{DATE}.csv")

    method_summary = authority["method_identity_summary"]
    parameter_summary = authority["parameter_summary"]
    metric_summary = authority["metric_summary"]
    claim_summary = authority["claim_summary"]
    contradiction_summary = authority["contradiction_summary"]

    assert method_summary["method_identities_discovered"] == len(methods)
    assert parameter_summary["total_parameters"] == len(parameters)
    assert metric_summary["total_metric_values_audited"] == len(metrics)
    assert contradiction_summary["found"] == len(contradictions)
    assert sum(claim_summary.values()) == len(claims)

    for filename, expected_hash in authority["output_hashes"].items():
        assert _sha256(AUTHORITY_ROOT / filename) == expected_hash


def test_method_identities_and_evidence_classes_are_explicit() -> None:
    methods = _csv(f"TRACKING_METHOD_IDENTITY_REGISTRY_{DATE}.csv")
    method_ids = [row["canonical_method_id"] for row in methods]
    allowed_evidence = {
        "MEASURED_FROM_FROZEN_ARTIFACT",
        "DERIVED_FROM_MEASURED_VALUES",
        "BOUND_BY_EFFECTIVE_CONFIG",
        "BOUND_BY_SOURCE_CODE",
        "REPORTED_BY_HISTORICAL_SUMMARY_ONLY",
        "INFERRED_WITH_SUPPORT",
        "UNRESOLVED",
        "UNSUPPORTED",
        "CONTRADICTED",
    }

    assert len(method_ids) == len(set(method_ids))
    assert {row["evidence_class"] for row in methods} <= allowed_evidence
    assert "HISTORICAL_H5B_H4_20260719" in method_ids
    assert "B1_CURRENT_STANDARD_V2_20260728" in method_ids
    assert "B1_EXECUTABLE_REPRODUCTION_FAILED_20260729" in method_ids


def test_parameter_grounding_is_complete_and_conservative() -> None:
    config = _csv(f"TRACKING_EFFECTIVE_CONFIGURATION_LEDGER_{DATE}.csv")
    support = _csv(f"TRACKING_PARAMETER_SCIENTIFIC_SUPPORT_{DATE}.csv")
    authority = _json(f"TRACKING_SCIENTIFIC_GROUNDING_AUTHORITY_{DATE}.json")
    active_methods = {
        "B0_CURRENT_STANDARD_V2_20260728",
        "B1_CURRENT_STANDARD_V2_20260728",
        "R0_CURRENT_STANDARD_V2_20260728",
        "R1_CURRENT_STANDARD_V2_20260728",
    }

    assert len(config) == len(support)
    for method_id in active_methods:
        rows = [row for row in support if row["method_id"] == method_id]
        assert len(rows) == 308
        assert all(row["frozen_status"] == "YES" for row in rows)
        assert all(row["per_video_parameter"] == "NO" for row in rows)

    gaps = [
        row
        for row in support
        if row["selection_basis"] == "UNKNOWN_SELECTION_BASIS"
    ]
    summary = authority["parameter_summary"]
    assert summary["unknown_selection_basis_parameters"] == len(gaps)
    assert summary["unsupported_parameters"] == len(
        [row for row in support if row["support_class"] == "UNSUPPORTED"]
    )
    assert gaps
    assert all(
        row["rationale_status"] == "UNSUPPORTED_OR_UNRECOVERED"
        for row in gaps
    )


def test_metric_prediction_and_evaluator_bindings() -> None:
    rows = _csv(f"TRACKING_METRIC_PROVENANCE_LEDGER_{DATE}.csv")
    measured = [
        row
        for row in rows
        if row["evidence_class"] == "MEASURED_FROM_FROZEN_ARTIFACT"
    ]
    current_standard_v2 = [
        row
        for row in measured
        if row["method_id"].endswith("STANDARD_V2_20260728")
    ]

    assert measured
    assert current_standard_v2
    assert all(row["prediction_hash"] for row in measured)
    assert all(row["artifact_hash"] for row in measured)
    assert all(
        row["evaluator_contract"] == "TRACKING_EVALUATOR_STANDARD_V2"
        for row in current_standard_v2
    )
    assert all(
        row["include_hidden_policy"] == "true"
        for row in current_standard_v2
    )


def test_claims_contradictions_and_stop_state_are_consistent() -> None:
    claims = _csv(f"TRACKING_SCIENTIFIC_CLAIM_AUTHORIZATION_{DATE}.csv")
    contradictions = _csv(f"TRACKING_AUTHORITY_CONTRADICTIONS_{DATE}.csv")
    stop = _json(f"TRACKING_SCIENTIFIC_STOP_STATE_{DATE}.json")
    two_by_two = _json(f"TRACKING_2X2_CLAIM_RECONCILIATION_{DATE}.json")

    claim_by_id = {row["claim_id"]: row for row in claims}
    assert claim_by_id["C03"]["status"] == "CONTRADICTED"
    assert claim_by_id["C04"]["status"] == "NOT_AUTHORIZED"
    assert claim_by_id["C05"]["status"] == "NOT_AUTHORIZED"
    assert claim_by_id["C11"]["status"] == "UNRESOLVED"
    assert claim_by_id["C12"]["status"] == "NOT_AUTHORIZED"
    assert any(
        row["resolution_status"] == "UNRESOLVED" for row in contradictions
    )

    assert stop["current_2x2_internal_authority"] == (
        "VALID_FOR_CURRENT_FROZEN_ARTIFACTS"
    )
    assert stop["current_b1_equals_historical_h5b_h4"] is False
    assert stop["historical_h5b_h4_executable_authority"] == "NOT_ESTABLISHED"
    assert stop["historical_h5b_h4_irrecoverable"] == "NOT_YET_PROVEN"
    assert stop["unseen_method_freeze_status"] == "SUSPENDED"
    assert stop["unseen_access_authorized"] is False
    assert two_by_two["current_2x2_represents_true_historical_best"] is False

    assert all(value == 0 for value in stop["execution_counts"].values())
