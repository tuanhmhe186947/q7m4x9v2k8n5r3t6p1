"""Authority-only standardization contracts for tracking methods."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from pig_behavior.tracking.method_registry import (
    ACTIVE_SCIENTIFIC_METHOD_IDS,
    PROVENANCE_ALIASES,
    SCIENTIFIC_METHOD_REGISTRY,
    validate_method_registry,
)

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_ROOT = ROOT / "docs" / "tracking" / "method_standardization"
EXPECTED_METHODS = (
    "bytetrack_raw",
    "hybrid_bytetrack",
    "realtime_fast",
    "rf_hybrid",
)


def _json(name: str) -> dict[str, object]:
    return json.loads((AUTHORITY_ROOT / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registry_has_exactly_four_canonical_methods() -> None:
    validate_method_registry()
    assert ACTIVE_SCIENTIFIC_METHOD_IDS == EXPECTED_METHODS
    assert tuple(SCIENTIFIC_METHOD_REGISTRY) == EXPECTED_METHODS
    assert set(PROVENANCE_ALIASES).isdisjoint(SCIENTIFIC_METHOD_REGISTRY)


def test_registry_authority_metadata_is_complete() -> None:
    required = (
        "canonical_version",
        "scientific_role",
        "scientific_status",
        "prediction_authority_type",
        "prediction_authority_path",
        "prediction_authority_hash",
        "execution_authority_status",
        "detector_contract",
        "tracker_contract",
        "state_lifecycle",
        "future_frame_policy",
        "causal",
        "development_evaluation_eligible",
        "runtime_benchmark_eligible",
        "deployment_eligible",
        "unseen_execution_eligible",
        "recommended_runtime_use",
        "known_limitations",
        "provenance_authority_path",
    )
    for contract in SCIENTIFIC_METHOD_REGISTRY.values():
        for field in required:
            assert getattr(contract, field) not in (None, "")


def test_prediction_authority_hashes_match_canonical_authority() -> None:
    authority = _json("CANONICAL_TRACKING_METHOD_AUTHORITY_20260730.json")
    registry_path = ROOT / "src" / "pig_behavior" / "tracking" / "method_registry.py"
    assert authority["authority_bindings"]["method_registry_sha256"] == _sha256(
        registry_path
    )
    methods = authority["methods"]
    assert isinstance(methods, dict)
    for method_id, contract in SCIENTIFIC_METHOD_REGISTRY.items():
        method = methods[method_id]
        assert method["prediction_authority_hash"] == (
            contract.prediction_authority_hash
        )


def test_execution_runtime_deployment_and_unseen_rules() -> None:
    raw = SCIENTIFIC_METHOD_REGISTRY["bytetrack_raw"]
    hybrid = SCIENTIFIC_METHOD_REGISTRY["hybrid_bytetrack"]
    realtime = SCIENTIFIC_METHOD_REGISTRY["realtime_fast"]
    rf_hybrid = SCIENTIFIC_METHOD_REGISTRY["rf_hybrid"]

    assert raw.runtime_benchmark_eligible is True
    assert realtime.runtime_benchmark_eligible is True
    assert realtime.deployment_eligible is True
    assert realtime.unseen_execution_eligible == "PENDING_SEPARATE_PREFLIGHT"
    for method in (hybrid, rf_hybrid):
        assert method.runtime_benchmark_eligible is False
        assert method.deployment_eligible is False
        assert method.unseen_execution_eligible == "NO"


def test_historical_hybrid_and_v2_disclosures_are_explicit() -> None:
    authority = _json("CANONICAL_TRACKING_METHOD_AUTHORITY_20260730.json")
    hybrid = authority["methods"]["hybrid_bytetrack"]
    rejected = authority["rejected_candidates"]["rf_hybrid_v2"]
    assert "EXACT_NUMERICAL_RUNTIME_NOT_RECOVERED" in (
        hybrid["execution_authority_status"]
    )
    assert hybrid["runtime_benchmark_eligible"] is False
    assert rejected["decision"] == "FAIL_RF_HYBRID_V2_MIXED_RESULT"
    assert rejected["active"] is False


def test_standard_v2_contract_and_missing_metric_encoding() -> None:
    results = _json("CANONICAL_TRACKING_DEVELOPMENT_RESULTS_20260730.json")
    contract = results["evaluation_contract"]
    assert contract["tracking_evaluator"] == "TRACKING_EVALUATOR_STANDARD_V2"
    assert contract["identity_episode_evaluator"] == (
        "IDENTITY_ERROR_EPISODES_V2"
    )
    assert contract["include_hidden"] is True
    assert contract["HOTA_threshold_count"] == 19
    assert contract["missing_metric_encoding"] == "NOT_AVAILABLE"
    provenance = results["provenance_only_results"]
    assert provenance[0]["other_metrics"] == "NOT_AVAILABLE"
    assert provenance[1]["all_metrics"] == "NOT_AVAILABLE"


def test_paper_tables_do_not_create_a_fifth_method() -> None:
    with (AUTHORITY_ROOT / "TRACKING_PAPER_TABLE_A_EXECUTABLE_METHODS.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        table_a = list(csv.DictReader(handle))
    with (
        AUTHORITY_ROOT / "TRACKING_PAPER_TABLE_B_DEVELOPMENT_REFERENCES.csv"
    ).open(encoding="utf-8", newline="") as handle:
        table_b = list(csv.DictReader(handle))

    assert [row["method"] for row in table_a] == [
        "bytetrack_raw",
        "realtime_fast",
    ]
    assert [row["method_or_candidate"] for row in table_b] == [
        "hybrid_bytetrack",
        "rf_hybrid_v1",
        "rf_hybrid_v2_candidate",
    ]
    assert "rf_hybrid_v2" not in SCIENTIFIC_METHOD_REGISTRY


def test_no_standardized_b1_or_symmetric_2x2_is_active() -> None:
    authority = _json("CANONICAL_TRACKING_METHOD_AUTHORITY_20260730.json")
    guards = authority["guards"]
    assert guards["standardized_b1_active"] is False
    assert guards["symmetric_2x2_active"] is False
    assert PROVENANCE_ALIASES["standardized_b1"]["kind"] == "FORENSIC_ONLY"
