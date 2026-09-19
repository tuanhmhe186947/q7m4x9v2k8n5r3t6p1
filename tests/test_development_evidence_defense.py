"""Contracts for the frozen four-method development evidence package."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pig_behavior.tracking.method_registry import (  # noqa: E402
    ACTIVE_SCIENTIFIC_METHOD_IDS,
    PROVENANCE_ALIASES,
    SCIENTIFIC_METHOD_REGISTRY,
)

EVIDENCE = ROOT / "docs" / "tracking" / "development_evidence_defense"
METHODS = (
    "bytetrack_raw",
    "hybrid_bytetrack",
    "realtime_fast",
    "rf_hybrid",
)


def _json(name: str) -> dict[str, object]:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def _csv(name: str) -> list[dict[str, str]]:
    with (EVIDENCE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_package_contains_every_required_scientific_artifact() -> None:
    required = {
        "DEVELOPMENT_EVIDENCE_INPUT_AUTHORITY_20260730.json",
        "DEVELOPMENT_VIDEO_SESSION_MAP_20260730.csv",
        "DEVELOPMENT_INDEPENDENCE_AUDIT_20260730.json",
        "DEVELOPMENT_PER_VIDEO_TRACKING_METRICS_20260730.csv",
        "DEVELOPMENT_PAIRED_COMPARISONS_20260730.csv",
        "DEVELOPMENT_WIN_TIE_LOSS_SUMMARY_20260730.csv",
        "DEVELOPMENT_CLUSTER_BOOTSTRAP_SUMMARY_20260730.csv",
        "DEVELOPMENT_LEAVE_ONE_CLUSTER_OUT_20260730.csv",
        "DEVELOPMENT_FRAGMENTATION_AND_COMPLETENESS_AUDIT_20260730.csv",
        "DEVELOPMENT_HYBRID_IDSW_ZERO_DEFENSE_20260730.json",
        "DEVELOPMENT_HIDDEN_SENSITIVITY_RESULTS_20260730.csv",
        "DEVELOPMENT_HIDDEN_SENSITIVITY_PER_VIDEO_20260730.csv",
        "DEVELOPMENT_GT_ERROR_AUDIT_ITEMS_20260730.csv",
        "DEVELOPMENT_IDENTITY_DIAGNOSTIC_METRIC_SPECIFICATION_20260730.md",
        "DEVELOPMENT_COMPLETE_METHOD_FAIRNESS_MATRIX_20260730.csv",
        "DEVELOPMENT_CLAIM_LIMITATION_MATRIX_20260730.csv",
        "DEVELOPMENT_REVIEWER_OBJECTION_RESPONSE_MATRIX_20260730.csv",
        "DEVELOPMENT_BASELINE_ADEQUACY_ASSESSMENT_20260730.json",
        "DEVELOPMENT_RUNTIME_CLAIM_AUDIT_20260730.json",
        "PAPER_TABLE_CANONICAL_DEVELOPMENT_RESULTS_20260730.csv",
        "PAPER_TABLE_PER_VIDEO_PAIRED_DIFFERENCES_20260730.csv",
        "PAPER_TABLE_HIDDEN_SENSITIVITY_20260730.csv",
        "PAPER_TABLE_HYBRID_REPRODUCIBILITY_20260730.csv",
        "PAPER_TABLE_CLAIMS_AND_LIMITATIONS_20260730.csv",
        "FIGURE_DATA_PER_VIDEO_HOTA_DELTAS_20260730.csv",
        "FIGURE_DATA_PER_VIDEO_IDF1_DELTAS_20260730.csv",
        "FIGURE_DATA_WRONG_ID_EXPOSURE_BY_VIDEO_20260730.csv",
        "FIGURE_DATA_FRAGMENTATION_VS_IDENTITY_QUALITY_20260730.csv",
        "FIGURE_DATA_LEAVE_ONE_CLUSTER_OUT_20260730.csv",
        "TRACKING_DEVELOPMENT_EVIDENCE_AND_REVIEWER_DEFENSE_REPORT_20260730.md",
        "DEVELOPMENT_EVIDENCE_ARTIFACT_MANIFEST_20260730.json",
        "DEVELOPMENT_EVIDENCE_FINAL_DECISION_20260730.json",
    }
    assert required.issubset({path.name for path in EVIDENCE.iterdir()})


def test_registry_remains_exactly_four_methods() -> None:
    assert ACTIVE_SCIENTIFIC_METHOD_IDS == METHODS
    assert tuple(SCIENTIFIC_METHOD_REGISTRY) == METHODS
    assert "rf_hybrid_v2" not in SCIENTIFIC_METHOD_REGISTRY
    assert "standardized_b1" not in SCIENTIFIC_METHOD_REGISTRY
    assert PROVENANCE_ALIASES["standardized_b1"]["kind"] == "FORENSIC_ONLY"


def test_per_video_population_and_schema_are_complete() -> None:
    rows = _csv("DEVELOPMENT_PER_VIDEO_TRACKING_METRICS_20260730.csv")
    required = {
        "method_id",
        "video_id",
        "session_key",
        "frame_count",
        "duration_seconds",
        "FPS",
        "HOTA",
        "DetA",
        "AssA",
        "LocA",
        "IDF1",
        "IDP",
        "IDR",
        "IDSW_STANDARD",
        "FP",
        "FN",
        "fragments",
        "wrong_identity_frames",
        "wrong_identity_seconds",
        "recovered_identity_episodes",
        "terminal_identity_episodes",
        "persistent_pairwise_swaps",
        "evaluation_status",
    }
    assert len(rows) == 13 * 4
    assert set(rows[0]) == required
    assert {row["method_id"] for row in rows} == set(METHODS)
    assert sum(int(row["frame_count"]) for row in rows if row["method_id"] == METHODS[0]) == 23400
    assert all(row["evaluation_status"] == "COMPARABLE_STANDARD_V2_PRIMARY" for row in rows)


def test_paired_analysis_uses_video_clusters_not_frames() -> None:
    paired = _csv("DEVELOPMENT_PAIRED_COMPARISONS_20260730.csv")
    summary = _csv("DEVELOPMENT_WIN_TIE_LOSS_SUMMARY_20260730.csv")
    bootstrap = _csv("DEVELOPMENT_CLUSTER_BOOTSTRAP_SUMMARY_20260730.csv")
    assert {row["comparison_id"] for row in paired} == {"C1", "C2", "C3"}
    assert {row["metric"] for row in summary} >= {"HOTA", "AssA", "IDF1"}
    assert all(row["resampling_unit"] == "VIDEO" for row in bootstrap)
    assert all(row["resample_count"] == "10000" for row in bootstrap)
    independence = _json("DEVELOPMENT_INDEPENDENCE_AUDIT_20260730.json")
    assert independence["FRAME_LEVEL_INFERENCE_ALLOWED"] == "NO"
    assert independence["PROVEN_SESSION_COUNT"] == 0
    assert not (EVIDENCE / "DEVELOPMENT_PER_SESSION_TRACKING_METRICS_20260730.csv").exists()


def test_golden_examples_cover_frozen_identity_contract() -> None:
    payload = _json("DEVELOPMENT_IDENTITY_DIAGNOSTIC_GOLDEN_EXAMPLES_20260730.json")
    example_ids = {row["id"] for row in payload["examples"]}
    assert example_ids == {
        "correct_continuity",
        "one_frame_switch",
        "recovered_switch",
        "terminal_wrong_owner",
        "fragmentation_without_switch",
        "hidden_interval",
        "prediction_gap",
        "global_identity_permutation",
        "simultaneous_pairwise_swap",
    }
    assert payload["contract_ids"] == [
        "TRACKING_EVALUATOR_STANDARD_V2",
        "TRACKING_MATCHING_STANDARD_V2",
        "IDENTITY_ERROR_EPISODES_V2",
    ]


def test_hidden_sensitivity_has_global_and_per_video_evidence() -> None:
    global_rows = _csv("DEVELOPMENT_HIDDEN_SENSITIVITY_RESULTS_20260730.csv")
    video_rows = _csv("DEVELOPMENT_HIDDEN_SENSITIVITY_PER_VIDEO_20260730.csv")
    assert len(global_rows) == 8
    assert len(video_rows) == 13 * 4
    assert {row["evaluation_scope"] for row in global_rows} == {
        "PRIMARY_INCLUDE_HIDDEN",
        "VISIBLE_ONLY_SENSITIVITY",
    }
    assert all(
        row["interpretation"] == "SENSITIVITY_ONLY_NOT_HUMAN_HIDDEN_VALIDATION"
        for row in video_rows
    )
    canonical = json.loads(
        (
            ROOT / "docs/tracking/method_standardization/"
            "CANONICAL_TRACKING_DEVELOPMENT_RESULTS_20260730.json"
        ).read_text(encoding="utf-8")
    )
    by_method = {row["method_id"]: row for row in canonical["active_method_results"]}
    for row in global_rows:
        authority = by_method[row["method_id"]]
        assert row["prediction_hash"] == authority["prediction_hash"]
        assert row["GT_hash"] == authority["GT_hash"]
        assert row["evaluator_hash"] == authority["evaluator_code_hash"]


def test_final_decision_preserves_all_nonexecution_guards() -> None:
    decision = _json("DEVELOPMENT_EVIDENCE_FINAL_DECISION_20260730.json")
    assert decision["ACTIVE_METHODS"] == list(METHODS)
    assert decision["PREDICTION_FILES_CHANGED"] == 0
    assert decision["GT_FILES_CHANGED"] == 0
    assert decision["DETECTOR_RUNS"] == 0
    assert decision["TRACKER_RUNS"] == 0
    assert decision["UNSEEN_FILES_ACCESSED"] == 0
    assert decision["MP4_FILES_CREATED"] == 0
    assert decision["GENERALIZATION_CLAIM_MADE"] == "NO"
    assert decision["HYBRID_EXACT_REPRODUCIBILITY"] == "NO"
    assert decision["HYBRID_METRIC_LEVEL_NEAR_PARITY"] == "YES"


def test_audit_pack_conserves_hybrid_wrong_rows_without_fake_review() -> None:
    rows = _csv("DEVELOPMENT_GT_ERROR_AUDIT_ITEMS_20260730.csv")
    hybrid = [row for row in rows if "ALL_24_HYBRID_WRONG_ID_FRAMES" in row["selection_reason"]]
    assert len(hybrid) == 24
    assert all(row["method_id"] == "hybrid_bytetrack" for row in hybrid)
    assert all(row["visual_review_status"] == "NOT_REVIEWED" for row in rows)


def test_report_and_reviewer_matrix_cover_all_required_concerns() -> None:
    report = (
        EVIDENCE / "TRACKING_DEVELOPMENT_EVIDENCE_AND_REVIEWER_DEFENSE_REPORT_20260730.md"
    ).read_text(encoding="utf-8")
    for number in range(1, 25):
        assert f"## {number}." in report
    objections = _csv("DEVELOPMENT_REVIEWER_OBJECTION_RESPONSE_MATRIX_20260730.csv")
    assert {row["objection_id"] for row in objections} == {
        "DEVELOPMENT_OVERFITTING",
        "PSEUDOREPLICATION",
        "SESSION_NONINDEPENDENCE",
        "AGGREGATE_DOMINATION",
        "IDSW_GAMING_BY_FRAGMENTATION",
        "DETECTION_ASSOCIATION_TRADEOFF",
        "METHOD_FAIRNESS",
        "HISTORICAL_RUNTIME_INCOMPLETENESS",
        "HIDDEN_GT_UNCERTAINTY",
        "CUSTOM_METRIC_VALIDITY",
        "REALTIME_CLAIM",
        "BASELINE_INSUFFICIENCY",
        "NO_UNSEEN_GENERALIZATION",
    }


def test_unavailable_values_are_never_encoded_as_numeric_zero() -> None:
    rows = _csv("DEVELOPMENT_FRAGMENTATION_AND_COMPLETENESS_AUDIT_20260730.csv")
    unavailable_fields = (
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "identity_continuity_through_occlusion_episodes",
    )
    for row in rows:
        for field in unavailable_fields:
            assert row[field] == "NOT_AVAILABLE"


def test_artifact_manifest_includes_final_decision_and_no_mutable_media() -> None:
    manifest = _json("DEVELOPMENT_EVIDENCE_ARTIFACT_MANIFEST_20260730.json")
    paths = {row["relative_path"] for row in manifest["artifacts"]}
    assert "DEVELOPMENT_EVIDENCE_FINAL_DECISION_20260730.json" in paths
    assert manifest["recursive_mp4_count"] == 0
    assert not list(EVIDENCE.rglob("*.xml"))
    assert not list(EVIDENCE.rglob("*.mp4"))
