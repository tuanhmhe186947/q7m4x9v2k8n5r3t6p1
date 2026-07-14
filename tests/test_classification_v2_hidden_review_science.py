from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.review.hidden_review_science import (
    HiddenScientificPolicy,
    build_hidden_scientific_design,
    evaluate_hidden_scientific_gate,
)


def _policy(
    *,
    random_upper: float = 0.50,
    high_risk_upper: float = 0.50,
) -> HiddenScientificPolicy:
    return HiddenScientificPolicy(
        bootstrap_iterations=200,
        random_false_negative_upper_threshold=random_upper,
        high_risk_yield_upper_threshold=high_risk_upper,
        min_random_reviewed_items=4,
        min_random_native_clusters=4,
        min_random_recording_clusters=2,
        min_high_risk_reviewed_items=4,
        min_high_risk_native_clusters=4,
        min_high_risk_recording_clusters=2,
    )


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort in ("hidden_no_random_audit", "hidden_no_high_risk"):
        for recording in range(4):
            for item in range(3):
                item_id = f"{cohort}-{recording}-{item}"
                is_random = cohort == "hidden_no_random_audit"
                rows.append(
                    {
                        "hidden_review_item_id": item_id,
                        "hidden_before_review": "No",
                        "hidden_after_review": "",
                        "hidden_review_status": "pending",
                        "hidden_review_cohort": cohort,
                        "source_type": "cvat_tracking_xml",
                        "hidden_review_stratum_key": f"video=recording-{recording}",
                        "temporal_unit_key": f"native-{recording}-{item}-{cohort}",
                        "hidden_sampling_probability": 0.5 if is_random else 1.0,
                        "hidden_sampling_weight": 2.0 if is_random else 1.0,
                        "hidden_sampling_stratum": (
                            "source_type=cvat_tracking_xml|"
                            f"hidden_review_stratum_key=video-{recording}|"
                            "hidden_false_negative_risk_band=high"
                        ),
                        "hidden_false_negative_risk_reasons": "pair_iou",
                        "hidden_false_negative_risk_band": "high",
                    }
                )
    return pd.DataFrame(rows)


def _selection_contract() -> dict[str, object]:
    return {
        "target_independent": True,
        "risk_input_columns": [
            "nearest_pair_iou",
            "hidden_before_review",
        ],
        "stratum_columns": [
            "source_type",
            "hidden_review_stratum_key",
            "hidden_false_negative_risk_band",
        ],
        "errors": [],
    }


def _design(
    manifest: pd.DataFrame,
    policy: HiddenScientificPolicy,
) -> dict[str, object]:
    return build_hidden_scientific_design(
        manifest,
        manifest_sha256="manifest-hash",
        policy_payload=policy.to_payload(),
        policy_sha256="policy-hash",
        selection_contract=_selection_contract(),
    )


def _decisions(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hidden_review_item_id": manifest["hidden_review_item_id"],
            "hidden_before_review": manifest["hidden_before_review"],
            "hidden_after_review": "No",
            "hidden_review_status": "reviewed",
        }
    )


def test_hidden_scientific_gate_passes_complete_clustered_review() -> None:
    manifest = _manifest()
    policy = _policy()
    audit = evaluate_hidden_scientific_gate(
        manifest,
        _decisions(manifest),
        _design(manifest, policy),
        manifest_sha256="manifest-hash",
        design_sha256="design-hash",
    )

    assert audit["status"] == "PASS"
    assert audit["training_snapshot_allowed"] is True
    random = audit["random_hidden_no_prevalence"]
    assert random["weighted_rate"] == 0.0
    assert random["recording_cluster_count"] == 4
    assert random["native_cluster_count"] == 12
    assert random["conservative_interval"][1] > 0.0
    assert audit["high_risk_correction_yield"][
        "is_population_prevalence"
    ] is False


def test_hidden_scientific_gate_blocks_partial_decisions() -> None:
    manifest = _manifest()
    decisions = _decisions(manifest).iloc[:-1].copy()
    audit = evaluate_hidden_scientific_gate(
        manifest,
        decisions,
        _design(manifest, _policy()),
        manifest_sha256="manifest-hash",
        design_sha256="design-hash",
    )

    assert audit["status"] == "BLOCKED_INCOMPLETE_OR_INSUFFICIENT_REVIEW"
    assert audit["training_snapshot_allowed"] is False
    assert "missing_decision_items=1" in audit["blockers"]


def test_hidden_scientific_gate_enforces_predeclared_upper_bound() -> None:
    manifest = _manifest()
    policy = _policy(random_upper=0.01, high_risk_upper=1.0)
    audit = evaluate_hidden_scientific_gate(
        manifest,
        _decisions(manifest),
        _design(manifest, policy),
        manifest_sha256="manifest-hash",
        design_sha256="design-hash",
    )

    assert audit["status"] == "FAIL_QUALITY_THRESHOLD"
    assert any(
        failure.startswith("random_false_negative_upper_exceeds_threshold")
        for failure in audit["threshold_failures"]
    )


def test_random_prevalence_uses_inverse_probability_weights_only() -> None:
    manifest = _manifest()
    random_mask = manifest["hidden_review_cohort"].eq(
        "hidden_no_random_audit"
    )
    first_random = manifest.index[random_mask][0]
    manifest.loc[first_random, "hidden_sampling_probability"] = 0.25
    manifest.loc[first_random, "hidden_sampling_weight"] = 4.0
    decisions = _decisions(manifest)
    item_id = manifest.loc[first_random, "hidden_review_item_id"]
    decisions.loc[
        decisions["hidden_review_item_id"].eq(item_id),
        "hidden_after_review",
    ] = "Yes"
    audit = evaluate_hidden_scientific_gate(
        manifest,
        decisions,
        _design(manifest, _policy(random_upper=1.0, high_risk_upper=1.0)),
        manifest_sha256="manifest-hash",
        design_sha256="design-hash",
    )

    random = audit["random_hidden_no_prevalence"]
    assert random["weighted_rate"] > random["unweighted_rate"]
    assert audit["high_risk_correction_yield"]["weighted_rate"] == 0.0


def test_random_prevalence_rejects_invalid_sampling_weight() -> None:
    manifest = _manifest()
    random_mask = manifest["hidden_review_cohort"].eq(
        "hidden_no_random_audit"
    )
    manifest.loc[
        manifest.index[random_mask][0],
        "hidden_sampling_weight",
    ] = 99.0
    audit = evaluate_hidden_scientific_gate(
        manifest,
        _decisions(manifest),
        _design(manifest, _policy()),
        manifest_sha256="manifest-hash",
        design_sha256="design-hash",
    )

    assert audit["status"] == "FAIL_CONTRACT"
    assert any(
        "sampling_probability_weight_mismatch" in error
        for error in audit["errors"]
    )


def test_hidden_scientific_design_rejects_target_informed_strata() -> None:
    manifest = _manifest()
    manifest.loc[0, "hidden_sampling_stratum"] += "|behavior=fight"

    with pytest.raises(ValueError, match="target-derived"):
        _design(manifest, _policy())


def test_hidden_scientific_gate_rejects_manifest_hash_drift() -> None:
    manifest = _manifest()
    audit = evaluate_hidden_scientific_gate(
        manifest,
        _decisions(manifest),
        _design(manifest, _policy()),
        manifest_sha256="different-hash",
        design_sha256="design-hash",
    )

    assert audit["status"] == "FAIL_CONTRACT"
    assert "hidden_manifest_hash_drift" in audit["errors"]


def test_full_hidden_design_rejects_insufficient_planned_support() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="insufficient planned support"):
        build_hidden_scientific_design(
            manifest,
            manifest_sha256="manifest-hash",
            policy_payload=HiddenScientificPolicy().to_payload(),
            policy_sha256="policy-hash",
            selection_contract=_selection_contract(),
        )


def test_smoke_hidden_design_can_report_but_never_authorize() -> None:
    manifest = _manifest()
    policy = HiddenScientificPolicy()
    design = build_hidden_scientific_design(
        manifest,
        manifest_sha256="manifest-hash",
        policy_payload=policy.to_payload(),
        policy_sha256="policy-hash",
        selection_contract=_selection_contract(),
        require_final_support=False,
    )
    audit = evaluate_hidden_scientific_gate(
        manifest,
        _decisions(manifest),
        design,
        manifest_sha256="manifest-hash",
        design_sha256="design-hash",
    )

    assert design["design_scope"] == "smoke"
    assert design["planned_support_meets_final_gate"] is False
    assert audit["status"] == "BLOCKED_INCOMPLETE_OR_INSUFFICIENT_REVIEW"
    assert "hidden_scientific_design_scope_not_full" in audit["blockers"]
