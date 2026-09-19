import pandas as pd

from pig_behavior.classification_v2.review.behavior_review_science import (
    BehaviorScientificPolicy,
    build_behavior_scientific_design,
    evaluate_behavior_scientific_gate,
)


def _policy(
    *,
    random_upper: float = 0.80,
) -> BehaviorScientificPolicy:
    return BehaviorScientificPolicy(
        bootstrap_iterations=200,
        random_intervention_upper_threshold=random_upper,
        min_random_reviewed_items=4,
        min_random_native_clusters=4,
        min_random_video_clusters=4,
        min_random_source_clusters=1,
        min_high_risk_reviewed_items=4,
        min_high_risk_native_clusters=4,
        min_high_risk_video_clusters=4,
        min_high_risk_source_clusters=1,
    )


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort in (
        "behavior_random_audit",
        "behavior_high_risk",
        "behavior_clean_control",
    ):
        for video_index in range(4):
            for item_index in range(3):
                random = cohort == "behavior_random_audit"
                probability = 0.25 if random and item_index == 0 else 0.5
                rows.append(
                    {
                        "review_unit_id": (
                            f"{cohort}-{video_index}-{item_index}"
                        ),
                        "temporal_unit_key": (
                            f"native-{cohort}-{video_index}-{item_index}"
                        ),
                        "behavior_review_cohort": cohort,
                        "include_in_review": True,
                        "source_type": "cvat_tracking_xml",
                        "video_key": f"video-{video_index}",
                        "behavior_label": "move",
                        "behavior_sampling_probability": (
                            probability if random else 1.0
                        ),
                        "behavior_sampling_weight": (
                            1.0 / probability if random else 1.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _selection_contract() -> dict[str, object]:
    return {
        "target_conditioned_selection": True,
        "random_residual_estimand": (
            "human_intervention_rate_in_post_high_risk_residual_pool"
        ),
        "errors": [],
    }


def _design(
    manifest: pd.DataFrame,
    policy: BehaviorScientificPolicy,
) -> dict[str, object]:
    return build_behavior_scientific_design(
        manifest,
        manifest_sha256="manifest-hash",
        policy_payload=policy.to_payload(),
        policy_sha256="policy-hash",
        selection_contract=_selection_contract(),
    )


def _decisions(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_unit_id": manifest["review_unit_id"],
            "manual_review_decision": "accept",
            "manual_corrected_behavior": "",
        }
    )


def test_behavior_scientific_gate_passes_and_separates_estimands() -> None:
    manifest = _manifest()
    decisions = _decisions(manifest)
    first_random = manifest.index[
        manifest["behavior_review_cohort"].eq("behavior_random_audit")
    ][0]
    decisions.loc[first_random, "manual_review_decision"] = "corrected"
    audit = evaluate_behavior_scientific_gate(
        manifest,
        decisions,
        _design(manifest, _policy()),
        manifest_sha256="manifest-hash",
        design_sha256="design-hash",
    )

    assert audit["status"] == "PASS"
    assert audit["training_snapshot_allowed"] is True
    random = audit["random_residual_intervention_rate"]
    assert random["weighted_rate"] > random["unweighted_rate"]
    assert random["is_population_prevalence"] is False
    assert audit["high_risk_intervention_yield"][
        "is_population_prevalence"
    ] is False
    assert audit["clean_control_audit_only"]["final_estimate"] is False
    assert any("do not certify" in warning for warning in audit["warnings"])


def test_behavior_scientific_gate_blocks_missing_decision() -> None:
    manifest = _manifest()
    decisions = _decisions(manifest).iloc[:-1].copy()
    audit = evaluate_behavior_scientific_gate(
        manifest,
        decisions,
        _design(manifest, _policy()),
        manifest_sha256="manifest-hash",
        design_sha256="design-hash",
    )

    assert audit["status"] == "BLOCKED_INCOMPLETE_OR_INSUFFICIENT_REVIEW"
    assert audit["training_snapshot_allowed"] is False
    assert "missing_behavior_decision_items=1" in audit["coverage"]["blockers"]


def test_behavior_scientific_gate_enforces_random_upper_bound() -> None:
    manifest = _manifest()
    decisions = _decisions(manifest)
    first_random = manifest.index[
        manifest["behavior_review_cohort"].eq("behavior_random_audit")
    ][0]
    decisions.loc[first_random, "manual_review_decision"] = "corrected"
    audit = evaluate_behavior_scientific_gate(
        manifest,
        decisions,
        _design(manifest, _policy(random_upper=0.01)),
        manifest_sha256="manifest-hash",
        design_sha256="design-hash",
    )

    assert audit["status"] == "FAIL_QUALITY_THRESHOLD"
    assert any(
        failure.startswith("random_intervention_upper_exceeds_threshold")
        for failure in audit["threshold_failures"]
    )


def test_behavior_scientific_design_requires_target_conditioned_contract() -> None:
    manifest = _manifest()
    invalid_contract = {
        "target_conditioned_selection": False,
        "random_residual_estimand": "residual",
        "errors": [],
    }

    try:
        build_behavior_scientific_design(
            manifest,
            manifest_sha256="manifest-hash",
            policy_payload=_policy().to_payload(),
            policy_sha256="policy-hash",
            selection_contract=invalid_contract,
        )
    except ValueError as error:
        assert "invalid selection contract" in str(error)
    else:
        raise AssertionError("invalid behavior selection contract was accepted")
