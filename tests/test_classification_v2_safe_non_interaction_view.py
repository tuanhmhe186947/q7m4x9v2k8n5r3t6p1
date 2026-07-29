from __future__ import annotations

import pandas as pd
import pytest

from pig_behavior.classification_v2.review.safe_non_interaction_view import (
    SafeNonInteractionViewError,
    audit_safe_non_interaction_view,
    build_safe_non_interaction_view,
)


def _row(
    review_unit_id: str,
    behavior: str,
    template: str,
    *,
    reasons: str = "motion_reason",
    predicates: str = "motion_contradiction",
) -> dict[str, object]:
    return {
        "review_unit_id": review_unit_id,
        "behavior_label": behavior,
        "review_template": template,
        "candidate_tier": "TIER_2_HIGH_RISK",
        "include_in_review": True,
        "review_reason_codes": reasons,
        "review_reason": reasons,
        "review_selection_predicates": predicates,
        "review_evidence_reason_auto": reasons,
        "interval_review_reason": "",
        "selection_predicate_version": "selection.v2",
        "selection_config_hash": "a" * 64,
        "review_predicate_interaction_contradiction": False,
        "review_predicate_partner_context_insufficient": False,
        "payload": review_unit_id,
    }


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("safe-motion", "move", "motion"),
            _row("safe-posture", "lying", "posture"),
            _row(
                "fight",
                "fight",
                "interaction",
                reasons="fight_without_persistent_contact_or_aggression",
                predicates="interaction_contradiction",
            ),
            _row(
                "social",
                "social-nose",
                "interaction",
                reasons="social_evidence_unavailable",
                predicates="partner_context_insufficient",
            ),
        ]
    )


def test_safe_view_preserves_exact_rows_and_order() -> None:
    candidates = _candidates()
    result = build_safe_non_interaction_view(
        candidates,
        producer_sha="b" * 40,
        input_sha256="c" * 64,
    )
    assert result.view["review_unit_id"].tolist() == [
        "safe-motion",
        "safe-posture",
    ]
    pd.testing.assert_frame_equal(
        result.view,
        candidates.iloc[:2].reset_index(drop=True),
    )
    assert result.audit["new_keys_added"] == 0
    assert result.audit["excluded_interaction_affected_count"] == 2


def test_explicit_interaction_dependency_excludes_noninteraction_label() -> None:
    row = _row(
        "cross-template",
        "move",
        "interaction",
        predicates="interaction_contradiction",
    )
    result = build_safe_non_interaction_view(
        pd.DataFrame([row]),
        producer_sha="b" * 40,
        input_sha256="c" * 64,
    )
    assert result.view.empty
    assert result.audit["unknown_dependency_count"] == 1


def test_missing_dependency_field_fails_closed_as_unknown() -> None:
    candidates = _candidates().drop(
        columns=["review_predicate_partner_context_insufficient"]
    )
    result = build_safe_non_interaction_view(
        candidates,
        producer_sha="b" * 40,
        input_sha256="c" * 64,
    )
    assert result.view.empty
    assert result.audit["unknown_dependency_count"] == len(candidates)


def test_checker_rejects_new_key() -> None:
    candidates = _candidates()
    view = candidates.iloc[:2].copy()
    view.loc[0, "review_unit_id"] = "not-current"
    audit = audit_safe_non_interaction_view(
        candidates,
        view,
        expected_candidate_sha256="d" * 64,
        actual_candidate_sha256="d" * 64,
    )
    assert not audit["valid"]
    assert audit["new_keys_added"] == 1


def test_checker_rejects_changed_source_value() -> None:
    candidates = _candidates()
    view = candidates.iloc[:2].copy()
    view.loc[0, "payload"] = "changed"
    audit = audit_safe_non_interaction_view(
        candidates,
        view,
        expected_candidate_sha256="d" * 64,
        actual_candidate_sha256="d" * 64,
    )
    assert not audit["valid"]
    assert any(
        error.startswith("safe_view_source_rows_changed")
        for error in audit["errors"]
    )


def test_checker_rejects_input_hash_drift() -> None:
    candidates = _candidates()
    audit = audit_safe_non_interaction_view(
        candidates,
        candidates.iloc[:2].copy(),
        expected_candidate_sha256="d" * 64,
        actual_candidate_sha256="e" * 64,
    )
    assert not audit["valid"]
    assert "candidate_manifest_hash_mismatch" in audit["errors"]


def test_duplicate_candidate_key_rejected() -> None:
    candidates = pd.concat([_candidates(), _candidates().iloc[[0]]])
    with pytest.raises(SafeNonInteractionViewError, match="duplicate"):
        build_safe_non_interaction_view(
            candidates,
            producer_sha="b" * 40,
            input_sha256="c" * 64,
        )


def test_view_is_deterministic() -> None:
    candidates = _candidates()
    first = build_safe_non_interaction_view(
        candidates,
        producer_sha="b" * 40,
        input_sha256="c" * 64,
    )
    second = build_safe_non_interaction_view(
        candidates,
        producer_sha="b" * 40,
        input_sha256="c" * 64,
    )
    pd.testing.assert_frame_equal(first.view, second.view)
    assert first.audit == second.audit
