"""Evaluation-population utilities for cross-length comparisons.

Two populations must accompany every length comparison:

``ALL_ELIGIBLE``
    all native units valid for that temporal length independently.

``COMMON_MATCHED_COHORT``
    the exact intersection of native units valid for every compared length.

A longer view may only be called better when its gain survives on
``COMMON_MATCHED_COHORT``; an ``ALL_ELIGIBLE``-only difference can be a pure
support artifact of longer views having fewer eligible windows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pig_behavior.classification_v2.temporal_views.registry import temporal_view_spec

MATCHED_COHORT_SCHEMA_VERSION = "classification_v2.matched_cohort.v1"

EVALUATION_POPULATIONS: tuple[str, ...] = ("ALL_ELIGIBLE", "COMMON_MATCHED_COHORT")


class EvaluationPopulationError(ValueError):
    """Raised when an eligibility table cannot support a valid comparison."""


@dataclass(frozen=True, slots=True)
class ViewEligibility:
    """Eligible native units for one view plus per-unit exclusion reasons."""

    view_name: str
    eligible_native_unit_ids: frozenset[str]
    exclusion_reasons: Mapping[str, str]

    def to_payload(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for reason in self.exclusion_reasons.values():
            counts[str(reason)] = counts.get(str(reason), 0) + 1
        return {
            "view_name": self.view_name,
            "eligible_native_units": len(self.eligible_native_unit_ids),
            "excluded_native_units": len(self.exclusion_reasons),
            "exclusion_reason_counts": dict(sorted(counts.items())),
        }


def build_view_eligibility(
    view_name: str,
    *,
    eligible_native_unit_ids: Sequence[str],
    exclusion_reasons: Mapping[str, str] | None = None,
) -> ViewEligibility:
    """Validate and wrap one view's eligibility set."""

    temporal_view_spec(view_name)
    ids = [str(value).strip() for value in eligible_native_unit_ids]
    if any(not value for value in ids):
        raise EvaluationPopulationError("eligible native unit ids must not be blank")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise EvaluationPopulationError(
            f"duplicate eligible native unit ids for {view_name}: {duplicates[:8]}"
        )
    reasons = {str(key): str(value) for key, value in (exclusion_reasons or {}).items()}
    overlap = sorted(set(ids) & set(reasons))
    if overlap:
        raise EvaluationPopulationError(
            f"native units cannot be both eligible and excluded for {view_name}: "
            f"{overlap[:8]}"
        )
    return ViewEligibility(
        view_name=view_name,
        eligible_native_unit_ids=frozenset(ids),
        exclusion_reasons=reasons,
    )


def all_eligible(eligibility: ViewEligibility) -> frozenset[str]:
    """Return the ``ALL_ELIGIBLE`` population for one view."""

    return eligibility.eligible_native_unit_ids


def common_matched_cohort(
    eligibilities: Sequence[ViewEligibility],
) -> frozenset[str]:
    """Return the exact intersection of eligible native units across views."""

    if not eligibilities:
        raise EvaluationPopulationError(
            "a common matched cohort needs at least one view eligibility set"
        )
    cohort = set(eligibilities[0].eligible_native_unit_ids)
    for item in eligibilities[1:]:
        cohort &= set(item.eligible_native_unit_ids)
    return frozenset(cohort)


def evaluation_population_report(
    eligibilities: Sequence[ViewEligibility],
) -> dict[str, Any]:
    """Return the per-view eligible/exclusion report plus the matched cohort."""

    cohort = common_matched_cohort(eligibilities)
    return {
        "schema_version": MATCHED_COHORT_SCHEMA_VERSION,
        "populations": list(EVALUATION_POPULATIONS),
        "views": [item.to_payload() for item in eligibilities],
        "common_matched_cohort_size": len(cohort),
        "common_matched_cohort_is_exact_intersection": True,
        "per_view_all_eligible": {
            item.view_name: len(item.eligible_native_unit_ids)
            for item in eligibilities
        },
        "per_view_dropped_relative_to_cohort": {
            item.view_name: len(item.eligible_native_unit_ids) - len(cohort)
            for item in eligibilities
        },
    }


@dataclass(frozen=True, slots=True)
class LengthComparison:
    """One measured comparison between a reference view and a candidate view."""

    reference_view: str
    candidate_view: str
    all_eligible_gain: float
    common_matched_cohort_gain: float | None
    minimum_practical_gain: float = 0.02

    def to_payload(self) -> dict[str, Any]:
        return {
            "reference_view": self.reference_view,
            "candidate_view": self.candidate_view,
            "all_eligible_gain": self.all_eligible_gain,
            "common_matched_cohort_gain": self.common_matched_cohort_gain,
            "minimum_practical_gain": self.minimum_practical_gain,
        }


def length_conclusion_guard(comparison: LengthComparison) -> dict[str, Any]:
    """Decide whether a longer view may be called successful.

    The guard refuses to mark success from ``ALL_ELIGIBLE`` alone, and refuses
    when the matched-cohort gain is missing.
    """

    reference = temporal_view_spec(comparison.reference_view)
    candidate = temporal_view_spec(comparison.candidate_view)
    if reference.family != candidate.family:
        raise EvaluationPopulationError(
            "target-view length and causal-history length are different "
            "scientific claims and must not be pooled: "
            f"{reference.name} ({reference.family}) vs "
            f"{candidate.name} ({candidate.family})"
        )
    matched = comparison.common_matched_cohort_gain
    reasons: list[str] = []
    if matched is None:
        reasons.append(
            "no COMMON_MATCHED_COHORT gain was supplied; an ALL_ELIGIBLE-only "
            "difference may be a support artifact"
        )
    elif matched < comparison.minimum_practical_gain:
        reasons.append(
            f"COMMON_MATCHED_COHORT gain {matched} is below the preregistered "
            f"minimum practical gain {comparison.minimum_practical_gain}"
        )
    successful = not reasons
    return {
        "schema_version": MATCHED_COHORT_SCHEMA_VERSION,
        **comparison.to_payload(),
        "family": reference.family,
        "successful": successful,
        "blocking_reasons": reasons,
        "decided_on_population": "COMMON_MATCHED_COHORT",
    }


__all__ = [
    "EVALUATION_POPULATIONS",
    "MATCHED_COHORT_SCHEMA_VERSION",
    "EvaluationPopulationError",
    "LengthComparison",
    "ViewEligibility",
    "all_eligible",
    "build_view_eligibility",
    "common_matched_cohort",
    "evaluation_population_report",
    "length_conclusion_guard",
]
