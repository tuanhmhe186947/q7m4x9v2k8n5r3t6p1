"""Canonical model-side temporal-view registry and contracts.

This package holds the *model-facing* view names used by configs, manifests and
metric tables (``T6_TARGET_CONTIGUOUS`` and friends). It is distinct from
``features.temporal_views``, which builds the data-side fixed-six slot manifests
during the production lineage.

Nothing here builds a production dataset; it defines the semantics that a
builder must satisfy and validates fixtures against them.
"""

from pig_behavior.classification_v2.temporal_views.builder_contract import (
    TemporalWindowSample,
    WindowContractError,
    deterministic_window_id,
    expected_frame_indices,
    validate_window,
)
from pig_behavior.classification_v2.temporal_views.matched_cohort import (
    EVALUATION_POPULATIONS,
    all_eligible,
    common_matched_cohort,
    evaluation_population_report,
    length_conclusion_guard,
)
from pig_behavior.classification_v2.temporal_views.registry import (
    CAUSAL_HISTORY_VIEWS,
    TARGET_CONTIGUOUS_VIEWS,
    TEMPORAL_VIEW_NAMES,
    TemporalViewSpec,
    temporal_view_registry_contract,
    temporal_view_spec,
)

__all__ = [
    "CAUSAL_HISTORY_VIEWS",
    "EVALUATION_POPULATIONS",
    "TARGET_CONTIGUOUS_VIEWS",
    "TEMPORAL_VIEW_NAMES",
    "TemporalViewSpec",
    "TemporalWindowSample",
    "WindowContractError",
    "all_eligible",
    "common_matched_cohort",
    "deterministic_window_id",
    "evaluation_population_report",
    "expected_frame_indices",
    "length_conclusion_guard",
    "temporal_view_registry_contract",
    "temporal_view_spec",
    "validate_window",
]
