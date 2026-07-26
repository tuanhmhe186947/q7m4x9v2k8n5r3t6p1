"""Deterministic split authority for the balanced causal main model.

Splits are computed from metadata tables only. No feature tensor, no label
distribution and no production run root is consulted.
"""

from pig_behavior.classification_v2.splits.date_grouped_split import (
    CALENDAR_GROUP_COUNT_EXPECTED,
    OUTER_FOLD_DATE_TOKENS,
    OUTER_PROTOCOL_NAME,
    SplitAuthorityConfig,
    SplitAuthorityError,
    SplitAuthorityTables,
    build_split_authority,
    canonical_calendar_date,
    per_calendar_date_statistics,
)
from pig_behavior.classification_v2.splits.split_audit import (
    SPLIT_AUDIT_CHECKS,
    audit_split_authority,
    require_split_authority,
)

__all__ = [
    "CALENDAR_GROUP_COUNT_EXPECTED",
    "OUTER_FOLD_DATE_TOKENS",
    "OUTER_PROTOCOL_NAME",
    "SPLIT_AUDIT_CHECKS",
    "SplitAuthorityConfig",
    "SplitAuthorityError",
    "SplitAuthorityTables",
    "audit_split_authority",
    "build_split_authority",
    "canonical_calendar_date",
    "per_calendar_date_statistics",
    "require_split_authority",
]
