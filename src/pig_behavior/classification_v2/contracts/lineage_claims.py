"""Strict optional claim metadata for classification_v2 artifacts.

Claims are audit metadata, never model features. When present, the scope and
human-review flag must travel as one uniform pair through every derived table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LINEAGE_CLAIM_COLUMNS: tuple[str, str] = (
    "lineage_scope",
    "human_review_complete",
)


@dataclass(frozen=True, slots=True)
class LineageClaims:
    """One uniform artifact-level claim pair."""

    lineage_scope: str
    human_review_complete: bool

    def as_dict(self) -> dict[str, str | bool]:
        """Return JSON- and dataframe-ready claim values."""

        return {
            "lineage_scope": self.lineage_scope,
            "human_review_complete": self.human_review_complete,
        }


def configured_lineage_claims(
    lineage_scope: object | None,
    human_review_complete: object | None,
    *,
    artifact_name: str,
) -> LineageClaims | None:
    """Validate an optional paired claim supplied by configuration."""

    scope_missing = lineage_scope is None
    review_missing = human_review_complete is None
    if scope_missing != review_missing:
        raise ValueError(
            f"{artifact_name} requires lineage_scope and "
            "human_review_complete together"
        )
    if scope_missing:
        return None
    if not isinstance(lineage_scope, str) or not lineage_scope.strip():
        raise ValueError(f"{artifact_name} lineage_scope must be nonblank")
    if not isinstance(human_review_complete, (bool, np.bool_)):
        raise ValueError(
            f"{artifact_name} human_review_complete must be a boolean"
        )
    reviewed = bool(human_review_complete)
    return LineageClaims(
        lineage_scope=lineage_scope.strip(),
        human_review_complete=reviewed,
    )


def resolve_optional_lineage_claims(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
) -> LineageClaims | None:
    """Resolve one strict claim pair from a dataframe, or return ``None``."""

    present = set(LINEAGE_CLAIM_COLUMNS).intersection(frame.columns)
    if not present:
        return None
    if present != set(LINEAGE_CLAIM_COLUMNS):
        missing = sorted(set(LINEAGE_CLAIM_COLUMNS).difference(present))
        raise ValueError(
            f"{artifact_name} has partial lineage claims; missing={missing}"
        )

    scopes = frame["lineage_scope"].map(_clean_scope_value)
    if scopes.eq("").any():
        raise ValueError(f"{artifact_name} has blank lineage_scope values")
    unique_scopes = sorted(set(scopes))
    if len(unique_scopes) != 1:
        raise ValueError(
            f"{artifact_name} has mixed lineage_scope values={unique_scopes}"
        )

    reviewed_values: list[bool] = []
    invalid_indices: list[str] = []
    for index, value in frame["human_review_complete"].items():
        try:
            reviewed_values.append(
                _strict_bool_value(value, artifact_name=artifact_name)
            )
        except ValueError:
            invalid_indices.append(str(index))
    if invalid_indices:
        raise ValueError(
            f"{artifact_name} has invalid human_review_complete values; "
            f"sample_indices={invalid_indices[:10]}"
        )
    unique_reviewed = set(reviewed_values)
    if len(unique_reviewed) != 1:
        raise ValueError(
            f"{artifact_name} has mixed human_review_complete values="
            f"{sorted(unique_reviewed)}"
        )
    return LineageClaims(
        lineage_scope=unique_scopes[0],
        human_review_complete=next(iter(unique_reviewed)),
    )


def attach_optional_lineage_claims(
    frame: pd.DataFrame,
    claims: LineageClaims | None,
) -> pd.DataFrame:
    """Return a copy with the uniform claim pair attached when configured."""

    out = frame.copy()
    if claims is None:
        return out
    out["lineage_scope"] = claims.lineage_scope
    out["human_review_complete"] = claims.human_review_complete
    return out


def require_lineage_claims_preserved(
    source: pd.DataFrame,
    derived: pd.DataFrame,
    *,
    source_name: str,
    derived_name: str,
) -> LineageClaims | None:
    """Fail when a derived table drops or changes an input claim pair."""

    source_claims = resolve_optional_lineage_claims(
        source,
        artifact_name=source_name,
    )
    derived_claims = resolve_optional_lineage_claims(
        derived,
        artifact_name=derived_name,
    )
    if source_claims != derived_claims:
        raise ValueError(
            f"lineage claims changed from {source_name} to {derived_name}: "
            f"source={source_claims}, derived={derived_claims}"
        )
    return source_claims


def add_optional_lineage_claims_to_audit(
    audit: dict[str, Any],
    frame: pd.DataFrame,
    *,
    artifact_name: str,
) -> dict[str, Any]:
    """Attach valid claims to an audit or append a fail-closed audit error."""

    errors = audit.setdefault("errors", [])
    if not isinstance(errors, list):
        raise TypeError("audit errors must be a list")
    try:
        claims = resolve_optional_lineage_claims(
            frame,
            artifact_name=artifact_name,
        )
    except ValueError as exc:
        errors.append(f"lineage_claim_contract={exc}")
        return audit
    if claims is not None:
        audit.update(claims.as_dict())
    return audit


def _clean_scope_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _strict_bool_value(value: object, *, artifact_name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(
        f"{artifact_name} human_review_complete must be strict true/false"
    )
