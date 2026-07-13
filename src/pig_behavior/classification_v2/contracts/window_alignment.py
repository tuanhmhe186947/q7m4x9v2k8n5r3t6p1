"""Ordered window-key contracts for positional multimodal artifacts."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def audit_ordered_window_ids(
    reference_name: str,
    reference: pd.Series,
    candidates: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    """Audit key uniqueness, membership, and order across positional tables."""

    reference_ids = _clean_window_ids(reference)
    errors = _window_key_errors(reference_ids, reference_name)
    comparisons: dict[str, dict[str, Any]] = {}
    reference_set = set(reference_ids)
    for name, values in (candidates or {}).items():
        candidate_ids = _clean_window_ids(values)
        candidate_errors = _window_key_errors(candidate_ids, name)
        missing = sorted(reference_set.difference(candidate_ids))
        extra = sorted(set(candidate_ids).difference(reference_set))
        order_mismatch = _ordered_mismatch_count(
            reference_ids,
            candidate_ids,
        )
        if missing:
            candidate_errors.append(f"missing_window_ids={len(missing)}")
        if extra:
            candidate_errors.append(f"extra_window_ids={len(extra)}")
        if order_mismatch:
            candidate_errors.append(
                f"window_order_mismatch_rows={order_mismatch}"
            )
        comparisons[name] = {
            "rows": int(len(candidate_ids)),
            "ordered_window_id_sha256": ordered_window_id_sha256(
                candidate_ids
            ),
            "missing_count": int(len(missing)),
            "extra_count": int(len(extra)),
            "order_mismatch_rows": int(order_mismatch),
            "errors": candidate_errors,
        }
        errors.extend(f"{name}:{error}" for error in candidate_errors)

    return {
        "reference": reference_name,
        "reference_rows": int(len(reference_ids)),
        "reference_ordered_window_id_sha256": ordered_window_id_sha256(
            reference_ids
        ),
        "comparisons": comparisons,
        "errors": errors,
        "valid": not errors,
    }


def require_ordered_window_ids(
    reference_name: str,
    reference: pd.Series,
    candidates: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    """Return audit evidence or reject positional multimodal misalignment."""

    audit = audit_ordered_window_ids(
        reference_name,
        reference,
        candidates,
    )
    if audit["errors"]:
        raise ValueError(
            f"ordered window alignment failed: {audit['errors']}"
        )
    return audit


def ordered_window_id_sha256(values: pd.Series) -> str:
    """Hash normalized ordered keys without exposing them as model features."""

    normalized = _clean_window_ids(values)
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _clean_window_ids(values: pd.Series) -> pd.Series:
    """Normalize keys without making missing values appear valid."""

    return values.fillna("").astype(str).str.strip().reset_index(drop=True)


def _window_key_errors(values: pd.Series, name: str) -> list[str]:
    """Return blank and duplicate violations for one positional artifact."""

    errors: list[str] = []
    blank = int(values.eq("").sum())
    duplicate = int(values.duplicated(keep=False).sum())
    if blank:
        errors.append(f"blank_{name}_window_ids={blank}")
    if duplicate:
        errors.append(f"duplicate_{name}_window_id_rows={duplicate}")
    return errors


def _ordered_mismatch_count(
    reference: pd.Series,
    candidate: pd.Series,
) -> int:
    """Count positional mismatches, including rows absent from either side."""

    size = max(len(reference), len(candidate))
    left = reference.reindex(range(size), fill_value="")
    right = candidate.reindex(range(size), fill_value="")
    return int(left.ne(right).sum())
