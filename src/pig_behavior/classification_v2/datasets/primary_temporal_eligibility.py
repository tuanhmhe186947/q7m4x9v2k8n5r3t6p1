"""Derive inner-only primary S1 eligibility from current window authorities.

The authoritative effective-window index preserves every exact-view row and
tensor position.  This module does not rebuild those rows or choose a primary
native owner.  It instead marks a row eligible only when all of its constituent
native units support one reviewed, resolved, eligible behavior target.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

PRIMARY_S1_ALLOWED_ROLES = frozenset({"train", "validation"})
_WINDOW_COLUMNS = (
    "window_id",
    "window_row_index",
    "view_type",
    "window_length_frames",
    "source_type",
    "temporal_unit_keys_json",
    "behavior_window_label",
    "window_valid_for_main_train",
    "window_sample_weight",
)
_ROLE_COLUMNS = (
    "outer_fold_id",
    "temporal_unit_key",
    "role",
    "behavior_label",
    "native_unit_valid_for_main_train",
    "native_unit_valid_for_main_eval",
)


class PrimaryTemporalEligibilityError(ValueError):
    """Raised when an S1 primary-eligibility or inner-role contract fails."""


@dataclass(frozen=True, slots=True)
class PrimaryTemporalEligibilityResult:
    """Row-preserving primary S1 eligibility overlay and its audit."""

    windows: pd.DataFrame
    audit: dict[str, Any]


def load_primary_s1_temporal_eligibility(
    effective_window_index_csv: Path,
    native_role_authority_csv: Path,
    *,
    fold_id: str = "FOLD_3",
    requested_roles: Iterable[str] = PRIMARY_S1_ALLOWED_ROLES,
    expected_window_index_sha256: str | None = None,
    expected_native_role_authority_sha256: str | None = None,
) -> PrimaryTemporalEligibilityResult:
    """Load metadata only after rejecting every forbidden S1 role request.

    The role request is validated before either metadata file is opened.  Model
    example payloads, feature tensors, labels from a held-out role, metrics, and
    prediction roots are never opened by this derived-eligibility producer.
    """

    allowed_roles = _validated_inner_roles(requested_roles)
    _verify_sha256(
        effective_window_index_csv,
        expected_window_index_sha256,
        "effective window index",
    )
    _verify_sha256(
        native_role_authority_csv,
        expected_native_role_authority_sha256,
        "native role authority",
    )
    windows = pd.read_csv(
        effective_window_index_csv,
        usecols=list(_WINDOW_COLUMNS),
        low_memory=False,
    )
    roles = pd.read_csv(
        native_role_authority_csv,
        usecols=list(_ROLE_COLUMNS),
        low_memory=False,
    )
    return build_primary_s1_temporal_eligibility(
        windows,
        roles,
        fold_id=fold_id,
        requested_roles=allowed_roles,
    )


def build_primary_s1_temporal_eligibility(
    windows: pd.DataFrame,
    native_roles: pd.DataFrame,
    *,
    fold_id: str = "FOLD_3",
    requested_roles: Iterable[str] = PRIMARY_S1_ALLOWED_ROLES,
) -> PrimaryTemporalEligibilityResult:
    """Create a row-preserving stable-single-label eligibility overlay."""

    allowed_roles = _validated_inner_roles(requested_roles)
    _require_columns(windows, _WINDOW_COLUMNS, "effective window index")
    _require_columns(native_roles, _ROLE_COLUMNS, "native role authority")
    if (
        windows["window_id"].isna().any()
        or windows["window_id"].astype(str).str.strip().eq("").any()
    ):
        raise PrimaryTemporalEligibilityError("effective window index has blank window_id")
    if windows["window_id"].astype(str).duplicated().any():
        raise PrimaryTemporalEligibilityError("effective window index has duplicate window_id")

    roles = native_roles.loc[
        native_roles["outer_fold_id"].astype(str).eq(str(fold_id))
    ].copy()
    if roles.empty:
        raise PrimaryTemporalEligibilityError(f"native role authority lacks {fold_id}")
    if (
        roles["temporal_unit_key"].isna().any()
        or roles["temporal_unit_key"].astype(str).str.strip().eq("").any()
    ):
        raise PrimaryTemporalEligibilityError("native role authority has blank temporal_unit_key")
    if roles["temporal_unit_key"].astype(str).duplicated().any():
        raise PrimaryTemporalEligibilityError(
            f"native role authority has duplicate {fold_id} temporal_unit_key"
        )

    records = roles.set_index("temporal_unit_key", drop=False).to_dict("index")
    output = windows.copy()
    derived_rows: list[dict[str, Any]] = []
    for row in output.itertuples(index=False):
        derived_rows.append(
            _classify_window(row, records=records, allowed_roles=allowed_roles)
        )
    derived = pd.DataFrame(derived_rows, index=output.index)
    output = pd.concat([output, derived], axis=1)
    _assert_row_preservation(windows, output)
    audit = _build_audit(output, fold_id=fold_id, requested_roles=allowed_roles)
    if audit["errors"]:
        raise PrimaryTemporalEligibilityError("; ".join(audit["errors"]))
    return PrimaryTemporalEligibilityResult(windows=output, audit=audit)


def build_primary_s1_validation_native_population(
    eligibility_windows: pd.DataFrame,
    native_roles: pd.DataFrame,
    *,
    fold_id: str = "FOLD_3",
    role: str = "validation",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the exact eligible native population for primary validation."""

    _validated_inner_roles([role])
    _require_columns(
        eligibility_windows,
        (
            "temporal_unit_keys_json",
            "primary_s1_role",
            "primary_s1_eligible",
        ),
        "primary eligibility overlay",
    )
    _require_columns(native_roles, _ROLE_COLUMNS, "native role authority")
    selected = eligibility_windows.loc[
        _strict_bool(eligibility_windows["primary_s1_eligible"], "primary_s1_eligible")
        & eligibility_windows["primary_s1_role"].astype(str).eq(role)
    ].copy()
    if selected.empty:
        raise PrimaryTemporalEligibilityError(
            f"no eligible primary S1 {role} windows"
        )
    roles = native_roles.loc[
        native_roles["outer_fold_id"].astype(str).eq(str(fold_id))
        & native_roles["role"].astype(str).eq(role)
    ].copy()
    if roles["temporal_unit_key"].astype(str).duplicated().any():
        raise PrimaryTemporalEligibilityError("duplicate validation native role key")
    role_lookup = roles.set_index("temporal_unit_key", drop=False)
    native_keys = sorted(
        {
            key
            for raw_keys in selected["temporal_unit_keys_json"]
            for key in _parse_native_keys(raw_keys)
        }
    )
    missing = sorted(set(native_keys).difference(role_lookup.index.astype(str)))
    if missing:
        raise PrimaryTemporalEligibilityError(
            f"eligible validation windows reference non-validation natives={len(missing)}"
        )
    native = role_lookup.loc[native_keys].copy()
    valid = _strict_bool(
        native["native_unit_valid_for_main_eval"],
        "native_unit_valid_for_main_eval",
    )
    if not valid.all():
        raise PrimaryTemporalEligibilityError(
            "primary validation population contains non-evaluable native units"
        )
    labels = native["behavior_label"].fillna("").astype(str).str.strip()
    if (~labels.isin(VALID_BEHAVIORS)).any():
        raise PrimaryTemporalEligibilityError(
            "primary validation population has unresolved native behavior labels"
        )
    result = (
        native[["temporal_unit_key", "behavior_label"]]
        .reset_index(drop=True)
        .sort_values("temporal_unit_key", kind="mergesort")
        .reset_index(drop=True)
    )
    return result, {
        "schema_version": "classification_v2.s1_primary_validation_population.v1",
        "fold_id": str(fold_id),
        "role": role,
        "expected_native_units": int(len(result)),
        "ordered_native_unit_key_sha256": _ordered_hash(
            result["temporal_unit_key"].astype(str).tolist()
        ),
        "errors": [],
        "valid": True,
    }


def build_primary_s1_view_role_overlay(
    eligibility_windows: pd.DataFrame,
    native_roles: pd.DataFrame,
    *,
    fold_id: str = "FOLD_3",
    view_type: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive view-local native eligibility for fold-local event weighting.

    The returned rows retain the immutable fold roles.  Only the two
    view-specific eligibility flags are overlaid so that a native excluded from
    this view before optimization is not silently expected during weighting or
    evaluation.
    """

    _require_columns(
        eligibility_windows,
        (
            "view_type",
            "temporal_unit_keys_json",
            "primary_s1_role",
            "primary_s1_eligible",
        ),
        "primary eligibility overlay",
    )
    _require_columns(native_roles, _ROLE_COLUMNS, "native role authority")
    if not str(view_type).strip():
        raise PrimaryTemporalEligibilityError("primary S1 view type is blank")
    roles = native_roles.loc[
        native_roles["outer_fold_id"].astype(str).eq(str(fold_id))
    ].copy()
    if roles.empty:
        raise PrimaryTemporalEligibilityError(f"native role authority lacks {fold_id}")
    inner = eligibility_windows.loc[
        eligibility_windows["view_type"].astype(str).eq(str(view_type))
        & _strict_bool(eligibility_windows["primary_s1_eligible"], "primary_s1_eligible")
        & eligibility_windows["primary_s1_role"].isin(PRIMARY_S1_ALLOWED_ROLES)
    ]
    train_keys = _keys_for_role(inner, "train")
    validation_keys = _keys_for_role(inner, "validation")
    represented = train_keys | validation_keys
    role_keys = roles["temporal_unit_key"].astype(str)
    original_train = _strict_bool(
        roles["native_unit_valid_for_main_train"],
        "native_unit_valid_for_main_train",
    )
    original_eval = _strict_bool(
        roles["native_unit_valid_for_main_eval"],
        "native_unit_valid_for_main_eval",
    )
    roles["native_unit_valid_for_main_train"] = original_train & role_keys.isin(train_keys)
    roles["native_unit_valid_for_main_eval"] = original_eval & role_keys.isin(represented)
    return roles, {
        "schema_version": "classification_v2.s1_primary_view_role_overlay.v1",
        "fold_id": str(fold_id),
        "view_type": str(view_type),
        "represented_train_native_units": int(len(train_keys)),
        "represented_validation_native_units": int(len(validation_keys)),
        "represented_inner_native_units": int(len(represented)),
        "unrepresented_native_units": int((~role_keys.isin(represented)).sum()),
        "event_weight_train_only": "PASS",
        "errors": [],
        "valid": True,
    }


def _classify_window(
    row: Any,
    *,
    records: dict[str, dict[str, Any]],
    allowed_roles: frozenset[str],
) -> dict[str, Any]:
    keys = _parse_native_keys(row.temporal_unit_keys_json)
    if not keys:
        return _derived_row(
            "",
            False,
            "OTHER_INVALID",
            "missing_constituent_native_keys",
            0.0,
            0,
        )
    constituent = [records.get(key) for key in keys]
    if any(record is None for record in constituent):
        return _derived_row(
            "", False, "ROLE_CONFLICT", "native_role_missing", 0.0, len(keys)
        )
    roles = {str(record["role"]) for record in constituent if record is not None}
    if len(roles) != 1:
        return _derived_row(
            "",
            False,
            "ROLE_CONFLICT",
            "constituent_roles_disagree",
            0.0,
            len(keys),
        )
    role = next(iter(roles))
    if role not in allowed_roles:
        return _derived_row(
            role,
            False,
            "OUTER_ROLE_REJECTED",
            "role_not_authorized_for_s1",
            0.0,
            len(keys),
        )

    window_label = str(getattr(row, "behavior_window_label", "")).strip()
    labels = [
        str(record["behavior_label"]).strip()
        for record in constituent
        if record is not None
    ]
    if (
        not window_label
        or window_label not in VALID_BEHAVIORS
        or any(not label for label in labels)
    ):
        return _derived_row(
            role, False, "UNRESOLVED_LABEL", "missing_or_invalid_label", 0.0, len(keys)
        )
    if any(label not in VALID_BEHAVIORS for label in labels):
        return _derived_row(
            role,
            False,
            "UNRESOLVED_LABEL",
            "constituent_label_not_canonical",
            0.0,
            len(keys),
        )
    if len(set(labels)) != 1 or labels[0] != window_label:
        return _derived_row(
            role,
            False,
            "MIXED_LABEL",
            "constituent_label_disagrees",
            0.0,
            len(keys),
        )

    valid_col = (
        "native_unit_valid_for_main_train"
        if role == "train"
        else "native_unit_valid_for_main_eval"
    )
    if not all(
        _strict_bool_value(record[valid_col], valid_col)
        for record in constituent
        if record is not None
    ):
        return _derived_row(
            role, False, "INELIGIBLE_CONSTITUENT", valid_col, 0.0, len(keys)
        )
    if not _strict_bool_value(
        row.window_valid_for_main_train, "window_valid_for_main_train"
    ):
        return _derived_row(
            role,
            False,
            "OTHER_INVALID",
            "source_window_not_primary_eligible",
            0.0,
            len(keys),
        )
    source_weight = _positive_weight(row.window_sample_weight)
    if source_weight <= 0.0:
        return _derived_row(
            role,
            False,
            "OTHER_INVALID",
            "source_window_nonpositive_weight",
            0.0,
            len(keys),
        )
    return _derived_row(role, True, "VALID_SINGLE_LABEL", "", source_weight, len(keys))


def _keys_for_role(windows: pd.DataFrame, role: str) -> set[str]:
    rows = windows.loc[windows["primary_s1_role"].astype(str).eq(role)]
    return {
        key
        for raw_keys in rows["temporal_unit_keys_json"]
        for key in _parse_native_keys(raw_keys)
    }


def _derived_row(
    role: str,
    eligible: bool,
    status: str,
    reason: str,
    weight: float,
    constituent_count: int,
) -> dict[str, Any]:
    return {
        "primary_s1_role": role,
        "primary_s1_eligible": bool(eligible),
        "primary_s1_eligibility_status": status,
        "primary_s1_eligibility_reason": reason,
        "primary_s1_effective_sample_weight": float(weight) if eligible else 0.0,
        "primary_s1_constituent_native_count": int(constituent_count),
    }


def _validated_inner_roles(requested_roles: Iterable[str]) -> frozenset[str]:
    values = frozenset(str(role).strip() for role in requested_roles)
    if not values:
        raise PrimaryTemporalEligibilityError("S1 requires at least one inner role")
    invalid = sorted(values.difference(PRIMARY_S1_ALLOWED_ROLES))
    if invalid:
        raise PrimaryTemporalEligibilityError(
            f"S1 outer role request rejected before metadata open={invalid}"
        )
    return values


def _parse_native_keys(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PrimaryTemporalEligibilityError("invalid temporal_unit_keys_json") from exc
    if not isinstance(parsed, list):
        raise PrimaryTemporalEligibilityError("temporal_unit_keys_json is not a list")
    keys = [str(key).strip() for key in parsed]
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise PrimaryTemporalEligibilityError("invalid constituent native key list")
    return keys


def _build_audit(
    windows: pd.DataFrame,
    *,
    fold_id: str,
    requested_roles: frozenset[str],
) -> dict[str, Any]:
    errors: list[str] = []
    positive_invalid = (~_strict_bool(windows["primary_s1_eligible"], "primary_s1_eligible")) & (
        pd.to_numeric(
            windows["primary_s1_effective_sample_weight"], errors="coerce"
        )
        .fillna(0.0)
        .gt(0.0)
    )
    if positive_invalid.any():
        errors.append(f"invalid_windows_with_positive_primary_weight={int(positive_invalid.sum())}")
    valid = _strict_bool(windows["primary_s1_eligible"], "primary_s1_eligible")
    return {
        "schema_version": "classification_v2.s1_primary_temporal_eligibility.v1",
        "fold_id": str(fold_id),
        "requested_roles": sorted(requested_roles),
        "input_window_rows": int(len(windows)),
        "primary_eligible_window_rows": int(valid.sum()),
        "mixed_label_primary_windows": int(
            windows["primary_s1_eligibility_status"].eq("MIXED_LABEL").sum()
        ),
        "validity_counts_by_view_role_source": _count_records(
            windows,
            [
                "view_type",
                "primary_s1_role",
                "source_type",
                "primary_s1_eligibility_status",
            ],
        ),
        "retained_window_feature_reuse": "PASS",
        "window_row_index_preserved": "PASS",
        "ordered_input_window_id_sha256": _ordered_hash(
            windows["window_id"].astype(str).tolist()
        ),
        "errors": errors,
        "valid": not errors,
    }


def _count_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    counts = (
        frame.groupby(columns, dropna=False, sort=True)
        .size()
        .reset_index(name="window_rows")
    )
    return counts.to_dict("records")


def _assert_row_preservation(before: pd.DataFrame, after: pd.DataFrame) -> None:
    if len(before) != len(after):
        raise PrimaryTemporalEligibilityError("primary eligibility changed window row count")
    if (
        before["window_id"].astype(str).tolist()
        != after["window_id"].astype(str).tolist()
    ):
        raise PrimaryTemporalEligibilityError("primary eligibility changed window order")
    if not np.array_equal(
        pd.to_numeric(before["window_row_index"], errors="raise").to_numpy(),
        pd.to_numeric(after["window_row_index"], errors="raise").to_numpy(),
    ):
        raise PrimaryTemporalEligibilityError("primary eligibility changed tensor row indices")


def _positive_weight(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and np.isfinite(float(numeric)) and numeric > 0.0:
        return float(numeric)
    return 0.0


def _strict_bool_value(value: object, name: str) -> bool:
    return bool(_strict_bool(pd.Series([value]), name).iloc[0])


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise PrimaryTemporalEligibilityError(f"invalid {name} boolean")
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    if (~normalized.isin(true_values | false_values)).any():
        raise PrimaryTemporalEligibilityError(f"invalid {name} boolean")
    return normalized.isin(true_values)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise PrimaryTemporalEligibilityError(f"{name} missing columns={missing}")


def _ordered_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _verify_sha256(path: Path, expected: str | None, name: str) -> None:
    if expected is None:
        return
    normalized = str(expected).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise PrimaryTemporalEligibilityError(f"invalid expected {name} sha256")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != normalized:
        raise PrimaryTemporalEligibilityError(f"{name} hash mismatch before metadata open")
