"""Burst-level behavior/posture target authority for Classification V2.

This module is deliberately independent of active review ledgers. Callers pass
explicit frozen or synthetic tables. Behavior remains directly supervised,
while posture is an independent masked target on the same native burst.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pig_behavior.classification_v2.schema import VALID_BEHAVIOR_SET

BEHAVIOR_POSTURE_CONTRACT_VERSION = (
    "classification_v2_behavior_posture_contract.v1"
)
POSTURE_LABEL_ORDER = ("lying", "sitting", "upright")
POSTURE_LABEL_SET = frozenset(POSTURE_LABEL_ORDER)
POSTURE_AUTHORITY_VALUES = frozenset(
    {"HUMAN_REVIEWED", "DERIVED_SAFE", "AUTO_VALIDATED", "UNRESOLVED"}
)
SAFE_DERIVATION_BEHAVIOR_AUTHORITIES = frozenset(
    {"FROZEN_HUMAN_REVIEWED", "SYNTHETIC_TEST"}
)
SAFE_POSTURE_BY_BEHAVIOR = {
    "lying": "lying",
    "sitting": "sitting",
    "stand": "upright",
    "eat": "upright",
}

BURST_KEY = "native_temporal_unit_key"
ANCHOR_KEY = "anchor_native_temporal_unit_key"


def derive_safe_posture(behavior: str) -> str | None:
    """Return a bounded, contract-approved posture derivation when one exists."""

    _require_behavior(behavior)
    return SAFE_POSTURE_BY_BEHAVIOR.get(behavior)


def build_burst_posture_authority(
    bursts: pd.DataFrame,
    overrides: pd.DataFrame | None = None,
    *,
    behavior_label_authority: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one posture-authority row per native burst.

    ``overrides`` is an explicit frozen or synthetic authority table. File-path
    protection is enforced by the CLI before any file is read.
    """

    if behavior_label_authority not in SAFE_DERIVATION_BEHAVIOR_AUTHORITIES:
        raise ValueError(
            "safe posture derivation requires frozen reviewed or synthetic "
            f"behavior authority, observed={behavior_label_authority}"
        )
    _require_columns(bursts, [BURST_KEY, "behavior_target"], table="bursts")
    base = bursts[[BURST_KEY, "behavior_target"]].copy()
    base[BURST_KEY] = _normalized_nonempty(base[BURST_KEY], name=BURST_KEY)
    base["behavior_target"] = _validated_behaviors(base["behavior_target"])
    base["behavior_label_authority"] = behavior_label_authority
    _require_unique(base, BURST_KEY, table="bursts")

    safe = base["behavior_target"].map(SAFE_POSTURE_BY_BEHAVIOR)
    base["posture_target"] = safe.fillna("")
    base["posture_valid_mask"] = safe.notna()
    base["posture_transition_flag"] = False
    base["posture_authority"] = safe.map(
        lambda value: "DERIVED_SAFE" if pd.notna(value) else "UNRESOLVED"
    )
    base["posture_authority_version"] = BEHAVIOR_POSTURE_CONTRACT_VERSION
    base["posture_proposal_confidence"] = pd.NA
    base["posture_review_status"] = safe.map(
        lambda value: "NOT_REQUIRED_SAFE" if pd.notna(value) else "PENDING"
    )

    override_count = 0
    if overrides is not None:
        base, override_count = _apply_overrides(base, overrides)

    _validate_authority_rows(base)
    audit = posture_authority_audit(base)
    audit["behavior_label_authority"] = behavior_label_authority
    audit["override_rows_applied"] = override_count
    return base, audit


def expand_posture_authority_to_windows(
    windows: pd.DataFrame,
    burst_authority: pd.DataFrame,
) -> pd.DataFrame:
    """Align burst authority to windows through an explicit anchor-unit key."""

    _require_columns(
        windows,
        ["window_id", ANCHOR_KEY, "behavior_target"],
        table="windows",
    )
    _require_columns(
        burst_authority,
        [
            BURST_KEY,
            "behavior_target",
            "posture_target",
            "posture_valid_mask",
            "posture_transition_flag",
            "posture_authority",
            "posture_authority_version",
            "behavior_label_authority",
        ],
        table="burst_authority",
    )
    work = windows.copy()
    work["window_id"] = _normalized_nonempty(work["window_id"], name="window_id")
    work[ANCHOR_KEY] = _normalized_nonempty(work[ANCHOR_KEY], name=ANCHOR_KEY)
    work["behavior_target"] = _validated_behaviors(work["behavior_target"])
    _require_unique(work, "window_id", table="windows")

    authority = burst_authority.copy()
    authority[BURST_KEY] = _normalized_nonempty(authority[BURST_KEY], name=BURST_KEY)
    _require_unique(authority, BURST_KEY, table="burst_authority")
    authority = authority.rename(
        columns={
            "behavior_target": "authority_behavior_target",
            BURST_KEY: ANCHOR_KEY,
        }
    )
    merged = work.merge(authority, on=ANCHOR_KEY, how="left", validate="many_to_one")
    missing = merged["posture_authority"].isna()
    if missing.any():
        raise ValueError(f"windows missing posture authority: count={int(missing.sum())}")
    mismatch = merged["behavior_target"].ne(merged["authority_behavior_target"])
    if mismatch.any():
        raise ValueError(
            "window/burst behavior mismatch: "
            f"count={int(mismatch.sum())}"
        )
    merged = merged.drop(columns="authority_behavior_target")
    _validate_authority_rows(merged)
    return merged


def align_window_posture_authority(
    windows: pd.DataFrame,
    window_authority: pd.DataFrame,
) -> pd.DataFrame:
    """Strictly align an already expanded posture authority by ``window_id``."""

    required = [
        "window_id",
        "behavior_target",
        "posture_target",
        "posture_valid_mask",
        "posture_transition_flag",
        "posture_authority",
        "posture_authority_version",
        "behavior_label_authority",
    ]
    _require_columns(windows, ["window_id", "behavior_target"], table="windows")
    _require_columns(window_authority, required, table="window_authority")
    base = windows[["window_id", "behavior_target"]].copy()
    base["window_id"] = _normalized_nonempty(base["window_id"], name="window_id")
    base["behavior_target"] = _validated_behaviors(base["behavior_target"])
    _require_unique(base, "window_id", table="windows")

    authority = window_authority[required].copy()
    authority["window_id"] = _normalized_nonempty(
        authority["window_id"], name="window_id"
    )
    _require_unique(authority, "window_id", table="window_authority")
    authority = authority.rename(
        columns={"behavior_target": "authority_behavior_target"}
    )
    merged = base.merge(authority, on="window_id", how="left", validate="one_to_one")
    missing = merged["posture_authority"].isna()
    if missing.any():
        raise ValueError(f"windows missing posture authority: count={int(missing.sum())}")
    mismatch = merged["behavior_target"].ne(merged["authority_behavior_target"])
    if mismatch.any():
        raise ValueError(
            "window/posture-authority behavior mismatch: "
            f"count={int(mismatch.sum())}"
        )
    merged = merged.drop(columns="authority_behavior_target")
    _validate_authority_rows(merged)
    return merged


def posture_authority_audit(authority: pd.DataFrame) -> dict[str, Any]:
    """Return bounded counts for a validated posture authority table."""

    valid = _strict_bool(authority["posture_valid_mask"], name="posture_valid_mask")
    transition = _strict_bool(
        authority["posture_transition_flag"],
        name="posture_transition_flag",
    )
    return {
        "schema_version": "classification_v2_posture_authority_audit.v1",
        "contract_version": BEHAVIOR_POSTURE_CONTRACT_VERSION,
        "rows": int(len(authority)),
        "valid_rows": int(valid.sum()),
        "unresolved_rows": int((~valid).sum()),
        "transition_rows": int(transition.sum()),
        "behavior_counts": authority["behavior_target"].value_counts().to_dict(),
        "posture_counts": (
            authority.loc[valid, "posture_target"].value_counts().to_dict()
        ),
        "authority_counts": authority["posture_authority"].value_counts().to_dict(),
        "errors": [],
    }


def _apply_overrides(
    base: pd.DataFrame,
    overrides: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    required = [
        BURST_KEY,
        "posture_target",
        "posture_valid_mask",
        "posture_transition_flag",
        "posture_authority",
    ]
    _require_columns(overrides, required, table="overrides")
    update = overrides[required].copy()
    update[BURST_KEY] = _normalized_nonempty(update[BURST_KEY], name=BURST_KEY)
    _require_unique(update, BURST_KEY, table="overrides")
    extra = sorted(set(update[BURST_KEY]).difference(base[BURST_KEY]))
    if extra:
        raise ValueError(f"override keys not present in bursts: {extra[:5]}")

    update = update.set_index(BURST_KEY)
    result = base.set_index(BURST_KEY)
    for key, row in update.iterrows():
        valid = _strict_bool_scalar(row["posture_valid_mask"], name="posture_valid_mask")
        transition = _strict_bool_scalar(
            row["posture_transition_flag"],
            name="posture_transition_flag",
        )
        authority = str(row["posture_authority"]).strip()
        target = str(row["posture_target"]).strip().lower()
        if authority not in POSTURE_AUTHORITY_VALUES:
            raise ValueError(f"invalid posture authority for {key}: {authority}")
        if transition and valid:
            raise ValueError(f"transition posture cannot be valid for {key}")
        if valid and target not in POSTURE_LABEL_SET:
            raise ValueError(f"invalid valid posture target for {key}: {target}")
        if not valid and target:
            raise ValueError(f"invalid posture target must be empty for {key}")
        expected = derive_safe_posture(str(result.at[key, "behavior_target"]))
        if expected is not None and valid and target != expected:
            raise ValueError(
                f"override contradicts safe posture for {key}: "
                f"expected={expected}, observed={target}"
            )
        result.at[key, "posture_target"] = target
        result.at[key, "posture_valid_mask"] = valid
        result.at[key, "posture_transition_flag"] = transition
        result.at[key, "posture_authority"] = authority
        result.at[key, "posture_review_status"] = (
            "RESOLVED" if valid else "UNRESOLVED"
        )
    return result.reset_index(), int(len(update))


def _validate_authority_rows(authority: pd.DataFrame) -> None:
    valid = _strict_bool(authority["posture_valid_mask"], name="posture_valid_mask")
    transition = _strict_bool(
        authority["posture_transition_flag"],
        name="posture_transition_flag",
    )
    targets = authority["posture_target"].fillna("").astype(str).str.strip().str.lower()
    invalid_target = valid & ~targets.isin(POSTURE_LABEL_SET)
    unexpected_target = ~valid & targets.ne("")
    if invalid_target.any() or unexpected_target.any():
        raise ValueError(
            "posture target/mask mismatch: "
            f"invalid_valid={int(invalid_target.sum())}, "
            f"target_while_invalid={int(unexpected_target.sum())}"
        )
    if (transition & valid).any():
        raise ValueError("transition rows must have posture_valid_mask=false")
    invalid_authority = ~authority["posture_authority"].isin(
        POSTURE_AUTHORITY_VALUES
    )
    if invalid_authority.any():
        values = sorted(authority.loc[invalid_authority, "posture_authority"].unique())
        raise ValueError(f"invalid posture authority values: {values}")
    invalid_behavior_authority = ~authority["behavior_label_authority"].isin(
        SAFE_DERIVATION_BEHAVIOR_AUTHORITIES
    )
    if invalid_behavior_authority.any():
        values = sorted(
            authority.loc[
                invalid_behavior_authority,
                "behavior_label_authority",
            ].unique()
        )
        raise ValueError(f"invalid behavior label authority values: {values}")
    safe = authority["behavior_target"].map(SAFE_POSTURE_BY_BEHAVIOR)
    contradiction = valid & safe.notna() & targets.ne(safe.fillna(""))
    if contradiction.any():
        raise ValueError(
            "safe behavior/posture contradiction: "
            f"count={int(contradiction.sum())}"
        )


def _validated_behaviors(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.lower()
    invalid = ~values.isin(VALID_BEHAVIOR_SET)
    if invalid.any():
        raise ValueError(f"invalid behavior targets: {sorted(values[invalid].unique())}")
    return values


def _require_behavior(behavior: str) -> None:
    if behavior not in VALID_BEHAVIOR_SET:
        raise ValueError(f"invalid behavior target: {behavior}")


def _normalized_nonempty(series: pd.Series, *, name: str) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise ValueError(f"{name} contains empty values")
    return values


def _require_columns(df: pd.DataFrame, columns: list[str], *, table: str) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{table} missing columns: {missing}")


def _require_unique(df: pd.DataFrame, column: str, *, table: str) -> None:
    duplicate_count = int(df[column].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{table} duplicate {column}: count={duplicate_count}")


def _strict_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{name} contains null values")
        return series.astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    invalid = ~normalized.isin(true_values | false_values)
    if invalid.any():
        raise ValueError(f"{name} contains invalid values: {sorted(normalized[invalid].unique())}")
    return normalized.isin(true_values)


def _strict_bool_scalar(value: object, *, name: str) -> bool:
    return bool(_strict_bool(pd.Series([value]), name=name).iloc[0])


__all__ = [
    "ANCHOR_KEY",
    "BEHAVIOR_POSTURE_CONTRACT_VERSION",
    "BURST_KEY",
    "POSTURE_AUTHORITY_VALUES",
    "POSTURE_LABEL_ORDER",
    "POSTURE_LABEL_SET",
    "SAFE_DERIVATION_BEHAVIOR_AUTHORITIES",
    "SAFE_POSTURE_BY_BEHAVIOR",
    "align_window_posture_authority",
    "build_burst_posture_authority",
    "derive_safe_posture",
    "expand_posture_authority_to_windows",
    "posture_authority_audit",
]
