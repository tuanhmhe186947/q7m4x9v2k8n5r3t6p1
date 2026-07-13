"""Fold-local native-event weighting without duplicating sequence windows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.window_alignment import (
    ordered_window_id_sha256,
)
from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

FOLD_EVENT_WEIGHT_SCHEMA_VERSION = "classification_v2_fold_event_weights_v1"
_ROLES = ("train", "validation", "test")


@dataclass(slots=True)
class FoldEventWeightTables:
    """Expanded fold/window weights plus event and class audit ledgers."""

    weights: pd.DataFrame
    class_summary: pd.DataFrame
    event_summary: pd.DataFrame
    audit: dict[str, Any]


def build_fold_event_weight_manifest(
    windows: pd.DataFrame,
    fold_roles: pd.DataFrame,
    *,
    selection: pd.DataFrame | None = None,
    window_id_col: str = "window_id",
    event_keys_col: str = "temporal_unit_keys_json",
    valid_col: str = "window_valid_for_main_train",
    selection_col: str = "fixed6_keep",
    label_col: str = "behavior_window_label",
    base_weight_col: str = "window_sample_weight",
    fold_col: str = "outer_fold_id",
    role_col: str = "role",
    native_key_col: str = "temporal_unit_key",
    class_weight_power: float = 0.5,
    class_weight_max: float = 5.0,
    sample_weight_max: float = 10.0,
) -> FoldEventWeightTables:
    """Allocate train-only native-event mass for every declared outer fold."""

    _validate_hyperparameters(
        class_weight_power=class_weight_power,
        class_weight_max=class_weight_max,
        sample_weight_max=sample_weight_max,
    )
    window_required = {
        window_id_col,
        event_keys_col,
        valid_col,
        label_col,
        base_weight_col,
    }
    role_required = {
        native_key_col,
        fold_col,
        role_col,
        "behavior_label",
        "native_unit_valid_for_main_eval",
    }
    missing_windows = sorted(window_required.difference(windows.columns))
    missing_roles = sorted(role_required.difference(fold_roles.columns))
    if missing_windows or missing_roles:
        raise ValueError(
            "fold event-weight input missing columns: "
            f"windows={missing_windows}, roles={missing_roles}"
        )

    work = windows.reset_index(drop=True).copy()
    window_ids = _validated_text_ids(work[window_id_col], name=window_id_col)
    source_valid = _strict_bool(work[valid_col], name=valid_col)
    selected = _validated_selection(
        selection,
        window_ids=window_ids,
        window_id_col=window_id_col,
        selection_col=selection_col,
    )
    event_lists = _parse_event_lists(work[event_keys_col])
    missing_valid_events = int(
        sum(
            is_valid and is_selected and not events
            for is_valid, is_selected, events in zip(
                source_valid,
                selected,
                event_lists,
                strict=True,
            )
        )
    )
    if missing_valid_events:
        raise ValueError(
            "fold event-weight valid windows missing native events="
            f"{missing_valid_events}"
        )
    labels = work[label_col].fillna("").astype(str).str.strip()
    invalid_train_labels = source_valid & selected & ~labels.isin(VALID_BEHAVIORS)
    if invalid_train_labels.any():
        values = sorted(labels[invalid_train_labels].unique())
        raise ValueError(f"invalid behavior labels on valid windows={values}")
    base_weight = pd.to_numeric(work[base_weight_col], errors="coerce")
    invalid_base = source_valid & selected & (
        ~np.isfinite(base_weight.to_numpy(dtype=float))
        | base_weight.lt(0.0)
    )
    if invalid_base.any():
        raise ValueError(
            "invalid base weights on valid windows="
            f"{int(invalid_base.sum())}"
        )
    base_weight = base_weight.fillna(0.0).astype(float)

    roles = _validated_roles(
        fold_roles,
        fold_col=fold_col,
        role_col=role_col,
        native_key_col=native_key_col,
    )
    folds = tuple(sorted(roles[fold_col].unique()))
    role_lookup = roles.set_index([fold_col, native_key_col])[role_col]
    native_valid_lookup = roles.set_index(
        [fold_col, native_key_col]
    )["native_unit_valid_for_main_eval"]
    label_lookup = _optional_native_label_lookup(
        roles,
        fold_col=fold_col,
        native_key_col=native_key_col,
    )

    expanded_parts: list[pd.DataFrame] = []
    for fold_id in folds:
        part = pd.DataFrame(
            {
                fold_col: fold_id,
                window_id_col: window_ids,
                event_keys_col: work[event_keys_col].astype(str),
                label_col: labels,
                base_weight_col: base_weight,
                "window_valid_for_source_train": source_valid,
                "window_selected_for_training_view": selected,
                "_window_order": np.arange(len(work), dtype=np.int64),
                "_event_keys": event_lists,
            }
        )
        part[role_col] = [
            _window_role(
                fold_id,
                events,
                role_lookup,
            )
            for events in event_lists
        ]
        part["window_native_units_valid_for_main_eval"] = [
            _window_native_units_valid(
                fold_id,
                events,
                native_valid_lookup,
            )
            for events in event_lists
        ]
        part["window_valid_for_event_weight"] = (
            part["window_valid_for_source_train"]
            & part["window_selected_for_training_view"]
            & part["window_native_units_valid_for_main_eval"]
        )
        part["window_valid_for_fold_training_weight"] = (
            part["window_valid_for_event_weight"]
            & part[role_col].eq("train")
        )
        _validate_native_label_agreement(
            part,
            fold_id=fold_id,
            event_keys_col="_event_keys",
            label_col=label_col,
            label_lookup=label_lookup,
        )
        expanded_parts.append(part)
    expanded = pd.concat(expanded_parts, ignore_index=True)

    weighted_parts: list[pd.DataFrame] = []
    class_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    fold_audits: dict[str, Any] = {}
    for fold_id in folds:
        part = expanded.loc[expanded[fold_col].eq(fold_id)].copy()
        weighted, classes, events, fold_audit = _weight_one_fold(
            part,
            roles.loc[roles[fold_col].eq(fold_id)],
            fold_id=fold_id,
            fold_col=fold_col,
            role_col=role_col,
            native_key_col=native_key_col,
            label_col=label_col,
            base_weight_col=base_weight_col,
            class_weight_power=class_weight_power,
            class_weight_max=class_weight_max,
            sample_weight_max=sample_weight_max,
        )
        weighted_parts.append(weighted)
        class_rows.extend(classes)
        event_rows.extend(events)
        fold_audits[fold_id] = fold_audit

    output = pd.concat(weighted_parts, ignore_index=True).sort_values(
        [fold_col, "_window_order"],
        kind="stable",
    )
    output = output.reset_index(drop=True)
    output["event_count_window"] = output["_event_keys"].map(len)
    output["event_overlap_cluster_id"] = output["_event_keys"].map(
        _canonical_event_json
    )
    output = output.drop(columns=["_window_order", "_event_keys"])
    output_columns = [
        fold_col,
        window_id_col,
        role_col,
        label_col,
        event_keys_col,
        "event_overlap_cluster_id",
        "event_count_window",
        "max_train_windows_per_event",
        base_weight_col,
        "window_valid_for_source_train",
        "window_selected_for_training_view",
        "window_native_units_valid_for_main_eval",
        "window_valid_for_event_weight",
        "window_valid_for_fold_training_weight",
        "fold_event_mass_weight",
        "fold_event_sample_weight_raw",
        "fold_event_sample_weight",
        "fold_class_weight",
        "fold_event_class_sample_weight_raw",
        "fold_event_class_sample_weight",
    ]
    output = output[output_columns]
    class_summary = pd.DataFrame(class_rows)
    event_summary = pd.DataFrame(event_rows)
    expected_rows = len(windows) * len(folds)
    errors: list[str] = []
    if len(output) != expected_rows:
        errors.append(
            f"expanded_row_count={len(output)} expected={expected_rows}"
        )
    duplicate_keys = int(
        output.duplicated([fold_col, window_id_col], keep=False).sum()
    )
    if duplicate_keys:
        errors.append(f"duplicate_fold_window_rows={duplicate_keys}")
    nontrain_nonzero = int(
        (
            ~output["window_valid_for_fold_training_weight"]
            & (
                output["fold_event_sample_weight"].ne(0.0)
                | output["fold_event_class_sample_weight"].ne(0.0)
            )
        ).sum()
    )
    if nontrain_nonzero:
        errors.append(f"nontraining_rows_with_weight={nontrain_nonzero}")
    errors.extend(
        f"fold={fold_id}:{error}"
        for fold_id, details in fold_audits.items()
        for error in details["errors"]
    )
    audit = {
        "schema_version": FOLD_EVENT_WEIGHT_SCHEMA_VERSION,
        "input_window_rows": int(len(windows)),
        "fold_count": int(len(folds)),
        "fold_ids": list(folds),
        "rows": int(len(output)),
        "expected_rows": int(expected_rows),
        "duplicate_fold_window_rows": duplicate_keys,
        "nontraining_rows_with_weight": nontrain_nonzero,
        "input_ordered_window_id_sha256": ordered_window_id_sha256(window_ids),
        "selected_window_rows": int(selected.sum()),
        "selection_column": selection_col,
        "selection_sha256": _selection_sha256(window_ids, selected),
        "fold_window_order_sha256": _fold_window_order_sha256(
            output,
            fold_col=fold_col,
            window_id_col=window_id_col,
        ),
        "hyperparameters": {
            "class_weight_power": float(class_weight_power),
            "class_weight_max": float(class_weight_max),
            "sample_weight_max": float(sample_weight_max),
        },
        "folds": fold_audits,
        "warnings": [
            "weights and behavior labels are supervision metadata, never model X",
            "validation and test rows are retained with zero training weight",
        ],
        "errors": errors,
        "valid": not errors,
    }
    if errors:
        raise ValueError("fold event-weight contract failed: " + "; ".join(errors))
    return FoldEventWeightTables(
        weights=output,
        class_summary=class_summary,
        event_summary=event_summary,
        audit=audit,
    )


def audit_fold_event_weight_manifest(
    persisted: pd.DataFrame,
    windows: pd.DataFrame,
    fold_roles: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
    **build_options: Any,
) -> dict[str, Any]:
    """Rebuild a fold manifest and compare exact order, values, and keys."""

    errors: list[str] = []
    try:
        rebuilt = build_fold_event_weight_manifest(
            windows,
            fold_roles,
            **build_options,
        )
    except ValueError as exc:
        return {
            "schema_version": "classification_v2_fold_event_weight_check_v1",
            "rows": int(len(persisted)),
            "errors": [f"rebuild_contract={exc}"],
            "valid": False,
        }
    expected = rebuilt.weights
    missing = sorted(set(expected.columns).difference(persisted.columns))
    extra = sorted(set(persisted.columns).difference(expected.columns))
    if missing:
        errors.append(f"missing_columns={missing}")
    if extra:
        errors.append(f"unexpected_columns={extra}")
    if len(persisted) != len(expected):
        errors.append(
            f"row_count={len(persisted)} expected={len(expected)}"
        )
    key_columns = ["outer_fold_id", "window_id"]
    order_mismatch = 0
    if not missing and len(persisted) == len(expected):
        actual_keys = persisted[key_columns].fillna("").astype(str)
        expected_keys = expected[key_columns].fillna("").astype(str)
        order_mismatch = int(actual_keys.ne(expected_keys).any(axis=1).sum())
        if order_mismatch:
            errors.append(f"fold_window_order_mismatch_rows={order_mismatch}")
    numeric_columns = [
        "event_count_window",
        "max_train_windows_per_event",
        "window_sample_weight",
        "fold_event_mass_weight",
        "fold_event_sample_weight_raw",
        "fold_event_sample_weight",
        "fold_class_weight",
        "fold_event_class_sample_weight_raw",
        "fold_event_class_sample_weight",
    ]
    numeric_mismatches: dict[str, int] = {}
    text_mismatches: dict[str, int] = {}
    if not missing and len(persisted) == len(expected):
        for column in numeric_columns:
            actual = pd.to_numeric(persisted[column], errors="coerce").to_numpy()
            wanted = pd.to_numeric(expected[column], errors="coerce").to_numpy()
            mismatch = ~np.isfinite(actual) | ~np.isclose(
                actual,
                wanted,
                atol=tolerance,
                rtol=0.0,
            )
            numeric_mismatches[column] = int(mismatch.sum())
        for column in sorted(set(expected.columns).difference(numeric_columns)):
            actual = persisted[column].fillna("").astype(str)
            wanted = expected[column].fillna("").astype(str)
            text_mismatches[column] = int(actual.ne(wanted).sum())
        errors.extend(
            f"numeric_mismatch_{column}={count}"
            for column, count in numeric_mismatches.items()
            if count
        )
        errors.extend(
            f"text_mismatch_{column}={count}"
            for column, count in text_mismatches.items()
            if count
        )
    return {
        "schema_version": "classification_v2_fold_event_weight_check_v1",
        "rows": int(len(persisted)),
        "expected_rows": int(len(expected)),
        "fold_window_order_mismatch_rows": order_mismatch,
        "numeric_mismatch_counts": numeric_mismatches,
        "text_mismatch_counts": text_mismatches,
        "rebuilt_audit": rebuilt.audit,
        "errors": errors,
        "valid": not errors,
    }


def _weight_one_fold(
    part: pd.DataFrame,
    fold_native_roles: pd.DataFrame,
    *,
    fold_id: str,
    fold_col: str,
    role_col: str,
    native_key_col: str,
    label_col: str,
    base_weight_col: str,
    class_weight_power: float,
    class_weight_max: float,
    sample_weight_max: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_valid = part["window_valid_for_fold_training_weight"].astype(bool)
    event_counts: Counter[str] = Counter()
    event_labels: dict[str, set[str]] = defaultdict(set)
    for events, label in zip(
        part.loc[train_valid, "_event_keys"],
        part.loc[train_valid, label_col],
        strict=True,
    ):
        event_counts.update(events)
        for event in events:
            event_labels[event].add(str(label))
    mixed_labels = {
        event: sorted(values)
        for event, values in event_labels.items()
        if len(values) != 1
    }
    if mixed_labels:
        raise ValueError(
            f"fold={fold_id} native events have multiple labels={mixed_labels}"
        )

    event_mass: list[float] = []
    max_counts: list[int] = []
    for is_train, events in zip(
        train_valid,
        part["_event_keys"],
        strict=True,
    ):
        if is_train:
            event_mass.append(
                float(sum(1.0 / event_counts[event] for event in events))
            )
        else:
            event_mass.append(0.0)
        max_counts.append(max((event_counts[event] for event in events), default=0))
    part["max_train_windows_per_event"] = max_counts
    part["fold_event_mass_weight"] = event_mass
    part["fold_event_sample_weight_raw"] = (
        part[base_weight_col] * part["fold_event_mass_weight"]
    )
    part["fold_event_sample_weight"] = _normalize_training_weights(
        part["fold_event_sample_weight_raw"].to_numpy(dtype=float),
        train_valid.to_numpy(dtype=bool),
        max_weight=sample_weight_max,
    )

    class_mass = (
        part.loc[train_valid]
        .groupby(label_col)["fold_event_mass_weight"]
        .sum()
        .reindex(VALID_BEHAVIORS, fill_value=0.0)
    )
    class_weights = _class_weights(
        class_mass,
        power=class_weight_power,
        max_weight=class_weight_max,
    )
    part["fold_class_weight"] = part[label_col].map(class_weights).fillna(0.0)
    part["fold_event_class_sample_weight_raw"] = (
        part["fold_event_sample_weight_raw"] * part["fold_class_weight"]
    )
    part["fold_event_class_sample_weight"] = _normalize_training_weights(
        part["fold_event_class_sample_weight_raw"].to_numpy(dtype=float),
        train_valid.to_numpy(dtype=bool),
        max_weight=sample_weight_max,
    )
    part.loc[~train_valid, "fold_class_weight"] = 0.0

    class_rows = [
        {
            fold_col: fold_id,
            "behavior_label": label,
            "train_window_rows": int(
                (train_valid & part[label_col].eq(label)).sum()
            ),
            "native_event_mass": float(class_mass[label]),
            "fold_class_weight": float(class_weights[label]),
            "event_only_weight_sum": float(
                part.loc[
                    train_valid & part[label_col].eq(label),
                    "fold_event_sample_weight",
                ].sum()
            ),
            "event_class_weight_sum": float(
                part.loc[
                    train_valid & part[label_col].eq(label),
                    "fold_event_class_sample_weight",
                ].sum()
            ),
        }
        for label in VALID_BEHAVIORS
    ]
    allocated_by_event: Counter[str] = Counter()
    for events in part.loc[train_valid, "_event_keys"]:
        for event in events:
            allocated_by_event[event] += 1.0 / event_counts[event]
    event_rows = _event_summary_rows(
        fold_native_roles,
        fold_id=fold_id,
        fold_col=fold_col,
        role_col=role_col,
        native_key_col=native_key_col,
        event_counts=event_counts,
        event_labels=event_labels,
        allocated_by_event=allocated_by_event,
    )
    represented_mass = float(sum(allocated_by_event.values()))
    expected_mass = float(len(event_counts))
    mass_error = abs(represented_mass - expected_mass)
    eligible_native = _eligible_native_events(
        fold_native_roles,
        native_key_col=native_key_col,
    )
    missing_eligible_train_events = sorted(
        event
        for event in eligible_native
        if event not in event_counts
        and str(
            fold_native_roles.loc[
                fold_native_roles[native_key_col].astype(str).eq(event),
                role_col,
            ].iloc[0]
        )
        == "train"
    )
    errors: list[str] = []
    if mass_error > 1e-8:
        errors.append(f"event_mass_conservation_error={mass_error}")
    if missing_eligible_train_events:
        errors.append(
            "eligible_train_native_events_without_valid_window="
            f"{len(missing_eligible_train_events)}"
        )
    event_weights = part.loc[train_valid, "fold_event_sample_weight"].to_numpy()
    combined = part.loc[
        train_valid,
        "fold_event_class_sample_weight",
    ].to_numpy()
    fold_audit = {
        "role_rows": {
            role: int(part[role_col].eq(role).sum())
            for role in (*_ROLES, "not_eligible")
        },
        "train_weight_rows": int(train_valid.sum()),
        "represented_train_native_events": int(len(event_counts)),
        "expected_event_mass": expected_mass,
        "allocated_event_mass": represented_mass,
        "event_mass_conservation_error": mass_error,
        "eligible_train_native_events_without_valid_window": (
            missing_eligible_train_events[:20]
        ),
        "class_native_event_mass": {
            label: float(class_mass[label])
            for label in VALID_BEHAVIORS
        },
        "class_weights": class_weights,
        "event_weight_mean": _safe_mean(event_weights),
        "event_weight_max": _safe_max(event_weights),
        "event_weight_effective_sample_size": _effective_sample_size(
            event_weights
        ),
        "event_class_weight_mean": _safe_mean(combined),
        "event_class_weight_max": _safe_max(combined),
        "event_class_effective_sample_size": _effective_sample_size(combined),
        "errors": errors,
        "valid": not errors,
    }
    return part, class_rows, event_rows, fold_audit


def _event_summary_rows(
    fold_native_roles: pd.DataFrame,
    *,
    fold_id: str,
    fold_col: str,
    role_col: str,
    native_key_col: str,
    event_counts: Counter[str],
    event_labels: dict[str, set[str]],
    allocated_by_event: Counter[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fold_native_roles.sort_values(native_key_col).itertuples(index=False):
        record = row._asdict()
        event = str(record[native_key_col])
        labels = sorted(event_labels.get(event, set()))
        rows.append(
            {
                fold_col: fold_id,
                native_key_col: event,
                role_col: str(record[role_col]),
                "behavior_label": labels[0] if len(labels) == 1 else "",
                "valid_training_window_count": int(event_counts[event]),
                "allocated_event_mass": float(allocated_by_event[event]),
            }
        )
    return rows


def _validated_roles(
    fold_roles: pd.DataFrame,
    *,
    fold_col: str,
    role_col: str,
    native_key_col: str,
) -> pd.DataFrame:
    roles = fold_roles.copy()
    roles[fold_col] = _validated_text_ids(
        roles[fold_col],
        name=fold_col,
        require_unique=False,
    )
    roles[native_key_col] = _validated_text_ids(
        roles[native_key_col],
        name=native_key_col,
        require_unique=False,
    )
    roles[role_col] = roles[role_col].fillna("").astype(str).str.strip()
    roles["native_unit_valid_for_main_eval"] = _strict_bool(
        roles["native_unit_valid_for_main_eval"],
        name="native_unit_valid_for_main_eval",
    )
    native_labels = roles["behavior_label"].fillna("").astype(str).str.strip()
    invalid_labels = sorted(set(native_labels).difference(VALID_BEHAVIORS))
    invalid_roles = sorted(set(roles[role_col]).difference(_ROLES))
    duplicate = int(
        roles.duplicated([fold_col, native_key_col], keep=False).sum()
    )
    if invalid_roles or invalid_labels or duplicate:
        raise ValueError(
            "fold role contract failed: "
            f"invalid_roles={invalid_roles}, invalid_labels={invalid_labels}, "
            f"duplicate_rows={duplicate}"
        )
    roles["behavior_label"] = native_labels
    return roles


def _window_role(
    fold_id: str,
    events: list[str],
    role_lookup: pd.Series,
) -> str:
    if not events:
        return "not_eligible"
    missing = [event for event in events if (fold_id, event) not in role_lookup.index]
    if missing:
        raise ValueError(
            f"fold={fold_id} window references events missing roles={missing}"
        )
    observed = {str(role_lookup.loc[(fold_id, event)]) for event in events}
    if len(observed) != 1:
        raise ValueError(
            f"fold={fold_id} overlapping window crosses roles={sorted(observed)}"
        )
    return next(iter(observed))


def _window_native_units_valid(
    fold_id: str,
    events: list[str],
    native_valid_lookup: pd.Series,
) -> bool:
    """Require every native unit represented by a window to be evaluation-valid."""

    if not events:
        return False
    return all(bool(native_valid_lookup.loc[(fold_id, event)]) for event in events)


def _validated_selection(
    selection: pd.DataFrame | None,
    *,
    window_ids: pd.Series,
    window_id_col: str,
    selection_col: str,
) -> pd.Series:
    """Align a temporal-view selection exactly without dropping audit rows."""

    if selection is None:
        return pd.Series(True, index=window_ids.index, dtype=bool)
    missing = sorted(
        {window_id_col, selection_col}.difference(selection.columns)
    )
    if missing:
        raise ValueError(f"temporal-view selection missing columns={missing}")
    selection_ids = _validated_text_ids(
        selection[window_id_col],
        name=f"selection.{window_id_col}",
    ).reset_index(drop=True)
    expected_ids = window_ids.reset_index(drop=True)
    if len(selection_ids) != len(expected_ids) or not selection_ids.equals(
        expected_ids
    ):
        raise ValueError(
            "temporal-view selection window order mismatch: "
            f"observed={len(selection_ids)}, expected={len(expected_ids)}"
        )
    return _strict_bool(
        selection[selection_col].reset_index(drop=True),
        name=selection_col,
    )


def _optional_native_label_lookup(
    roles: pd.DataFrame,
    *,
    fold_col: str,
    native_key_col: str,
) -> pd.Series | None:
    if "behavior_label" not in roles.columns:
        return None
    labels = roles["behavior_label"].fillna("").astype(str).str.strip()
    return pd.Series(
        labels.to_numpy(),
        index=pd.MultiIndex.from_frame(roles[[fold_col, native_key_col]]),
    )


def _validate_native_label_agreement(
    part: pd.DataFrame,
    *,
    fold_id: str,
    event_keys_col: str,
    label_col: str,
    label_lookup: pd.Series | None,
) -> None:
    if label_lookup is None:
        return
    mismatches: list[str] = []
    valid = part["window_valid_for_fold_training_weight"]
    for events, label in zip(
        part.loc[valid, event_keys_col],
        part.loc[valid, label_col],
        strict=True,
    ):
        for event in events:
            native_label = str(label_lookup.loc[(fold_id, event)])
            if native_label and native_label != str(label):
                mismatches.append(f"{event}:{native_label}!={label}")
    if mismatches:
        raise ValueError(
            f"fold={fold_id} native/window label mismatch={mismatches[:20]}"
        )


def _eligible_native_events(
    fold_native_roles: pd.DataFrame,
    *,
    native_key_col: str,
) -> set[str]:
    if "native_unit_valid_for_main_eval" not in fold_native_roles.columns:
        return set()
    valid = _strict_bool(
        fold_native_roles["native_unit_valid_for_main_eval"],
        name="native_unit_valid_for_main_eval",
    )
    return set(fold_native_roles.loc[valid, native_key_col].astype(str))


def _class_weights(
    class_mass: pd.Series,
    *,
    power: float,
    max_weight: float,
) -> dict[str, float]:
    positive = class_mass[class_mass > 0.0]
    if positive.empty:
        raise ValueError("cannot compute class weights without train event mass")
    median = float(positive.median())
    result: dict[str, float] = {}
    for label in VALID_BEHAVIORS:
        mass = float(class_mass[label])
        if mass <= 0.0:
            result[label] = 0.0
        else:
            result[label] = float(
                min(max_weight, (median / mass) ** power)
            )
    return result


def _normalize_training_weights(
    raw: np.ndarray,
    train_mask: np.ndarray,
    *,
    max_weight: float,
) -> np.ndarray:
    output = np.zeros(len(raw), dtype=np.float64)
    selected = np.asarray(raw[train_mask], dtype=np.float64)
    if selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("training weights are empty or nonfinite")
    if (selected <= 0.0).any():
        raise ValueError("every valid training row must have positive raw weight")
    if max_weight < 1.0:
        raise ValueError("sample_weight_max must be at least one")
    low = 0.0
    high = max_weight / float(selected.min())
    for _ in range(100):
        scale = (low + high) / 2.0
        mean = float(np.minimum(selected * scale, max_weight).mean())
        if mean < 1.0:
            low = scale
        else:
            high = scale
    normalized = np.minimum(selected * high, max_weight)
    if abs(float(normalized.mean()) - 1.0) > 1e-9:
        raise ValueError(
            "bounded weight normalization failed: "
            f"mean={float(normalized.mean())}"
        )
    output[train_mask] = normalized
    return output


def _parse_event_lists(values: pd.Series) -> list[list[str]]:
    parsed_rows: list[list[str]] = []
    errors: list[int] = []
    for index, value in enumerate(values):
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            parsed_rows.append([])
            continue
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if not isinstance(parsed, list):
            parsed_rows.append([])
            errors.append(index)
            continue
        cleaned = [str(item).strip() for item in parsed]
        if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
            parsed_rows.append([])
            errors.append(index)
            continue
        parsed_rows.append(sorted(cleaned))
    if errors:
        raise ValueError(f"invalid temporal_unit_keys_json rows={errors[:20]}")
    return parsed_rows


def _validated_text_ids(
    values: pd.Series,
    *,
    name: str,
    require_unique: bool = True,
) -> pd.Series:
    text = values.fillna("").astype(str).str.strip()
    blank = int(text.eq("").sum())
    duplicate = int(text.duplicated(keep=False).sum()) if require_unique else 0
    if blank or duplicate:
        raise ValueError(
            f"{name} identity failed: blank={blank}, duplicate_rows={duplicate}"
        )
    return text


def _strict_bool(values: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        if values.isna().any():
            raise ValueError(f"{name} contains null booleans")
        return values.astype(bool)
    normalized = values.fillna("").astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    invalid = ~normalized.isin(true_values | false_values)
    if invalid.any():
        raise ValueError(
            f"{name} contains invalid booleans={sorted(normalized[invalid].unique())}"
        )
    return normalized.isin(true_values)


def _validate_hyperparameters(
    *,
    class_weight_power: float,
    class_weight_max: float,
    sample_weight_max: float,
) -> None:
    if class_weight_power < 0.0:
        raise ValueError("class_weight_power must be non-negative")
    if class_weight_max <= 0.0:
        raise ValueError("class_weight_max must be positive")
    if sample_weight_max < 1.0:
        raise ValueError("sample_weight_max must be at least one")


def _canonical_event_json(events: list[str]) -> str:
    return json.dumps(events, ensure_ascii=True, separators=(",", ":"))


def _fold_window_order_sha256(
    frame: pd.DataFrame,
    *,
    fold_col: str,
    window_id_col: str,
) -> str:
    rows = (
        f"{fold_id}\t{window_id}"
        for fold_id, window_id in frame[[fold_col, window_id_col]].itertuples(
            index=False,
            name=None,
        )
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _selection_sha256(window_ids: pd.Series, selected: pd.Series) -> str:
    rows = (
        f"{window_id}\t{int(is_selected)}"
        for window_id, is_selected in zip(
            window_ids.astype(str),
            selected.astype(bool),
            strict=True,
        )
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _effective_sample_size(weights: np.ndarray) -> float:
    if weights.size == 0 or float(np.square(weights).sum()) <= 0.0:
        return 0.0
    return float(weights.sum() ** 2 / np.square(weights).sum())


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(values.max()) if values.size else 0.0
