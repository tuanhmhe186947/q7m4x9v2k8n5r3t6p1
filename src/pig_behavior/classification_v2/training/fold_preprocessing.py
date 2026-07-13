"""Fold-local, hash-bound preprocessing for spatial sequence features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pig_behavior.classification_v2.contracts.output_safety import (
    require_output_paths_available,
)
from pig_behavior.classification_v2.contracts.window_alignment import (
    ordered_window_id_sha256,
)

PREPROCESSING_SCHEMA_VERSION = "classification_v2_fold_preprocessing_v1"
_VALID_ROLES = frozenset({"train", "validation", "test"})


@dataclass(frozen=True, slots=True)
class FoldPreprocessingState:
    """Immutable semantic state fitted from one fold's training rows only."""

    fold_id: str
    snapshot_sha256: str
    config_sha256: str
    spatial_audit_sha256: str
    all_window_id_sha256: str
    train_window_id_sha256: str
    role_assignment_sha256: str
    row_count: int
    train_row_count: int
    role_counts: dict[str, int]
    feature_groups: tuple[str, ...]
    standardized_groups: tuple[str, ...]
    group_contracts: dict[str, dict[str, Any]]
    statistics: dict[str, dict[str, Any]]
    state_sha256: str

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe payload with its semantic digest."""

        payload = self.semantic_payload()
        return {**payload, "state_sha256": self.state_sha256}

    def semantic_payload(self) -> dict[str, Any]:
        """Return fields covered by ``state_sha256``."""

        return {
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "fold_id": self.fold_id,
            "snapshot_sha256": self.snapshot_sha256,
            "config_sha256": self.config_sha256,
            "spatial_audit_sha256": self.spatial_audit_sha256,
            "all_window_id_sha256": self.all_window_id_sha256,
            "train_window_id_sha256": self.train_window_id_sha256,
            "role_assignment_sha256": self.role_assignment_sha256,
            "row_count": self.row_count,
            "train_row_count": self.train_row_count,
            "role_counts": self.role_counts,
            "feature_groups": list(self.feature_groups),
            "standardized_groups": list(self.standardized_groups),
            "group_contracts": self.group_contracts,
            "statistics": self.statistics,
            "fit_contract": {
                "role": "train",
                "eligible_only": True,
                "imputation": "training_mean",
                "scaling": "training_population_standard_deviation",
                "absent_slot_value_after_transform": 0.0,
                "validation_test_excluded_from_fit": True,
            },
        }

    def transform_torch(
        self,
        features: Mapping[str, torch.Tensor],
        *,
        length_mask: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Transform one batch while keeping padding and missing slots at zero."""

        if length_mask.shape != observed_mask.shape:
            raise ValueError(
                "preprocessing mask shape mismatch: "
                f"length={tuple(length_mask.shape)}, "
                f"observed={tuple(observed_mask.shape)}"
            )
        valid_slots = length_mask.bool() & observed_mask.bool()
        transformed: dict[str, torch.Tensor] = {}
        for group in self.feature_groups:
            if group not in features:
                raise ValueError(f"preprocessing input missing feature group={group}")
            values = features[group]
            contract = self.group_contracts[group]
            expected_dim = int(contract["dimension"])
            if values.ndim != 3 or values.shape[-1] != expected_dim:
                raise ValueError(
                    f"preprocessing shape mismatch for {group}: "
                    f"observed={tuple(values.shape)}, expected_last_dim={expected_dim}"
                )
            if tuple(values.shape[:2]) != tuple(valid_slots.shape):
                raise ValueError(
                    f"preprocessing slot shape mismatch for {group}: "
                    f"features={tuple(values.shape[:2])}, "
                    f"mask={tuple(valid_slots.shape)}"
                )
            clean = values
            if group in self.statistics:
                statistics = self.statistics[group]
                mean = torch.as_tensor(
                    statistics["mean"],
                    dtype=values.dtype,
                    device=values.device,
                )
                scale = torch.as_tensor(
                    statistics["scale"],
                    dtype=values.dtype,
                    device=values.device,
                )
                clean = torch.where(torch.isfinite(clean), clean, mean)
                clean = (clean - mean) / scale
            else:
                clean = torch.where(
                    torch.isfinite(clean),
                    clean,
                    torch.zeros((), dtype=values.dtype, device=values.device),
                )
            transformed[group] = torch.where(
                valid_slots.unsqueeze(-1),
                clean,
                torch.zeros((), dtype=values.dtype, device=values.device),
            )
        return transformed


def fit_fold_preprocessing(
    frame: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    feature_names: Mapping[str, Sequence[str]],
    *,
    fold_id: str,
    snapshot_sha256: str,
    config_sha256: str,
    spatial_audit_sha256: str,
    feature_groups: Sequence[str],
    standardized_groups: Sequence[str],
    role_col: str = "grouped_role",
    eligible_col: str = "eligible",
    window_id_col: str = "window_id",
) -> FoldPreprocessingState:
    """Fit means and scales from all eligible ``role=train`` rows."""

    _require_nonblank_lineage(
        fold_id=fold_id,
        snapshot_sha256=snapshot_sha256,
        config_sha256=config_sha256,
        spatial_audit_sha256=spatial_audit_sha256,
    )
    required_columns = {window_id_col, role_col, eligible_col}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"preprocessing frame missing columns={missing_columns}")
    _validate_window_ids(frame[window_id_col])
    eligible = _as_bool(frame[eligible_col])
    roles = frame[role_col].fillna("").astype(str).str.strip()
    invalid_roles = sorted(set(roles[eligible]).difference(_VALID_ROLES))
    if invalid_roles:
        raise ValueError(f"eligible rows have invalid grouped roles={invalid_roles}")
    train_mask = eligible & roles.eq("train")
    if not train_mask.any():
        raise ValueError("cannot fit fold preprocessing on zero training rows")

    groups = _ordered_unique(feature_groups, name="feature_groups")
    standardized = _ordered_unique(
        standardized_groups,
        name="standardized_groups",
    )
    unknown_standardized = sorted(set(standardized).difference(groups))
    if unknown_standardized:
        raise ValueError(
            "standardized groups are outside feature groups: "
            f"{unknown_standardized}"
        )
    length_mask, observed_mask = _validate_array_contract(
        arrays,
        expected_rows=len(frame),
    )
    train_indices = np.flatnonzero(train_mask.to_numpy())
    valid_train_slots = (
        length_mask[train_indices].astype(bool)
        & observed_mask[train_indices].astype(bool)
    )

    group_contracts: dict[str, dict[str, Any]] = {}
    statistics: dict[str, dict[str, Any]] = {}
    for group in groups:
        if group not in arrays:
            raise ValueError(f"preprocessing arrays missing group={group}")
        values = np.asarray(arrays[group])
        names = tuple(str(value) for value in feature_names.get(group, ()))
        _validate_group_array(
            group,
            values,
            names,
            expected_rows=len(frame),
            expected_slots=length_mask.shape[1],
        )
        group_contracts[group] = {
            "feature_names": list(names),
            "dimension": int(values.shape[-1]),
            "dtype": str(values.dtype),
        }
        if group in standardized:
            statistics[group] = _fit_group_statistics(
                group,
                values[train_indices][valid_train_slots],
                names,
            )

    role_counts = {
        role: int((eligible & roles.eq(role)).sum())
        for role in ("train", "validation", "test")
    }
    semantic = {
        "schema_version": PREPROCESSING_SCHEMA_VERSION,
        "fold_id": fold_id,
        "snapshot_sha256": snapshot_sha256,
        "config_sha256": config_sha256,
        "spatial_audit_sha256": spatial_audit_sha256,
        "all_window_id_sha256": ordered_window_id_sha256(frame[window_id_col]),
        "train_window_id_sha256": ordered_window_id_sha256(
            frame.loc[train_mask, window_id_col]
        ),
        "role_assignment_sha256": _role_assignment_sha256(
            frame[window_id_col],
            roles,
            eligible,
        ),
        "row_count": int(len(frame)),
        "train_row_count": int(train_mask.sum()),
        "role_counts": role_counts,
        "feature_groups": list(groups),
        "standardized_groups": list(standardized),
        "group_contracts": group_contracts,
        "statistics": statistics,
        "fit_contract": {
            "role": "train",
            "eligible_only": True,
            "imputation": "training_mean",
            "scaling": "training_population_standard_deviation",
            "absent_slot_value_after_transform": 0.0,
            "validation_test_excluded_from_fit": True,
        },
    }
    digest = _payload_sha256(semantic)
    return FoldPreprocessingState(
        fold_id=fold_id,
        snapshot_sha256=snapshot_sha256,
        config_sha256=config_sha256,
        spatial_audit_sha256=spatial_audit_sha256,
        all_window_id_sha256=semantic["all_window_id_sha256"],
        train_window_id_sha256=semantic["train_window_id_sha256"],
        role_assignment_sha256=semantic["role_assignment_sha256"],
        row_count=int(len(frame)),
        train_row_count=int(train_mask.sum()),
        role_counts=role_counts,
        feature_groups=groups,
        standardized_groups=standardized,
        group_contracts=group_contracts,
        statistics=statistics,
        state_sha256=digest,
    )


def write_fold_preprocessing_state(
    path: Path,
    state: FoldPreprocessingState,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist a preprocessing state without silent replacement."""

    require_output_paths_available([path], overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state.to_payload(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_fold_preprocessing_state(
    path: Path,
    *,
    expected_fold_id: str | None = None,
    expected_snapshot_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> FoldPreprocessingState:
    """Load a state and reject tampering or requested-lineage drift."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    state = _state_from_payload(payload)
    expected = {
        "fold_id": expected_fold_id,
        "snapshot_sha256": expected_snapshot_sha256,
        "config_sha256": expected_config_sha256,
    }
    mismatches = {
        name: {"expected": value, "observed": getattr(state, name)}
        for name, value in expected.items()
        if value is not None and getattr(state, name) != value
    }
    if mismatches:
        raise ValueError(f"fold preprocessing lineage mismatch={mismatches}")
    return state


def ensure_fold_preprocessing_state(
    path: Path,
    state: FoldPreprocessingState,
) -> str:
    """Write a new state or require an existing state to match exactly."""

    if not path.exists():
        write_fold_preprocessing_state(path, state)
        return "written"
    existing = load_fold_preprocessing_state(
        path,
        expected_fold_id=state.fold_id,
        expected_snapshot_sha256=state.snapshot_sha256,
        expected_config_sha256=state.config_sha256,
    )
    if existing.state_sha256 != state.state_sha256:
        raise ValueError(
            "existing fold preprocessing differs from current fitted state: "
            f"existing={existing.state_sha256}, current={state.state_sha256}"
        )
    return "matched_existing"


def file_sha256(path: Path) -> str:
    """Hash one lineage artifact without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_from_payload(payload: dict[str, Any]) -> FoldPreprocessingState:
    required = {
        "schema_version",
        "fold_id",
        "snapshot_sha256",
        "config_sha256",
        "spatial_audit_sha256",
        "all_window_id_sha256",
        "train_window_id_sha256",
        "role_assignment_sha256",
        "row_count",
        "train_row_count",
        "role_counts",
        "feature_groups",
        "standardized_groups",
        "group_contracts",
        "statistics",
        "fit_contract",
        "state_sha256",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"fold preprocessing state missing fields={missing}")
    if payload["schema_version"] != PREPROCESSING_SCHEMA_VERSION:
        raise ValueError(
            "fold preprocessing schema mismatch: "
            f"{payload['schema_version']}"
        )
    semantic = {key: value for key, value in payload.items() if key != "state_sha256"}
    observed_digest = str(payload["state_sha256"])
    expected_digest = _payload_sha256(semantic)
    if observed_digest != expected_digest:
        raise ValueError(
            "fold preprocessing state hash mismatch: "
            f"expected={expected_digest}, observed={observed_digest}"
        )
    return FoldPreprocessingState(
        fold_id=str(payload["fold_id"]),
        snapshot_sha256=str(payload["snapshot_sha256"]),
        config_sha256=str(payload["config_sha256"]),
        spatial_audit_sha256=str(payload["spatial_audit_sha256"]),
        all_window_id_sha256=str(payload["all_window_id_sha256"]),
        train_window_id_sha256=str(payload["train_window_id_sha256"]),
        role_assignment_sha256=str(payload["role_assignment_sha256"]),
        row_count=int(payload["row_count"]),
        train_row_count=int(payload["train_row_count"]),
        role_counts={str(k): int(v) for k, v in payload["role_counts"].items()},
        feature_groups=tuple(str(value) for value in payload["feature_groups"]),
        standardized_groups=tuple(
            str(value) for value in payload["standardized_groups"]
        ),
        group_contracts=dict(payload["group_contracts"]),
        statistics=dict(payload["statistics"]),
        state_sha256=observed_digest,
    )


def _fit_group_statistics(
    group: str,
    selected: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    if selected.ndim != 2 or selected.shape[0] == 0:
        raise ValueError(f"no observed training slots for spatial group={group}")
    finite = np.isfinite(selected)
    finite_counts = finite.sum(axis=0)
    empty_dimensions = [
        feature_names[index]
        for index in np.flatnonzero(finite_counts == 0)
    ]
    if empty_dimensions:
        raise ValueError(
            f"no finite training values for {group} features={empty_dimensions}"
        )
    clean = np.where(finite, selected, np.nan)
    mean = np.nanmean(clean, axis=0)
    scale = np.nanstd(clean, axis=0)
    constant = ~np.isfinite(scale) | (scale <= 1e-8)
    scale = np.where(constant, 1.0, scale)
    return {
        "feature_names": list(feature_names),
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "finite_value_count": finite_counts.astype(int).tolist(),
        "imputed_nonfinite_count": (~finite).sum(axis=0).astype(int).tolist(),
        "constant_feature_names": [
            feature_names[index]
            for index in np.flatnonzero(constant)
        ],
        "observed_slot_count": int(selected.shape[0]),
    }


def _validate_array_contract(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    missing = sorted({"length_mask", "observed_mask"}.difference(arrays))
    if missing:
        raise ValueError(f"preprocessing arrays missing masks={missing}")
    length_mask = np.asarray(arrays["length_mask"])
    observed_mask = np.asarray(arrays["observed_mask"])
    if length_mask.ndim != 2 or observed_mask.shape != length_mask.shape:
        raise ValueError(
            "preprocessing mask contract failed: "
            f"length={length_mask.shape}, observed={observed_mask.shape}"
        )
    if length_mask.shape[0] != expected_rows:
        raise ValueError(
            "preprocessing mask row mismatch: "
            f"observed={length_mask.shape[0]}, expected={expected_rows}"
        )
    if (observed_mask.astype(bool) & ~length_mask.astype(bool)).any():
        raise ValueError("preprocessing observed mask extends outside length mask")
    return length_mask, observed_mask


def _validate_group_array(
    group: str,
    values: np.ndarray,
    feature_names: Sequence[str],
    *,
    expected_rows: int,
    expected_slots: int,
) -> None:
    if values.ndim != 3:
        raise ValueError(f"spatial group {group} must have rank 3, got {values.shape}")
    if values.shape[:2] != (expected_rows, expected_slots):
        raise ValueError(
            f"spatial group {group} row/slot mismatch: {values.shape[:2]} "
            f"!= {(expected_rows, expected_slots)}"
        )
    if len(feature_names) != values.shape[-1] or len(set(feature_names)) != len(
        feature_names
    ):
        raise ValueError(
            f"spatial feature order mismatch for {group}: "
            f"names={list(feature_names)}, dimension={values.shape[-1]}"
        )
    if any(not name.strip() for name in feature_names):
        raise ValueError(f"blank spatial feature name in group={group}")


def _validate_window_ids(values: pd.Series) -> None:
    text = values.fillna("").astype(str).str.strip()
    blank = int(text.eq("").sum())
    duplicate = int(text.duplicated(keep=False).sum())
    if blank or duplicate:
        raise ValueError(
            "preprocessing window identity failed: "
            f"blank={blank}, duplicate_rows={duplicate}"
        )


def _require_nonblank_lineage(**values: str) -> None:
    blank = sorted(name for name, value in values.items() if not str(value).strip())
    if blank:
        raise ValueError(f"preprocessing lineage fields are blank={blank}")


def _ordered_unique(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    ordered = tuple(str(value).strip() for value in values)
    if any(not value for value in ordered) or len(set(ordered)) != len(ordered):
        raise ValueError(f"{name} must be nonblank and unique in declared order")
    return ordered


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _role_assignment_sha256(
    window_ids: pd.Series,
    roles: pd.Series,
    eligible: pd.Series,
) -> str:
    rows = (
        f"{window_id}\t{role}\t{int(is_eligible)}"
        for window_id, role, is_eligible in zip(
            window_ids.astype(str),
            roles.astype(str),
            eligible.astype(bool),
            strict=True,
        )
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    allowed = {"true", "1", "yes", "y", "t", "false", "0", "no", "n", "f"}
    invalid = ~normalized.isin(allowed)
    if invalid.any():
        raise ValueError(f"invalid boolean values={sorted(normalized[invalid].unique())}")
    return normalized.isin({"true", "1", "yes", "y", "t"})
