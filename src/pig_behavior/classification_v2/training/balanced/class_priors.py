"""Training-fold class priors computed from native temporal units only.

The single rule enforced here is that class priors and any weight derived from
them may be fitted **only** on training-fold native temporal units. Validation,
test, all-data population, source-box counts, and uncorrected window rows are
rejected by name so a mistake is loud rather than silent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pig_behavior.classification_v2.schema import VALID_BEHAVIORS

CLASS_PRIOR_SCHEMA_VERSION = "classification_v2.balanced_class_priors.v1"

#: The only permitted fitting role.
TRAIN_FOLD_NATIVE_UNIT_ROLE = "TRAIN_FOLD_NATIVE_UNITS"

#: Roles that are explicitly rejected, with the reason reported to the caller.
FORBIDDEN_PRIOR_ROLES: dict[str, str] = {
    "VALIDATION": "validation folds must not inform training priors",
    "TEST": "test folds must not inform training priors",
    "ALL_DATA": "all-data population leaks evaluation folds into the priors",
    "SOURCE_BOX_COUNTS": (
        "source-box counts are a different imbalance unit and must not be used "
        "to tune the loss"
    ),
    "WINDOW_ROWS_WITHOUT_NATIVE_UNIT_CORRECTION": (
        "overlapping windows inflate frequent native units; correct to native "
        "temporal units first"
    ),
}


class ClassPriorError(ValueError):
    """Raised when class priors are fitted from a forbidden population."""


@dataclass(frozen=True, slots=True)
class ClassPriors:
    """Immutable training-fold priors over the canonical ten-class order."""

    class_order: tuple[str, ...]
    counts: tuple[float, ...]
    frequencies: tuple[float, ...]
    role: str
    native_unit_count: int
    fold_id: str
    state_sha256: str

    @property
    def num_classes(self) -> int:
        return len(self.class_order)

    def counts_array(self) -> np.ndarray:
        return np.asarray(self.counts, dtype=np.float64)

    def log_prior(self) -> np.ndarray:
        return np.log(np.asarray(self.frequencies, dtype=np.float64))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CLASS_PRIOR_SCHEMA_VERSION,
            "class_order": list(self.class_order),
            "counts": list(self.counts),
            "frequencies": list(self.frequencies),
            "role": self.role,
            "native_unit_count": self.native_unit_count,
            "fold_id": self.fold_id,
            "fit_contract": {
                "train_fold_native_units_only": True,
                "validation_excluded": True,
                "test_excluded": True,
                "source_box_counts_excluded": True,
                "uncorrected_window_rows_excluded": True,
            },
            "state_sha256": self.state_sha256,
        }


def compute_class_priors(
    *,
    native_unit_ids: Sequence[str],
    native_unit_labels: Sequence[int],
    role: str = TRAIN_FOLD_NATIVE_UNIT_ROLE,
    fold_id: str = "unspecified",
    class_order: Sequence[str] = tuple(VALID_BEHAVIORS),
) -> ClassPriors:
    """Fit priors over distinct training-fold native temporal units.

    One native unit contributes exactly one count. Duplicated native-unit ids
    are rejected, which is what prevents overlapping windows from being counted
    as independent evidence.
    """

    if role != TRAIN_FOLD_NATIVE_UNIT_ROLE:
        reason = FORBIDDEN_PRIOR_ROLES.get(role, "unknown class-prior role")
        raise ClassPriorError(
            f"class priors may only be fitted with role="
            f"{TRAIN_FOLD_NATIVE_UNIT_ROLE}; requested role={role} ({reason})"
        )
    order = tuple(str(name) for name in class_order)
    if len(set(order)) != len(order) or not order:
        raise ClassPriorError(f"invalid class order={list(order)}")
    ids = [str(value).strip() for value in native_unit_ids]
    labels = np.asarray(native_unit_labels, dtype=np.int64)
    if len(ids) != labels.shape[0]:
        raise ClassPriorError(
            f"native_unit_ids ({len(ids)}) and native_unit_labels "
            f"({labels.shape[0]}) must be aligned"
        )
    if not ids:
        raise ClassPriorError("class priors need at least one native unit")
    if any(not value for value in ids):
        raise ClassPriorError("native_unit_ids must not contain blank ids")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ClassPriorError(
            "duplicated native_unit_ids would double-count overlapping windows: "
            f"{duplicates[:8]}"
        )
    if labels.min() < 0 or labels.max() >= len(order):
        raise ClassPriorError(
            f"native unit labels must be in [0,{len(order)}); observed "
            f"[{int(labels.min())},{int(labels.max())}]"
        )
    counts = np.bincount(labels, minlength=len(order)).astype(np.float64)
    total = float(counts.sum())
    frequencies = counts / total
    payload = {
        "schema_version": CLASS_PRIOR_SCHEMA_VERSION,
        "class_order": list(order),
        "counts": counts.tolist(),
        "frequencies": frequencies.tolist(),
        "role": role,
        "native_unit_count": int(len(ids)),
        "fold_id": str(fold_id),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ClassPriors(
        class_order=order,
        counts=tuple(float(value) for value in counts),
        frequencies=tuple(float(value) for value in frequencies),
        role=role,
        native_unit_count=int(len(ids)),
        fold_id=str(fold_id),
        state_sha256=digest,
    )


def require_train_fold_priors(priors: ClassPriors) -> ClassPriors:
    """Reject any prior object that was not fitted on training native units."""

    if priors.role != TRAIN_FOLD_NATIVE_UNIT_ROLE:
        raise ClassPriorError(
            f"loss weights require role={TRAIN_FOLD_NATIVE_UNIT_ROLE}; "
            f"observed role={priors.role}"
        )
    if min(priors.counts) <= 0.0:
        empty = [
            name
            for name, count in zip(priors.class_order, priors.counts, strict=True)
            if count <= 0.0
        ]
        raise ClassPriorError(
            f"training fold has no native units for classes={empty}; a weight "
            "derived from a zero count is undefined"
        )
    return priors


__all__ = [
    "CLASS_PRIOR_SCHEMA_VERSION",
    "FORBIDDEN_PRIOR_ROLES",
    "TRAIN_FOLD_NATIVE_UNIT_ROLE",
    "ClassPriorError",
    "ClassPriors",
    "compute_class_priors",
    "require_train_fold_priors",
]
