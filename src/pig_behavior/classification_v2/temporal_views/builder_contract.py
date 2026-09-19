"""Contract a temporal-view builder must satisfy, validated against fixtures.

This module does not build production windows. It states, in code, what a valid
window looks like so a fixture (or a future production builder) can be checked
against the canonical registry.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pig_behavior.classification_v2.temporal_views.registry import (
    TemporalViewSpec,
    temporal_view_spec,
)

WINDOW_CONTRACT_SCHEMA_VERSION = "classification_v2.temporal_window_contract.v1"


class WindowContractError(ValueError):
    """Raised when a candidate window violates its declared view semantics."""


@dataclass(frozen=True, slots=True)
class TemporalWindowSample:
    """One candidate window, expressed the way a builder would emit it.

    ``target_labels`` and ``history_labels`` are metadata used to *validate* the
    window. They are never model inputs; :func:`validate_window` asserts that no
    label field appears in ``model_input_fields``.
    """

    view_name: str
    actor_authority: str
    split_group_id: str
    endpoint_frame_index: int
    target_frame_indices: tuple[int, ...]
    target_timestamps_sec: tuple[float, ...]
    target_labels: tuple[str, ...]
    history_frame_indices: tuple[int, ...] = ()
    history_timestamps_sec: tuple[float, ...] = ()
    history_labels: tuple[str, ...] = ()
    target_actor_authority: tuple[str, ...] = ()
    target_split_group_ids: tuple[str, ...] = ()
    train_eligible: bool = True
    burst_start_frame_index: int | None = None
    available_history_frames: int | None = None
    model_input_fields: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def expected_frame_indices(
    spec: TemporalViewSpec,
    endpoint_frame_index: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(target_indices, history_indices)`` for one endpoint."""

    target = tuple(
        endpoint_frame_index + offset for offset in spec.target_offsets_from_endpoint
    )
    history = tuple(
        endpoint_frame_index + offset for offset in spec.history_offsets_from_endpoint
    )
    return target, history


def deterministic_window_id(sample: TemporalWindowSample) -> str:
    """Return a stable window id derived only from view identity and frames."""

    payload = "|".join(
        [
            WINDOW_CONTRACT_SCHEMA_VERSION,
            sample.view_name,
            sample.actor_authority,
            sample.split_group_id,
            ",".join(str(index) for index in sample.target_frame_indices),
            ",".join(str(index) for index in sample.history_frame_indices),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def validate_window(sample: TemporalWindowSample) -> dict[str, Any]:
    """Validate one candidate window and return a structured report."""

    spec = temporal_view_spec(sample.view_name)
    errors: list[str] = []
    warnings: list[str] = []

    expected_target, expected_history = expected_frame_indices(
        spec,
        sample.endpoint_frame_index,
    )
    if tuple(sample.target_frame_indices) != expected_target:
        errors.append(
            f"target frame offsets do not match {spec.name}: "
            f"expected={list(expected_target)} "
            f"observed={list(sample.target_frame_indices)}"
        )
    if tuple(sample.history_frame_indices) != expected_history:
        errors.append(
            f"history frame offsets do not match {spec.name}: "
            f"expected={list(expected_history)} "
            f"observed={list(sample.history_frame_indices)}"
        )

    if spec.legacy_burst_offsets is not None:
        if sample.burst_start_frame_index is None:
            errors.append(
                f"{spec.name} is a legacy burst view and requires "
                "burst_start_frame_index"
            )
        else:
            expected_burst = tuple(
                sample.burst_start_frame_index + offset
                for offset in spec.legacy_burst_offsets
            )
            if tuple(sample.target_frame_indices) != expected_burst:
                errors.append(
                    f"{spec.name} burst offsets must be "
                    f"{list(spec.legacy_burst_offsets)} inside the native burst; "
                    f"expected indices={list(expected_burst)} "
                    f"observed={list(sample.target_frame_indices)}"
                )

    errors.extend(_timestamp_errors(spec, sample))
    errors.extend(_future_frame_errors(sample))
    errors.extend(_label_errors(spec, sample))
    errors.extend(_authority_errors(sample))

    if not sample.train_eligible:
        errors.append("window is not train-eligible")

    if spec.history_length:
        available = sample.available_history_frames
        if available is not None and available < spec.history_length:
            warnings.append(
                f"insufficient causal history: available={available} "
                f"required={spec.history_length}; the builder must mask the "
                "missing history rather than pad it with valid-looking zeros"
            )
    forbidden_inputs = sorted(
        name
        for name in sample.model_input_fields
        if any(token in name.lower() for token in ("label", "behavior", "review"))
    )
    if forbidden_inputs:
        errors.append(
            f"label/review fields must never enter model X: {forbidden_inputs}"
        )

    return {
        "schema_version": WINDOW_CONTRACT_SCHEMA_VERSION,
        "view_name": spec.name,
        "window_id": deterministic_window_id(sample),
        "expected_target_frame_indices": list(expected_target),
        "expected_history_frame_indices": list(expected_history),
        "future_frame_dependence": 0 if not errors else None,
        "history_labels_in_x": False,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def require_window(sample: TemporalWindowSample) -> dict[str, Any]:
    """Return a clean report or raise one precise error."""

    report = validate_window(sample)
    if not report["valid"]:
        raise WindowContractError(
            f"temporal window contract failed for {sample.view_name}: "
            + "; ".join(report["errors"])
        )
    return report


def _timestamp_errors(
    spec: TemporalViewSpec,
    sample: TemporalWindowSample,
) -> list[str]:
    errors: list[str] = []
    for role, stamps, indices in (
        ("target", sample.target_timestamps_sec, sample.target_frame_indices),
        ("history", sample.history_timestamps_sec, sample.history_frame_indices),
    ):
        if not stamps:
            if indices:
                errors.append(f"{role} timestamps are required for {spec.name}")
            continue
        if len(stamps) != len(indices):
            errors.append(
                f"{role} timestamps ({len(stamps)}) must align with frame "
                f"indices ({len(indices)})"
            )
            continue
        ordered = all(
            earlier < later for earlier, later in zip(stamps, stamps[1:], strict=False)
        )
        if not ordered:
            errors.append(f"{role} timestamps must be strictly increasing")
    if (
        sample.history_timestamps_sec
        and sample.target_timestamps_sec
        and sample.history_timestamps_sec[-1] >= sample.target_timestamps_sec[0]
    ):
        errors.append(
            "history must end strictly before the target start in real elapsed "
            f"seconds: history_end={sample.history_timestamps_sec[-1]} "
            f"target_start={sample.target_timestamps_sec[0]}"
        )
    return errors


def _future_frame_errors(sample: TemporalWindowSample) -> list[str]:
    errors: list[str] = []
    future_target = [
        index
        for index in sample.target_frame_indices
        if index > sample.endpoint_frame_index
    ]
    future_history = [
        index
        for index in sample.history_frame_indices
        if index > sample.endpoint_frame_index
    ]
    if future_target:
        errors.append(f"target contains future frames={future_target}")
    if future_history:
        errors.append(
            "history contains frames occurring after the declared prediction "
            f"endpoint={future_history}"
        )
    if sample.history_frame_indices and sample.target_frame_indices:
        if max(sample.history_frame_indices) >= min(sample.target_frame_indices):
            errors.append(
                "history endpoint must be strictly before the target start: "
                f"history_end={max(sample.history_frame_indices)} "
                f"target_start={min(sample.target_frame_indices)}"
            )
    return errors


def _label_errors(
    spec: TemporalViewSpec,
    sample: TemporalWindowSample,
) -> list[str]:
    errors: list[str] = []
    labels = {str(label).strip() for label in sample.target_labels}
    if not labels or any(not label for label in labels):
        errors.append("target must carry one resolved behavior label")
    elif len(labels) != 1:
        errors.append(
            f"cross-label target window rejected for {spec.name}: "
            f"labels={sorted(labels)}"
        )
    return errors


def _authority_errors(sample: TemporalWindowSample) -> list[str]:
    errors: list[str] = []
    authorities = set(sample.target_actor_authority) or {sample.actor_authority}
    if len(authorities) != 1 or sample.actor_authority not in authorities:
        errors.append(
            "every target frame must belong to one stable actor authority: "
            f"declared={sample.actor_authority} observed={sorted(authorities)}"
        )
    groups = set(sample.target_split_group_ids) or {sample.split_group_id}
    if len(groups) != 1 or sample.split_group_id not in groups:
        errors.append(
            "every target frame must belong to one split group: "
            f"declared={sample.split_group_id} observed={sorted(groups)}"
        )
    return errors


def builder_contract_summary(view_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Serialize the builder requirements for the implementation document."""

    from pig_behavior.classification_v2.temporal_views.registry import (
        TEMPORAL_VIEW_NAMES,
    )

    names = tuple(view_names or TEMPORAL_VIEW_NAMES)
    return {
        "schema_version": WINDOW_CONTRACT_SCHEMA_VERSION,
        "views": [temporal_view_spec(name).to_payload() for name in names],
        "required_builder_behaviour": [
            "select exact frame indices for the declared view",
            "record selected offsets and real elapsed timestamps",
            "recompute motion pairs inside the view",
            "recompute ROI and partner transitions inside the view",
            "recompute temporal aggregates and availability masks",
            "assign one deterministic window id",
            "reject cross-label target windows",
            "mask insufficient causal history instead of padding it",
        ],
        "forbidden_builder_behaviour": [
            "import pair or aggregate features from another view",
            "truncate a longer aggregate to synthesize a shorter view",
            "allow a pair whose first endpoint is outside the view",
            "mix sparse and contiguous views in one primary population",
            "place any label or review field into model X",
        ],
        "production_datasets_built": False,
    }


__all__ = [
    "WINDOW_CONTRACT_SCHEMA_VERSION",
    "TemporalWindowSample",
    "WindowContractError",
    "builder_contract_summary",
    "deterministic_window_id",
    "expected_frame_indices",
    "require_window",
    "validate_window",
]
