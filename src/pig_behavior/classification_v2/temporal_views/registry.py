"""Canonical temporal-view registry with exact, non-negotiable names.

Offsets are expressed **relative to the prediction endpoint**, so the endpoint
always has offset ``0`` and every other frame is strictly negative. That makes
"no future frame" a property the registry itself can state.

Legacy-only views additionally record their ``legacy_burst_offsets`` inside a
native 16-frame burst, because ``S6_AT_16_SPARSE`` and ``HISTORICAL_C6_SCREEN``
are only distinguishable at burst level: relative to their own endpoints the
historical screen looks contiguous, but it is a different sampling of a
different part of the burst.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TEMPORAL_VIEW_SCHEMA_VERSION = "classification_v2.temporal_view_registry.v1"

TARGET_CONTIGUOUS_VIEWS: tuple[str, ...] = (
    "T6_TARGET_CONTIGUOUS",
    "T8_TARGET_CONTIGUOUS",
    "T12_TARGET_CONTIGUOUS",
    "T16_TARGET_CONTIGUOUS",
)

CAUSAL_HISTORY_VIEWS: tuple[str, ...] = (
    "T6_TARGET_PLUS_H5",
    "T6_TARGET_PLUS_H6",
    "T6_TARGET_PLUS_H12",
    "T6_TARGET_PLUS_H24",
)

LEGACY_ONLY_VIEWS: tuple[str, ...] = (
    "S6_AT_16_SPARSE",
    "HISTORICAL_C6_SCREEN",
)

TEMPORAL_VIEW_NAMES: tuple[str, ...] = (
    *TARGET_CONTIGUOUS_VIEWS,
    *CAUSAL_HISTORY_VIEWS,
    *LEGACY_ONLY_VIEWS,
)

#: Exact offsets inside one native legacy 16-frame burst.
S6_AT_16_BURST_OFFSETS: tuple[int, ...] = (0, 3, 6, 9, 12, 15)
S6_AT_16_PAIR_DELTAS: tuple[int, ...] = (3, 3, 3, 3, 3)
HISTORICAL_C6_BURST_OFFSETS: tuple[int, ...] = (5, 6, 7, 8, 9, 10)
HISTORICAL_C6_PAIR_DELTAS: tuple[int, ...] = (1, 1, 1, 1, 1)

VIEW_FAMILIES: tuple[str, ...] = (
    "TARGET_CONTIGUOUS",
    "TARGET_PLUS_CAUSAL_HISTORY",
    "LEGACY_SPARSE_ABLATION",
    "HISTORICAL_SCREEN",
)


@dataclass(frozen=True, slots=True)
class TemporalViewSpec:
    """Exact semantics of one canonical temporal view."""

    name: str
    family: str
    target_length: int
    history_length: int
    target_offsets_from_endpoint: tuple[int, ...]
    history_offsets_from_endpoint: tuple[int, ...]
    pair_deltas: tuple[int, ...]
    contiguous_target: bool
    causal: bool
    legacy_only: bool
    primary_cross_source_eligible: bool
    metrics_transferable: bool
    uses_real_elapsed_seconds: bool
    legacy_burst_offsets: tuple[int, ...] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.family not in VIEW_FAMILIES:
            raise ValueError(f"unknown temporal view family={self.family}")
        if len(self.target_offsets_from_endpoint) != self.target_length:
            raise ValueError(f"{self.name}: target offsets must match target_length")
        if self.target_offsets_from_endpoint[-1] != 0:
            raise ValueError(f"{self.name}: target must end at the prediction endpoint")
        if any(offset > 0 for offset in self.target_offsets_from_endpoint):
            raise ValueError(f"{self.name}: target contains a future frame")
        if any(offset > 0 for offset in self.history_offsets_from_endpoint):
            raise ValueError(f"{self.name}: history contains a future frame")
        if len(self.history_offsets_from_endpoint) != self.history_length:
            raise ValueError(f"{self.name}: history offsets must match history_length")
        if self.history_offsets_from_endpoint:
            latest_history = max(self.history_offsets_from_endpoint)
            earliest_target = min(self.target_offsets_from_endpoint)
            if latest_history >= earliest_target:
                raise ValueError(
                    f"{self.name}: history must end strictly before the target start"
                )

    @property
    def total_length(self) -> int:
        return self.target_length + self.history_length

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "target_length": self.target_length,
            "history_length": self.history_length,
            "target_offsets_from_endpoint": list(self.target_offsets_from_endpoint),
            "history_offsets_from_endpoint": list(self.history_offsets_from_endpoint),
            "pair_deltas": list(self.pair_deltas),
            "contiguous_target": self.contiguous_target,
            "causal": self.causal,
            "future_frame_dependence": 0,
            "legacy_only": self.legacy_only,
            "legacy_burst_offsets": (
                None
                if self.legacy_burst_offsets is None
                else list(self.legacy_burst_offsets)
            ),
            "primary_cross_source_eligible": self.primary_cross_source_eligible,
            "metrics_transferable": self.metrics_transferable,
            "uses_real_elapsed_seconds": self.uses_real_elapsed_seconds,
            "label_source": "target_only",
            "note": self.note,
        }


def _contiguous_target_spec(length: int) -> TemporalViewSpec:
    offsets = tuple(range(-(length - 1), 1))
    return TemporalViewSpec(
        name=f"T{length}_TARGET_CONTIGUOUS",
        family="TARGET_CONTIGUOUS",
        target_length=length,
        history_length=0,
        target_offsets_from_endpoint=offsets,
        history_offsets_from_endpoint=(),
        pair_deltas=tuple([1] * (length - 1)),
        contiguous_target=True,
        causal=True,
        legacy_only=False,
        primary_cross_source_eligible=True,
        metrics_transferable=True,
        uses_real_elapsed_seconds=True,
        note="trailing causal target window ending at prediction time",
    )


def _history_spec(history_length: int) -> TemporalViewSpec:
    target_offsets = tuple(range(-5, 1))
    history_offsets = tuple(range(-5 - history_length, -5))
    return TemporalViewSpec(
        name=f"T6_TARGET_PLUS_H{history_length}",
        family="TARGET_PLUS_CAUSAL_HISTORY",
        target_length=6,
        history_length=history_length,
        target_offsets_from_endpoint=target_offsets,
        history_offsets_from_endpoint=history_offsets,
        pair_deltas=tuple([1] * 5),
        contiguous_target=True,
        causal=True,
        legacy_only=False,
        primary_cross_source_eligible=history_length == 5,
        metrics_transferable=True,
        uses_real_elapsed_seconds=True,
        note=(
            "label comes only from the T6 target; history may carry a different "
            "earlier behavior and its labels never enter model X"
        ),
    )


_REGISTRY: dict[str, TemporalViewSpec] = {}
for _length in (6, 8, 12, 16):
    _spec = _contiguous_target_spec(_length)
    _REGISTRY[_spec.name] = _spec
for _history in (5, 6, 12, 24):
    _spec = _history_spec(_history)
    _REGISTRY[_spec.name] = _spec

_REGISTRY["S6_AT_16_SPARSE"] = TemporalViewSpec(
    name="S6_AT_16_SPARSE",
    family="LEGACY_SPARSE_ABLATION",
    target_length=6,
    history_length=0,
    target_offsets_from_endpoint=tuple(
        offset - S6_AT_16_BURST_OFFSETS[-1] for offset in S6_AT_16_BURST_OFFSETS
    ),
    history_offsets_from_endpoint=(),
    pair_deltas=S6_AT_16_PAIR_DELTAS,
    contiguous_target=False,
    causal=True,
    legacy_only=True,
    primary_cross_source_eligible=False,
    metrics_transferable=True,
    uses_real_elapsed_seconds=True,
    legacy_burst_offsets=S6_AT_16_BURST_OFFSETS,
    note=(
        "legacy-only sparse ablation inside one 16-frame burst; never call this "
        "contiguous T6 and never make it the primary cross-source view"
    ),
)
_REGISTRY["HISTORICAL_C6_SCREEN"] = TemporalViewSpec(
    name="HISTORICAL_C6_SCREEN",
    family="HISTORICAL_SCREEN",
    target_length=6,
    history_length=0,
    target_offsets_from_endpoint=tuple(
        offset - HISTORICAL_C6_BURST_OFFSETS[-1]
        for offset in HISTORICAL_C6_BURST_OFFSETS
    ),
    history_offsets_from_endpoint=(),
    pair_deltas=HISTORICAL_C6_PAIR_DELTAS,
    contiguous_target=True,
    causal=True,
    legacy_only=True,
    primary_cross_source_eligible=False,
    metrics_transferable=False,
    uses_real_elapsed_seconds=True,
    legacy_burst_offsets=HISTORICAL_C6_BURST_OFFSETS,
    note=(
        "historical legacy middle-six screening offsets [5,6,7,8,9,10]; neither "
        "its metrics nor its name transfer to the mixed reviewed lineage"
    ),
)


def temporal_view_spec(name: str) -> TemporalViewSpec:
    """Return one canonical view spec or reject an unknown/ambiguous name."""

    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown temporal view={name}; canonical names are "
            f"{list(TEMPORAL_VIEW_NAMES)}"
        ) from exc


def views_are_distinct(first: str, second: str) -> bool:
    """Return whether two registered views have different sampling identity."""

    left = temporal_view_spec(first)
    right = temporal_view_spec(second)
    return (
        left.name != right.name
        or left.legacy_burst_offsets != right.legacy_burst_offsets
        or left.target_offsets_from_endpoint != right.target_offsets_from_endpoint
        or left.pair_deltas != right.pair_deltas
    )


def temporal_view_registry_contract() -> dict[str, Any]:
    """Serialize the registry for configs, manifests and metric tables."""

    return {
        "schema_version": TEMPORAL_VIEW_SCHEMA_VERSION,
        "view_names": list(TEMPORAL_VIEW_NAMES),
        "target_view_ablation": list(TARGET_CONTIGUOUS_VIEWS),
        "causal_history_ablation": list(CAUSAL_HISTORY_VIEWS),
        "legacy_only_views": list(LEGACY_ONLY_VIEWS),
        "s6_at_16_offsets": list(S6_AT_16_BURST_OFFSETS),
        "s6_at_16_pair_deltas": list(S6_AT_16_PAIR_DELTAS),
        "historical_c6_offsets": list(HISTORICAL_C6_BURST_OFFSETS),
        "historical_c6_metrics_transferred": False,
        "historical_c6_name_transferred": False,
        "ambiguous_6c_name_used": False,
        "cross_length_families_separated": True,
        "future_frame_dependence": 0,
        "views": [_REGISTRY[name].to_payload() for name in TEMPORAL_VIEW_NAMES],
    }


__all__ = [
    "CAUSAL_HISTORY_VIEWS",
    "HISTORICAL_C6_BURST_OFFSETS",
    "HISTORICAL_C6_PAIR_DELTAS",
    "LEGACY_ONLY_VIEWS",
    "S6_AT_16_BURST_OFFSETS",
    "S6_AT_16_PAIR_DELTAS",
    "TARGET_CONTIGUOUS_VIEWS",
    "TEMPORAL_VIEW_NAMES",
    "TEMPORAL_VIEW_SCHEMA_VERSION",
    "VIEW_FAMILIES",
    "TemporalViewSpec",
    "temporal_view_registry_contract",
    "temporal_view_spec",
    "views_are_distinct",
]
