"""Sequentially compose completed Behavior-review decision layers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

TERMINAL_DECISIONS = frozenset({"accept", "corrected", "exclude"})
DECISION_REQUIRED_COLUMNS = (
    "review_unit_id",
    "temporal_unit_key",
    "behavior_label",
    "manual_review_decision",
    "manual_corrected_behavior",
)
QUALITY_REQUIRED_COLUMNS = (
    "review_unit_id",
    "original_behavior",
    "reviewed_behavior",
    "source_label_error_confirmed",
)
SOURCE_REQUIRED_COLUMNS = (
    "review_unit_id",
    "temporal_unit_key",
    "behavior_label",
)
SCOPE_OUTCOME_COLUMNS = frozenset(
    {
        "manual_review_decision",
        "manual_corrected_behavior",
        "manual_label_strength",
        "manual_training_action",
        "manual_sample_weight",
        "manual_note",
        "human_decision_synthesized",
    }
)


class CompositeReviewContractError(ValueError):
    """Raised when sequential review lineage cannot be composed safely."""


@dataclass(frozen=True, slots=True)
class ReviewLayer:
    """One completed review layer in chronological application order."""

    name: str
    decisions: pd.DataFrame
    quality: pd.DataFrame


def compose_behavior_review_layers(
    source_units: pd.DataFrame,
    layers: Sequence[ReviewLayer],
) -> dict[str, Any]:
    """Compose review layers relative to immutable original source labels."""
    _require_columns(source_units, SOURCE_REQUIRED_COLUMNS, "source_units")
    if not layers:
        raise CompositeReviewContractError("review_layers_empty")

    source = source_units.copy()
    source["temporal_unit_key"] = _normalized(source["temporal_unit_key"])
    source["review_unit_id"] = _normalized(source["review_unit_id"])
    source["behavior_label"] = _normalized(source["behavior_label"])
    if source["temporal_unit_key"].eq("").any():
        raise CompositeReviewContractError("source_temporal_unit_key_blank")
    if source["temporal_unit_key"].duplicated().any():
        raise CompositeReviewContractError("source_temporal_unit_key_duplicate")

    source_by_key = source.set_index("temporal_unit_key", drop=False)
    state: dict[str, str] = {}
    original: dict[str, str] = {}
    excluded: set[str] = set()
    ever_changed_from_source: set[str] = set()
    last_decision: dict[str, dict[str, str]] = {}
    last_quality: dict[str, dict[str, str]] = {}
    layer_names_by_key: dict[str, list[str]] = {}
    lineage_rows: list[dict[str, Any]] = []
    layer_audits: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for layer_order, layer in enumerate(layers, start=1):
        decisions, quality = _validated_layer(layer)
        overlap = int(decisions["temporal_unit_key"].isin(seen_keys).sum())
        quality_by_id = quality.set_index("review_unit_id", drop=False)

        for row in decisions.to_dict(orient="records"):
            key = row["temporal_unit_key"]
            if key not in source_by_key.index:
                raise CompositeReviewContractError(
                    f"layer_key_missing_from_source={layer.name}:{key}"
                )
            source_label = _text(source_by_key.at[key, "behavior_label"])
            incoming = _text(row["behavior_label"])
            prior = state.get(key, source_label)
            if incoming != prior:
                raise CompositeReviewContractError(
                    "layer_input_behavior_mismatch="
                    f"{layer.name}:{key}:{incoming}:{prior}"
                )

            decision = _text(row["manual_review_decision"]).casefold()
            corrected = _text(row["manual_corrected_behavior"])
            if decision == "corrected":
                output = corrected
            elif decision == "exclude":
                output = ""
            else:
                output = prior

            review_id = _text(row["review_unit_id"])
            quality_row = {
                column: _text(quality_by_id.at[review_id, column])
                for column in quality.columns
            }
            if quality_row["original_behavior"] != incoming:
                raise CompositeReviewContractError(
                    f"quality_input_behavior_mismatch={layer.name}:{key}"
                )
            if decision != "exclude" and quality_row["reviewed_behavior"] != output:
                raise CompositeReviewContractError(
                    f"quality_output_behavior_mismatch={layer.name}:{key}"
                )

            original.setdefault(key, source_label)
            state[key] = output
            if output and output != source_label:
                ever_changed_from_source.add(key)
            if decision == "exclude":
                excluded.add(key)
            else:
                excluded.discard(key)
            last_decision[key] = {column: _text(value) for column, value in row.items()}
            last_quality[key] = quality_row
            layer_names_by_key.setdefault(key, []).append(layer.name)
            lineage_rows.append(
                {
                    "temporal_unit_key": key,
                    "layer_order": layer_order,
                    "layer_name": layer.name,
                    "layer_review_unit_id": review_id,
                    "layer_decision": decision,
                    "layer_input_behavior": incoming,
                    "layer_output_behavior": output,
                    "layer_changed_behavior": incoming != output,
                }
            )

        seen_keys.update(decisions["temporal_unit_key"])
        layer_audits.append(
            {
                "layer_order": layer_order,
                "layer_name": layer.name,
                "decision_rows": int(len(decisions)),
                "quality_rows": int(len(quality)),
                "overlap_with_prior_layers": overlap,
                "unique_keys_after_layer": int(len(seen_keys)),
            }
        )

    ordered_keys = sorted(
        state,
        key=lambda key: _source_sort_key(source_by_key.loc[key]),
    )
    decision_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for index, key in enumerate(ordered_keys, start=1):
        source_row = source_by_key.loc[key]
        source_label = original[key]
        final_label = state[key]
        latest = last_decision[key]
        latest_quality = last_quality[key]
        is_excluded = key in excluded
        changed = not is_excluded and final_label != source_label
        composite_decision = "exclude" if is_excluded else "corrected" if changed else "accept"
        decision_rows.append(
            {
                "review_item_id": f"composite_review_{index:07d}",
                "review_unit_id": _text(source_row["review_unit_id"]),
                "review_unit_type": _text(source_row.get("review_unit_type", "")),
                "temporal_unit_key": key,
                "source_type": _text(source_row.get("source_type", "")),
                "video_key": _text(source_row.get("video_key", "")),
                "track_id": _text(source_row.get("track_id", "")),
                "unit_start_frame": _text(source_row.get("unit_start_frame", "")),
                "display_frame_indices": _text(
                    source_row.get("display_frame_indices", "")
                ),
                "behavior_label": source_label,
                "review_reason": "SEQUENTIAL_COMPOSITE_REVIEW_AUTHORITY",
                "manual_review_decision": composite_decision,
                "manual_corrected_behavior": final_label if changed else "",
                "manual_label_strength": latest.get("manual_label_strength", ""),
                "manual_sample_weight": latest.get("manual_sample_weight", ""),
                "manual_note": latest.get("manual_note", ""),
            }
        )
        quality_rows.append(
            {
                "review_unit_id": _text(source_row["review_unit_id"]),
                "original_behavior": source_label,
                "reviewed_behavior": final_label if not is_excluded else source_label,
                "label_status": (
                    "TECHNICAL_DEFECT"
                    if is_excluded
                    else "SOURCE_LABEL_ERROR_CONFIRMED"
                    if changed
                    else "SUPPORTED"
                ),
                "source_label_error_confirmed": "YES" if changed else "NO",
                "error_pattern": latest_quality.get("error_pattern", ""),
                "review_confidence": latest_quality.get("review_confidence", ""),
                "selection_assessment": latest_quality.get(
                    "selection_assessment", ""
                ),
                "composite_layer_names": "+".join(layer_names_by_key[key]),
            }
        )

    scope_columns = [
        column for column in source.columns if column not in SCOPE_OUTCOME_COLUMNS
    ]
    scope = source.loc[
        source["temporal_unit_key"].isin(ordered_keys), scope_columns
    ].copy()
    scope["_composite_order"] = scope["temporal_unit_key"].map(
        {key: index for index, key in enumerate(ordered_keys)}
    )
    scope = scope.sort_values("_composite_order").drop(columns="_composite_order")

    decisions_out = pd.DataFrame(decision_rows)
    quality_out = pd.DataFrame(quality_rows)
    lineage_out = pd.DataFrame(lineage_rows)
    audit = {
        "source_unit_rows": int(len(source)),
        "composite_reviewed_keys": int(len(ordered_keys)),
        "composite_corrected_from_source": int(
            decisions_out["manual_review_decision"].eq("corrected").sum()
        ),
        "ever_changed_from_source": int(len(ever_changed_from_source)),
        "reverted_to_source_by_later_review": int(
            len(ever_changed_from_source)
            - decisions_out["manual_review_decision"].eq("corrected").sum()
        ),
        "composite_excluded": int(
            decisions_out["manual_review_decision"].eq("exclude").sum()
        ),
        "layer_audits": layer_audits,
        "input_label_mismatch_count": 0,
        "quality_semantic_mismatch_count": 0,
    }
    return {
        "scope": scope,
        "decisions": decisions_out,
        "quality": quality_out,
        "lineage": lineage_out,
        "audit": audit,
    }


def _validated_layer(layer: ReviewLayer) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _text(layer.name):
        raise CompositeReviewContractError("review_layer_name_blank")
    _require_columns(layer.decisions, DECISION_REQUIRED_COLUMNS, layer.name)
    _require_columns(layer.quality, QUALITY_REQUIRED_COLUMNS, f"{layer.name}_quality")
    decisions = layer.decisions.copy()
    quality = layer.quality.copy()
    for column in DECISION_REQUIRED_COLUMNS:
        decisions[column] = _normalized(decisions[column])
    for column in QUALITY_REQUIRED_COLUMNS:
        quality[column] = _normalized(quality[column])
    if decisions["review_unit_id"].duplicated().any():
        raise CompositeReviewContractError(f"layer_decision_duplicate={layer.name}")
    if decisions["temporal_unit_key"].duplicated().any():
        raise CompositeReviewContractError(f"layer_temporal_duplicate={layer.name}")
    if quality["review_unit_id"].duplicated().any():
        raise CompositeReviewContractError(f"layer_quality_duplicate={layer.name}")
    if set(decisions["review_unit_id"]) != set(quality["review_unit_id"]):
        raise CompositeReviewContractError(f"layer_quality_coverage={layer.name}")
    invalid = sorted(
        set(decisions["manual_review_decision"].str.casefold()) - TERMINAL_DECISIONS
    )
    if invalid:
        raise CompositeReviewContractError(
            f"layer_terminal_decisions={layer.name}:{','.join(invalid)}"
        )
    corrected = decisions["manual_review_decision"].str.casefold().eq("corrected")
    if decisions.loc[corrected, "manual_corrected_behavior"].eq("").any():
        raise CompositeReviewContractError(f"layer_correction_blank={layer.name}")
    if decisions.loc[~corrected, "manual_corrected_behavior"].ne("").any():
        raise CompositeReviewContractError(f"layer_unexpected_correction={layer.name}")
    return decisions, quality


def _source_sort_key(row: pd.Series) -> tuple[str, str, int, int, str]:
    return (
        _text(row.get("video_key", "")),
        _text(row.get("object_track_key", row.get("track_id", ""))),
        _integer(row.get("unit_start_frame", 0)),
        _integer(row.get("unit_end_frame", 0)),
        _text(row.get("temporal_unit_key", "")),
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise CompositeReviewContractError(
            f"{label}_missing_columns={','.join(missing)}"
        )


def _normalized(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _integer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
