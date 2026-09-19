"""Deterministic complete-unit scopes for short classification_v2 checks.

The selector is only for engineering smoke runs. It never changes labels and
never defines training splits. CVAT blocks retain every actor in one six-frame
scene interval; legacy blocks retain every tracklet in one recovered burst.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pig_behavior.classification_v2.contracts.lineage_claims import (
    attach_optional_lineage_claims,
    configured_lineage_claims,
    resolve_optional_lineage_claims,
)

SUPPORTED_SOURCES: tuple[str, ...] = (
    "cvat_tracking_xml",
    "legacy_recovered",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "source_type",
    "dataset_id",
    "video_key",
    "frame_index",
    "pig_id",
    "track_id",
    "behavior",
)


@dataclass(frozen=True, slots=True)
class TemporalSmokeScopeConfig:
    """Control bounded source-balanced smoke selection."""

    blocks_per_source: int = 4
    cvat_label_stride: int = 6
    legacy_expected_sequence_length: int = 16
    required_sources: tuple[str, ...] = SUPPORTED_SOURCES
    lineage_scope: str | None = None
    human_review_complete: bool | None = None

    def validate(self) -> None:
        """Reject settings that could create empty or ambiguous units."""

        if self.blocks_per_source <= 0:
            raise ValueError("blocks_per_source must be > 0")
        if self.cvat_label_stride <= 0:
            raise ValueError("cvat_label_stride must be > 0")
        if self.legacy_expected_sequence_length <= 0:
            raise ValueError("legacy_expected_sequence_length must be > 0")
        if not self.required_sources:
            raise ValueError("required_sources must not be empty")
        if len(set(self.required_sources)) != len(self.required_sources):
            raise ValueError("required_sources must not contain duplicates")
        unknown = sorted(set(self.required_sources).difference(SUPPORTED_SOURCES))
        if unknown:
            raise ValueError(f"unsupported required_sources={unknown}")
        configured_lineage_claims(
            self.lineage_scope,
            self.human_review_complete,
            artifact_name="temporal scope configuration",
        )


def select_temporal_smoke_scope(
    frame_features: pd.DataFrame,
    *,
    config: TemporalSmokeScopeConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select complete temporal scene blocks without truncating native units."""

    cfg = config or TemporalSmokeScopeConfig()
    cfg.validate()
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame_features.columns))
    if missing:
        raise ValueError(f"frame_features missing smoke columns: {missing}")
    if frame_features.empty:
        raise ValueError("frame_features must not be empty")

    input_claims = resolve_optional_lineage_claims(
        frame_features,
        artifact_name="temporal scope input",
    )
    configured_claims = configured_lineage_claims(
        cfg.lineage_scope,
        cfg.human_review_complete,
        artifact_name="temporal scope configuration",
    )
    if (
        input_claims is not None
        and configured_claims is not None
        and input_claims != configured_claims
    ):
        raise ValueError(
            "temporal scope configured claims conflict with input claims"
        )
    claims = configured_claims or input_claims

    work = frame_features.copy()
    work["_smoke_source_position"] = np.arange(len(work), dtype="int64")
    source = work["source_type"].fillna("").astype(str).str.strip()
    unknown_sources = sorted(set(source).difference(SUPPORTED_SOURCES))
    errors: list[str] = []
    warnings: list[str] = []
    if unknown_sources:
        errors.append(f"unsupported_sources={unknown_sources}")

    block_tables: list[pd.DataFrame] = []
    prepared_parts: list[pd.DataFrame] = []
    for source_type in cfg.required_sources:
        source_rows = work.loc[source.eq(source_type)].copy()
        if source_rows.empty:
            errors.append(f"missing_source={source_type}")
            continue
        prepared, blocks = _prepare_source_blocks(
            source_rows,
            source_type=source_type,
            config=cfg,
        )
        prepared_parts.append(prepared)
        block_tables.append(blocks)

    prepared = (
        pd.concat(prepared_parts, ignore_index=False, sort=False)
        if prepared_parts
        else work.iloc[0:0].copy()
    )
    blocks = (
        pd.concat(block_tables, ignore_index=True, sort=False)
        if block_tables
        else pd.DataFrame()
    )

    selected_keys: list[str] = []
    selected_by_source: dict[str, list[str]] = {}
    for source_type in cfg.required_sources:
        candidates = blocks.loc[
            blocks["source_type"].eq(source_type)
            & blocks["block_valid"]
        ].copy()
        chosen = _greedy_behavior_blocks(candidates, cfg.blocks_per_source)
        selected_by_source[source_type] = chosen
        selected_keys.extend(chosen)
        if not chosen:
            errors.append(f"no_complete_smoke_block={source_type}")
        elif len(chosen) < cfg.blocks_per_source:
            warnings.append(
                f"smoke_blocks_below_requested={source_type}:"
                f"{len(chosen)}/{cfg.blocks_per_source}"
            )

    selected_mask = prepared["_smoke_block_key"].isin(selected_keys)
    selected = prepared.loc[selected_mask].copy()
    selected = selected.sort_values(
        "_smoke_source_position",
        kind="mergesort",
    )
    helper_columns = [
        column for column in selected.columns if column.startswith("_smoke_")
    ]
    selected = selected.drop(columns=helper_columns).reset_index(drop=True)
    selected = attach_optional_lineage_claims(selected, claims)

    selected_blocks = blocks.loc[
        blocks["block_key"].isin(selected_keys)
    ].copy()
    invalid_blocks = (
        blocks.loc[~blocks["block_valid"]]
        if "block_valid" in blocks.columns
        else blocks.iloc[0:0]
    )
    if not selected_blocks.empty and not selected_blocks["block_valid"].all():
        errors.append("selected_incomplete_temporal_block")
    if len(selected) > len(frame_features):
        errors.append("smoke_scope_row_count_exceeds_input")

    audit = {
        "schema_version": "classification_v2_temporal_smoke_scope_v2",
        "input_rows": int(len(frame_features)),
        "selected_rows": int(len(selected)),
        "not_selected_rows": int(len(frame_features) - len(selected)),
        "input_source_counts": _counts(frame_features, "source_type"),
        "selected_source_counts": _counts(selected, "source_type"),
        "selected_behavior_counts": _counts(selected, "behavior"),
        "available_block_counts": _block_counts(blocks, "source_type"),
        "invalid_block_counts": _block_counts(
            invalid_blocks,
            "source_type",
        ),
        "selected_block_counts": _block_counts(
            selected_blocks,
            "source_type",
        ),
        "selected_native_unit_counts": _unit_counts(selected_blocks),
        "selected_multi_actor_block_counts": _multi_actor_counts(
            selected_blocks
        ),
        "selected_blocks": selected_by_source,
        "parameters": {
            "blocks_per_source": cfg.blocks_per_source,
            "cvat_label_stride": cfg.cvat_label_stride,
            "required_sources": list(cfg.required_sources),
            "legacy_expected_sequence_length": (
                cfg.legacy_expected_sequence_length
            ),
            "lineage_scope": (
                claims.lineage_scope if claims is not None else None
            ),
            "human_review_complete": (
                claims.human_review_complete if claims is not None else None
            ),
        },
        "errors": errors,
        "warnings": warnings,
    }
    if claims is not None:
        audit.update(claims.as_dict())
    return selected, audit


def _prepare_source_blocks(
    rows: pd.DataFrame,
    *,
    source_type: str,
    config: TemporalSmokeScopeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign source-specific block/unit keys and audit completeness."""

    out = rows.copy()
    frame_index = pd.to_numeric(out["frame_index"], errors="coerce")
    invalid_frame = (
        frame_index.isna()
        | frame_index.mod(1).ne(0)
        | frame_index.lt(0)
    )
    out["_smoke_frame_invalid"] = invalid_frame
    out["_smoke_frame_value"] = frame_index.round().astype("Int64")
    dataset = _text(out["dataset_id"])
    video = _text(out["video_key"])
    pig = _text(out["pig_id"])
    track = _text(out["track_id"])

    if source_type == "cvat_tracking_xml":
        anchor = (
            np.floor(frame_index / config.cvat_label_stride)
            * config.cvat_label_stride
        ).round().astype("Int64")
        out["_smoke_expected_start"] = anchor
        out["_smoke_expected_end"] = anchor + config.cvat_label_stride - 1
        out["_smoke_unit_frame"] = out["_smoke_frame_value"]
        out["_smoke_block_key"] = (
            source_type
            + "|"
            + dataset
            + "|"
            + video
            + "|anchor="
            + anchor.astype(str)
        )
        expected_length = config.cvat_label_stride
    else:
        if "clip_id" not in out.columns:
            raise ValueError("legacy smoke scope requires clip_id")
        if "relative_frame_index" not in out.columns:
            raise ValueError(
                "legacy smoke scope requires relative_frame_index"
            )
        clip = _text(out["clip_id"])
        relative = pd.to_numeric(
            out["relative_frame_index"],
            errors="coerce",
        )
        invalid_relative = (
            relative.isna()
            | relative.mod(1).ne(0)
            | relative.lt(0)
        )
        out["_smoke_frame_invalid"] |= invalid_relative | clip.eq("")
        out["_smoke_unit_frame"] = relative.round().astype("Int64")
        out["_smoke_expected_start"] = 0
        out["_smoke_expected_end"] = (
            config.legacy_expected_sequence_length - 1
        )
        out["_smoke_block_key"] = (
            source_type
            + "|"
            + dataset
            + "|"
            + video
            + "|clip="
            + clip
        )
        expected_length = config.legacy_expected_sequence_length

    out["_smoke_expected_length"] = expected_length

    out["_smoke_unit_key"] = (
        out["_smoke_block_key"]
        + "|pig="
        + pig
        + "|track="
        + track
    )
    missing_actor = pig.eq("") & track.eq("")
    out.loc[missing_actor, "_smoke_frame_invalid"] = True

    units = (
        out.groupby("_smoke_unit_key", dropna=False, sort=True)
        .agg(
            block_key=("_smoke_block_key", "first"),
            row_count=("_smoke_unit_frame", "size"),
            frame_count=("_smoke_unit_frame", "nunique"),
            min_frame=("_smoke_unit_frame", "min"),
            max_frame=("_smoke_unit_frame", "max"),
            invalid_frame=("_smoke_frame_invalid", "any"),
            expected_start=("_smoke_expected_start", "first"),
            expected_end=("_smoke_expected_end", "first"),
            expected_length=("_smoke_expected_length", "first"),
        )
        .reset_index()
        .rename(columns={"_smoke_unit_key": "unit_key"})
    )
    units["unit_valid"] = (
        ~units["invalid_frame"].fillna(True).astype(bool)
        & units["row_count"].eq(units["expected_length"])
        & units["frame_count"].eq(units["expected_length"])
        & units["min_frame"].eq(units["expected_start"])
        & units["max_frame"].eq(units["expected_end"])
    ).fillna(False)
    behavior_sets = (
        out.groupby("_smoke_block_key", dropna=False, sort=True)["behavior"]
        .agg(lambda values: tuple(sorted(set(_text(values)) - {""})))
        .rename("behavior_set")
    )
    block_units = units.groupby("block_key", dropna=False, sort=True).agg(
        unit_count=("unit_key", "size"),
        block_valid=("unit_valid", "all"),
    )
    block_rows = (
        out.groupby("_smoke_block_key", dropna=False, sort=True)
        .size()
        .rename("row_count")
    )
    blocks = pd.concat(
        [block_rows, block_units, behavior_sets],
        axis=1,
        join="outer",
    ).reset_index(names="block_key")
    blocks.insert(0, "source_type", source_type)
    blocks["unit_count"] = blocks["unit_count"].fillna(0).astype(int)
    blocks["block_valid"] = blocks["block_valid"].fillna(False).astype(bool)
    blocks["multi_actor"] = blocks["unit_count"].gt(1)
    return out, blocks


def _greedy_behavior_blocks(
    candidates: pd.DataFrame,
    count: int,
) -> list[str]:
    """Choose deterministic blocks that maximize behavior and social coverage."""

    remaining = candidates.sort_values("block_key", kind="mergesort").copy()
    selected: list[str] = []
    covered: set[str] = set()
    while len(selected) < count and not remaining.empty:
        scores = remaining.apply(
            lambda row: (
                len(set(row["behavior_set"]) - covered),
                int(bool(row["multi_actor"])),
                len(set(row["behavior_set"])),
                int(row["unit_count"]),
            ),
            axis=1,
        )
        best_index = max(scores.index, key=lambda index: scores.loc[index])
        best = remaining.loc[best_index]
        selected.append(str(best["block_key"]))
        covered.update(best["behavior_set"])
        remaining = remaining.drop(index=best_index)
    return selected


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].fillna("<NA>").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.sort_index().items()}


def _block_counts(blocks: pd.DataFrame, column: str) -> dict[str, int]:
    if blocks.empty or column not in blocks.columns:
        return {}
    return _counts(blocks, column)


def _unit_counts(blocks: pd.DataFrame) -> dict[str, int]:
    if blocks.empty:
        return {}
    totals = blocks.groupby("source_type", sort=True)["unit_count"].sum()
    return {str(key): int(value) for key, value in totals.items()}


def _multi_actor_counts(blocks: pd.DataFrame) -> dict[str, int]:
    if blocks.empty:
        return {}
    totals = blocks.groupby("source_type", sort=True)["multi_actor"].sum()
    return {str(key): int(value) for key, value in totals.items()}
