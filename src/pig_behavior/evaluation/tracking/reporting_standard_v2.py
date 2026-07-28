"""Versioned report artifacts for Tracking Evaluator Standard V2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, fields
from typing import Any

import pandas as pd

from .config import TrackingEvaluationPipelineConfig
from .contracts import (
    EVALUATOR_CONTRACT_ID,
    HOTA_ALPHAS,
    IDENTITY_EPISODE_CONTRACT_ID,
    MATCHING_CONTRACT_ID,
)
from .evaluator_standard_v2 import StandardV2Evaluation
from .identity_episodes_v2 import (
    IdentityAuthority,
    IdentityErrorEpisode,
    MatchedIdentityRow,
    PairwiseIdentitySwapEvent,
)


def _format_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return str(value)
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_table(dataframe: pd.DataFrame, columns: list[str]) -> str:
    if dataframe.empty:
        return "_No rows._"
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for _, row in dataframe.loc[:, columns].iterrows():
        values = [_format_value(row[column]) for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def hota_alpha_dataframe(
    evaluations: Sequence[StandardV2Evaluation],
) -> pd.DataFrame:
    """Return one auditable row per sequence and alpha."""
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        result = evaluation.hota_result
        for index, alpha in enumerate(result.alphas):
            rows.append(
                {
                    "video_stem": evaluation.metrics.video_stem,
                    "alpha": alpha,
                    "hota": result.hota[index],
                    "deta": result.deta[index],
                    "assa": result.assa[index],
                    "loca": result.loca[index],
                    "tp": result.tp[index],
                    "fp": result.fp[index],
                    "fn": result.fn[index],
                }
            )
    return pd.DataFrame(rows)


def identity_authority_dataframe(
    evaluations: Sequence[StandardV2Evaluation],
) -> pd.DataFrame:
    """Return frozen prediction-to-GT authority rows."""
    rows = [
        asdict(authority)
        for evaluation in evaluations
        for authority in evaluation.episode_result.authorities
    ]
    return pd.DataFrame(
        rows,
        columns=[field.name for field in fields(IdentityAuthority)],
    )


def identity_episode_dataframe(
    evaluations: Sequence[StandardV2Evaluation],
) -> pd.DataFrame:
    """Return GT-primary identity-error episodes."""
    rows = [
        asdict(episode)
        for evaluation in evaluations
        for episode in evaluation.episode_result.episodes
    ]
    return pd.DataFrame(
        rows,
        columns=[field.name for field in fields(IdentityErrorEpisode)],
    )


def pairwise_swap_dataframe(
    evaluations: Sequence[StandardV2Evaluation],
) -> pd.DataFrame:
    """Return reciprocal pair events without duplicating GT-primary rows."""
    rows = [
        asdict(event)
        for evaluation in evaluations
        for event in evaluation.episode_result.pairwise_events
    ]
    return pd.DataFrame(
        rows,
        columns=[field.name for field in fields(PairwiseIdentitySwapEvent)],
    )


def identity_ambiguity_dataframe(
    evaluations: Sequence[StandardV2Evaluation],
) -> pd.DataFrame:
    """Return unresolved identity rows retained outside authoritative ranking."""
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        for ambiguity in evaluation.episode_result.ambiguous_rows:
            rows.append(
                {
                    **asdict(ambiguity.row),
                    "ambiguity_reason": ambiguity.reason,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            *(field.name for field in fields(MatchedIdentityRow)),
            "ambiguity_reason",
        ],
    )


def build_markdown_report_standard_v2(
    assets_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config: TrackingEvaluationPipelineConfig,
) -> str:
    """Build a compact report that cannot be mistaken for Legacy V1."""
    aggregate = metrics_df.loc[metrics_df["video_stem"] == "ALL"]
    per_video = metrics_df.loc[metrics_df["video_stem"] != "ALL"]
    metric_columns = [
        "video_stem",
        "hota_pct",
        "deta_pct",
        "assa_pct",
        "loca_pct",
        "idf1_pct",
        "id_precision_pct",
        "id_recall_pct",
        "idsw_standard",
        "fp",
        "fn",
        "fragments",
        "wrong_id_matched_frames",
        "terminal_identity_error_episode_count",
        "persistent_pairwise_identity_swap_count",
    ]
    available_columns = [
        column for column in metric_columns if column in metrics_df.columns
    ]
    lines = [
        "# Tracking Evaluation — Standard V2",
        "",
        f"- Evaluator contract: `{EVALUATOR_CONTRACT_ID}`",
        f"- Matching contract: `{MATCHING_CONTRACT_ID}`",
        f"- Identity episode contract: `{IDENTITY_EPISODE_CONTRACT_ID}`",
        f"- HOTA alphas: `{list(HOTA_ALPHAS)}`",
        f"- Include Hidden: `{config.include_hidden}`",
        "- Legacy HOTA and legacy swap fields are not present.",
        "",
        "## Dataset-authoritative aggregate",
        "",
        _markdown_table(aggregate, available_columns),
        "",
        "## Per-video metrics",
        "",
        _markdown_table(per_video, available_columns),
        "",
        "## Evaluated assets",
        "",
        _markdown_table(assets_df, list(assets_df.columns))
        if not assets_df.empty
        else "_No assets._",
        "",
        "Per-video means or medians are descriptive only. The `ALL` row combines",
        "HOTA sufficient statistics before alpha averaging and recomputes",
        "identity ratios from sequence-local count totals.",
    ]
    return "\n".join(lines) + "\n"
