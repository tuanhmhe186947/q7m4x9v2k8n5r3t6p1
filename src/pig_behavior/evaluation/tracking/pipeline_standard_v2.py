"""Active reporting pipeline for Tracking Evaluator Standard V2."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_guard import assert_no_mp4_artifacts
from .assets import TrackingPair, video_metadata
from .config import TrackingEvaluationPipelineConfig
from .contracts import (
    EVALUATOR_CONTRACT_ID,
    HOTA_ALPHAS,
    IDENTITY_EPISODE_CONTRACT_ID,
    MATCHING_CONTRACT_ID,
    resolve_evaluator_code_sha,
    validate_report_contract,
)
from .cvat_io import parse_cvat_video_xml
from .evaluator_standard_v2 import (
    StandardV2Evaluation,
    aggregate_tracking_standard_v2,
    evaluate_tracking_standard_v2,
    metrics_to_dataframe_standard_v2,
)
from .lineage import (
    finalize_run_manifest,
    prepare_run_manifest,
    validate_metric_universe,
    write_artifact_manifest,
)
from .reporting_standard_v2 import (
    build_markdown_report_standard_v2,
    hota_alpha_dataframe,
    identity_ambiguity_dataframe,
    identity_authority_dataframe,
    identity_episode_dataframe,
    pairwise_swap_dataframe,
)


def _jsonable_config(config: TrackingEvaluationPipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "video_path",
        "gt_xml",
        "gt_dir",
        "video_dir",
        "prediction_root",
        "output_root",
        "weights_path",
        "weights_v26_path",
        "mask_path",
    ):
        value = payload.get(key)
        if value is not None:
            payload[key] = str(value)
    if payload.get("video_paths") is not None:
        payload["video_paths"] = [
            str(path) for path in payload["video_paths"]
        ]
    return payload


def _video_fps(pair: TrackingPair) -> float | None:
    metadata = video_metadata(pair.video_path)
    value = metadata.get("video_fps")
    if value is None:
        return None
    fps = float(value)
    return fps if fps > 0 else None


def evaluate_pair_standard_v2(
    pair: TrackingPair,
    config: TrackingEvaluationPipelineConfig,
    *,
    evaluator_code_sha: str,
) -> StandardV2Evaluation | None:
    """Evaluate one existing prediction with the complete unfiltered population."""
    if pair.pred_xml is None or not pair.pred_xml.is_file():
        return None
    parse_kwargs = {
        "include_hidden": True,
        "start_frame": config.evaluation_start_frame,
        "end_frame": config.evaluation_end_frame,
    }
    gt = parse_cvat_video_xml(pair.gt_xml, **parse_kwargs)
    pred = parse_cvat_video_xml(pair.pred_xml, **parse_kwargs)
    return evaluate_tracking_standard_v2(
        gt,
        pred,
        video_stem=pair.video_stem,
        include_hidden=config.include_hidden,
        detection_iou_threshold=config.iou_threshold,
        frames_per_second=_video_fps(pair),
        evaluator_code_sha=evaluator_code_sha,
    )


def _write_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    empty_columns: tuple[str, ...],
) -> None:
    if dataframe.empty and not list(dataframe.columns):
        dataframe = pd.DataFrame(columns=list(empty_columns))
    dataframe.to_csv(path, index=False)


def save_standard_v2_report(
    *,
    pairs: list[TrackingPair],
    assets_df: pd.DataFrame,
    evaluations: list[StandardV2Evaluation],
    aggregate: StandardV2Evaluation,
    metrics_df: pd.DataFrame,
    runtime_telemetry_df: pd.DataFrame,
    config: TrackingEvaluationPipelineConfig,
) -> Path:
    """Write one internally homogeneous Standard V2 report."""
    run_dir = config.output_root
    all_evaluations = [*evaluations, aggregate]
    validate_report_contract(metrics_df.to_dict(orient="records"))

    assets_df.to_csv(run_dir / "tracking_eval_assets.csv", index=False)
    metrics_df.to_csv(run_dir / "tracking_metrics.csv", index=False)
    hota_alpha_dataframe(all_evaluations).to_csv(
        run_dir / "tracking_hota_by_alpha.csv",
        index=False,
    )
    _write_dataframe(
        identity_authority_dataframe(evaluations),
        run_dir / "tracking_identity_authority_v2.csv",
        empty_columns=(
            "sequence_key",
            "gt_id",
            "pred_id",
            "source",
            "established_frame",
        ),
    )
    _write_dataframe(
        identity_episode_dataframe(evaluations),
        run_dir / "tracking_identity_error_episodes_v2.csv",
        empty_columns=(
            "event_id",
            "sequence_key",
            "gt_id",
            "status",
            "duration_frames",
            "duration_seconds",
        ),
    )
    _write_dataframe(
        pairwise_swap_dataframe(evaluations),
        run_dir / "tracking_pairwise_identity_swaps_v2.csv",
        empty_columns=(
            "event_id",
            "sequence_key",
            "gt_ids",
            "persistent",
            "persistence_basis",
        ),
    )
    _write_dataframe(
        identity_ambiguity_dataframe(evaluations),
        run_dir / "tracking_identity_ambiguities_v2.csv",
        empty_columns=(
            "sequence_key",
            "frame",
            "gt_id",
            "pred_id",
            "ambiguity_reason",
        ),
    )
    runtime_telemetry_df.to_csv(
        run_dir / "tracking_runtime_telemetry.csv",
        index=False,
    )
    (run_dir / "tracking_report.md").write_text(
        build_markdown_report_standard_v2(assets_df, metrics_df, config),
        encoding="utf-8",
    )
    config_payload = {
        **_jsonable_config(config),
        "evaluator_metadata": {
            key: value
            for key, value in asdict(aggregate.metrics).items()
            if key
            in {
                "evaluator_contract_id",
                "identity_episode_contract_id",
                "matching_contract_id",
                "hota_threshold_set",
                "include_hidden",
                "sequence_boundary_policy",
                "idsw_policy",
                "identity_authority_policy",
                "reference_parity_status",
                "evaluator_code_sha",
                "metric_config_sha256",
            }
        },
    }
    (run_dir / "tracking_eval_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    contract_payload = {
        "evaluator_contract_id": EVALUATOR_CONTRACT_ID,
        "matching_contract_id": MATCHING_CONTRACT_ID,
        "identity_episode_contract_id": IDENTITY_EPISODE_CONTRACT_ID,
        "hota_threshold_set": list(HOTA_ALPHAS),
        "trackeval_source": "https://github.com/JonathonLuiten/TrackEval.git",
        "trackeval_commit_sha": (
            "12c8791b303e0a0b50f753af204249e622d0281a"
        ),
        "trackeval_license": "MIT, Copyright 2020 Jonathon Luiten",
        "mixed_version_report": False,
        "legacy_outputs_rewritten": 0,
    }
    (run_dir / "tracking_evaluator_contract.json").write_text(
        json.dumps(contract_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def run_pipeline_standard_v2(
    config: TrackingEvaluationPipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Evaluate active prediction outputs without invoking any legacy metric path."""
    from .pipeline import (
        build_pairs,
        ensure_predictions,
        pairs_to_dataframe,
        runtime_telemetry_to_dataframe,
    )

    if config.evaluator_contract_id != EVALUATOR_CONTRACT_ID:
        raise ValueError("New tracking reports require Standard V2")
    pairs = build_pairs(config)
    prepare_run_manifest(pairs, config)
    pairs = ensure_predictions(pairs, config)
    assets_df = pairs_to_dataframe(pairs)
    code_sha = resolve_evaluator_code_sha()
    evaluations = [
        result
        for pair in pairs
        if (
            result := evaluate_pair_standard_v2(
                pair,
                config,
                evaluator_code_sha=code_sha,
            )
        )
        is not None
    ]
    if not evaluations:
        raise ValueError("Standard V2 requires at least one prediction pair")
    aggregate = aggregate_tracking_standard_v2(evaluations)
    metrics_df = metrics_to_dataframe_standard_v2(
        [*evaluations, aggregate]
    )
    validate_metric_universe(metrics_df, pairs)
    runtime_telemetry_df = runtime_telemetry_to_dataframe(pairs)
    run_dir = save_standard_v2_report(
        pairs=pairs,
        assets_df=assets_df,
        evaluations=evaluations,
        aggregate=aggregate,
        metrics_df=metrics_df,
        runtime_telemetry_df=runtime_telemetry_df,
        config=config,
    )
    assert_no_mp4_artifacts(run_dir, context="tracking evaluation report V2")
    finalize_run_manifest(run_dir)
    write_artifact_manifest(run_dir, pairs)
    return assets_df, metrics_df, run_dir
