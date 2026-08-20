"""M0-F1 V4 local runner using the canonical production training path."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pig_behavior.classification_v2.training.config import (  # noqa: E402
    ClassificationV2TrainingConfig,
    load_training_config,
    training_config_to_jsonable,
)
from pig_behavior.classification_v2.training.data_module import (  # noqa: E402
    StrictTrainingDataModule,
)
from pig_behavior.classification_v2.training.trainer import (  # noqa: E402
    _behavior_class_weights,
    _build_model,
    _evaluate,
    _task_specs,
    _train_epoch,
)

SEED = 240494961
EXPECTED_PARAM_COUNT = 43136168
EXPECTED_TRAIN_COUNT = 18694
EXPECTED_TEST_COUNT = 14593
HELDOUT_DATE = "2019-11-29"
FULL_EPOCH_BUDGET = 30
DATA_EXTERNAL_PREFIX = "EXTERNAL_M0_F1_DATA_ROOT"
RGB_EXTERNAL_PREFIX = "EXTERNAL_M0_F1_RGB_ROOT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M0-F1 V4 canonical local runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/classification_v2/m0_outer_f1_scientific_v4.json",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root for the explicit external M0 data artifacts.",
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=None,
        help="Root for the explicit external window-major RGB artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run the bounded two-epoch CPU preflight subset.",
    )
    return parser.parse_args()


def _repo_path(value: Path) -> Path:
    return value if value.is_absolute() else REPO_ROOT / value


def _external_root(args: argparse.Namespace) -> Path | None:
    raw = args.data_root or os.environ.get("M0_F1_DATA_ROOT")
    return _repo_path(Path(raw)).resolve() if raw else None


def _rgb_root(args: argparse.Namespace) -> Path | None:
    raw = args.rgb_root or os.environ.get("M0_F1_RGB_ROOT")
    return _repo_path(Path(raw)).resolve() if raw else None


def _resolve_external_paths(
    config: ClassificationV2TrainingConfig,
    data_root: Path | None,
    rgb_root: Path | None,
) -> ClassificationV2TrainingConfig:
    dataset = config.dataset
    updates: dict[str, Path] = {}
    data_marker = f"{DATA_EXTERNAL_PREFIX}/"
    rgb_marker = f"{RGB_EXTERNAL_PREFIX}/"
    for field_name in dataset.__dataclass_fields__:
        value = getattr(dataset, field_name)
        if not isinstance(value, Path):
            continue
        text = value.as_posix()
        if text.startswith(data_marker):
            updates[field_name] = (
                data_root / text[len(data_marker) :]
                if data_root is not None
                else REPO_ROOT / text
            )
        elif text.startswith(rgb_marker):
            updates[field_name] = (
                rgb_root / text[len(rgb_marker) :]
                if rgb_root is not None
                else REPO_ROOT / text
            )
        elif field_name in {
            "snapshot_json",
            "trainer_contract_json",
            "grouped_fold_roles",
        }:
            updates[field_name] = _repo_path(value)
    return replace(config, dataset=replace(dataset, **updates))


def _manifest_counts(config: ClassificationV2TrainingConfig) -> tuple[int, int]:
    frame = pd.read_csv(config.dataset.grouped_fold_roles, usecols=["target_id", "split"])
    train_count = int(frame["split"].eq("train").sum())
    test_count = int(frame["split"].eq("test").sum())
    if (train_count, test_count) != (EXPECTED_TRAIN_COUNT, EXPECTED_TEST_COUNT):
        raise ValueError(
            f"F1 role counts mismatch: train={train_count}, test={test_count}"
        )
    print(f"F1_TRAIN_COUNT = {train_count}", flush=True)
    print(f"F1_TEST_COUNT = {test_count}", flush=True)
    print(f"F1_HELDOUT_DATE = {HELDOUT_DATE}", flush=True)
    return train_count, test_count


def _require_external_artifacts(config: ClassificationV2TrainingConfig) -> list[str]:
    fast_path = config.dataset.window_major_rgb_cache is not None
    skipped_fast_path_fields = {
        "actor_packed_cache",
        "actor_packed_index",
        "visual_cache_manifest",
        "visual_packed_cache",
        "visual_packed_index",
        "frame_context_csv",
        "window_context_csv",
    }
    missing: list[str] = []
    for field_name in config.dataset.__dataclass_fields__:
        if fast_path and field_name in skipped_fast_path_fields:
            continue
        value = getattr(config.dataset, field_name)
        if isinstance(value, Path) and not value.exists():
            missing.append(f"dataset.{field_name}={value}")
    return missing


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return torch.device(name)


def _stats(tensor: torch.Tensor) -> list[float | int]:
    values = tensor.detach().float().cpu()
    return [
        float(values.min().item()),
        float(values.max().item()),
        int(torch.count_nonzero(values).item()),
    ]


def _weight_stats(tensor: torch.Tensor) -> list[float | int]:
    values = tensor.detach().float().cpu()
    return [
        float(values.min().item()),
        float(values.max().item()),
        float(values.mean().item()),
        int(torch.count_nonzero(values).item()),
    ]


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
    config: ClassificationV2TrainingConfig,
    selection_policy: str,
    selected_completed_epochs: int,
    selected_epoch_index: int,
    checkpoint_completed_epochs: int,
    outer_test_used_for_selection: bool,
    scientific_outer_test_evaluated: bool,
    scientific_result: bool,
    purpose: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "seed": SEED,
            "fold": "F1",
            "config": training_config_to_jsonable(config),
            "selection_policy": selection_policy,
            "selected_completed_epochs": selected_completed_epochs,
            "selected_epoch_index": selected_epoch_index,
            "checkpoint_completed_epochs": checkpoint_completed_epochs,
            "outer_test_used_for_selection": outer_test_used_for_selection,
            "scientific_outer_test_evaluated": scientific_outer_test_evaluated,
            "scientific_result": scientific_result,
            "purpose": purpose,
        },
        path,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _repo_path(args.config).resolve()
    config = _resolve_external_paths(
        load_training_config(config_path),
        _external_root(args),
        _rgb_root(args),
    )
    train_count, test_count = _manifest_counts(config)
    missing = _require_external_artifacts(config)
    if missing:
        print("M0_F1_V4_RUN = BLOCKED", flush=True)
        print("EXTERNAL_DATA_DEPENDENCIES =", json.dumps(missing), flush=True)
        raise RuntimeError("required production data artifacts are missing")
    if args.preflight and args.device != "cpu":
        raise ValueError("--preflight requires --device cpu")
    device = _resolve_device(args.device)
    runtime_config = replace(
        config,
        execution=replace(
            config.execution,
            mode="smoke" if args.preflight else "full_oof",
        ),
        optimization=replace(
            config.optimization,
            epochs=2 if args.preflight else FULL_EPOCH_BUDGET,
        ),
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    data = StrictTrainingDataModule(runtime_config, device=device)
    try:
        full_train_indices = data.split_indices("train")
        full_test_indices = data.split_indices("test")
        if len(full_train_indices) != train_count or len(full_test_indices) != test_count:
            raise ValueError(
                f"DataModule F1 counts mismatch: train={len(full_train_indices)}, "
                f"test={len(full_test_indices)}"
            )
        train_indices = (
            data.balanced_smoke_indices(train=True)
            if args.preflight
            else full_train_indices
        )
        test_indices = (
            data.balanced_smoke_split("test")
            if args.preflight
            else full_test_indices
        )
        preprocessing = data.fit_fold_preprocessor()
        fit_rows = int(preprocessing.train_row_count)
        if fit_rows != EXPECTED_TRAIN_COUNT:
            raise ValueError(f"fold preprocessing fit rows mismatch: {fit_rows}")
        probe = data.batch(train_indices[: min(len(train_indices), 2)])
        print("INTERACTION_CONTEXT_SOURCE = REAL_DATA_MODULE", flush=True)
        print(
            "INTERACTION_CONTEXT_STATS = "
            f"{_stats(probe.model_inputs['interaction_context_features'])}",
            flush=True,
        )
        print(f"SAMPLE_WEIGHT_STATS = {_weight_stats(probe.sample_weight)}", flush=True)
        print("UNION_AVAILABILITY_SOURCE = REAL_DATA_MODULE", flush=True)
        print(
            "UNION_AVAILABILITY_STATS = "
            f"{_stats(probe.model_inputs['visual_context_available_mask'])}",
            flush=True,
        )
        model = _build_model(runtime_config, probe, data).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"MODEL_PARAMETER_COUNT = {parameter_count}", flush=True)
        if parameter_count != EXPECTED_PARAM_COUNT:
            raise ValueError("canonical M0 parameter count mismatch")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=runtime_config.optimization.learning_rate,
            weight_decay=runtime_config.optimization.weight_decay,
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda" and runtime_config.optimization.precision == "amp"),
        )
        behavior_weights = _behavior_class_weights(
            data, full_train_indices, runtime_config, device
        )
        task_specs = _task_specs(runtime_config)
        default_output = REPO_ROOT / (
            "outputs/classification_v2/m0_outer_folds_20260820/"
            + ("m0_outer_f1_v4_preflight" if args.preflight else "m0_outer_f1_v4_run")
        )
        output_dir = _repo_path(args.output_dir or default_output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        global_step = 0
        for epoch in range(runtime_config.optimization.epochs):
            train_result, global_step = _train_epoch(
                model,
                optimizer,
                scaler,
                data,
                train_indices,
                behavior_weights,
                {},
                task_specs,
                runtime_config,
                device,
                epoch=epoch,
                global_step=global_step,
            )
            record = {
                "epoch": epoch,
                "train_loss": train_result["train_loss_mean"],
                "completed_epochs": epoch + 1,
                "global_step": global_step,
                "scientific_result": not args.preflight,
                "purpose": (
                    "checkpoint_and_evaluator_lifecycle_gate"
                    if args.preflight
                    else "fixed_30_epoch_outer_protocol"
                ),
            }
            history.append(record)
            (output_dir / "epoch_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            _save_checkpoint(
                output_dir / "last.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics=record,
                config=runtime_config,
                selection_policy=(
                    "checkpoint_and_evaluator_lifecycle_gate"
                    if args.preflight
                    else "fixed_30_epoch_outer_protocol"
                ),
                selected_completed_epochs=(
                    epoch + 1
                    if args.preflight
                    else runtime_config.optimization.epochs
                ),
                selected_epoch_index=epoch,
                checkpoint_completed_epochs=epoch + 1,
                outer_test_used_for_selection=False,
                scientific_outer_test_evaluated=False,
                scientific_result=not args.preflight,
                purpose=(
                    "checkpoint_and_evaluator_lifecycle_gate"
                    if args.preflight
                    else "fixed_30_epoch_outer_protocol"
                ),
            )
            if args.preflight:
                history_state = json.loads(
                    (output_dir / "epoch_history.json").read_text(encoding="utf-8")
                )
                expected_epochs = epoch + 1
                if [item["epoch"] for item in history_state] != list(
                    range(expected_epochs)
                ):
                    raise RuntimeError("preflight epoch history persistence mismatch")
                best_path = output_dir / "best_validation.pt"
                shutil.copyfile(output_dir / "last.pt", best_path)
                checkpoint_states = [
                    torch.load(output_dir / "last.pt", map_location=device, weights_only=False),
                    torch.load(best_path, map_location=device, weights_only=False),
                ]
                for checkpoint_state in checkpoint_states:
                    if checkpoint_state["checkpoint_completed_epochs"] != expected_epochs:
                        raise RuntimeError("preflight checkpoint epoch metadata mismatch")
                    if checkpoint_state["scientific_result"] is not False:
                        raise RuntimeError("preflight checkpoint marked scientific")
                    if checkpoint_state["purpose"] != "checkpoint_and_evaluator_lifecycle_gate":
                        raise RuntimeError("preflight checkpoint purpose mismatch")
                    if checkpoint_state["outer_test_used_for_selection"] is not False:
                        raise RuntimeError("preflight checkpoint selection leakage")
                print(
                    f"PREFLIGHT_EPOCH_{epoch}_CHECKPOINT_LIFECYCLE = PASS | "
                    f"completed_epochs={expected_epochs}",
                    flush=True,
                )
            print(
                f"EPOCH_{epoch}_COMPLETED = PASS | global_step={global_step}",
                flush=True,
            )
        best_path = output_dir / "best_validation.pt"
        last_path = output_dir / "last.pt"
        if not last_path.exists():
            raise RuntimeError("last.pt missing after fixed training budget")
        shutil.copyfile(last_path, best_path)
        selected = torch.load(best_path, map_location=device, weights_only=False)
        if selected["selected_completed_epochs"] != runtime_config.optimization.epochs:
            raise RuntimeError("selected checkpoint epoch budget mismatch")
        model.load_state_dict(selected["model_state_dict"])
        model.eval()
        predictions, native_predictions, final_metrics, _ = _evaluate(
            model, data, test_indices, runtime_config, device, split="test"
        )
        if not args.preflight and len(predictions) != EXPECTED_TEST_COUNT:
            raise RuntimeError("full F1 outer-test evaluation row count mismatch")
        prediction_name = (
            "preflight_predictions.csv"
            if args.preflight
            else "f1_outer_test_predictions.csv"
        )
        native_prediction_name = (
            "preflight_native_predictions.csv"
            if args.preflight
            else "f1_outer_test_native_predictions.csv"
        )
        predictions.to_csv(output_dir / prediction_name, index=False)
        native_predictions.to_csv(output_dir / native_prediction_name, index=False)
        metrics_artifact = {
            "metrics": final_metrics,
            "scientific_result": not args.preflight,
            "purpose": (
                "checkpoint_and_evaluator_lifecycle_gate"
                if args.preflight
                else "fixed_30_epoch_outer_protocol"
            ),
            "selection_policy": (
                "checkpoint_and_evaluator_lifecycle_gate"
                if args.preflight
                else "fixed_30_epoch_outer_protocol"
            ),
            "selected_completed_epochs": runtime_config.optimization.epochs,
            "selected_epoch_index": runtime_config.optimization.epochs - 1,
            "outer_test_evaluation_count": 1,
            "outer_test_used_for_selection": False,
            "scientific_outer_test_evaluated": not args.preflight,
        }
        (output_dir / "f1_outer_test_metrics.json").write_text(
            json.dumps(metrics_artifact, indent=2), encoding="utf-8"
        )
        summary = {
            "status": "PASS",
            "runner": "M0-F1-V4",
            "preflight": args.preflight,
            "device": str(device),
            "seed": SEED,
            "heldout_date": HELDOUT_DATE,
            "train_count": train_count,
            "test_count": test_count,
            "train_rows_used": int(len(train_indices)),
            "test_rows_used": int(len(test_indices)),
            "preprocessing_fit_rows": fit_rows,
            "parameter_count": parameter_count,
            "temporal_encoder": runtime_config.model.temporal_encoder_name,
            "sample_weight_policy": runtime_config.loss.sample_weight_policy,
            "final_metrics": final_metrics,
            "epoch_history": history,
            "scientific_result": not args.preflight,
            "purpose": (
                "checkpoint_and_evaluator_lifecycle_gate"
                if args.preflight
                else "fixed_30_epoch_outer_protocol"
            ),
            "selection_policy": (
                "checkpoint_and_evaluator_lifecycle_gate"
                if args.preflight
                else "fixed_30_epoch_outer_protocol"
            ),
            "selected_completed_epochs": runtime_config.optimization.epochs,
            "selected_epoch_index": runtime_config.optimization.epochs - 1,
            "outer_test_evaluation_count": 1,
            "outer_test_used_for_selection": False,
            "scientific_outer_test_evaluated": not args.preflight,
            "best_checkpoint_load": "PASS",
            "last_checkpoint_load": "PASS",
            "final_evaluator_loads_best": True,
        }
        (output_dir / "m0_f1_v4_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    finally:
        data.close()


def main() -> None:
    try:
        summary = run(parse_args())
    except Exception as exc:
        print(f"V4_RUN_BLOCKED = {type(exc).__name__}: {exc}", flush=True)
        raise
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
