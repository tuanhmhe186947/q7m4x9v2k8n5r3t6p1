"""M0-F1 V4 local runner using the canonical production training path."""

from __future__ import annotations

import argparse
import json
import os
import random
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

from pig_behavior.classification_v2.training.config import (
    ClassificationV2TrainingConfig,
    load_training_config,
    training_config_to_jsonable,
)
from pig_behavior.classification_v2.training.data_module import StrictTrainingDataModule
from pig_behavior.classification_v2.training.trainer import (
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
EXTERNAL_PREFIX = "EXTERNAL_M0_F1_DATA_ROOT"


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
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/classification_v2/m0_outer_folds_20260820/m0_outer_f1_v4_run",
    )
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    return parser.parse_args()


def _repo_path(value: Path) -> Path:
    return value if value.is_absolute() else REPO_ROOT / value


def _external_root(args: argparse.Namespace) -> Path | None:
    raw = args.data_root or os.environ.get("M0_F1_DATA_ROOT")
    return _repo_path(Path(raw)).resolve() if raw else None


def _resolve_external_paths(
    config: ClassificationV2TrainingConfig,
    data_root: Path | None,
) -> ClassificationV2TrainingConfig:
    dataset = config.dataset
    updates: dict[str, Path] = {}
    marker = f"{EXTERNAL_PREFIX}/"
    for field_name in dataset.__dataclass_fields__:
        value = getattr(dataset, field_name)
        if not isinstance(value, Path):
            continue
        text = value.as_posix()
        if text.startswith(marker):
            updates[field_name] = (
                data_root / text[len(marker) :]
                if data_root is not None
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
    missing: list[str] = []
    for field_name in config.dataset.__dataclass_fields__:
        value = getattr(config.dataset, field_name)
        if isinstance(value, Path) and not value.exists():
            missing.append(f"dataset.{field_name}={value}")
    return missing


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
        },
        path,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _repo_path(args.config).resolve()
    data_root = _external_root(args)
    config = _resolve_external_paths(load_training_config(config_path), data_root)
    train_count, test_count = _manifest_counts(config)
    missing = _require_external_artifacts(config)
    if missing:
        print("M0_F1_V4_LOCAL_GATE = BLOCKED", flush=True)
        print("EXTERNAL_DATA_DEPENDENCIES =", json.dumps(missing), flush=True)
        print("MISSING_EXTERNAL_DATA_ARTIFACTS =", json.dumps(missing), flush=True)
        raise RuntimeError("required production data artifacts are missing")

    device = torch.device("cpu")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    data = StrictTrainingDataModule(config, device=device)
    try:
        train_indices = data.split_indices("train")
        test_indices = data.split_indices("test")
        if len(train_indices) != train_count or len(test_indices) != test_count:
            raise ValueError(
                f"DataModule F1 counts mismatch: train={len(train_indices)}, "
                f"test={len(test_indices)}"
            )
        preprocessing = data.fit_fold_preprocessor()
        fit_rows = int(preprocessing.train_row_count)
        if fit_rows != EXPECTED_TRAIN_COUNT:
            raise ValueError(f"fold preprocessing fit rows mismatch: {fit_rows}")

        train_subset = data.balanced_smoke_indices(train=True)
        test_subset = data.balanced_smoke_split("test")
        probe = data.batch(train_subset)
        print(
            "INTERACTION_CONTEXT_SOURCE = REAL_DATA_MODULE",
            flush=True,
        )
        print(
            "INTERACTION_CONTEXT_STATS = "
            f"{_stats(probe.model_inputs['interaction_context_features'])}",
            flush=True,
        )
        print(f"SAMPLE_WEIGHT_STATS = {_weight_stats(probe.sample_weight)}", flush=True)
        print(
            "UNION_AVAILABILITY_SOURCE = REAL_DATA_MODULE",
            flush=True,
        )
        print(
            "UNION_AVAILABILITY_STATS = "
            f"{_stats(probe.model_inputs['visual_context_available_mask'])}",
            flush=True,
        )

        model = _build_model(config, probe, data).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"MODEL_PARAMETER_COUNT = {parameter_count}", flush=True)
        if parameter_count != EXPECTED_PARAM_COUNT:
            raise ValueError("canonical M0 parameter count mismatch")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        behavior_weights = _behavior_class_weights(
            data, train_indices, config, device
        )
        output_dir = _repo_path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict[str, Any]] = []
        best_score: tuple[float, float] | None = None
        task_specs = _task_specs(config)
        for epoch in range(args.epochs):
            train_result, _ = _train_epoch(
                model,
                optimizer,
                scaler,
                data,
                train_subset,
                behavior_weights,
                {},
                task_specs,
                config,
                device,
                epoch=epoch,
                global_step=0,
            )
            _, _, val_metrics, _ = _evaluate(
                model, data, test_subset, config, device, split="test"
            )
            macro_f1 = float(val_metrics["test_native_unit_macro_f1_global"])
            nll = float(val_metrics["test_native_unit_nll"])
            record = {
                "epoch": epoch,
                "train_loss": train_result["train_loss_mean"],
                "val_macro_f1": macro_f1,
                "val_nll": nll,
                "validation_ran": True,
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
                config=config,
            )
            score = (macro_f1, -nll)
            if best_score is None or score > best_score:
                best_score = score
                _save_checkpoint(
                    output_dir / "best_validation.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    metrics=record,
                    config=config,
                )
            print(
                f"EPOCH_{epoch}_VALIDATION = PASS | macro_f1={macro_f1:.6f} | nll={nll:.6f}",
                flush=True,
            )

        best_path = output_dir / "best_validation.pt"
        last_path = output_dir / "last.pt"
        best = torch.load(best_path, map_location=device, weights_only=False)
        torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        model.eval()
        _, _, final_metrics, _ = _evaluate(
            model, data, test_subset, config, device, split="test"
        )
        summary = {
            "status": "PASS",
            "runner": "M0-F1-V4",
            "seed": SEED,
            "heldout_date": HELDOUT_DATE,
            "train_count": train_count,
            "test_count": test_count,
            "preprocessing_fit_rows": fit_rows,
            "parameter_count": parameter_count,
            "temporal_encoder": config.model.temporal_encoder_name,
            "sample_weight_policy": config.loss.sample_weight_policy,
            "final_metrics": final_metrics,
            "epoch_history": history,
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
