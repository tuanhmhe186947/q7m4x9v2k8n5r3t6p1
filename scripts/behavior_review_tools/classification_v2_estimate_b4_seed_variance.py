from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.config import load_training_config, training_config_to_jsonable
from pig_behavior.classification_v2.training.trainer import run_training


DEFAULT_SEEDS = (20260710, 20260711, 20260712)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate B4 inner-validation seed variance with bounded smoke runs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification_v2/baseline_actor_spatial.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classification_v2/model_smoke/b4_seed_variance"),
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--smoke-per-class", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--fold-id", default="q2_outer_00")
    parser.add_argument("--execute", action="store_true", help="Run bounded seed smokes. Omit for plan-only audit.")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    base = load_training_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_dir = args.output_dir / f"seed_{seed}"
        config = replace(
            base,
            optimization=replace(
                base.optimization,
                seed=seed,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
            ),
            execution=replace(
                base.execution,
                mode="smoke",
                smoke_steps=args.smoke_steps,
                smoke_per_class=args.smoke_per_class,
                fold_id=args.fold_id,
                output_dir=run_dir,
                resume=False,
            ),
        )
        row: dict[str, Any] = {
            "seed": seed,
            "output_dir": str(run_dir),
            "config": training_config_to_jsonable(config),
        }
        if args.execute:
            audit = run_training(config)
            validation_metrics = [
                float(item["validation_window_macro_f1"])
                for item in audit.get("history", [])
                if "validation_window_macro_f1" in item
            ]
            row.update(
                {
                    "device": audit.get("device"),
                    "hardware": audit.get("hardware"),
                    "git": audit.get("git"),
                    "errors": audit.get("errors", []),
                    "validation_window_macro_f1": validation_metrics[-1] if validation_metrics else None,
                    "outer_test_metrics_present_but_ignored": bool(audit.get("outer_test_metrics")),
                }
            )
        rows.append(row)

    metrics = [
        float(row["validation_window_macro_f1"])
        for row in rows
        if row.get("validation_window_macro_f1") is not None
    ]
    errors = _validate(rows, metrics, executed=args.execute, expected_count=len(seeds))
    result = {
        "schema_version": "classification_v2_b4_inner_validation_seed_variance_v1",
        "mode": "execute" if args.execute else "plan_only",
        "config": str(args.config),
        "runtime_python_executable": sys.executable,
        "fold_id": args.fold_id,
        "seeds": seeds,
        "metric": "validation_window_macro_f1",
        "outer_test_used_for_threshold_tuning": False,
        "outer_test_metrics_ignored": True,
        "full_oof_executed": False,
        "rows": rows,
        "summary": _summary(metrics),
        "errors": errors,
        "valid": not errors,
    }
    output_path = args.output_dir / "b4_seed_variance_audit.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["schema_version", "mode", "summary", "errors", "valid"]}, indent=2))
    if errors:
        raise SystemExit(1)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds: {seeds}")
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _summary(metrics: list[float]) -> dict[str, Any]:
    if not metrics:
        return {"count": 0}
    return {
        "count": len(metrics),
        "mean": float(statistics.fmean(metrics)),
        "min": float(min(metrics)),
        "max": float(max(metrics)),
        "population_stdev": float(statistics.pstdev(metrics)),
        "sample_stdev": float(statistics.stdev(metrics)) if len(metrics) > 1 else 0.0,
        "range": float(max(metrics) - min(metrics)),
    }


def _validate(
    rows: list[dict[str, Any]],
    metrics: list[float],
    *,
    executed: bool,
    expected_count: int,
) -> list[str]:
    errors: list[str] = []
    if not executed:
        return errors
    if len(rows) != expected_count or len(metrics) != expected_count:
        errors.append(f"incomplete_seed_runs=rows:{len(rows)},metrics:{len(metrics)},expected:{expected_count}")
    for row in rows:
        if row.get("errors"):
            errors.append(f"seed_{row.get('seed')}_errors={row.get('errors')}")
        if row.get("device") != "cuda":
            errors.append(f"seed_{row.get('seed')}_device={row.get('device')}")
        if row.get("git", {}).get("dirty") is not False:
            errors.append(f"seed_{row.get('seed')}_git_dirty={row.get('git', {}).get('dirty')}")
    return errors


if __name__ == "__main__":
    main()
