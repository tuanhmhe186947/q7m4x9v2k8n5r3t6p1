from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training.config import load_training_config, training_config_to_jsonable
from pig_behavior.classification_v2.training.trainer import run_training


BASELINE_CONFIGS = {
    "B2": Path("configs/classification_v2/baseline_spatial_tcn.json"),
    "B3": Path("configs/classification_v2/baseline_actor_image.json"),
    "B4": Path("configs/classification_v2/baseline_actor_spatial.json"),
    "B5": Path("configs/classification_v2/baseline_actor_spatial_partner_context.json"),
    "B6": Path("configs/classification_v2/baseline_actor_spatial_partner_multitask.json"),
    "B7": Path("configs/classification_v2/full_candidate_domain_controls.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or run bounded one-fold Q2 B2-B7 engineering smokes."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classification_v2/model_smoke/q2_baselines"))
    parser.add_argument("--baseline", action="append", choices=sorted(BASELINE_CONFIGS), default=None)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--smoke-per-class", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--fold-id", default="q2_outer_00")
    parser.add_argument("--execute", action="store_true", help="Run the bounded smokes. Omit for plan-only audit.")
    args = parser.parse_args()

    selected = args.baseline or list(BASELINE_CONFIGS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_rows: list[dict[str, Any]] = []
    run_audits: list[dict[str, Any]] = []
    for baseline_id in selected:
        config_path = BASELINE_CONFIGS[baseline_id]
        base = load_training_config(config_path)
        run_dir = args.output_dir / baseline_id
        config = replace(
            base,
            optimization=replace(
                base.optimization,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
            ),
            execution=replace(
                base.execution,
                mode="smoke",
                smoke_steps=args.smoke_steps,
                smoke_per_class=args.smoke_per_class,
                output_dir=run_dir,
                fold_id=args.fold_id,
                resume=False,
            ),
        )
        plan_rows.append(
            {
                "baseline_id": baseline_id,
                "config_json": str(config_path),
                "output_dir": str(run_dir),
                "execute_requested": bool(args.execute),
                "cmd": _cmd_for_baseline(config_path, baseline_id, args),
                "effective_config": training_config_to_jsonable(config),
            }
        )
        if args.execute:
            # This path is intentionally bounded to smoke mode and one fold.
            # Full OOF remains gated by the separate full-run preflight.
            run_audits.append({"baseline_id": baseline_id, "audit": run_training(config)})

    result = {
        "schema_version": "classification_v2_q2_baseline_smoke_orchestration_v1",
        "mode": "execute" if args.execute else "plan_only",
        "selected_baselines": selected,
        "full_oof_executed": False,
        "outer_test_threshold_tuning": False,
        "plan": plan_rows,
        "run_audits": run_audits,
    }
    output_path = args.output_dir / "q2_baseline_smoke_orchestration_audit.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["schema_version", "mode", "selected_baselines"]}, indent=2))


def _cmd_for_baseline(config_path: Path, baseline_id: str, args: argparse.Namespace) -> str:
    return (
        "cd /d C:\\Users\\ironh\\Downloads\\PIG_Behavior_Project && "
        "set PYTHONPATH=%CD%\\src && "
        "python scripts\\behavior_review_tools\\classification_v2_run_q2_baseline_smokes.py "
        f"--baseline {baseline_id} "
        f"--output-dir {args.output_dir} "
        f"--smoke-steps {args.smoke_steps} "
        f"--smoke-per-class {args.smoke_per_class} "
        f"--batch-size {args.batch_size} "
        f"--eval-batch-size {args.eval_batch_size} "
        f"--fold-id {args.fold_id} "
        "--execute"
    )


if __name__ == "__main__":
    main()
