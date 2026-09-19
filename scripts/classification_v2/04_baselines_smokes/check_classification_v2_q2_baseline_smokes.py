"""Validate bounded Q2 baseline smoke audit artifacts.

This checker is intentionally stricter than a generic smoke check: it verifies
that B2-B7 all ran in bounded smoke mode, did not execute full OOF, and used the
requested accelerator when CUDA smoke evidence is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_BASELINES = ["B2", "B3", "B4", "B5", "B6", "B7"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Q2 B2-B7 baseline smoke audit.")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "outputs/classification_v2/model_smoke/q2_baselines_cuda_clean/q2_baseline_smoke_orchestration_audit.json"
        ),
    )
    parser.add_argument("--require-device", default="cuda", choices=["cuda", "cpu", "any"])
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/classification_v2/model_design/q2_baseline_smoke_check_audit.json"),
    )
    args = parser.parse_args()

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    errors: list[str] = []
    selected = list(audit.get("selected_baselines", []))
    if selected != EXPECTED_BASELINES:
        errors.append(f"selected_baselines={selected}")
    if audit.get("mode") != "execute":
        errors.append(f"mode_not_execute={audit.get('mode')}")
    if audit.get("full_oof_executed") is not False:
        errors.append("full_oof_executed")
    if audit.get("outer_test_threshold_tuning") is not False:
        errors.append("outer_test_threshold_tuning_enabled")

    rows = []
    for row in audit.get("run_audits", []):
        baseline_id = str(row.get("baseline_id"))
        run = row.get("audit", {})
        row_errors = _check_run_audit(run, require_device=args.require_device)
        if row_errors:
            errors.extend(f"{baseline_id}:{err}" for err in row_errors)
        rows.append(
            {
                "baseline_id": baseline_id,
                "device": run.get("device"),
                "hardware": run.get("hardware"),
                "git_dirty": run.get("git", {}).get("dirty"),
                "run_errors": run.get("errors", []),
                "train_steps": [item.get("train_steps") for item in run.get("history", [])],
            }
        )
    if [row["baseline_id"] for row in rows] != EXPECTED_BASELINES:
        errors.append(f"run_audit_baselines={[row['baseline_id'] for row in rows]}")

    result = {
        "schema_version": "classification_v2_q2_baseline_smoke_check_audit_v1",
        "audit_json": str(args.audit_json),
        "require_device": args.require_device,
        "runtime_python_executable": audit.get("runtime_python_executable"),
        "baseline_count": len(rows),
        "errors": errors,
        "valid": not errors,
        "baselines": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


def _check_run_audit(run: dict[str, Any], *, require_device: str) -> list[str]:
    errors: list[str] = []
    if run.get("errors"):
        errors.append(f"run_errors={run.get('errors')}")
    if run.get("valid") is not True:
        errors.append("run_not_valid")
    if require_device != "any" and run.get("device") != require_device:
        errors.append(f"device={run.get('device')}")
    if run.get("git", {}).get("dirty") is not False:
        errors.append(f"git_dirty={run.get('git', {}).get('dirty')}")
    config = run.get("config", {})
    execution = config.get("execution", {})
    if execution.get("mode") != "smoke":
        errors.append(f"execution_mode={execution.get('mode')}")
    if execution.get("smoke_steps") != 2:
        errors.append(f"smoke_steps={execution.get('smoke_steps')}")
    if not run.get("history"):
        errors.append("missing_history")
    return errors


if __name__ == "__main__":
    main()
