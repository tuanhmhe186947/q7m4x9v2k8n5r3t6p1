"""CPU-only real-data preflight for the fixed PRE-S1 calibration route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pig_behavior.classification_v2.training.pre_s1_calibration import (
    create_calibration_plan,
    load_canonical_population,
    preflight_calibration,
    run_real_data_cpu_preflight,
)


def parse_args() -> argparse.Namespace:
    """Parse only non-scientific paths and the exact preflight token."""

    parser = argparse.ArgumentParser(
        description="Run PRE-S1 population/RGB CPU preflight without training."
    )
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--data-bindings", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--execution-authorization", required=True)
    return parser.parse_args()


def _write_report(path: Path, report: dict[str, object]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"immutable preflight report already exists={path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"preflight report parent missing={path.parent}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """Open only inner metadata and a bounded packed-RGB sample on CPU."""

    args = parse_args()
    if args.execution_authorization != "PRE_S1_CALIBRATION_PREFLIGHT_AUTHORIZED":
        raise SystemExit("PRE-S1 CPU preflight requires its exact authorization token")
    plan = create_calibration_plan(
        args.authority,
        repository_root=args.repository_root.resolve(),
        outputs_root=args.outputs_root.resolve(),
        device_name="cuda",
        data_bindings_path=args.data_bindings,
    )
    hashes = preflight_calibration(plan)
    population = load_canonical_population(plan, hashes)
    try:
        report = run_real_data_cpu_preflight(plan, population)
    finally:
        population.close()
    _write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
