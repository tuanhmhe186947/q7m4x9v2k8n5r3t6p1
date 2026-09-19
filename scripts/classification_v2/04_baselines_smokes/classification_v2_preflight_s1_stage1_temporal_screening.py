"""Run the bounded CPU-only preflight for all Stage-1 temporal arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pig_behavior.classification_v2.training import stage1_temporal_screening as stage1

CPU_PREFLIGHT_AUTHORIZATION = "S1_STAGE1_CPU_PREFLIGHT_AUTHORIZED"


def parse_args() -> argparse.Namespace:
    """Parse the no-GPU Stage-1 preflight contract."""

    parser = argparse.ArgumentParser(
        description="CPU-preflight all registered Stage-1 temporal-screening arms."
    )
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--bindings-manifest", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--cpu-smoke-steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    """Validate real metadata/cache binding and one CPU step per temporal view."""

    args = parse_args()
    if args.authorization != CPU_PREFLIGHT_AUTHORIZATION:
        raise ValueError("Stage-1 CPU preflight authorization token mismatch")
    report_path = args.report_path.resolve()
    if report_path.exists():
        raise FileExistsError(f"immutable Stage-1 preflight report exists={report_path}")
    if any(token in str(report_path).lower() for token in ("outer", "q2_outer_00")):
        raise ValueError("Stage-1 preflight report path has forbidden outer scope")
    bundle_path = args.bindings_manifest.resolve()
    bundle = _read_json(bundle_path)
    views = bundle.get("views")
    if not isinstance(views, dict) or set(views) != set(stage1.VIEW_SPECS):
        raise ValueError("Stage-1 bindings manifest does not cover all registered views")
    reports: dict[str, object] = {}
    for view in stage1.VIEW_SPECS:
        binding_entry = views[view]
        if not isinstance(binding_entry, dict):
            raise ValueError(f"Stage-1 binding entry is invalid={view}")
        data_path = _relative_binding_path(
            bundle_path.parent,
            str(binding_entry.get("data_bindings_relative_path", "")),
        )
        if _sha256(data_path) != str(binding_entry.get("data_bindings_sha256", "")):
            raise ValueError(f"Stage-1 data binding hash mismatch={view}")
        plan = stage1.create_stage1_plan(
            args.authority,
            view=view,
            repository_root=args.repository_root,
            outputs_root=args.outputs_root,
            trial_id=f"s1_stage1_{view.lower()}_cpu_preflight",
            device_name="cpu",
            data_bindings_path=data_path,
            engineering_smoke=False,
            allow_existing_output=True,
        )
        hashes = stage1.preflight_stage1(plan)
        population = stage1.load_stage1_population(plan, hashes)
        try:
            reports[view] = {
                "preflight": stage1.run_real_data_cpu_preflight(
                    plan,
                    population,
                    sample_size=args.sample_size,
                ),
                "cpu_engineering_smoke": stage1.run_real_data_cpu_engineering_smoke(
                    plan,
                    population,
                    steps=args.cpu_smoke_steps,
                ),
                "input_hashes": dict(population.data_hashes),
                "data_bindings_path": str(data_path),
                "data_bindings_sha256": _sha256(data_path),
            }
        finally:
            population.close()
    payload = {
        "schema_version": "classification_v2.s1_stage1_cpu_preflight.v1",
        "status": "PASS",
        "authorization": CPU_PREFLIGHT_AUTHORIZATION,
        "authority": {"path": str(args.authority), "sha256": _sha256(args.authority)},
        "bindings_manifest": {"path": str(bundle_path), "sha256": _sha256(bundle_path)},
        "gpu_used": False,
        "views": reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(report_path, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "report_path": str(report_path),
                "report_sha256": _sha256(report_path),
                "views": list(reports),
                "gpu_used": False,
            },
            indent=2,
        )
    )


def _relative_binding_path(base: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("binding manifest must use a relative data-binding path")
    resolved_base = base.resolve()
    path = (resolved_base / relative).resolve()
    if not path.is_relative_to(resolved_base):
        raise ValueError("binding manifest path escapes its bundle root")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required={path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
