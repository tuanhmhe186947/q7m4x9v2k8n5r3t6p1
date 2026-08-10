"""Materialize immutable inner-only RGB bindings for all registered Stage-1 views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.training import stage1_rgb_binding
from pig_behavior.classification_v2.training import stage1_temporal_screening as stage1

INITIAL_STAGE1_SCREEN = "INITIAL_STAGE1_SCREEN"
STAGE1_CONFIRMATION = "STAGE1_CONFIRMATION"


def parse_args() -> argparse.Namespace:
    """Parse only authority-bound Stage-1 RGB binding inputs."""

    parser = argparse.ArgumentParser(
        description="Materialize hash-bound Stage-1 T6/T8/T12/T16 RGB bindings."
    )
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--rgb-source-root", type=Path, required=True)
    parser.add_argument("--source-integrity-evidence", type=Path, required=True)
    parser.add_argument("--input-parity-evidence", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--binding-id", required=True)
    parser.add_argument(
        "--execution-phase",
        choices=(INITIAL_STAGE1_SCREEN, STAGE1_CONFIRMATION),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=stage1.SEED)
    parser.add_argument("--confirmation-authority", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    """Construct all four bindings after all metadata preflights have passed."""

    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable Stage-1 binding root already exists={output_root}")
    if any(token in args.binding_id.lower() for token in ("outer", "test")):
        raise ValueError("binding identifier must not include outer/test scope")
    confirmation_authority = _confirmation_authority_for_phase(
        execution_phase=args.execution_phase,
        seed=args.seed,
        confirmation_authority=args.confirmation_authority,
    )
    source_integrity = _read_json(args.source_integrity_evidence)
    input_parity = _read_json(args.input_parity_evidence) if args.input_parity_evidence else None
    views = _materialization_views(args.execution_phase)
    prepared: list[tuple[stage1.Stage1Plan, stage1.Stage1Rows]] = []
    for view in views:
        plan = stage1.create_stage1_plan(
            args.authority,
            view=view,
            seed=args.seed,
            repository_root=args.repository_root,
            outputs_root=args.outputs_root,
            trial_id=f"s1_stage1_{view.lower()}_binding_preflight",
            device_name="cpu",
            engineering_smoke=False,
            allow_existing_output=True,
            confirmation_authority_path=confirmation_authority,
        )
        prepared.append((plan, stage1.load_stage1_inner_rows(plan, stage1.preflight_stage1(plan))))

    reports: dict[str, dict[str, object]] = {}
    provenance_by_view: dict[str, dict[str, str]] = {}
    for plan, rows in prepared:
        requested_roles = pd.concat(
            [
                rows.train.loc[:, ["window_id", "primary_s1_role"]],
                rows.validation.loc[:, ["window_id", "primary_s1_role"]],
            ],
            ignore_index=True,
        )
        report = stage1_rgb_binding.materialize_stage1_rgb_binding(
            output_dir=output_root / f"{args.binding_id}_{plan.view.lower()}",
            rgb_source_root=args.rgb_source_root,
            requested_roles=requested_roles,
            authority_sha256=plan.authority_sha256,
            provenance_hashes=rows.data_hashes,
            view=plan.view,
            sequence_length=plan.sequence_length,
            expected_train_windows=len(rows.train),
            expected_validation_windows=len(rows.validation),
            input_parity_evidence=input_parity,
            source_integrity_evidence=source_integrity,
        )
        reports[plan.view] = report
        provenance_by_view[plan.view] = dict(rows.data_hashes)
    bundle = {
        "schema_version": "classification_v2.s1_stage1_rgb_binding_bundle.v1",
        "authority": {
            "path": str(args.authority),
            "sha256": _sha256(args.authority),
        },
        "materialization": {
            "execution_phase": args.execution_phase,
            "seed": args.seed,
            "confirmation_authority_sha256": (
                prepared[0][0].seed_authorization.confirmation_authority_sha256
            ),
        },
        "views": {
            view: {
                "scientific_binding_relative_path": str(
                    Path(str(report["scientific_binding_path"])).relative_to(output_root)
                ).replace("\\", "/"),
                "scientific_binding_sha256": report["scientific_binding_sha256"],
                "data_bindings_relative_path": str(
                    Path(str(report["data_bindings_path"])).relative_to(output_root)
                ).replace("\\", "/"),
                "data_bindings_sha256": report["data_bindings_sha256"],
                "coverage": report["coverage"],
                "provenance_hashes": provenance_by_view[view],
            }
            for view, report in reports.items()
        },
    }
    bundle_path = output_root / "stage1_temporal_rgb_bindings.json"
    _write_json_atomic(bundle_path, bundle)
    print(
        json.dumps(
            {
                "status": "PASS",
                "binding_bundle_path": str(bundle_path),
                "binding_bundle_sha256": _sha256(bundle_path),
                "views": reports,
            },
            indent=2,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required={path}")
    return value


def _confirmation_authority_for_phase(
    *,
    execution_phase: str,
    seed: int,
    confirmation_authority: Path | None,
) -> Path | None:
    """Require explicit phase/authority provenance before binding RGB metadata."""

    if execution_phase == INITIAL_STAGE1_SCREEN:
        if seed != stage1.SEED:
            raise ValueError("initial Stage-1 materialization requires the initial seed")
        if confirmation_authority is not None:
            raise ValueError("initial Stage-1 materialization must not bind confirmation authority")
        return None
    if execution_phase == STAGE1_CONFIRMATION:
        if seed == stage1.SEED:
            raise ValueError("Stage-1 confirmation materialization requires a future seed")
        if confirmation_authority is None:
            raise ValueError("Stage-1 confirmation materialization requires confirmation authority")
        return confirmation_authority.resolve()
    raise ValueError(f"unsupported Stage-1 materialization phase={execution_phase}")


def _materialization_views(execution_phase: str) -> tuple[str, ...]:
    """Retain all initial views but only the registered confirmation candidates."""

    if execution_phase == INITIAL_STAGE1_SCREEN:
        return tuple(stage1.VIEW_SPECS)
    if execution_phase == STAGE1_CONFIRMATION:
        return ("T6", "T16")
    raise ValueError(f"unsupported Stage-1 materialization phase={execution_phase}")


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
