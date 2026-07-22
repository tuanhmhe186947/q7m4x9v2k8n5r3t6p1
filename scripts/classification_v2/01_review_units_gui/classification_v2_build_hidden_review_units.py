"""Build two-sided Hidden review cohorts from enhanced frame features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.contracts.identifiers import scene_frame_key
from pig_behavior.classification_v2.review.hidden_review_builder import (
    HiddenReviewConfig,
    audit_hidden_input_structure,
    balanced_hidden_smoke_scope,
    build_hidden_review_frame_context,
    build_hidden_review_manifest,
)
from pig_behavior.classification_v2.review.hidden_review_science import (
    build_hidden_scientific_design,
    load_hidden_scientific_policy,
    sha256_file,
)

TEMPLATE_FILENAMES = {
    "hidden_yes_confirmation": "hidden_yes_review_template.csv",
    "hidden_no_high_risk": "hidden_no_risk_review_template.csv",
    "hidden_no_random_audit": "hidden_no_random_audit_template.csv",
    "hidden_no_clean_control": "hidden_no_clean_control_template.csv",
}

FAILURE_AUDIT_FILENAME = "hidden_review_build_failure.json"
DESIGN_SCOPES = ("smoke", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic Hidden=Yes confirmation and Hidden=No "
            "false-negative audit cohorts."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--design-scope",
        choices=DESIGN_SCOPES,
        required=True,
        help=(
            "Scientific validation scope. 'smoke' keeps structural gates but "
            "does not require final-support quotas; 'full' requires all final "
            "support thresholds and cannot be combined with row caps."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--trusted-yes-per-stratum", type=int, default=1)
    parser.add_argument("--random-no-per-stratum", type=int, default=10)
    parser.add_argument("--clean-control-per-stratum", type=int, default=1)
    parser.add_argument("--max-high-risk-per-stratum", type=int, default=16)
    parser.add_argument("--high-risk-threshold", type=float, default=0.35)
    parser.add_argument("--clean-control-max-risk", type=float, default=0.10)
    parser.add_argument(
        "--scientific-policy-json",
        type=Path,
        default=Path(
            "configs/classification_v2/"
            "hidden_review_scientific_policy_v1.json"
        ),
        help="Predeclared uncertainty, support, and quality thresholds.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Optional debug input row cap. This never changes --design-scope "
            "or scientific thresholds."
        ),
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help=(
            "Optional debug cap per source. This never changes "
            "--design-scope or scientific thresholds."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in output-dir for the same lineage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    if not args.scientific_policy_json.exists():
        raise FileNotFoundError(args.scientific_policy_json)
    output_names = _canonical_output_names()
    output_paths = [args.output_dir / name for name in output_names]
    failure_path = args.output_dir / FAILURE_AUDIT_FILENAME
    _guard_outputs(
        [*output_paths, failure_path],
        overwrite=args.overwrite,
    )

    try:
        payloads, manifest, frame_context, audit, scientific_design = (
            _build_output_payloads(args)
        )
        _publish_output_transaction(
            args.output_dir,
            payloads,
            overwrite=args.overwrite,
        )
    except Exception as error:
        _write_failure_audit(args, output_paths, error)
        raise

    print(f"[PASS] Hidden review manifest rows={len(manifest)} cohorts={audit['cohort_counts']}")
    print(f"[PASS] frame context rows={len(frame_context)}")
    print(f"[PASS] audit={args.output_dir / 'hidden_review_template_audit.json'}")
    print(
        "[PASS] scientific design="
        f"{args.output_dir / 'hidden_review_scientific_design.json'}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be > 0")
    if args.max_rows_per_source is not None and args.max_rows_per_source <= 0:
        raise ValueError("--max-rows-per-source must be > 0")
    if args.max_rows is not None and args.max_rows_per_source is not None:
        raise ValueError("Use only one input row cap")
    if args.design_scope == "full" and _input_was_bounded(args):
        raise ValueError(
            "--design-scope full cannot be combined with --max-rows or "
            "--max-rows-per-source"
        )


def _canonical_output_names() -> list[str]:
    return [
        "hidden_review_unit_manifest.csv",
        "hidden_review_frame_context.csv",
        "hidden_review_template_audit.json",
        "hidden_review_scientific_design.json",
        *TEMPLATE_FILENAMES.values(),
    ]


def _build_output_payloads(
    args: argparse.Namespace,
) -> tuple[
    dict[str, bytes],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    _, policy_payload, policy_sha256 = load_hidden_scientific_policy(
        args.scientific_policy_json
    )
    all_frames = pd.read_csv(args.input_csv, low_memory=False)
    frames = _bound_input(all_frames, args)
    structural_audit = audit_hidden_input_structure(frames)
    if structural_audit["errors"]:
        raise ValueError(
            "Hidden input structural audit failed: "
            f"{structural_audit['errors']}"
        )

    config = HiddenReviewConfig(
        random_seed=args.random_seed,
        trusted_yes_per_stratum=args.trusted_yes_per_stratum,
        random_no_per_stratum=args.random_no_per_stratum,
        clean_control_per_stratum=args.clean_control_per_stratum,
        max_high_risk_per_stratum=args.max_high_risk_per_stratum,
        high_risk_threshold=args.high_risk_threshold,
        clean_control_max_risk=args.clean_control_max_risk,
    )
    manifest, templates, audit = build_hidden_review_manifest(
        frames,
        config=config,
    )
    frame_context = build_hidden_review_frame_context(frames, manifest)
    manifest_bytes = _dataframe_bytes(manifest)
    require_final_support = args.design_scope == "full"
    scientific_design = build_hidden_scientific_design(
        manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        policy_payload=policy_payload,
        policy_sha256=policy_sha256,
        selection_contract=audit["selection_contract"],
        require_final_support=require_final_support,
    )

    audit.update(
        {
            "input_csv": str(args.input_csv),
            "input_csv_sha256": sha256_file(args.input_csv),
            "output_dir": str(args.output_dir),
            "input_rows_before_bounding": int(len(all_frames)),
            "input_rows_after_bounding": int(len(frames)),
            "design_scope": args.design_scope,
            "require_final_support": require_final_support,
            "max_rows": args.max_rows,
            "max_rows_per_source": args.max_rows_per_source,
            "input_was_bounded": _input_was_bounded(args),
            "input_bounding_mode": _input_bounding_mode(args),
            "scientific_policy_json": str(args.scientific_policy_json),
            "scientific_policy_sha256": policy_sha256,
            "final_support_policy_version": policy_payload["schema_version"],
            "structural_checks_pass": True,
            "structural_audit": structural_audit,
            "final_support_checks_required": require_final_support,
            "final_support_checks_pass": scientific_design[
                "planned_support_meets_final_gate"
            ],
            "frame_context_rows": int(len(frame_context)),
            "frame_context_frames": int(
                scene_frame_key(frame_context).nunique(dropna=True)
            ),
            "frame_context_objects": int(
                frame_context["frame_uid"].nunique(dropna=True)
            ),
            "output_transaction_status": "committed",
            "outputs_published": True,
        }
    )

    payloads = {
        "hidden_review_unit_manifest.csv": manifest_bytes,
        "hidden_review_frame_context.csv": _dataframe_bytes(frame_context),
        "hidden_review_scientific_design.json": _json_bytes(scientific_design),
    }
    for cohort, filename in TEMPLATE_FILENAMES.items():
        payloads[filename] = _dataframe_bytes(templates[cohort])
    audit["published_file_hashes"] = {
        name: _sha256_bytes(payload)
        for name, payload in sorted(payloads.items())
    }
    payloads["hidden_review_template_audit.json"] = _json_bytes(audit)
    return payloads, manifest, frame_context, audit, scientific_design


def _bound_input(
    frames: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.max_rows is not None:
        return frames.head(args.max_rows).copy()
    if args.max_rows_per_source is not None:
        return balanced_hidden_smoke_scope(
            frames,
            args.max_rows_per_source,
        )
    return frames.copy()


def _publish_output_transaction(
    output_dir: Path,
    payloads: dict[str, bytes],
    *,
    overwrite: bool,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.hidden-review-staging-",
            dir=output_dir.parent,
        )
    )
    backup_dir = staging_dir / "backups"
    published: list[Path] = []
    backups: dict[Path, Path] = {}
    try:
        for name, payload in payloads.items():
            staged = staging_dir / name
            staged.write_bytes(payload)
            if _sha256_bytes(staged.read_bytes()) != _sha256_bytes(payload):
                raise RuntimeError(f"staging hash mismatch: {name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            backup_dir.mkdir()
            for name in payloads:
                target = output_dir / name
                if target.exists():
                    backup = backup_dir / name
                    _replace_for_commit(target, backup)
                    backups[target] = backup
        for name in payloads:
            target = output_dir / name
            _replace_for_commit(staging_dir / name, target)
            published.append(target)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _replace_for_commit(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _write_failure_audit(
    args: argparse.Namespace,
    output_paths: list[Path],
    error: Exception,
) -> None:
    no_outputs_published = not any(path.exists() for path in output_paths)
    payload = {
        "schema_version": "classification_v2.hidden_review_build_failure.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha": _git_sha(),
        "design_scope": args.design_scope,
        "require_final_support": args.design_scope == "full",
        "final_support_checks_required": args.design_scope == "full",
        "final_support_checks_pass": False,
        "structural_checks_pass": "structural audit failed" not in str(error),
        "input_bounding_mode": _input_bounding_mode(args),
        "input_was_bounded": _input_was_bounded(args),
        "max_rows": args.max_rows,
        "max_rows_per_source": args.max_rows_per_source,
        "input_csv": str(args.input_csv),
        "input_csv_sha256": _sha256_if_file(args.input_csv),
        "scientific_policy_json": str(args.scientific_policy_json),
        "scientific_policy_sha256": _sha256_if_file(
            args.scientific_policy_json
        ),
        "final_support_policy_version": _policy_version_if_file(
            args.scientific_policy_json
        ),
        "exception_type": type(error).__name__,
        "exception": str(error),
        "validation_errors": [str(error)],
        "output_transaction_status": "aborted",
        "outputs_published": False,
        "no_outputs_published": no_outputs_published,
    }
    if not no_outputs_published:
        raise RuntimeError(
            "Hidden output transaction failed and canonical outputs remain"
        ) from error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(
        args.output_dir / FAILURE_AUDIT_FILENAME,
        _json_bytes(payload),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _input_was_bounded(args: argparse.Namespace) -> bool:
    return args.max_rows is not None or args.max_rows_per_source is not None


def _input_bounding_mode(args: argparse.Namespace) -> str:
    if args.max_rows is not None:
        return "max_rows"
    if args.max_rows_per_source is not None:
        return "max_rows_per_source"
    return "none"


def _dataframe_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_if_file(path: Path) -> str | None:
    return sha256_file(path) if path.exists() and path.is_file() else None


def _policy_version_if_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("schema_version")


def _git_sha() -> str | None:
    repository_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _guard_outputs(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        display = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {display}. Use --overwrite explicitly.")


if __name__ == "__main__":
    main()
