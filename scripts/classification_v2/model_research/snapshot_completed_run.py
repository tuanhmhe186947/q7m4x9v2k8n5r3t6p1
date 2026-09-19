"""Read-only run-snapshot tool for a COMPLETED classification_v2 stage.

The tool never auto-discovers a run root and never inspects the currently active
production output. Every path is supplied explicitly by the operator, and a
stage still marked ``RUNNING`` is refused unless the operator passes
``--allow-running-metadata-only``. Even then the tool records metadata only: it
does not hash partially written outputs, does not mark PASS, and does not
publish a canonical snapshot.

Example
-------
``python scripts/classification_v2/model_research/snapshot_completed_run.py
--run-root D:/runs/cls_v2/20260726 --manifest-path D:/runs/.../manifest.json
--log-path D:/runs/.../stage.log --command-line "python ... run_lineage_stage.py"
--expected-stage native_evidence --output snapshot.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SNAPSHOT_SCHEMA_VERSION = "classification_v2.run_snapshot.v1"

RUNNING_STATUS_VALUES = frozenset({"RUNNING", "IN_PROGRESS", "STARTED"})
COMPLETED_STATUS_VALUES = frozenset({"PASS", "COMPLETED", "SUCCESS", "OK"})

SNAPSHOT_FIELDS: tuple[str, ...] = (
    "RUN_ID",
    "BASE_GIT_SHA",
    "CONFIG_SHA256",
    "COMMAND_LINE",
    "START_TIME",
    "END_TIME",
    "EXIT_CODE",
    "INPUT_PATHS",
    "OUTPUT_PATHS",
    "OUTPUT_HASHES",
    "MANIFEST_PATH",
    "LOG_PATH",
    "MOTION_SCHEMA_VERSION",
    "MOTION_SCHEMA_HASH",
    "STAGE_STATUS",
    "PASS_FAIL_REASON",
)


class SnapshotRefused(RuntimeError):
    """Raised when snapshotting would read or publish an unsafe artifact."""


@dataclass(slots=True)
class SnapshotRequest:
    """Explicit operator-supplied inputs. Nothing is discovered."""

    run_root: Path
    manifest_path: Path
    log_path: Path
    command_line: str
    expected_stage: str
    allow_running_metadata_only: bool = False
    output_path: Path | None = None
    extra_input_paths: tuple[Path, ...] = field(default_factory=tuple)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SnapshotRefused(f"manifest_path is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotRefused(f"manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SnapshotRefused(f"manifest must be a JSON object: {path}")
    return payload


def _stage_payload(manifest: dict[str, Any], expected_stage: str) -> dict[str, Any]:
    stages = manifest.get("stages")
    if isinstance(stages, dict) and expected_stage in stages:
        stage = stages[expected_stage]
        if not isinstance(stage, dict):
            raise SnapshotRefused(f"stage entry must be an object: {expected_stage}")
        return stage
    declared = manifest.get("stage_id") or manifest.get("stage")
    if declared is not None and str(declared) != expected_stage:
        raise SnapshotRefused(
            f"manifest stage={declared!r} does not match --expected-stage="
            f"{expected_stage!r}"
        )
    return manifest


def _stage_status(stage: dict[str, Any]) -> str:
    for key in ("status", "stage_status", "state", "result"):
        value = stage.get(key)
        if value is not None:
            return str(value).strip().upper()
    return "UNKNOWN"


def build_snapshot(request: SnapshotRequest) -> dict[str, Any]:
    """Build one snapshot payload from explicitly supplied artifacts."""

    if not request.run_root.is_dir():
        raise SnapshotRefused(f"run_root is not a directory: {request.run_root}")
    if not request.log_path.is_file():
        raise SnapshotRefused(f"log_path is not a file: {request.log_path}")
    if not request.command_line.strip():
        raise SnapshotRefused("--command-line must not be empty")
    if not request.expected_stage.strip():
        raise SnapshotRefused("--expected-stage must not be empty")

    manifest = _load_manifest(request.manifest_path)
    stage = _stage_payload(manifest, request.expected_stage)
    status = _stage_status(stage)
    running = status in RUNNING_STATUS_VALUES

    if running and not request.allow_running_metadata_only:
        raise SnapshotRefused(
            f"stage {request.expected_stage} is marked {status}. Refusing to "
            "snapshot a running stage. Re-run with "
            "--allow-running-metadata-only for a metadata-only, "
            "non-canonical, non-PASS record."
        )

    output_paths = [str(value) for value in _as_list(stage.get("output_paths"))]
    input_paths = [str(value) for value in _as_list(stage.get("input_paths"))]
    input_paths.extend(str(path) for path in request.extra_input_paths)

    output_hashes: dict[str, str | None] = {}
    hashing_skipped_reason: str | None = None
    if running:
        hashing_skipped_reason = (
            "stage is RUNNING; partially written outputs are never hashed"
        )
        output_hashes = {path: None for path in output_paths}
    else:
        for path in output_paths:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = request.run_root / path
            output_hashes[path] = (
                _sha256_file(candidate) if candidate.is_file() else None
            )

    pass_fail_reason = str(
        stage.get("pass_fail_reason")
        or stage.get("reason")
        or ("stage still running: no verdict recorded" if running else "")
    )
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "RUN_ID": str(stage.get("run_id") or manifest.get("run_id") or ""),
        "BASE_GIT_SHA": str(
            stage.get("base_git_sha") or manifest.get("base_git_sha") or ""
        ),
        "CONFIG_SHA256": str(
            stage.get("config_sha256") or manifest.get("config_sha256") or ""
        ),
        "COMMAND_LINE": request.command_line,
        "START_TIME": str(stage.get("start_time") or ""),
        "END_TIME": str(stage.get("end_time") or ""),
        "EXIT_CODE": stage.get("exit_code"),
        "INPUT_PATHS": input_paths,
        "OUTPUT_PATHS": output_paths,
        "OUTPUT_HASHES": output_hashes,
        "MANIFEST_PATH": str(request.manifest_path),
        "LOG_PATH": str(request.log_path),
        "MOTION_SCHEMA_VERSION": str(
            stage.get("motion_schema_version")
            or manifest.get("motion_schema_version")
            or ""
        ),
        "MOTION_SCHEMA_HASH": str(
            stage.get("motion_schema_hash") or manifest.get("motion_schema_hash") or ""
        ),
        "STAGE_STATUS": "RUNNING_METADATA_ONLY" if running else status,
        "PASS_FAIL_REASON": pass_fail_reason,
        "snapshot_contract": {
            "expected_stage": request.expected_stage,
            "auto_discovery_used": False,
            "active_production_output_inspected": False,
            "metadata_only": running,
            "outputs_hashed": not running,
            "hashing_skipped_reason": hashing_skipped_reason,
            "marked_pass": bool(not running and status in COMPLETED_STATUS_VALUES),
            "published_as_canonical": False,
        },
    }
    if running:
        snapshot["snapshot_contract"]["warning"] = (
            "metadata-only snapshot of a RUNNING stage; not a verdict and not "
            "canonical"
        )
    return snapshot


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot a COMPLETED classification_v2 stage from explicitly "
            "supplied artifacts. Nothing is auto-discovered."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--command-line", type=str, required=True)
    parser.add_argument("--expected-stage", type=str, required=True)
    parser.add_argument(
        "--allow-running-metadata-only",
        action="store_true",
        help=(
            "record metadata for a stage still marked RUNNING; outputs are not "
            "hashed, the snapshot is not marked PASS, and it is not canonical"
        ),
    )
    parser.add_argument("--input-path", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = SnapshotRequest(
        run_root=args.run_root,
        manifest_path=args.manifest_path,
        log_path=args.log_path,
        command_line=args.command_line,
        expected_stage=args.expected_stage,
        allow_running_metadata_only=args.allow_running_metadata_only,
        output_path=args.output,
        extra_input_paths=tuple(args.input_path),
    )
    try:
        snapshot = build_snapshot(request)
    except SnapshotRefused as exc:
        print(f"SNAPSHOT_REFUSED: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(snapshot, indent=2, ensure_ascii=True, allow_nan=False)
    if request.output_path is not None:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
