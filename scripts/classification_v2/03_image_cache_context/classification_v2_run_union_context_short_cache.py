from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from pig_behavior.classification_v2.datasets.visual_interaction_context import (
    VisualInteractionCacheConfig,
    build_visual_interaction_cache,
)

SCHEMA_VERSION = "classification_v2.legacy_development_l6.union_context_short_cache_config.v1"
LINEAGE_SCOPE = "legacy-only-unreviewed-development"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one immutable legacy union-context cache replica."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replica", choices=("primary", "repeat"), required=True)
    args = parser.parse_args()
    audit = run_union_context_short_cache(args.config, args.replica)
    print(json.dumps(audit, indent=2))


def run_union_context_short_cache(
    config_path: Path,
    replica: str,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload, repo_root = _load_config(config_path)
    _validate_bound_inputs(payload, repo_root)
    _validate_implementation(payload, repo_root)
    guard = _git_guard(payload, repo_root)
    if not guard["valid"]:
        raise ValueError(f"union-context cache git guard failed: {guard['errors']}")
    selection_path = _bound_path(
        repo_root,
        payload["inputs"]["selection_csv"]["path"],
    )
    frame_path = _bound_path(
        repo_root,
        payload["inputs"]["frame_context_manifest"]["path"],
    )
    selection = pd.read_csv(selection_path, low_memory=False)
    if list(selection.columns) != ["image_context_id"]:
        raise ValueError("union-context cache selection columns are not identifier-only")
    if len(selection) != int(payload["cache_contract"]["selected_rows"]):
        raise ValueError("union-context cache selection row count drift")
    frame_paths = pd.read_csv(
        frame_path,
        usecols=["image_context_id", "resolved_media_path"],
        low_memory=False,
    )
    selected_ids = set(selection["image_context_id"].astype(str))
    selected_frame_paths = (
        frame_paths.loc[
            frame_paths["image_context_id"].astype(str).isin(selected_ids),
            "resolved_media_path",
        ]
        .fillna("")
        .astype(str)
    )
    if len(selected_frame_paths) != len(selection):
        raise ValueError("union-context cache selection is not covered by frame paths")
    _require_media_access(selected_frame_paths)

    output_key = f"{replica}_root_relative_path"
    output_root = _bound_path(repo_root, payload["outputs"][output_key])
    if output_root.exists():
        raise FileExistsError(f"union-context cache output already exists: {output_root}")
    cache = payload["cache_contract"]
    start = time.perf_counter()
    build_audit = build_visual_interaction_cache(
        VisualInteractionCacheConfig(
            frame_context_csv=frame_path,
            output_dir=output_root,
            selection_csv=selection_path,
            image_size=int(cache["image_size"]),
            padding_ratio=float(cache["padding_ratio"]),
            max_contexts=cache["max_contexts"],
            source_type=str(payload["source_type"]),
            preview_limit=int(cache["preview_limit"]),
            checkpoint_every=int(cache["checkpoint_every"]),
            resume=False,
        )
    )
    runtime_seconds = time.perf_counter() - start
    expected_ready = int(cache["expected_ready_rows"])
    if not build_audit.get("valid"):
        raise ValueError(f"union-context cache build failed: {build_audit['errors']}")
    if int(build_audit.get("selected_rows", -1)) != len(selection):
        raise ValueError("union-context cache manifest row count drift")
    if int(build_audit.get("available_rows", -1)) != expected_ready:
        raise ValueError(
            "union-context cache media readiness failed: "
            f"{build_audit.get('available_rows')}!={expected_ready}"
        )
    if build_audit.get("selection_sha256") != payload["inputs"]["selection_csv"]["sha256"]:
        raise ValueError("union-context cache selection hash drift")
    if build_audit.get("lineage_scope") != LINEAGE_SCOPE:
        raise ValueError("union-context cache lineage scope drift")
    if build_audit.get("human_review_complete") is not False:
        raise ValueError("union-context cache review claim drift")

    run_manifest = {
        "schema_version": (
            "classification_v2.legacy_development_l6."
            "union_context_short_cache_run.v1"
        ),
        "status": "PASS_LEGACY_DEVELOPMENT_L6_UNION_CONTEXT_SHORT_CACHE",
        "run_id": f"l6_union_context_{replica}_20260716",
        "replica": replica,
        "lineage_scope": LINEAGE_SCOPE,
        "canonical_source_name": str(payload["canonical_source_name"]),
        "source_type": str(payload["source_type"]),
        "dataset_id": str(payload["dataset_id"]),
        "human_review_complete": False,
        "reviewed_or_final_claim_allowed": False,
        "q2_claim_allowed": False,
        "canonical_full_oof_authorized": False,
        "outer_holdout_predictions_authorized": False,
        "config_path": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "code_sha": _git(repo_root, "rev-parse", "HEAD").strip(),
        "process_id": os.getpid(),
        "runtime_seconds": runtime_seconds,
        "source_media_paths": int(selected_frame_paths.nunique()),
        "source_media_reads": int(build_audit["video_decode_count"]),
        "selection_csv_sha256": payload["inputs"]["selection_csv"]["sha256"],
        "build_audit_sha256": _file_sha256(output_root / "visual_context_cache_audit.json"),
        "build_audit": build_audit,
        "output_root": str(output_root),
        "output_manifest": str(output_root / "visual_context_manifest.csv"),
        "errors": [],
        "valid": True,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )
    return run_manifest


def _load_config(path: Path) -> tuple[dict[str, Any], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("union-context cache config schema mismatch")
    if payload.get("lineage_scope") != LINEAGE_SCOPE:
        raise ValueError("union-context cache lineage scope mismatch")
    for name in (
        "human_review_complete",
        "reviewed_or_final_claim_allowed",
        "q2_claim_allowed",
        "canonical_full_oof_authorized",
        "outer_holdout_predictions_authorized",
    ):
        if payload.get(name) is not False:
            raise ValueError(f"union-context cache claim flag drift: {name}")
    required = {
        "inputs",
        "implementation",
        "cache_contract",
        "outputs",
        "execution_guard",
    }
    if not required.issubset(payload):
        raise ValueError("union-context cache config is incomplete")
    return payload, path.parents[2]


def _validate_bound_inputs(payload: dict[str, Any], repo_root: Path) -> None:
    for name, entry in payload["inputs"].items():
        path = _bound_path(repo_root, entry["path"])
        if _file_sha256(path) != entry["sha256"]:
            raise ValueError(f"union-context cache input hash mismatch: {name}")


def _validate_implementation(payload: dict[str, Any], repo_root: Path) -> None:
    for name, entry in payload["implementation"].items():
        path = _bound_path(repo_root, entry["path"])
        if entry["sha256"] == "TO_BE_FILLED":
            raise ValueError(f"union-context cache implementation hash missing: {name}")
        if _file_sha256(path) != entry["sha256"]:
            raise ValueError(f"union-context cache implementation hash mismatch: {name}")


def _require_media_access(paths: pd.Series) -> None:
    errors: list[str] = []
    for value in sorted(set(paths)):
        if not value:
            errors.append("blank_resolved_media_path")
            continue
        try:
            exists = Path(value).is_file()
        except OSError as exc:
            errors.append(f"media_access_error={value}:{exc}")
            continue
        if not exists:
            errors.append(f"media_not_accessible={value}")
    if errors:
        raise PermissionError(
            "union-context cache media preflight failed before output creation: "
            f"{errors[:5]}"
        )


def _git_guard(payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    guard = payload["execution_guard"]
    entries = [
        line
        for line in _git(
            repo_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ).splitlines()
        if line.strip()
    ]
    observed = sorted(_status_path(line) for line in entries)
    allowed = sorted(
        str(path).replace("\\", "/") for path in guard["allowed_dirty_paths"]
    )
    unexpected = sorted(set(observed) - set(allowed))
    required = [
        str(path).replace("\\", "/")
        for path in guard["required_tracked_paths"]
    ]
    untracked = [
        path
        for path in required
        if subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", path],
            capture_output=True,
            check=False,
            text=True,
        ).returncode
        != 0
    ]
    errors: list[str] = []
    if unexpected:
        errors.append(f"unexpected_dirty_paths={unexpected}")
    if untracked:
        errors.append(f"required_paths_untracked={untracked}")
    return {
        "code_sha": _git(repo_root, "rev-parse", "HEAD").strip(),
        "dirty_entries": entries,
        "allowed_dirty_paths": allowed,
        "observed_dirty_paths": observed,
        "unexpected_dirty_paths": unexpected,
        "required_tracked_paths": required,
        "untracked_required_paths": untracked,
        "errors": errors,
        "valid": not errors,
    }


def _bound_path(repo_root: Path, value: str) -> Path:
    path = (repo_root / value).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes repository: {value}") from error
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _status_path(line: str) -> str:
    value = line[3:].strip()
    if " -> " in value:
        value = value.rsplit(" -> ", maxsplit=1)[1]
    return value.replace("\\", "/")


if __name__ == "__main__":
    main()
