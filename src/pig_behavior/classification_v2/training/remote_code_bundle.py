"""Build and verify one Git-native remote runtime bundle for post-S1 runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA = "classification_v2.remote_code_bundle.v1"
PACKAGER_VERSION = 1
RUNTIME_PATHS = (
    "src/pig_behavior",
    "scripts/classification_v2",
    "pyproject.toml",
    "uv.lock",
    "docs/classification_v2/corrected_pooled_route_20260806",
)
CRITICAL_PATHS = (
    "scripts/classification_v2/04_baselines_smokes/"
    "classification_v2_run_post_s1_resolution_screen.py",
    "src/pig_behavior/classification_v2/datasets/resolution_pipeline.py",
    "src/pig_behavior/classification_v2/training/stage1_temporal_screening.py",
    "src/pig_behavior/classification_v2/training/remote_input_resolution.py",
    "src/pig_behavior/classification_v2/training/post_s1_host_binding.py",
    "src/pig_behavior/classification_v2/training/post_s1_resolution_screening.py",
)
CRITICAL_MODULES = (
    "pig_behavior",
    "pig_behavior.classification_v2.datasets.resolution_pipeline",
    "pig_behavior.classification_v2.training.stage1_temporal_screening",
    "pig_behavior.classification_v2.training.remote_input_resolution",
    "pig_behavior.classification_v2.training.post_s1_host_binding",
    "pig_behavior.classification_v2.training.post_s1_resolution_screening",
)


class RemoteCodeBundleError(ValueError):
    """Raised when a canonical runtime archive cannot be trusted."""


def build_remote_code_bundle(
    *, repository_root: Path,
    requested_git_sha: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Export a complete runtime surface from Git objects, never worktree dirt."""

    repository_root = Path(repository_root).resolve()
    canonical_sha = _canonical_sha(repository_root, requested_git_sha)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RemoteCodeBundleError(f"bundle output already exists={output_dir}")
    output_dir.mkdir(parents=True)
    archive_path = output_dir / f"post_s1_runtime_{canonical_sha}.tar.gz"
    _write_git_archive(repository_root, canonical_sha, archive_path)
    members = _archive_members(archive_path)
    _validate_members(members)
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "canonical_git_sha": canonical_sha,
        "bundle_filename": archive_path.name,
        "bundle_sha256": _sha256_file(archive_path),
        "bundle_size_bytes": archive_path.stat().st_size,
        "file_count": len(members),
        "included_top_level_paths": list(RUNTIME_PATHS),
        "critical_file_sha256": {
            path: _git_blob_sha256(repository_root, canonical_sha, path)
            for path in CRITICAL_PATHS
        },
        "packaging_method": "git_archive_tar_gzip",
        "packager_version": PACKAGER_VERSION,
        "working_tree_dirt_included": False,
    }
    manifest_path = output_dir / "remote_code_bundle_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return {"archive_path": archive_path, "manifest_path": manifest_path, **manifest}


def verify_remote_code_bundle(
    *,
    archive_path: Path,
    manifest_path: Path,
    expected_git_sha: str,
) -> dict[str, Any]:
    """Verify a bundle and prove its imports from an isolated extraction root."""

    archive_path = Path(archive_path).resolve()
    manifest = _read_json(Path(manifest_path))
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise RemoteCodeBundleError("unsupported remote code bundle schema")
    if manifest.get("canonical_git_sha") != expected_git_sha:
        raise RemoteCodeBundleError("remote code bundle Git SHA drifted")
    if manifest.get("bundle_sha256") != _sha256_file(archive_path):
        raise RemoteCodeBundleError("remote code bundle SHA256 audit failed")
    if manifest.get("working_tree_dirt_included") is not False:
        raise RemoteCodeBundleError("remote code bundle dirt policy drifted")
    members = _archive_members(archive_path)
    _validate_members(members)
    with tempfile.TemporaryDirectory(prefix="post_s1_bundle_") as temporary:
        extraction_root = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extraction_root, filter="data")
        _prove_isolated_imports(extraction_root)
        _prove_cli_help(extraction_root)
    return {"status": "PASS", "file_count": len(members), "manifest": manifest}


def _write_git_archive(repository_root: Path, git_sha: str, archive_path: Path) -> None:
    command = ["git", "archive", "--format=tar", git_sha, "--", *RUNTIME_PATHS]
    with subprocess.Popen(
        command,
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        assert process.stdout is not None
        with gzip.GzipFile(archive_path, "wb", mtime=0) as compressed:
            shutil.copyfileobj(process.stdout, compressed)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RemoteCodeBundleError(f"git archive failed={stderr.strip()}")


def _prove_isolated_imports(extraction_root: Path) -> None:
    module_list = repr(CRITICAL_MODULES)
    root_literal = repr(str(extraction_root))
    code = (
        "import importlib,pathlib,sys;"
        f"root=pathlib.Path({root_literal}).resolve();"
        "sys.path[:]=[str(root/'src')]+[path for path in sys.path if path];"
        f"modules={module_list};"
        "paths={name:pathlib.Path(importlib.import_module(name).__file__).resolve() "
        "for name in modules};"
        "assert all(path.is_relative_to(root) for path in paths.values()),paths;"
        "print(paths)"
    )
    _run_isolated(extraction_root, code)


def _prove_cli_help(extraction_root: Path) -> None:
    script = extraction_root / CRITICAL_PATHS[0]
    code = (
        "import runpy,sys;sys.argv=["
        + repr(str(script))
        + ", '--help'];runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    _run_isolated(extraction_root, code)


def _run_isolated(extraction_root: Path, code: str) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=extraction_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RemoteCodeBundleError(
            f"isolated bundle proof failed={result.stderr.strip() or result.stdout.strip()}"
        )


def _canonical_sha(repository_root: Path, requested_git_sha: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{requested_git_sha}^{{commit}}"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RemoteCodeBundleError("requested canonical Git SHA is unavailable")
    return result.stdout.strip()


def _git_blob_sha256(repository_root: Path, git_sha: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{git_sha}:{path}"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RemoteCodeBundleError(f"critical runtime path missing={path}")
    return hashlib.sha256(result.stdout).hexdigest()


def _archive_members(archive_path: Path) -> list[str]:
    with tarfile.open(archive_path, "r:gz") as archive:
        return sorted(member.name for member in archive.getmembers() if member.isfile())


def _validate_members(members: list[str]) -> None:
    if not members:
        raise RemoteCodeBundleError("remote code bundle is empty")
    if any(path.startswith(("outputs/", "datasets/", ".git/", ".venv/")) for path in members):
        raise RemoteCodeBundleError("remote code bundle contains forbidden content")
    missing = [path for path in CRITICAL_PATHS if path not in members]
    if missing:
        raise RemoteCodeBundleError(f"remote code bundle misses critical paths={missing}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RemoteCodeBundleError(f"invalid bundle manifest={path}") from error
    if not isinstance(value, dict):
        raise RemoteCodeBundleError("bundle manifest must be an object")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "BUNDLE_SCHEMA",
    "CRITICAL_MODULES",
    "CRITICAL_PATHS",
    "RUNTIME_PATHS",
    "RemoteCodeBundleError",
    "build_remote_code_bundle",
    "verify_remote_code_bundle",
]
