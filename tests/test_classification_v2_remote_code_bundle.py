from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from pig_behavior.classification_v2.training import remote_code_bundle as bundle


def _run(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _minimal_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run("git", "init", cwd=repository)
    _run("git", "config", "user.email", "test@example.invalid", cwd=repository)
    _run("git", "config", "user.name", "Test", cwd=repository)
    for relative in bundle.CRITICAL_PATHS:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# tracked\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    authority = repository / "docs/classification_v2/corrected_pooled_route_20260806/authority.json"
    authority.parent.mkdir(parents=True, exist_ok=True)
    authority.write_text("{}\n", encoding="utf-8")
    _run("git", "add", ".", cwd=repository)
    _run("git", "commit", "-m", "fixture", cwd=repository)
    return repository, _run("git", "rev-parse", "HEAD", cwd=repository)


def test_bundle_is_git_native_and_excludes_untracked_dirt(tmp_path: Path) -> None:
    repository, git_sha = _minimal_repository(tmp_path)
    (repository / "src/pig_behavior/untracked_owner_file.py").write_text(
        "owner dirt\n",
        encoding="utf-8",
    )
    report = bundle.build_remote_code_bundle(
        repository_root=repository,
        requested_git_sha=git_sha,
        output_dir=tmp_path / "bundle",
    )

    manifest = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["canonical_git_sha"] == git_sha
    assert manifest["working_tree_dirt_included"] is False
    assert bundle._archive_members(Path(report["archive_path"])) == sorted(
        bundle._archive_members(Path(report["archive_path"]))
    )
    assert "src/pig_behavior/untracked_owner_file.py" not in bundle._archive_members(
        Path(report["archive_path"])
    )
    assert set(bundle.CRITICAL_PATHS).issubset(
        bundle._archive_members(Path(report["archive_path"]))
    )


def test_tampered_or_wrong_sha_bundle_fails_closed(tmp_path: Path) -> None:
    repository, git_sha = _minimal_repository(tmp_path)
    report = bundle.build_remote_code_bundle(
        repository_root=repository,
        requested_git_sha=git_sha,
        output_dir=tmp_path / "bundle",
    )
    archive = Path(report["archive_path"])
    archive.write_bytes(archive.read_bytes() + b"tamper")

    with pytest.raises(bundle.RemoteCodeBundleError, match="SHA256"):
        bundle.verify_remote_code_bundle(
            archive_path=archive,
            manifest_path=Path(report["manifest_path"]),
            expected_git_sha=git_sha,
        )


def test_bundle_from_wrong_git_sha_fails_closed(tmp_path: Path) -> None:
    repository, git_sha = _minimal_repository(tmp_path)
    report = bundle.build_remote_code_bundle(
        repository_root=repository,
        requested_git_sha=git_sha,
        output_dir=tmp_path / "bundle",
    )

    with pytest.raises(bundle.RemoteCodeBundleError, match="Git SHA drifted"):
        bundle.verify_remote_code_bundle(
            archive_path=Path(report["archive_path"]),
            manifest_path=Path(report["manifest_path"]),
            expected_git_sha="0" * 40,
        )


def test_isolated_import_closure_passes_for_current_canonical_commit(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    git_sha = _run("git", "rev-parse", "HEAD", cwd=repository)
    report = bundle.build_remote_code_bundle(
        repository_root=repository,
        requested_git_sha=git_sha,
        output_dir=tmp_path / "bundle",
    )

    verified = bundle.verify_remote_code_bundle(
        archive_path=Path(report["archive_path"]),
        manifest_path=Path(report["manifest_path"]),
        expected_git_sha=git_sha,
    )

    assert verified["status"] == "PASS"
    shutil.rmtree(Path(report["archive_path"]).parent)
