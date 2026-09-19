"""Read-only run-snapshot tool: refuses running stages, never auto-discovers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "classification_v2"
    / "model_research"
    / "snapshot_completed_run.py"
)


def _load_module():
    name = "classification_v2_snapshot_completed_run"
    spec = importlib.util.spec_from_file_location(name, SNAPSHOT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # ``dataclass(slots=True)`` resolves annotations through ``sys.modules``,
    # so the module must be registered before it is executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _fake_run(tmp_path: Path, status: str) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "run_root"
    run_root.mkdir(exist_ok=True)
    output = run_root / "stage_output.csv"
    output.write_text("window_id,cx_n\nw0,0.5\n", encoding="utf-8")
    manifest = run_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_id": "run_2026_07_26_a",
                "base_git_sha": "4fc94c2d3c9ebba5d6dc2ba6ad0aa5e7149e9c98",
                "config_sha256": "a" * 64,
                "motion_schema_version": "classification_v2.motion_tensor.v2",
                "motion_schema_hash": "b" * 64,
                "stages": {
                    "native_evidence": {
                        "status": status,
                        "start_time": "2026-07-26T01:00:00Z",
                        "end_time": "2026-07-26T02:00:00Z",
                        "exit_code": 0,
                        "input_paths": ["inputs/frame_local.csv"],
                        "output_paths": [str(output)],
                        "pass_fail_reason": "all native evidence gates passed",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    log = run_root / "stage.log"
    log.write_text("stage finished\n", encoding="utf-8")
    return run_root, manifest, log


def _request(tmp_path: Path, status: str, **overrides):
    run_root, manifest, log = _fake_run(tmp_path, status)
    return MODULE.SnapshotRequest(
        run_root=run_root,
        manifest_path=manifest,
        log_path=log,
        command_line="python scripts/classification_v2/run_lineage_stage.py ...",
        expected_stage="native_evidence",
        **overrides,
    )


def test_completed_stage_is_snapshotted_with_hashes(tmp_path: Path) -> None:
    snapshot = MODULE.build_snapshot(_request(tmp_path, "PASS"))
    for field in MODULE.SNAPSHOT_FIELDS:
        assert field in snapshot
    assert snapshot["STAGE_STATUS"] == "PASS"
    assert snapshot["RUN_ID"] == "run_2026_07_26_a"
    assert snapshot["MOTION_SCHEMA_VERSION"] == "classification_v2.motion_tensor.v2"
    hashes = snapshot["OUTPUT_HASHES"]
    assert hashes and all(value and len(value) == 64 for value in hashes.values())
    contract = snapshot["snapshot_contract"]
    assert contract["auto_discovery_used"] is False
    assert contract["active_production_output_inspected"] is False
    assert contract["marked_pass"] is True
    assert contract["published_as_canonical"] is False


def test_running_stage_is_refused_without_the_explicit_flag(tmp_path: Path) -> None:
    with pytest.raises(MODULE.SnapshotRefused, match="RUNNING"):
        MODULE.build_snapshot(_request(tmp_path, "RUNNING"))


def test_running_stage_with_flag_is_metadata_only(tmp_path: Path) -> None:
    snapshot = MODULE.build_snapshot(
        _request(tmp_path, "RUNNING", allow_running_metadata_only=True)
    )
    assert snapshot["STAGE_STATUS"] == "RUNNING_METADATA_ONLY"
    contract = snapshot["snapshot_contract"]
    assert contract["metadata_only"] is True
    assert contract["outputs_hashed"] is False
    assert contract["marked_pass"] is False
    assert contract["published_as_canonical"] is False
    assert all(value is None for value in snapshot["OUTPUT_HASHES"].values())
    assert "partially written" in contract["hashing_skipped_reason"]


def test_stage_mismatch_and_missing_paths_fail_closed(tmp_path: Path) -> None:
    request = _request(tmp_path, "PASS")
    request.expected_stage = "behavior_review_units"
    snapshot = MODULE.build_snapshot(request)
    assert snapshot["STAGE_STATUS"] == "UNKNOWN"

    missing = _request(tmp_path, "PASS")
    missing.log_path = tmp_path / "does_not_exist.log"
    with pytest.raises(MODULE.SnapshotRefused, match="log_path"):
        MODULE.build_snapshot(missing)

    blank = _request(tmp_path, "PASS")
    blank.command_line = "  "
    with pytest.raises(MODULE.SnapshotRefused, match="command-line"):
        MODULE.build_snapshot(blank)


def test_cli_requires_every_explicit_argument() -> None:
    parser = MODULE.build_parser()
    required = {
        action.dest
        for action in parser._actions  # noqa: SLF001 - argparse introspection
        if getattr(action, "required", False)
    }
    assert required == {
        "run_root",
        "manifest_path",
        "log_path",
        "command_line",
        "expected_stage",
    }
    with pytest.raises(SystemExit):
        parser.parse_args([])
