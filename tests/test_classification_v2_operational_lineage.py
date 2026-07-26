from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pig_behavior.classification_v2.lineage_config import (
    load_config,
    reject_stale_path,
    resolve_run_root,
)

SCRIPTS = Path(__file__).parents[1] / "scripts" / "classification_v2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lineage_preflight  # noqa: E402
import run_lineage_stage  # noqa: E402


def _config() -> dict:
    _, config = load_config()
    return config


def _load_train_ready_candidate_module() -> object:
    path = (
        SCRIPTS
        / "02_train_ready_exports"
        / "classification_v2_export_train_ready_candidate.py"
    )
    spec = importlib.util.spec_from_file_location("train_ready_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_source_paths_and_xml_selection() -> None:
    config = _config()
    source = config["source"]
    assert source["legacy_export"].startswith(
        "outputs/legacy_16f_rebuild/"
    )
    assert source["legacy_crop_root"].startswith(
        "outputs/legacy_16f_rebuild/"
    )
    assert len(source["cvat_behavior_xml"]) == 12
    assert source["cvat_behavior_xml"] == sorted(source["cvat_behavior_xml"])


def test_stale_source_path_rejected() -> None:
    with pytest.raises(ValueError, match="STALE_SOURCE_PATH"):
        reject_stale_path("data/raw/legacy_full_multigt_masked_nodup_16f/crops")


def test_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / "crops").mkdir(parents=True)
    (source_root / "videos").mkdir()
    (source_root / "cvat").mkdir()
    (source_root / "legacy.csv").write_text("a\n1\n", encoding="utf-8")
    (source_root / "audit.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    (source_root / "roi.json").write_text("{}", encoding="utf-8")
    (source_root / "cvat" / "a.xml").write_text(
        "<annotations><track><box/></track></annotations>",
        encoding="utf-8",
    )
    config = _config()
    config = copy.deepcopy(config)
    config["source"].update(
        {
            "legacy_export": "legacy.csv",
            "legacy_completion_audit": "audit.json",
            "legacy_crop_root": "crops",
            "cvat_behavior_root": "cvat",
            "cvat_behavior_xml": ["cvat/a.xml"],
            "roi": "roi.json",
            "video_root": "videos",
            "expected_legacy_sha256": "wrong",
            "expected_legacy_rows": 1,
            "expected_legacy_crop_files": 0,
            "expected_cvat_xml_count": 1,
            "expected_cvat_box_rows": 1,
            "expected_roi_sha256": hashlib.sha256(b"{}").hexdigest(),
            "expected_cvat_xml_fingerprint": "wrong",
            "expected_crop_fingerprint": hashlib.sha256(b"").hexdigest(),
            "expected_mixed_rows": 2,
        }
    )
    report = lineage_preflight.source_bundle_report(source_root, config)
    assert report["valid"] is False


def test_unauthorized_stage_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    root, config = load_config()
    monkeypatch.setattr(run_lineage_stage, "load_config", lambda _: (root, config))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_lineage_stage.py", "--config", "ignored", "--stage", "source_merge"],
    )
    assert run_lineage_stage.main() == 2


def test_collision_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, config = load_config()
    config = copy.deepcopy(config)
    config["authorization"]["authorizes_source_rebuild"] = True
    monkeypatch.setenv("CLASSIFICATION_V2_RUN_ROOT", str(tmp_path))
    output = tmp_path / "candidates" / "source_merge"
    output.mkdir(parents=True)
    (output / "merged_frame_objects.csv").write_text("collision", encoding="utf-8")
    monkeypatch.setattr(run_lineage_stage, "load_config", lambda _: (root, config))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_lineage_stage.py", "--config", "ignored", "--stage", "source_merge"],
    )
    assert run_lineage_stage.main() == 3


def test_one_stage_command_has_exactly_twelve_xmls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, config = load_config()
    config = copy.deepcopy(config)
    config["authorization"]["authorizes_source_rebuild"] = True
    monkeypatch.setenv("CLASSIFICATION_V2_RUN_ROOT", str(tmp_path))
    command = run_lineage_stage._command(root, config, "source_merge")
    assert command.count("--cvat-tracking-xml") == 12
    assert "--cvat-tracking-dir" not in command
    assert command[-1].endswith("merged_frame_objects_lineage.json")


def test_frame_local_schema_is_a_candidate_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, config = load_config()
    monkeypatch.setenv("CLASSIFICATION_V2_RUN_ROOT", str(tmp_path))

    command = run_lineage_stage._command(root, config, "frame_local")
    schema_path = Path(command[command.index("--schema-json") + 1])

    assert schema_path == (
        tmp_path
        / "candidates"
        / "frame_local"
        / "frame_local_primitives_schema.json"
    )
    assert "docs/classification_v2/scientific_contract_v1" not in (
        schema_path.as_posix()
    )


def test_dry_run_does_not_invoke_downstream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, config = load_config()
    config = copy.deepcopy(config)
    config["authorization"]["authorizes_source_rebuild"] = True
    monkeypatch.setenv("CLASSIFICATION_V2_RUN_ROOT", str(tmp_path))
    called = []
    monkeypatch.setattr(
        run_lineage_stage.subprocess,
        "run",
        lambda *args, **kwargs: called.append(args) or None,
    )
    monkeypatch.setattr(run_lineage_stage, "load_config", lambda _: (root, config))
    monkeypatch.setattr(
        run_lineage_stage,
        "source_bundle_report",
        lambda *_: {"valid": True},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_lineage_stage.py",
            "--config",
            "ignored",
            "--stage",
            "source_merge",
            "--dry-run",
        ],
    )
    assert run_lineage_stage.main() == 0
    assert called == []
    assert "automatic_downstream_execution" in capsys.readouterr().out


def test_windows_run_root_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    root, config = load_config()
    monkeypatch.setenv(
        "CLASSIFICATION_V2_RUN_ROOT",
        r"C:\pig_runs\classification_v2_synthetic",
    )
    assert str(resolve_run_root(root, config)).endswith(
        "classification_v2_synthetic"
    )


def test_synthetic_lineage_interface(tmp_path: Path) -> None:
    root, config = load_config()
    config = copy.deepcopy(config)
    config["authorization"]["authorizes_source_rebuild"] = True
    config["run_root_default"] = str(tmp_path)
    errors = lineage_preflight.validate_config(root, config)
    assert errors == []
    assert run_lineage_stage._command(root, config, "source_merge")


def test_all_stage_commands_are_complete_and_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, config = load_config()
    monkeypatch.setenv("CLASSIFICATION_V2_RUN_ROOT", str(tmp_path))
    commands = {
        stage: run_lineage_stage._command(root, config, stage)
        for stage in lineage_preflight.EXPECTED_STAGE_IDS
    }
    assert set(commands) == set(lineage_preflight.EXPECTED_STAGE_IDS)
    assert all(len(command) > 2 for command in commands.values())
    assert all(command[0] == sys.executable for command in commands.values())
    assert len({tuple(command) for command in commands.values()}) == 14
    assert len(run_lineage_stage._commands(root, config, "source_merge")) == 1
    assert len(run_lineage_stage._commands(root, config, "train_ready")) == 2


def test_bounded_synthetic_lineage_uses_runner_interface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, loaded = load_config()
    config = copy.deepcopy(loaded)
    for flag in config["authorization"]:
        config["authorization"][flag] = True
    monkeypatch.setenv("CLASSIFICATION_V2_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(run_lineage_stage, "load_config", lambda _: (root, config))
    monkeypatch.setattr(
        run_lineage_stage,
        "source_bundle_report",
        lambda *_: {"valid": True},
    )
    monkeypatch.setattr(run_lineage_stage, "_upstream_errors", lambda *_: [])
    invoked: list[list[str]] = []
    current_stage = ""

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        invoked.append(command)
        artifact = run_lineage_stage._artifact_path(root, config, current_stage)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("synthetic\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    def fake_builder(**kwargs: object) -> None:
        manifest = Path(str(kwargs["candidate_manifest_path"]))
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text('{"synthetic":true}', encoding="utf-8")

    monkeypatch.setattr(run_lineage_stage.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_lineage_stage,
        "build_candidate_artifact_manifest",
        fake_builder,
    )
    for stage_key in lineage_preflight.EXPECTED_STAGE_IDS:
        current_stage = stage_key
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_lineage_stage.py",
                "--config",
                "ignored",
                "--stage",
                stage_key,
            ],
        )
        assert run_lineage_stage.main() == 0
    assert len(invoked) == 15


def test_train_ready_candidate_uses_explicit_whitelist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_train_ready_candidate_module()
    windows = tmp_path / "windows.csv"
    windows.write_text(
        "window_length_frames,behavior_window_label,"
        "window_valid_for_main_train,window_sample_weight\n"
        "6,walk,true,1.0\n",
        encoding="utf-8",
    )
    trainer_contract = tmp_path / "trainer_contract.json"
    trainer_contract.write_text(
        json.dumps({"tabular_feature_whitelist": ["window_length_frames"]}),
        encoding="utf-8",
    )
    output = tmp_path / "train_ready"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classification_v2_export_train_ready_candidate.py",
            "--input-csv",
            str(windows),
            "--output-dir",
            str(output),
            "--trainer-contract-json",
            str(trainer_contract),
        ],
    )
    assert module.main() == 0
    assert (output / "X_window_features.csv").read_text(
        encoding="utf-8"
    ) == "window_length_frames\n6\n"


def test_no_automatic_promotion_or_downstream_policy() -> None:
    config = _config()
    assert config["policy"]["automatic_promotion"] is False
    assert config["policy"]["automatic_downstream_execution"] is False
    assert all(value is False for value in config["authorization"].values())
