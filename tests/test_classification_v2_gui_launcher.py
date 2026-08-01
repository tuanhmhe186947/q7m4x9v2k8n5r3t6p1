from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "classification_v2"
    / "01_review_units_gui"
    / "launch_classification_v2_gui.py"
)
DEFAULT_PROFILE = (
    ROOT / "configs" / "classification_v2" / "gui_operator_profile_v1.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("classification_v2_gui_launcher", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profile(tmp_path: Path) -> dict[str, object]:
    common = {}
    for name in (
        "review_units_csv",
        "frame_features_csv",
        "video_root",
        "raw_root",
        "roi_coco_json",
    ):
        path = tmp_path / name
        path.touch()
        common[name] = str(path)
    return {
        "schema_version": "classification_v2.gui_operator_profile.v1",
        "default_reviewer": "TuanHM",
        "common": common,
        "behavior": {"output_dir": str(tmp_path / "behavior-output")},
        "mini_cvat": {"output_root": str(tmp_path / "mini-cvat-output")},
    }


def test_default_profile_pins_adjusted_toy_roi() -> None:
    profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    assert profile["common"]["roi_coco_json"].endswith(
        "ROI_annotations.toy_adjusted.coco.json"
    )


def test_behavior_command_expands_profile_without_opening_gui(tmp_path: Path) -> None:
    module = _load_module()
    profile = _profile(tmp_path)
    command, required = module._behavior_command(
        profile,
        max_items=12,
        start_review_unit_id="unit_review_00030931",
        prepare_frame_cache_only=False,
    )
    assert command[0] == sys.executable
    assert command[1].endswith("review_final_behavior_gui_v1.py")
    assert command[command.index("--max-items") + 1] == "12"
    assert command[command.index("--start-review-unit-id") + 1] == (
        "unit_review_00030931"
    )
    assert str(tmp_path / "roi_coco_json") in command
    assert module.BEHAVIOR_GUI in required


def test_mini_cvat_command_uses_isolated_session_and_explicit_apply(
    tmp_path: Path,
) -> None:
    module = _load_module()
    profile = _profile(tmp_path)
    source_csv = tmp_path / "source.csv"
    source_xml = tmp_path / "source.xml"
    source_csv.touch()
    source_xml.touch()
    command, required = module._mini_cvat_command(
        profile,
        session_name="fight_move_identity_18db1b2",
        reviewer="TuanHM",
        review_item_ids=["unit_review_00030931", "unit_review_00030932"],
        editable_pig_ids=["ID_4", "ID_5", "ID_6"],
        apply_source_csvs=[source_csv],
        apply_source_xml=source_xml,
        apply_group_id="",
    )
    output = Path(command[command.index("--output-dir") + 1])
    assert output.name == "fight_move_identity_18db1b2"
    assert command.count("--review-item-id") == 2
    assert command.count("--editable-pig-id") == 3
    assert command.count("--apply-source-csv") == 1
    assert source_csv.resolve() in required
    assert source_xml.resolve() in required


def test_mini_cvat_rejects_partial_source_apply(tmp_path: Path) -> None:
    module = _load_module()
    with pytest.raises(module.LauncherError, match="requires_both_csv_and_xml"):
        module._mini_cvat_command(
            _profile(tmp_path),
            session_name="case-a",
            reviewer="TuanHM",
            review_item_ids=["unit-a"],
            editable_pig_ids=["ID_4"],
            apply_source_csvs=[tmp_path / "source.csv"],
            apply_source_xml=None,
            apply_group_id="",
        )


@pytest.mark.parametrize("session_name", ["../escape", "nested/case", "..", ""])
def test_mini_cvat_rejects_unsafe_session_name(
    tmp_path: Path,
    session_name: str,
) -> None:
    module = _load_module()
    with pytest.raises(module.LauncherError, match="unsafe_session_name"):
        module._session_output_dir(_profile(tmp_path), session_name)


def test_behavior_dry_run_never_spawns_gui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile(tmp_path)), encoding="utf-8")

    def fail_if_spawned(*_args, **_kwargs):
        raise AssertionError("dry-run spawned a process")

    monkeypatch.setattr(module.subprocess, "run", fail_if_spawned)
    result = module.main(
        ["--profile", str(profile_path), "behavior", "--dry-run"]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert "DRY_RUN=YES" in output
    assert "review_final_behavior_gui_v1.py" in output
