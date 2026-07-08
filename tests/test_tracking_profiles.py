import importlib.util
from pathlib import Path

from pig_behavior.tracking.profiles import (
    EVAL_CONFIG_OVERRIDES,
    PRESENTATION_PROFILES,
    get_presentation_profile,
)


def test_presentation_profiles_map_to_clear_modes() -> None:
    assert PRESENTATION_PROFILES["bytetrack_raw"]["mode"] == "bytetrack_raw"
    assert PRESENTATION_PROFILES["realtime"]["mode"] == "realtime"
    assert PRESENTATION_PROFILES["hybrid_bytetrack"]["mode"] == "hybrid_bytetrack"

    assert PRESENTATION_PROFILES["bytetrack_raw"]["eval_config"] == "bytetrack_raw"
    assert PRESENTATION_PROFILES["realtime"]["eval_config"] == "realtime_quality_delayed"
    assert PRESENTATION_PROFILES["hybrid_bytetrack"]["eval_config"] == "hybrid_bytetrack_best"


def test_profile_configs_keep_expected_behavior_separation() -> None:
    raw = EVAL_CONFIG_OVERRIDES["bytetrack_raw"]
    realtime = EVAL_CONFIG_OVERRIDES["realtime_quality_delayed"]
    hybrid = EVAL_CONFIG_OVERRIDES["hybrid_bytetrack_best"]

    assert raw["enable_offline_smoothing"] is False
    assert raw["hidden_suffix_id_swap_repair"] is False
    assert raw["realtime_motion_pair_stabilizer"] is False

    assert realtime["enable_offline_smoothing"] is False
    assert realtime["realtime_motion_pair_stabilizer"] is True
    assert realtime["realtime_motion_pair_simple_min_gain"] == 0.005

    assert hybrid["enable_offline_smoothing"] is True
    assert hybrid["overlap_small_box_suppression"] is True
    assert hybrid["hidden_suffix_id_swap_repair"] is True
    assert hybrid["suffix_pair_swap_repair"] is True


def test_get_presentation_profile_returns_mutable_copy() -> None:
    profile = get_presentation_profile("realtime")
    profile["mode"] = "changed"

    assert PRESENTATION_PROFILES["realtime"]["mode"] == "realtime"


def test_run_tracking_mode_lists_profiles_without_video_selection() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args, extra_args = module.parse_args(["--list-modes"])

    assert args.list_modes is True
    assert extra_args == []


def test_run_tracking_mode_accepts_mode_name() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args, extra_args = module.parse_args(
        ["--mode", "realtime_fast", "--task", "eval", "-v", "Pigs291119_000263_30fps"]
    )

    assert args.mode == "realtime_fast"
    assert args.task == "eval"
    assert extra_args == []


def test_run_tracking_mode_compare_modes_default_set() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args, _ = module.parse_args(
        ["--task", "compare", "-v", "Pigs291119_000263_30fps"]
    )

    assert module._selected_modes(args) == [
        "bytetrack_raw",
        "realtime",
        "hybrid_bytetrack",
    ]


def test_run_tracking_mode_has_clear_two_task_model() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args, _ = module.parse_args(
        ["--mode", "hybrid_bytetrack", "--task", "track", "-v", "Pigs291119_000263_30fps"]
    )

    assert args.task == "track"
    assert args.eval_existing is False


def test_run_tracking_mode_rejects_compare_modes_without_compare_task() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    try:
        module.parse_args(
            ["--compare-modes", "--task", "eval", "-v", "Pigs291119_000263_30fps"]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--compare-modes should require --task compare")


def test_run_tracking_mode_compare_summary_writes_csv_and_markdown(tmp_path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    metrics_dir = (
        tmp_path
        / "realtime"
        / "realtime_quality_delayed"
        / "iou0_area0_condarea0_merge0"
    )
    metrics_dir.mkdir(parents=True)
    metrics_csv = metrics_dir / "tracking_metrics.csv"
    metrics_csv.write_text(
        "video_stem,remapped_idsw,remapped_hota_pct,remapped_idf1_pct,fp,fn,evaluated_frames\n"
        "ALL,2,98.5,99.0,1,3,1800\n",
        encoding="utf-8",
    )

    csv_path, md_path = module._write_compare_summary(
        tmp_path,
        {
            "realtime": {
                "tracking_mode": "realtime",
                "eval_config": "realtime_quality_delayed",
            }
        },
    )

    assert csv_path.exists()
    assert md_path.exists()
    summary = csv_path.read_text(encoding="utf-8")
    assert "presentation_mode,tracking_mode,eval_config" in summary
    assert "realtime,realtime,realtime_quality_delayed" in summary
    assert "ALL,2,98.5,99.0,1,3,1800" in summary
