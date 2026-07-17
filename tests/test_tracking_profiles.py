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
    assert PRESENTATION_PROFILES["realtime_balanced"]["mode"] == "realtime"
    assert PRESENTATION_PROFILES["hybrid_bytetrack"]["mode"] == "hybrid_bytetrack"

    assert PRESENTATION_PROFILES["bytetrack_raw"]["eval_config"] == "bytetrack_raw"
    assert PRESENTATION_PROFILES["realtime"]["eval_config"] == "realtime_quality_delayed"
    assert PRESENTATION_PROFILES["realtime_balanced"]["eval_config"] == "realtime_balanced"
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
    assert realtime["realtime_motion_pair_simple_min_gain"] == 0.003

    assert hybrid["enable_offline_smoothing"] is True
    assert hybrid["overlap_small_box_suppression"] is True
    assert hybrid["hidden_suffix_id_swap_repair"] is True
    assert hybrid["suffix_pair_swap_repair"] is True
    assert hybrid["identity_swap_guard_skip_mixed_occlusion_hold"] is True
    assert hybrid["identity_swap_guard_skip_mixed_occlusion_hold_far_only"] is True
    assert hybrid["identity_swap_guard_far_x_threshold"] == 0.67


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
        [
            "--mode",
            "realtime_fast",
            "--task",
            "eval",
            "-v",
            "Pigs291119_000263_30fps",
        ]
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

    args, _ = module.parse_args(["--task", "compare", "-v", "Pigs291119_000263_30fps"])

    assert module._selected_modes(args) == [
        "bytetrack_raw",
        "realtime",
        "hybrid_bytetrack",
    ]


def test_run_tracking_mode_compare_accepts_all_realtime_variants() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args, _ = module.parse_args(
        [
            "--task",
            "compare",
            "--compare-modes",
            "realtime_fast,realtime_balanced,realtime_quality_delayed",
            "-v",
            "Pigs291119_000263_30fps",
        ]
    )

    assert module._selected_modes(args) == [
        "realtime_fast",
        "realtime_balanced",
        "realtime_quality_delayed",
    ]


def test_run_tracking_mode_science_metadata_marks_raw_baseline() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    metadata = module._mode_science_metadata(
        "bytetrack_raw",
        "bytetrack_raw",
        "bytetrack_raw",
        module.get_eval_config("bytetrack_raw"),
    )

    assert metadata["baseline_role"] == "raw_bytetrack_baseline_same_detector_pipeline"
    assert metadata["causality_level"] == "online_raw"
    assert metadata["uses_offline_smoothing"] == "false"
    assert metadata["uses_identity_repair"] == "false"
    assert metadata["uses_delayed_repair"] == "false"


def test_run_tracking_mode_science_metadata_marks_global_graph_truthfully() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    metadata = module._mode_science_metadata(
        "realtime_quality_delayed",
        "realtime",
        "realtime_quality_delayed",
        module.get_eval_config("realtime_quality_delayed"),
    )

    assert metadata["baseline_role"] == "realtime_quality_delayed_candidate"
    assert metadata["causality_level"] == "post_video_global_graph"
    assert metadata["output_timing_contract"] == "post_video_global_graph"
    assert metadata["declared_delay_frames"] == "-1"
    assert metadata["latency_window_frames"] == ""


def test_run_tracking_mode_has_clear_two_task_model() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    args, _ = module.parse_args(
        [
            "--mode",
            "hybrid_bytetrack",
            "--task",
            "track",
            "-v",
            "Pigs291119_000263_30fps",
        ]
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
        module.parse_args(["--compare-modes", "--task", "eval", "-v", "Pigs291119_000263_30fps"])
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

    metrics_dir = tmp_path / "realtime" / "realtime_quality_delayed" / "iou0_area0_condarea0_merge0"
    metrics_dir.mkdir(parents=True)
    metrics_csv = metrics_dir / "tracking_metrics.csv"
    metrics_csv.write_text(
        "video_stem,gt_detections,pred_detections,matches,precision_pct,recall_pct,"
        "mota_pct,motp_iou_pct,idf1_pct,hota_pct,idsw,remapped_idsw,"
        "remapped_mota_pct,remapped_hota_pct,remapped_idf1_pct,remapped_assa_pct,"
        "idmap_coverage_pct,fp,fn,fragments,remapped_fragments,"
        "gap_tolerant_fragments,remapped_gap_tolerant_fragments,tracklets,"
        "remapped_tracklets,evaluated_frames\n"
        "ALL,100,98,95,96.94,95.0,92.0,81.0,93.0,94.0,4,2,94.0,98.5,99.0,"
        "98.0,100.0,1,3,5,2,3,1,8,6,1800\n",
        encoding="utf-8",
    )
    metrics_csv.write_text(
        metrics_csv.read_text(encoding="utf-8")
        + "Pigs291119_000263_30fps,100,98,95,96.94,95.0,92.0,81.0,93.0,"
        "94.0,4,2,94.0,98.5,99.0,98.0,100.0,1,3,5,2,3,1,8,6,1800\n",
        encoding="utf-8",
    )
    assets_csv = metrics_dir / "tracking_eval_assets.csv"
    assets_csv.write_text(
        "video_stem,video_frame_count,video_fps\nPigs291119_000263_30fps,1800,30\n",
        encoding="utf-8",
    )

    csv_path, md_path = module._write_compare_summary(
        tmp_path,
        {
            "realtime": {
                "tracking_mode": "realtime",
                "eval_config": "realtime_quality_delayed",
                "baseline_role": "realtime_quality_delayed_candidate",
                "causality_level": "short_delay_realtime",
                "uses_offline_smoothing": "false",
                "uses_identity_repair": "true",
                "uses_delayed_repair": "true",
                "detect_every_n_frames": "1",
                "latency_window_frames": "30",
            }
        },
        {
            "realtime": {
                "status": "ok",
                "return_code": "0",
                "compare_elapsed_sec": "30.0000",
            }
        },
    )
    runtime_csv_path, runtime_md_path = module._write_runtime_summary(
        tmp_path,
        {
            "realtime": {
                "tracking_mode": "realtime",
                "eval_config": "realtime_quality_delayed",
                "baseline_role": "realtime_quality_delayed_candidate",
                "causality_level": "short_delay_realtime",
            }
        },
        {
            "realtime": {
                "status": "ok",
                "return_code": "0",
                "compare_elapsed_sec": "30.0000",
            }
        },
    )
    scientific_csv_path, scientific_md_path = module._write_scientific_summary(tmp_path)

    assert csv_path.exists()
    assert md_path.exists()
    assert runtime_csv_path.exists()
    assert runtime_md_path.exists()
    assert scientific_csv_path.exists()
    assert scientific_md_path.exists()
    summary = csv_path.read_text(encoding="utf-8")
    assert "presentation_mode,baseline_role,causality_level" in summary
    assert "compare_elapsed_sec,compare_evaluated_fps,compare_realtime_factor" in summary
    assert "realtime,realtime_quality_delayed_candidate,short_delay_realtime" in summary
    assert "ALL,100,98,95,96.94,95.0,92.0,81.0,93.0,94.0,2,4,94.0,98.5" in summary
    runtime_summary = runtime_csv_path.read_text(encoding="utf-8")
    assert (
        "realtime,realtime_quality_delayed_candidate,short_delay_realtime,"
        "realtime,realtime_quality_delayed,ok,0,30.0000" in runtime_summary
    )
    assert ",1800,60.0000,1800,60.0000,2.0000" in runtime_summary
    scientific_summary = scientific_csv_path.read_text(encoding="utf-8")
    assert "evaluated_video_count,evaluated_frames" in scientific_summary
    assert "realtime,realtime_quality_delayed_candidate,short_delay_realtime" in scientific_summary
