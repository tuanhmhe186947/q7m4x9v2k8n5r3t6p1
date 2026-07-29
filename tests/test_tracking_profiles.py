import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from pig_behavior.evaluation.tracking.lineage import finalize_run_manifest
from pig_behavior.tracking import TrackingConfig, validate_config
from pig_behavior.tracking.method_registry import (
    ACTIVE_SCIENTIFIC_METHOD_IDS,
    SCIENTIFIC_METHOD_REGISTRY,
)
from pig_behavior.tracking.profiles import (
    EVAL_CONFIG_OVERRIDES,
    PRESENTATION_PROFILES,
    RetiredTrackingProfileError,
    get_eval_config,
    get_presentation_profile,
)
from pig_behavior.tracking.profiles.realtime import REALTIME_FAST_CONFIG


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_presentation_profiles_map_to_clear_modes() -> None:
    assert set(PRESENTATION_PROFILES) == {
        "bytetrack_raw",
        "realtime_fast",
        "hybrid_bytetrack",
    }
    assert PRESENTATION_PROFILES["bytetrack_raw"]["mode"] == "bytetrack_raw"
    assert PRESENTATION_PROFILES["realtime_fast"]["mode"] == "realtime"
    assert PRESENTATION_PROFILES["hybrid_bytetrack"]["mode"] == "hybrid_bytetrack"

    assert PRESENTATION_PROFILES["bytetrack_raw"]["eval_config"] == "bytetrack_raw"
    assert PRESENTATION_PROFILES["realtime_fast"]["eval_config"] == "realtime_fast"
    assert PRESENTATION_PROFILES["hybrid_bytetrack"]["eval_config"] == "hybrid_bytetrack_best"


def test_scientific_method_registry_exposes_clean_three_method_core() -> None:
    expected = {
        "bytetrack_raw",
        "hybrid_bytetrack",
        "realtime_fast",
    }

    assert set(ACTIVE_SCIENTIFIC_METHOD_IDS) == expected
    assert set(SCIENTIFIC_METHOD_REGISTRY) == expected
    assert set(PRESENTATION_PROFILES) == expected

    hybrid = SCIENTIFIC_METHOD_REGISTRY["hybrid_bytetrack"]
    assert hybrid.scientific_role == "COMPLETE_OPTIMIZED_OFFLINE_METHOD"
    assert hybrid.future_frame_policy == (
        "POST_VIDEO_ALLOWED_BY_ACCEPTED_LINEAGE_ONLY"
    )
    assert "LIVE_YOLO_TRACK" in hybrid.stage_graph
    assert "H5B_HIDDEN_SUFFIX_OVERLAP_PERSISTENCE" in hybrid.stage_graph
    assert "H4_FAR_CAMERA_GEOMETRY_REPLAY" in hybrid.stage_graph
    assert len(hybrid.stage_graph) == 21
    assert hybrid.execution_authority_status == (
        "HISTORICAL_ARTIFACT_AUTHORITY_ESTABLISHED",
        "ALGORITHMIC_LINEAGE_RECOVERED",
        "EXACT_NUMERICAL_RUNTIME_NOT_RECOVERED",
    )

    realtime = SCIENTIFIC_METHOD_REGISTRY["realtime_fast"]
    assert realtime.future_frame_policy == "CAUSAL_ZERO_DELAY"
    assert "CADENCED_YOLO_PREDICT" in realtime.stage_graph
    assert "LIVE_YOLO_TRACK" not in realtime.stage_graph


def test_realtime_fast_profile_resolves_deterministically(tmp_path: Path) -> None:
    candidate = get_eval_config("realtime_fast")
    synthetic_video = tmp_path / "unused.mp4"
    synthetic_weights = tmp_path / "unused.pt"
    synthetic_video.write_bytes(b"")
    synthetic_weights.write_bytes(b"")
    cfg = TrackingConfig(
        mode="realtime",
        video_path=synthetic_video,
        weights_path=synthetic_weights,
        **candidate,
    )
    validate_config(cfg)

    assert cfg.detect_every_n_frames == 2
    assert cfg.causal_hidden_detection_reservation is False
    assert cfg.enable_offline_smoothing is False
    assert cfg.realtime_motion_pair_stabilizer is False
    assert candidate == REALTIME_FAST_CONFIG
    assert candidate is not REALTIME_FAST_CONFIG


def test_retained_profile_hashes_match_frozen_authority() -> None:
    expected = {
        "bytetrack_raw": (
            "547ae86e3be26671a9a148cb0e613ea1c602a0ff842a977ce9b7f1d217c10e41"
        ),
        "realtime_fast": (
            "9bf4ce6d07423ab517b4705c716e3eb012349b756b7c0591cc3458eac207808d"
        ),
        "hybrid_bytetrack_best": (
            "4eb3d4e2262485d48d425be06fd8a6b3adfd8a01a27b28e76b5a8d55958d1d55"
        ),
    }
    for name, expected_hash in expected.items():
        config = {
            key: EVAL_CONFIG_OVERRIDES[name][key]
            for key in sorted(EVAL_CONFIG_OVERRIDES[name])
        }
        assert _canonical_hash(config) == expected_hash


def test_profile_configs_keep_expected_behavior_separation() -> None:
    raw = EVAL_CONFIG_OVERRIDES["bytetrack_raw"]
    realtime_fast = EVAL_CONFIG_OVERRIDES["realtime_fast"]
    hybrid = EVAL_CONFIG_OVERRIDES["hybrid_bytetrack_best"]

    assert raw["enable_offline_smoothing"] is False
    assert raw["hidden_suffix_id_swap_repair"] is False
    assert raw["realtime_motion_pair_stabilizer"] is False

    for non_hybrid in (raw, realtime_fast):
        assert "near_wall_hidden_geometry_refine" not in non_hybrid
        assert "far_camera_hidden_geometry_refine" not in non_hybrid
        assert "hidden_suffix_id_swap_use_overlap_persistence" not in non_hybrid
    assert "realtime_core_unassigned_tiebreak" not in hybrid
    assert "realtime_core_unassigned_require_score_nondecrease" not in hybrid

    assert realtime_fast["realtime_core_unassigned_tiebreak"] is True
    assert (
        realtime_fast["realtime_core_unassigned_require_score_nondecrease"]
        is True
    )
    assert realtime_fast["realtime_core_unassigned_max_cost_delta"] == 0.01
    assert realtime_fast["realtime_core_unassigned_min_appearance_gain"] == 0.01
    assert realtime_fast["realtime_core_unassigned_min_detection_iou"] == 0.30
    assert realtime_fast["realtime_core_unassigned_max_selected_cost"] == 0.40
    assert realtime_fast["realtime_core_pairwise_tiebreak"] is True
    assert (
        realtime_fast["realtime_core_pairwise_max_total_cost_increase"]
        == 0.05
    )
    assert (
        realtime_fast["realtime_core_pairwise_min_total_appearance_gain"]
        == 0.10
    )
    assert realtime_fast["realtime_core_pairwise_min_detection_iou"] == 0.30

    assert realtime_fast["realtime_visible_better_competitor_prefer"] is True
    assert realtime_fast["realtime_visible_close_competitor_guard"] is True
    assert realtime_fast["realtime_visible_close_competitor_margin"] == 0.08
    assert realtime_fast["realtime_visible_close_competitor_max_cost"] == 0.40
    assert realtime_fast["realtime_lk_point_batching"] is True
    assert (
        realtime_fast["realtime_visible_close_competitor_min_center_x_ratio"]
        == 0.67
    )

    assert hybrid["enable_offline_smoothing"] is True
    assert hybrid["overlap_small_box_suppression"] is True
    assert hybrid["hidden_suffix_id_swap_repair"] is True
    assert hybrid["hidden_suffix_id_swap_use_overlap_persistence"] is True
    assert hybrid["hidden_suffix_id_swap_min_overlap_persistence_frames"] == 2
    assert hybrid["suffix_pair_swap_repair"] is True
    assert hybrid["identity_swap_guard_skip_mixed_occlusion_hold"] is True
    assert hybrid["identity_swap_guard_skip_mixed_occlusion_hold_far_only"] is True
    assert hybrid["identity_swap_guard_far_x_threshold"] == 0.67
    assert hybrid["near_wall_hidden_geometry_refine"] is True
    assert hybrid["near_wall_hidden_geometry_max_gap_frames"] == 30
    assert hybrid["near_wall_hidden_geometry_distance_bbox_scale"] == 0.25
    assert hybrid["near_wall_hidden_geometry_min_width_excess"] == 0.08
    assert hybrid["near_wall_hidden_geometry_max_center_shift"] == 0.04
    assert hybrid["near_wall_hidden_geometry_original_weight"] == 0.50
    assert hybrid["far_camera_hidden_geometry_refine"] is True
    assert hybrid["far_camera_hidden_geometry_x_threshold"] == 0.67
    assert hybrid["far_camera_hidden_geometry_max_future_gap_frames"] == 15
    assert hybrid["far_camera_hidden_geometry_min_height_excess"] == 0.15
    assert hybrid["far_camera_hidden_geometry_min_visible_overlap_iou"] == 0.65
    assert hybrid["far_camera_hidden_geometry_min_overlap_reduction"] == 0.10
    assert hybrid["far_camera_hidden_geometry_max_center_shift"] == 0.12
    assert hybrid["far_camera_hidden_geometry_original_weight"] == 0.10


def test_get_presentation_profile_returns_mutable_copy() -> None:
    profile = get_presentation_profile("realtime_fast")
    profile["mode"] = "changed"

    assert PRESENTATION_PROFILES["realtime_fast"]["mode"] == "realtime"


def test_retired_profiles_fail_with_migration_messages() -> None:
    expected_messages = {
        "realtime": "Use 'realtime_fast'.",
        "realtime_balanced": "historical and unavailable",
        "realtime_quality_delayed": "historical and unavailable",
        "realtime_fast_h1_r2": "rejected experimental profile",
    }
    for name, expected in expected_messages.items():
        try:
            get_presentation_profile(name)
        except RetiredTrackingProfileError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"retired profile remained active: {name}")


def test_retired_names_are_absent_from_active_eval_configs() -> None:
    retired = {
        "realtime",
        "realtime_balanced",
        "realtime_quality_delayed",
        "realtime_fast_h1_r2",
    }
    assert retired.isdisjoint(PRESENTATION_PROFILES)
    assert retired.isdisjoint(EVAL_CONFIG_OVERRIDES)


def test_unknown_profile_still_fails_normally() -> None:
    try:
        get_presentation_profile("arbitrary_unknown_profile")
    except KeyError as exc:
        assert exc.args == ("arbitrary_unknown_profile",)
    else:
        raise AssertionError("unknown profile unexpectedly resolved")


def test_historical_manifest_retains_retired_profile_name(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"status": "planned", "profile": "realtime"}),
        encoding="utf-8",
    )

    finalize_run_manifest(tmp_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["profile"] == "realtime"


def test_current_command_templates_select_realtime_fast() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "scripts/_shortcuts/run_realtime_default.bat",
        "scripts/_shortcuts/run_realtime_skip_frames.bat",
    ):
        command = (root / relative_path).read_text(encoding="utf-8")
        assert "--mode realtime" in command
        assert "--eval-config realtime_fast" in command

    readme = (root / "scripts" / "README.md").read_text(encoding="utf-8")
    assert (
        "--compare-modes bytetrack_raw,realtime_fast,hybrid_bytetrack"
        in readme
    )
    evaluator = (root / "scripts" / "evaluate_tracking.py").read_text(
        encoding="utf-8"
    )
    assert "--mode realtime --eval-config realtime_fast" in evaluator


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


def test_run_tracking_mode_retired_alias_recommends_realtime_fast() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    try:
        module._active_profile_arg("realtime")
    except argparse.ArgumentTypeError as exc:
        assert str(exc) == (
            "Profile 'realtime' has been retired. Use 'realtime_fast'."
        )
    else:
        raise AssertionError("retired realtime alias remained selectable")


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
        "realtime_fast",
        "hybrid_bytetrack",
    ]


def test_run_tracking_mode_compare_rejects_retired_profile() -> None:
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

    try:
        module._selected_modes(args)
    except ValueError as exc:
        assert "realtime_balanced" in str(exc)
        assert "historical and unavailable" in str(exc)
    else:
        raise AssertionError("retired compare profile remained selectable")


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
        "diagnostic_global_graph",
        "realtime",
        "diagnostic_global_graph",
        {
            **module.get_eval_config("realtime_fast"),
            "realtime_motion_pair_stabilizer": True,
        },
    )

    assert metadata["baseline_role"] == "post_video_global_graph_candidate"
    assert metadata["causality_level"] == "post_video_global_graph"
    assert metadata["output_timing_contract"] == "post_video_global_graph"
    assert metadata["declared_delay_frames"] == "-1"
    assert metadata["latency_window_frames"] == ""


def test_run_tracking_mode_science_metadata_marks_fixed_lag_truthfully() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tracking_mode.py"
    spec = importlib.util.spec_from_file_location("run_tracking_mode_script", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    overrides = module.get_eval_config("realtime_fast")
    overrides["realtime_motion_pair_stabilizer"] = True
    overrides["realtime_motion_pair_fixed_lag_frames"] = 15

    metadata = module._mode_science_metadata(
        "diagnostic_fixed_lag",
        "realtime",
        "diagnostic_fixed_lag",
        overrides,
    )

    assert metadata["baseline_role"] == "realtime_quality_fixed_lag_candidate"
    assert metadata["causality_level"] == "fixed_lag_realtime"
    assert metadata["output_timing_contract"] == "fixed_lag_framewise"
    assert metadata["declared_delay_frames"] == "15"
    assert metadata["latency_window_frames"] == "15"


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

    metrics_dir = (
        tmp_path
        / "diagnostic"
        / "post_video_global_graph"
        / "iou0_area0_condarea0_merge0"
    )
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
            "diagnostic": {
                "tracking_mode": "realtime",
                "eval_config": "diagnostic_global_graph",
                "baseline_role": "delayed_repair_diagnostic",
                "causality_level": "post_video_global_graph",
                "uses_offline_smoothing": "false",
                "uses_identity_repair": "true",
                "uses_delayed_repair": "true",
                "detect_every_n_frames": "1",
                "latency_window_frames": "30",
            }
        },
        {
            "diagnostic": {
                "status": "ok",
                "return_code": "0",
                "compare_elapsed_sec": "30.0000",
            }
        },
    )
    runtime_csv_path, runtime_md_path = module._write_runtime_summary(
        tmp_path,
        {
            "diagnostic": {
                "tracking_mode": "realtime",
                "eval_config": "diagnostic_global_graph",
                "baseline_role": "delayed_repair_diagnostic",
                "causality_level": "post_video_global_graph",
            }
        },
        {
            "diagnostic": {
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
    assert (
        "diagnostic,delayed_repair_diagnostic,post_video_global_graph"
        in summary
    )
    assert "ALL,100,98,95,96.94,95.0,92.0,81.0,93.0,94.0,2,4,94.0,98.5" in summary
    runtime_summary = runtime_csv_path.read_text(encoding="utf-8")
    assert (
        "diagnostic,delayed_repair_diagnostic,post_video_global_graph,"
        "realtime,diagnostic_global_graph,ok,0,30.0000"
        in runtime_summary
    )
    assert ",1800,60.0000,1800,60.0000,2.0000" in runtime_summary
    scientific_summary = scientific_csv_path.read_text(encoding="utf-8")
    assert "evaluated_video_count,evaluated_frames" in scientific_summary
    assert (
        "diagnostic,delayed_repair_diagnostic,post_video_global_graph"
        in scientific_summary
    )
