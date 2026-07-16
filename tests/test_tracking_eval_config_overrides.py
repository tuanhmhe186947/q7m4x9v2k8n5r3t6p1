import importlib.util
from pathlib import Path

import pytest

from pig_behavior.evaluation.tracking.cli import (
    parse_args as parse_pipeline_args,
)
from pig_behavior.evaluation.tracking.cli import (
    parse_profile_overrides,
    selected_rule_combos,
)
from pig_behavior.tracking.config import TrackingConfig

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_tracking.py"
SPEC = importlib.util.spec_from_file_location("evaluate_tracking_script", SCRIPT_PATH)
assert SPEC is not None
evaluate_tracking_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_tracking_script)
EVAL_CONFIG_OVERRIDES = evaluate_tracking_script.EVAL_CONFIG_OVERRIDES
_selected_eval_configs = evaluate_tracking_script._selected_eval_configs
parse_args = evaluate_tracking_script.parse_args
_selected_rule_combos = evaluate_tracking_script._selected_rule_combos


def test_parse_profile_overrides_coerces_tracking_config_values() -> None:
    allowed_fields = set(TrackingConfig.__dataclass_fields__.keys())

    overrides = parse_profile_overrides(
        [
            "det_conf=0.20",
            "max_raw_detections=64",
            "identity_swap_guard=true",
            "mask_path=null",
        ],
        allowed_fields,
    )

    assert overrides == {
        "det_conf": 0.20,
        "max_raw_detections": 64,
        "identity_swap_guard": True,
        "mask_path": None,
    }


def test_direct_pipeline_cli_keeps_condarea_off_by_default() -> None:
    args = parse_pipeline_args(["--video", "input.mp4"])

    assert not args.use_conditional_area_occlusion_freeze


def test_parse_profile_overrides_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown TrackingConfig override"):
        parse_profile_overrides(["not_a_tracking_field=1"], {"det_conf"})


def test_evaluate_tracking_named_configs_include_base_and_candidate() -> None:
    selected = _selected_eval_configs(
        ["base,iou0_area0_condarea0_merge0_smooth_det020_loose_motion"]
    )

    assert selected == [
        "base",
        "iou0_area0_condarea0_merge0_smooth_det020_loose_motion",
    ]
    candidate = EVAL_CONFIG_OVERRIDES[
        "iou0_area0_condarea0_merge0_smooth_det020_loose_motion"
    ]
    assert candidate["det_conf"] == 0.20
    assert candidate["max_raw_detections"] == 64
    assert candidate["low_conf_max_center_jump"] == 0.10
    assert candidate["low_conf_max_box_jump_scale"] == 2.00


def test_evaluate_tracking_defaults_to_exact_single_config() -> None:
    args, extra_args = parse_args(["-v", "Pigs291119_000263_30fps"])

    assert args.benchmark_compatible is False
    assert args.single_config is False
    assert extra_args == []


def test_evaluate_tracking_benchmark_matrix_is_explicit() -> None:
    args, _ = parse_args(
        ["-v", "Pigs291119_000263_30fps", "--benchmark-compatible"]
    )

    assert args.benchmark_compatible is True


def test_evaluate_tracking_accepts_single_rule_combo() -> None:
    args, _ = parse_args(
        [
            "-v",
            "Pigs291119_000263_30fps",
            "--rule-combo",
            "iou0_area0_condarea0_merge0",
        ]
    )

    assert _selected_rule_combos(args.rule_combo) == ["iou0_area0_condarea0_merge0"]


def test_tracking_pipeline_normalizes_rule_combos() -> None:
    assert selected_rule_combos(
        ["iou0_area0_condarea0_merge0,iou1_area0_condarea0_merge0"]
    ) == ["iou0_area0_condarea0_merge0", "iou1_area0_condarea0_merge0"]
