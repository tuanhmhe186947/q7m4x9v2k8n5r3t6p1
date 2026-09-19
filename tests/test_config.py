import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _load_config_module():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from pig_behavior import config

    return config


config = _load_config_module()
COARSE_LABELS = config.COARSE_LABELS
BEHAVIOR_CLASSIFIER_WEIGHTS = config.BEHAVIOR_CLASSIFIER_WEIGHTS
BEHAVIOR_SEQUENCE_FEATURES = config.BEHAVIOR_SEQUENCE_FEATURES
BEHAVIOR_SEQUENCE_LABELS = config.BEHAVIOR_SEQUENCE_LABELS
BEHAVIOR_SEQUENCE_LENGTH = config.BEHAVIOR_SEQUENCE_LENGTH
BEHAVIOR_SEQUENCE_OFFSETS = config.BEHAVIOR_SEQUENCE_OFFSETS
BEHAVIOR_SEQUENCE_STRIDE_FRAMES = config.BEHAVIOR_SEQUENCE_STRIDE_FRAMES
CONFIG_PROJECT_ROOT = config.PROJECT_ROOT
DEFAULT_DETECTOR_MODEL = config.DEFAULT_DETECTOR_MODEL
FINE_LABELS = config.FINE_LABELS
TABULAR_FEATURES = config.TABULAR_FEATURES
TrainConfig = config.TrainConfig
YOLO_WEIGHTS = config.YOLO_WEIGHTS


def test_project_root_points_to_repository_root() -> None:
    assert CONFIG_PROJECT_ROOT == PROJECT_ROOT


def test_label_sets_match_configured_class_counts() -> None:
    fine_cfg = TrainConfig(use_coarse_labels=False)
    coarse_cfg = TrainConfig(use_coarse_labels=True)

    assert fine_cfg.labels == FINE_LABELS
    assert fine_cfg.num_classes == 8
    assert coarse_cfg.labels == COARSE_LABELS
    assert coarse_cfg.num_classes == 4


def test_tabular_feature_contract_is_stable() -> None:
    assert TABULAR_FEATURES == [
        "in_feeder",
        "in_drinker",
        "in_toy",
        "speed_feat",
        "min_dist_other",
        "num_close_other",
    ]


def test_pt_model_roles_are_separate() -> None:
    assert BEHAVIOR_CLASSIFIER_WEIGHTS.name == "pig_behavior_sequence.pt"
    assert "behavior" in BEHAVIOR_CLASSIFIER_WEIGHTS.parts
    assert DEFAULT_DETECTOR_MODEL == YOLO_WEIGHTS
    assert DEFAULT_DETECTOR_MODEL.name == "pig_detector_yolo.pt"
    assert "detector" in DEFAULT_DETECTOR_MODEL.parts


def test_behavior_sequence_contract_matches_training_notebook() -> None:
    assert BEHAVIOR_SEQUENCE_LENGTH == 6
    assert BEHAVIOR_SEQUENCE_STRIDE_FRAMES == 3
    assert BEHAVIOR_SEQUENCE_OFFSETS == (-3, -2, -1, 0, 1, 2)
    assert BEHAVIOR_SEQUENCE_LABELS == [
        "drink",
        "eat",
        "fight",
        "social-nose",
        "explore",
        "lying",
        "stand",
        "move",
        "sitting",
        "playwithtoy",
    ]
    assert BEHAVIOR_SEQUENCE_FEATURES == [
        "cx_n",
        "cy_n",
        "bw_n",
        "bh_n",
        "speed_feat",
        "min_dist_other",
        "num_close_other",
        "in_feeder",
        "in_drinker",
        "in_toy",
    ]
