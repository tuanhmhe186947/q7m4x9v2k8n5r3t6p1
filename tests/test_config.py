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
CONFIG_PROJECT_ROOT = config.PROJECT_ROOT
FINE_LABELS = config.FINE_LABELS
TABULAR_FEATURES = config.TABULAR_FEATURES
TrainConfig = config.TrainConfig


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
