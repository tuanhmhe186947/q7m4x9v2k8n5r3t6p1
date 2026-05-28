"""Central configuration for the pig behavior classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = DATA_DIR / "annotations"

CSV_PATH = PROCESSED_DATA_DIR / "behavior_with_feats_rectROI.csv"
IMAGES_DIR = RAW_DATA_DIR / "images_clean"
COCO_ANNOTATIONS = ANNOTATION_DIR / "scene_objects.coco.json"
BACKGROUND_PATH = ANNOTATION_DIR / "background.png"
MASK_PATH = ANNOTATION_DIR / "mask.png"
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "weights.pt"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EXPORT_DIR = OUTPUT_DIR / "export"
LOG_DIR = OUTPUT_DIR / "logs"

FINE_LABELS: list[str] = [
    "lying",
    "eat",
    "drink",
    "explore",
    "sitting",
    "stand",
    "social-nose",
    "playwithtoy",
]

COARSE_LABELS: list[str] = [
    "resting",
    "feeding",
    "locomotion",
    "social",
]

TABULAR_FEATURES: list[str] = [
    "in_feeder",
    "in_drinker",
    "in_toy",
    "speed_feat",
    "min_dist_other",
    "num_close_other",
]


@dataclass(slots=True)
class TrainConfig:
    """Training, export, and inference settings."""

    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    test_split: float = 0.2
    val_split: float = 0.1
    random_seed: int = 42
    use_coarse_labels: bool = False

    rotation_range: float = 30.0
    horizontal_flip: bool = True
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    zoom_range: float = 0.15

    backbone: str = "MobileNetV3Small"
    use_hybrid: bool = True
    dropout_rate: float = 0.3
    dense_units: int = 128
    tabular_dense_units: int = 32

    learning_rate: float = 1e-4
    fine_tune_lr: float = 1e-5
    fine_tune_at_layer: int = -20
    phase1_epochs: int = 10
    phase2_epochs: int = 20
    patience: int = 5

    quantize: bool = True
    export_onnx: bool = False
    dry_run: bool = False
    csv_path: Path = CSV_PATH
    images_dir: Path = IMAGES_DIR

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def label_column(self) -> str:
        return "behavior_coarse" if self.use_coarse_labels else "behavior"

    @property
    def labels(self) -> list[str]:
        return COARSE_LABELS if self.use_coarse_labels else FINE_LABELS


def ensure_output_dirs() -> None:
    """Create all runtime output directories."""
    for path in (OUTPUT_DIR, CHECKPOINT_DIR, EXPORT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
