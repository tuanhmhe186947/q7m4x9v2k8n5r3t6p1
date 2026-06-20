"""Central configuration for the pig behavior project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = DATA_DIR / "annotations"
SCENE_ANNOTATION_DIR = ANNOTATION_DIR / "scene"
ROI_ANNOTATION_DIR = ANNOTATION_DIR / "roi"
SCHEMA_ANNOTATION_DIR = ANNOTATION_DIR / "schemas"
TRACKING_ANNOTATION_DIR = ANNOTATION_DIR / "tracking"
CLASSIFICATION_ANNOTATION_DIR = ANNOTATION_DIR / "classification"
VIDEO_DIR = DATA_DIR / "videos"

MODEL_DIR = PROJECT_ROOT / "models"
BEHAVIOR_MODEL_DIR = MODEL_DIR / "behavior"
DETECTOR_MODEL_DIR = MODEL_DIR / "detector"

CLASSIFICATION_PROCESSED_DIR = PROCESSED_DATA_DIR / "classification"
BEHAVIOR_CLEAN_CSV_NAME = "behavior_clean_merged.csv"
BEHAVIOR_FEATURE_CSV_NAME = "behavior_with_feats_rectROI.csv"
IMAGES_DIR = RAW_DATA_DIR / "images_clean"
ROI_COCO_ANNOTATIONS = ROI_ANNOTATION_DIR / "ROI_annotations.coco.json"
CLASSIFICATION_COCO_ANNOTATIONS = (
    CLASSIFICATION_ANNOTATION_DIR / "pig_annotations.clean.coco.json"
)
SCENE_COCO_ANNOTATIONS = SCENE_ANNOTATION_DIR / "scene_objects.coco.json"
# Backward-compatible name used by the real-time tracking service for ROI boxes.
COCO_ANNOTATIONS = ROI_COCO_ANNOTATIONS
BACKGROUND_PATH = SCENE_ANNOTATION_DIR / "background.png"
MASK_PATH = SCENE_ANNOTATION_DIR / "mask.png"
YOLO_WEIGHTS = DETECTOR_MODEL_DIR / "pig_detector_yolo.pt"
BEHAVIOR_CLASSIFIER_WEIGHTS = BEHAVIOR_MODEL_DIR / "pig_behavior_sequence.pt"
DETECTION_TRACKING_WEIGHTS = YOLO_WEIGHTS
DEFAULT_PT_MODEL = BEHAVIOR_CLASSIFIER_WEIGHTS
DEFAULT_DETECTOR_MODEL = DETECTION_TRACKING_WEIGHTS
DEFAULT_VIDEO_PATH = VIDEO_DIR / "pigs101219_full.mp4"

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

BEHAVIOR_SEQUENCE_LABELS: list[str] = [
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

BEHAVIOR_SEQUENCE_FEATURES: list[str] = [
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

BEHAVIOR_SEQUENCE_LENGTH = 6
BEHAVIOR_SEQUENCE_STRIDE_FRAMES = 3
BEHAVIOR_SEQUENCE_OFFSETS: tuple[int, ...] = (-3, -2, -1, 0, 1, 2)


def latest_classification_run_dir(
    processed_dir: Path = CLASSIFICATION_PROCESSED_DIR,
) -> Path | None:
    """Return the newest timestamped classification dataset directory."""
    if not processed_dir.exists():
        return None
    candidates = [
        path
        for path in processed_dir.iterdir()
        if path.is_dir() and (path / BEHAVIOR_FEATURE_CSV_NAME).exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name, reverse=True)[0]


def latest_classification_csv(
    csv_name: str = BEHAVIOR_FEATURE_CSV_NAME,
    processed_dir: Path = CLASSIFICATION_PROCESSED_DIR,
) -> Path:
    """Resolve the newest timestamped classification CSV with legacy fallback."""
    if processed_dir.exists():
        candidates = [
            path / csv_name
            for path in processed_dir.iterdir()
            if path.is_dir() and (path / csv_name).exists()
        ]
        if candidates:
            return sorted(
                candidates,
                key=lambda path: path.parent.name,
                reverse=True,
            )[0]
    return PROCESSED_DATA_DIR / csv_name


CSV_PATH = latest_classification_csv()


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
    csv_path: Path = field(default_factory=latest_classification_csv)
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
